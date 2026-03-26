# finetune_controller.py
import os
import asyncio
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
import re
from shared_state import GPUStatus, GPU_STATE

router = APIRouter()


def _is_user_training(username: str) -> bool:
    """兼容新的 GPU_STATE 结构：通过 status + current_user 判断训练状态。"""
    return (
        GPU_STATE.get("status") == GPUStatus.TRAINING
        and GPU_STATE.get("current_user") == username
    )


def _resolve_user_model_path(username: str) -> Optional[str]:
    """
    查找用户微调完成后的 checkpoint-final 目录。
    目录结构：output/{username}/{base_model_name}/checkpoint-final
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    user_output_dir = os.path.join(project_root, "output", username)

    direct_path = os.path.join(user_output_dir, "checkpoint-final")
    if os.path.exists(direct_path):
        return direct_path

    if not os.path.isdir(user_output_dir):
        return None

    for entry in os.listdir(user_output_dir):
        candidate = os.path.join(user_output_dir, entry, "checkpoint-final")
        if os.path.exists(candidate):
            return candidate
    return None

class FinetuneRequest(BaseModel):
    username: str
    learning_rate: float = Field(default=2e-4, description="学习率")
    epochs: int = Field(default=20, ge=1, le=200, description="训练轮数")
    batch_size: int = Field(default=8, ge=1, le=64, description="批次大小")
    accumulation_steps: int = Field(default=1, ge=1, description="梯度累积步数")
    warmup_steps: int = Field(default=20, ge=0, description="预热步数")
    use_8bit: bool = Field(default=False, description="是否开启 8bit 量化 (省显存)")
    use_adalora: bool = Field(default=False, description="是否使用 AdaLora")
    fp16: bool = Field(default=True, description="是否使用混合精度训练")
    min_audio_len: float = Field(default=0.5, ge=0.1, description="最小音频长度(秒)")
    max_audio_len: float = Field(default=30.0, le=30.0, description="最大音频长度(秒)")


async def run_finetune_process(req: FinetuneRequest):
    """
    这是一个后台任务，负责实际拉起 finetune.py 进程并等待它结束。
    """
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset", req.username)
    output_dir = os.path.join(PROJECT_ROOT, "output", req.username)
    web_log_path = os.path.join(dataset_dir, "training_log.jsonl")

    # 先在后端侧主动清空旧日志，避免 SSE 先读到旧文件后被训练进程 truncate 导致读指针卡在 EOF
    os.makedirs(dataset_dir, exist_ok=True)
    with open(web_log_path, "w", encoding="utf-8"):
        pass

    # 构造执行命令
    cmd = [
        "python", "finetune.py",
        f"--train_data={os.path.join(dataset_dir, 'train.json')}",
        f"--test_data={os.path.join(dataset_dir, 'test.json')}",
        f"--output_dir={output_dir}",
        f"--web_log_path={web_log_path}",
        # 动态接收前端传来的值
        f"--warmup_steps={req.warmup_steps}",
        f"--learning_rate={req.learning_rate}",
        f"--min_audio_len={req.min_audio_len}",
        f"--max_audio_len={req.max_audio_len}",
        
        f"--use_adalora={str(req.use_adalora)}",
        f"--use_8bit={str(req.use_8bit)}",
        f"--fp16={str(req.fp16)}",

        f"--num_train_epochs={req.epochs}",
        f"--per_device_train_batch_size={req.batch_size}",
        f"--per_device_eval_batch_size={req.batch_size}",
        f"--gradient_accumulation_steps={req.accumulation_steps}"
    ]

    try:
        # 使用 asyncio.create_subprocess_exec 异步拉起进程，不阻塞主线程
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # 等待微调进程结束 (这里可能需要几十分钟到几个小时)
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"[{req.username}] 微调成功！")
        else:
            print(f"[{req.username}] 报错：{stderr.decode()}")
    except Exception as e:
        print(f"执行微调脚本出错: {e}")
    finally:
        # --- 释放 GPU 锁 ---
        # 无论成功、失败还是崩溃，必须确保锁被释放！
        GPU_STATE["status"] = GPUStatus.IDLE
        GPU_STATE["current_user"] = None


@router.post("/api/start_finetune")
async def start_finetune(req: FinetuneRequest, background_tasks: BackgroundTasks):
    # --- 检查 GPU 锁 ---
    if GPU_STATE["status"] != GPUStatus.IDLE:
        raise HTTPException(
            status_code=423, 
            detail=f"当前 GPU 正处于 {GPU_STATE['status']} 状态，被用户 [{GPU_STATE['current_user']}] 占用，请稍后再试！"
        )
    
    # 获取到了 GPU 空闲状态，上锁
    GPU_STATE["status"] = GPUStatus.TRAINING
    GPU_STATE["current_user"] = req.username
    
    # 丢给后台任务去执行，FastAPI 接口立刻向前端返回成功
    background_tasks.add_task(run_finetune_process, req)
    
    return {"message": "微调任务已在后台启动！"}

@router.get("/api/gpu_status")
async def get_gpu_status():
    """让前端可以查询当前是否有人在训练"""
    return GPU_STATE


async def log_generator(username: str):
    """
    异步生成器：持续监控 training_log.jsonl，实现类似 tail -f 的效果。
    """
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(PROJECT_ROOT, "dataset", username, "training_log.jsonl")

    # 1. 等待日志文件生成
    # Hugging Face Trainer 需要跑完第一个 logging_steps 才会创建文件
    wait_time = 0
    while not os.path.exists(log_path):
        # 如果等待期间训练已经意外终止，直接退出
        if not _is_user_training(username):
            yield f"data: {{\"status\": \"error\", \"message\": \"训练未启动或提前异常终止\"}}\n\n"
            return
            
        if wait_time > 120:  # 设置 2 分钟超时
            yield f"data: {{\"status\": \"error\", \"message\": \"等待日志文件超时\"}}\n\n"
            return
            
        await asyncio.sleep(1)
        wait_time += 1

    # 2. 文件存在，开始持续读取
    with open(log_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                # 如果文件被 truncate（例如新训练刚启动清空日志），读指针可能落在文件末尾之外
                # 需要把指针拉回文件开头，才能继续读到新写入的数据
                current_pos = f.tell()
                try:
                    file_size = os.path.getsize(log_path)
                except OSError:
                    file_size = current_pos
                if file_size < current_pos:
                    f.seek(0)

                # 读到了文件末尾（EOF）
                # 检查当前用户的训练任务是否还在继续
                if not _is_user_training(username):
                    # 训练已经结束，发送完成信号跳出循环
                    yield f"data: {{\"status\": \"finished\", \"message\": \"训练已完成\"}}\n\n"
                    break

                # 训练还在进行，只是当前没有新日志，稍微让出 CPU 稍后再试
                await asyncio.sleep(0.5)
                continue

            # 读到了新的一行，按照 SSE 规范格式化并 yield 出去
            # SSE 规范要求以 "data: " 开头，以 "\n\n" 结尾
            try:
                parsed = json.loads(line.strip())
                normalized = json.dumps(parsed, ensure_ascii=False, allow_nan=False)
                yield f"data: {normalized}\n\n"
            except (json.JSONDecodeError, ValueError):
                # 跳过损坏行或包含非法 JSON 数值的历史日志，避免前端解析中断
                continue


@router.get("/api/train_stream")
async def train_stream(username: str):
    """
    前端通过 EventSource 调用的 SSE 流式接口。
    """
    # 返回 StreamingResponse，指定媒体类型为 text/event-stream
    return StreamingResponse(
        log_generator(username),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 如果前面有 Nginx，这个头可以关闭代理缓冲，避免 SSE 被攒包后才下发
            "X-Accel-Buffering": "no"
        }
    )


# 模型检查与评估模块
class EvaluateRequest(BaseModel):
    username: str
    batch_size: int = 8


@router.get("/api/check_model")
async def check_model(username: str):
    """前端轮询或初始化时调用，检查该用户是否已经有训练好的模型"""

    model_path = _resolve_user_model_path(username)
    return {"has_model": model_path is not None}


@router.post("/api/evaluate")
async def evaluate_model(req: EvaluateRequest):
    """拉起 evaluation.py 并截获结果 (因为测试集较小，这里直接使用 await 阻塞等待结果返回前端)"""
    if GPU_STATE["status"] != GPUStatus.IDLE:
        raise HTTPException(
            status_code=423, 
            detail=f"GPU 正忙 ({GPU_STATE['status']})，被 [{GPU_STATE['current_user']}] 占用"
        )
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset", req.username)
    model_path = _resolve_user_model_path(req.username)
    
    if not model_path:
        raise HTTPException(status_code=400, detail="未找到您的专属模型，请先完成微调训练！")

    GPU_STATE["status"] = GPUStatus.EVALUATING
    GPU_STATE["current_user"] = req.username

    cmd = [
        "python", "evaluation.py",
        f"--test_data={os.path.join(dataset_dir, 'test.json')}",
        f"--model_path={model_path}",
        f"--batch_size={req.batch_size}"
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        output_str = stdout.decode('utf-8')
        
        if process.returncode != 0:
            print(f"评估报错: {stderr.decode('utf-8')}")
            raise HTTPException(status_code=500, detail="模型评估过程中发生错误")

        # 使用正则截获终端打印的 CER 结果：例如 "评估结果：cer=0.12345"
        cer_match = re.search(r"评估结果：cer=([\d\.]+)", output_str)
        if cer_match:
            cer_value = float(cer_match.group(1))
            return {"message": "评估完成", "cer": cer_value}
        else:
            raise HTTPException(status_code=500, detail="未能成功解析评估结果")

    finally:
        # 执行完毕，释放 GPU
        GPU_STATE["status"] = GPUStatus.IDLE
        GPU_STATE["current_user"] = None
