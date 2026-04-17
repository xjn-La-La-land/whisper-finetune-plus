# finetune_controller.py
import os
import asyncio
import json
import sqlite3
import time
import sys
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from shared_state import GPUStatus, GPU_STATE

router = APIRouter()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tasks.db")
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, "whisper-base-models")
DOWNLOAD_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "download_whisper_models.py")
SUPPORTED_BASE_MODELS = [
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "whisper-base",
    "whisper-tiny",
    "whisper-small",
    "whisper-medium",
]
BASE_MODEL_DOWNLOAD_STATE = {
    model_name: {"status": "idle", "message": ""}
    for model_name in SUPPORTED_BASE_MODELS
}


def _is_user_training(username: str) -> bool:
    """兼容新的 GPU_STATE 结构：通过 status + current_user 判断训练状态。"""
    return (
        GPU_STATE.get("status") == GPUStatus.TRAINING
        and GPU_STATE.get("current_user") == username
    )


def _upsert_user_model(username: str, model_name: str):
    now_ts = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''
        INSERT INTO models (username, model_name, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(username, model_name)
        DO UPDATE SET updated_at = excluded.updated_at
        ''',
        (username, model_name, now_ts, now_ts)
    )
    conn.commit()
    conn.close()


def _model_name_exists(username: str, model_name: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT 1 FROM models WHERE username = ? AND model_name = ? LIMIT 1',
        (username, model_name)
    )
    exists = c.fetchone() is not None
    conn.close()
    return exists

class FinetuneRequest(BaseModel):
    username: str
    model_name: str = Field(..., min_length=1, max_length=80, description="用户自定义模型名")
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
    base_model: str = Field(..., min_length=1, description="Whisper 基础模型目录名")

class PublishModelRequest(BaseModel):
    username: str
    model_name: str


def _list_base_model_dirs():
    if not os.path.isdir(BASE_MODELS_DIR):
        return []
    models = []
    for entry in os.scandir(BASE_MODELS_DIR):
        if entry.is_dir():
            models.append(entry.name)
    return sorted(models)


def _build_base_model_payload():
    local_models = set(_list_base_model_dirs())
    all_models = []
    for model_name in SUPPORTED_BASE_MODELS:
        state = BASE_MODEL_DOWNLOAD_STATE.get(model_name, {"status": "idle", "message": ""})
        all_models.append({
            "name": model_name,
            "is_downloaded": model_name in local_models,
            "download_status": state.get("status", "idle"),
            "download_message": state.get("message", ""),
        })

    return {
        "base_models_dir": BASE_MODELS_DIR,
        "models": sorted(local_models),
        "all_models": all_models,
    }


async def _run_base_model_download(model_name: str):
    cmd = ["python", DOWNLOAD_SCRIPT_PATH, model_name]
    BASE_MODEL_DOWNLOAD_STATE[model_name] = {
        "status": "downloading",
        "message": "正在下载模型文件，请稍候...",
    }

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            BASE_MODEL_DOWNLOAD_STATE[model_name] = {
                "status": "completed",
                "message": "模型下载完成",
            }
            print(f"[base-model-download] {model_name} 下载完成")
        else:
            error_message = stderr.decode("utf-8", errors="ignore").strip() or stdout.decode("utf-8", errors="ignore").strip()
            BASE_MODEL_DOWNLOAD_STATE[model_name] = {
                "status": "failed",
                "message": error_message or "模型下载失败",
            }
            print(f"[base-model-download] {model_name} 下载失败: {error_message}")
    except Exception as e:
        BASE_MODEL_DOWNLOAD_STATE[model_name] = {
            "status": "failed",
            "message": str(e),
        }
        print(f"[base-model-download] {model_name} 下载异常: {e}")


async def run_finetune_process(req: FinetuneRequest):
    """
    这是一个后台任务，负责实际拉起 finetune.py 进程并等待它结束。
    """
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset", req.username)
    output_dir = os.path.join(PROJECT_ROOT, "output", req.username, req.model_name)
    web_log_path = os.path.join(dataset_dir, "training_log.jsonl")
    base_model_path = os.path.join(BASE_MODELS_DIR, req.base_model)

    # 先在后端侧主动清空旧日志，避免 SSE 先读到旧文件后被训练进程 truncate 导致读指针卡在 EOF
    with open(web_log_path, "w", encoding="utf-8"):
        pass

    # 构造执行命令
    cmd = [
        "python", "finetune.py",
        f"--train_data={os.path.join(dataset_dir, 'train.json')}",
        f"--test_data={os.path.join(dataset_dir, 'test.json')}",
        f"--output_dir={output_dir}",
        f"--web_log_path={web_log_path}",
        f"--base_model={base_model_path}",
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
            if os.path.exists(output_dir):
                _upsert_user_model(req.username, req.model_name)
                
                # --- 新增：自动导出 TFLite ---
                try:
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                    from tflite_export import run_tflite_export
                    tflite_output = os.path.join(output_dir, "whisper_model.tflite")
                    checkpoint_final = os.path.join(output_dir, "checkpoint-final")
                    
                    print(f"[{req.username}] 正在自动将模型转换为 TFLite 格式...")
                    # 调用之前封装好的函数
                    success, msg = run_tflite_export(
                        base_model_path=base_model_path,
                        checkpoint_path=checkpoint_final,
                        output_tflite_path=tflite_output
                    )
                    if success:
                        print(f"[{req.username}] TFLite 自动导出完成：{tflite_output}")
                    else:
                        print(f"[{req.username}] TFLite 自动导出失败：{msg}")
                except Exception as ex:
                    print(f"[{req.username}] TFLite 导出过程中出现异常: {ex}")
                # -----------------------------
            else:
                print(f"[{req.username}] 警告：未找到模型输出目录，无法写入模型记录")
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
    model_name = req.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="模型名称不能为空")
    if _model_name_exists(req.username, model_name):
        raise HTTPException(status_code=400, detail="模型名称已存在，请换一个名字")
    req.model_name = model_name
    req.base_model = req.base_model.strip()
    if not req.base_model:
        raise HTTPException(status_code=400, detail="请先选择基础模型")
    available_base_models = set(_list_base_model_dirs())
    if req.base_model not in available_base_models:
        raise HTTPException(status_code=400, detail=f"基础模型不存在: {req.base_model}")

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


@router.get("/api/base_models")
async def get_base_models():
    return _build_base_model_payload()


class BaseModelDownloadRequest(BaseModel):
    model_name: str = Field(..., min_length=1, description="要下载的基础模型名称")


@router.post("/api/base_models/download")
async def download_base_model(req: BaseModelDownloadRequest):
    model_name = req.model_name.strip()
    if model_name not in SUPPORTED_BASE_MODELS:
        raise HTTPException(status_code=400, detail=f"不支持的基础模型: {model_name}")

    local_models = set(_list_base_model_dirs())
    if model_name in local_models:
        BASE_MODEL_DOWNLOAD_STATE[model_name] = {
            "status": "completed",
            "message": "模型已存在于本地目录",
        }
        return {"message": f"{model_name} 已存在，无需重复下载"}

    current_state = BASE_MODEL_DOWNLOAD_STATE.get(model_name, {})
    if current_state.get("status") == "downloading":
        raise HTTPException(status_code=409, detail=f"{model_name} 正在下载中，请稍后刷新")

    asyncio.create_task(_run_base_model_download(model_name))
    return {"message": f"{model_name} 下载任务已启动"}

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


@router.get("/api/train_history")
async def get_train_history(username: str):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(PROJECT_ROOT, "dataset", username, "training_log.jsonl")

    if not os.path.exists(log_path):
        return {"train_loss": [], "eval_loss": []}

    train_loss = []
    eval_loss = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except Exception:
                continue

            if data.get("step") is not None and data.get("loss") is not None:
                train_loss.append([data["step"], data["loss"]])

            if data.get("step") is not None and data.get("eval_loss") is not None:
                eval_loss.append([data["step"], data["eval_loss"]])

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss
    }


@router.post("/api/publish_model")
async def publish_model(req: PublishModelRequest):
    # 检查 TFLite 模型文件是否存在
    tflite_path = os.path.join(PROJECT_ROOT, "output", req.username, req.model_name, "whisper_model.tflite")
    if not os.path.exists(tflite_path):
        raise HTTPException(status_code=400, detail="该模型尚未生成 TFLite 文件，无法发布")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. 将该用户的所有模型设为未发布
    c.execute('UPDATE models SET is_published = 0 WHERE username = ?', (req.username,))
    
    # 2. 将选中的模型设为发布状态，并更新版本号 (以时间戳为例)
    version_tag = str(int(time.time()))
    c.execute(
        'UPDATE models SET is_published = 1, version_tag = ? WHERE username = ? AND model_name = ?',
        (version_tag, req.username, req.model_name)
    )
    
    if c.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="未找到对应的模型记录")
        
    conn.commit()
    conn.close()
    
    return {"message": "模型发布成功", "version_tag": version_tag}


# 模型检查
@router.get("/api/check_model")
async def check_model(username: str):
    """前端轮询或初始化时调用，检查该用户是否已经有训练好的模型"""
    user_output_dir = os.path.join(PROJECT_ROOT, "output", username)
    if not os.path.isdir(user_output_dir):
        return {"has_model": False}
    has_content = any(os.scandir(user_output_dir))
    return {"has_model": has_content}
