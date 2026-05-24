# inference_controller.py
import os
import gc
import json
import sqlite3
import torch
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from zhconv import convert
from utils.data_utils import remove_punctuation
from utils.db import get_db
from utils.auth import get_current_user
from utils.path_safety import OUTPUT_BASE, safe_join, safe_resolve_under

from shared_state import GPUStatus, GPU_STATE, GPU_LOCK, INFERENCE_CACHE

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, "whisper-base-models")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tasks.db")

DEFAULT_BATCH_SIZE = 16


def _detect_attn_implementation() -> str:
    """根据环境自动选择最优的 attention 实现。

    优先级：
      1. flash_attention_2 —— 最快，但需要 `pip install flash-attn`（建议 GPU 是
         Ampere/Ada/Hopper，对 Blackwell 等新架构需要 flash-attn 2.7+）
      2. sdpa —— PyTorch 2.x 内置的 scaled_dot_product_attention，所有主流
         GPU 都原生支持，性能比朴素 eager 快很多，比 flash-attn 慢 10~20%
      3. eager —— 朴素实现，仅作兜底

    选 SDPA 是默认安全路径：零依赖、跨架构、跨 CUDA 版本都能跑。
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# 模块加载时确定一次即可——切换需要重启服务
ATTN_IMPLEMENTATION = _detect_attn_implementation()
print(f"[inference] 使用 attention 实现: {ATTN_IMPLEMENTATION}")


def resolve_lora_base_model_path(model_path: str):
    """如果是 LoRA 目录，返回其 base_model_name_or_path；否则返回 None。"""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return None

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    return adapter_config.get("base_model_name_or_path")


def resolve_base_model_paths():
    """扫描基础模型目录，返回 {model_name: model_path}。"""
    if not os.path.isdir(BASE_MODELS_DIR):
        return {}
    models = {}
    for entry in os.scandir(BASE_MODELS_DIR):
        if entry.is_dir():
            models[entry.name] = entry.path
    return models


async def resolve_user_model_path(username: str):
    """从数据库查询用户所有可用微调模型，并返回发布状态。"""
    async with get_db(row_factory=sqlite3.Row) as conn:
        cursor = await conn.execute(
            '''
            SELECT model_name, is_published, version_tag
            FROM models
            WHERE username = ?
            ORDER BY updated_at DESC, id DESC
            ''',
            (username,)
        )
        rows = [dict(row) for row in await cursor.fetchall()]

    valid_models = []
    user_output_root = safe_join(OUTPUT_BASE, username)
    for row in rows:
        # 历史 DB 行的 model_name 可能没经过 MODEL_NAME_RE 校验，跳过非法行
        # 而不是让一条脏数据让整个 list_models 接口 400
        model_dir = safe_resolve_under(user_output_root, row["model_name"])
        if not model_dir:
            print(f"[user_models] 跳过非法 model_name: {row['model_name']}")
            continue

        checkpoint_path = safe_join(model_dir, "checkpoint-final")
        tflite_path = safe_join(model_dir, "whisper_model.tflite")

        if os.path.exists(checkpoint_path):
            valid_models.append({
                "model_name": row["model_name"],
                "model_path": checkpoint_path,
                "is_published": bool(row["is_published"]),
                "version_tag": row["version_tag"],
                "has_tflite": os.path.exists(tflite_path)
            })
    return valid_models


@router.get("/api/user_models")
async def get_user_models(current_user: str = Depends(get_current_user)):
    return {"models": await resolve_user_model_path(current_user)}


@router.get("/api/latest_model_info")
async def get_latest_model_info(current_user: str = Depends(get_current_user)):
    """供客户端轮询：获取当前发布的最新模型信息"""
    async with get_db() as conn:
        cursor = await conn.execute(
            'SELECT model_name, version_tag FROM models WHERE username = ? AND is_published = 1 LIMIT 1',
            (current_user,)
        )
        row = await cursor.fetchone()

    if not row:
        return {"has_published": False}

    model_name, version_tag = row
    user_output_root = safe_join(OUTPUT_BASE, current_user)
    tflite_path = safe_resolve_under(user_output_root, os.path.join(model_name, "whisper_model.tflite"))
    if not tflite_path or not os.path.exists(tflite_path):
        return {"has_published": False}
        
    file_size = os.path.getsize(tflite_path)

    return {
        "has_published": True,
        "model_name": model_name,
        "version_tag": version_tag,
        "file_size": file_size
    }


@router.get("/api/download_published_model")
async def download_published_model(current_user: str = Depends(get_current_user)):
    """供客户端调用：下载已发布的 TFLite 模型文件"""
    async with get_db() as conn:
        cursor = await conn.execute(
            'SELECT model_name FROM models WHERE username = ? AND is_published = 1 LIMIT 1',
            (current_user,)
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="该用户尚未发布任何模型")

    model_name = row[0]
    user_output_root = safe_join(OUTPUT_BASE, current_user)
    tflite_path = safe_resolve_under(user_output_root, os.path.join(model_name, "whisper_model.tflite"))
    if not tflite_path or not os.path.exists(tflite_path):
        raise HTTPException(status_code=404, detail="已发布的 TFLite 模型文件不存在")

    return FileResponse(
        path=tflite_path,
        filename=f"whisper_{model_name}.tflite",
        media_type="application/octet-stream"
    )


def load_model_to_gpu(model_path: str):
    """清理显存并加载新模型"""
    print(f"正在从显存卸载旧模型，准备加载: {model_path}")
    
    # 1. 彻底清空旧模型
    if INFERENCE_CACHE["pipeline"] is not None:
        del INFERENCE_CACHE["pipeline"].model
        del INFERENCE_CACHE["pipeline"].feature_extractor
        del INFERENCE_CACHE["pipeline"].tokenizer
        del INFERENCE_CACHE["pipeline"]
        INFERENCE_CACHE["pipeline"] = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. 加载新模型
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_path 既可能是基础模型路径，也可能是 LoRA 权重目录
    lora_model_path = None
    base_model_path = model_path
    resolved_base_model_path = resolve_lora_base_model_path(model_path)
    if resolved_base_model_path:
        lora_model_path = model_path
        base_model_path = resolved_base_model_path

    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        # 新版 transformers 用 attn_implementation 取代 use_flash_attention_2 旧参数
        # ATTN_IMPLEMENTATION 在模块加载时自动检测：有 flash_attn 就用 flash_attention_2，
        # 否则 fallback 到 PyTorch 内置的 sdpa（零依赖、跨架构）
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    if lora_model_path:
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    if hasattr(processor, "get_decoder_prompt_ids"):
        model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="zh",
            task="transcribe"
        )
    model.generation_config.suppress_tokens = []

    model.to(device)

    infer_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=DEFAULT_BATCH_SIZE,
        torch_dtype=torch_dtype,
        device=device
    )
    
    INFERENCE_CACHE["pipeline"] = infer_pipe
    INFERENCE_CACHE["loaded_model_path"] = model_path
    print("模型加载完成！")


@router.post("/api/recognition")
async def api_recognition(
    to_simple: int = Form(1),
    remove_pun: int = Form(0),
    num_beams: int = Form(1),
    model_name: str = Form("whisper-large-v3"),
    audio: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    username = current_user
    # 1. 解析目标模型路径（无需 GPU 锁，纯查表）
    user_models = await resolve_user_model_path(username)
    user_model_by_name = {m["model_name"]: m for m in user_models}
    base_models = resolve_base_model_paths()
    selected_type = None

    if model_name in user_model_by_name:
        target_model_path = user_model_by_name[model_name]["model_path"]
        selected_type = "finetuned"
    elif model_name in base_models:
        target_model_path = base_models[model_name]
        selected_type = "base"
    else:
        raise HTTPException(status_code=400, detail="未找到指定模型，请重新选择")

    # 2. 原子化检查 + 上锁：只在锁内做状态转移，加载/推理放到锁外执行
    async with GPU_LOCK:
        if GPU_STATE["status"] != GPUStatus.IDLE:
            raise HTTPException(
                status_code=423,
                detail=f"GPU 正被用户 [{GPU_STATE['current_user']}] {GPU_STATE['status']} 占用，请稍后再试！"
            )
        GPU_STATE["status"] = GPUStatus.INFERENCING
        GPU_STATE["current_user"] = username

    # 3. 锁外执行实际工作 (热插拔 + 推理)。无论成功失败都要在 finally 释放状态。
    try:
        # 动态热插拔：如果显存里的模型不是我们想要的，就重新加载
        if INFERENCE_CACHE["loaded_model_path"] != target_model_path:
            try:
                load_model_to_gpu(target_model_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

        data = await audio.read()
        generate_kwargs = {"task": "transcribe", "num_beams": num_beams, "language": "chinese"}

        result = INFERENCE_CACHE["pipeline"](data, return_timestamps=True, generate_kwargs=generate_kwargs)
        print("RAW RESULT =", result)

        results = []
        chunks = result.get("chunks", [])
        for chunk in chunks:
            text = chunk["text"]
            if to_simple == 1:
                text = convert(text, "zh-cn")
            if remove_pun == 1:
                text = remove_punctuation(text)
            results.append({
                "text": text,
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1]
            })

        return {
            "code": 0,
            "results": results,
            "used_model": model_name,
            "used_model_type": selected_type
        }
    finally:
        # 推理结束 / 异常退出，都要释放 GPU 状态（不卸载模型，仅置回 IDLE）
        async with GPU_LOCK:
            GPU_STATE["status"] = GPUStatus.IDLE
            GPU_STATE["current_user"] = None
