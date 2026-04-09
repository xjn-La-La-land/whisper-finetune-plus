# inference_controller.py
import os
import gc
import json
import sqlite3
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from zhconv import convert
from utils.data_utils import remove_punctuation

from shared_state import GPUStatus, GPU_STATE, INFERENCE_CACHE

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, "whisper-base-models")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tasks.db")

DEFAULT_BATCH_SIZE = 16


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


def resolve_user_model_path(username: str):
    """从数据库查询用户所有可用微调模型。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        '''
        SELECT model_name
        FROM models
        WHERE username = ?
        ORDER BY updated_at DESC, id DESC
        ''',
        (username,)
    )
    rows = [dict(row) for row in c.fetchall()]
    conn.close()

    valid_models = []
    for row in rows:
        model_path = os.path.join(PROJECT_ROOT, "output", username, row["model_name"], "checkpoint-final")
        if os.path.exists(model_path):
            valid_models.append({
                "model_name": row["model_name"],
                "model_path": model_path
            })
    return valid_models


@router.get("/api/user_models")
async def get_user_models(username: str):
    return {"models": resolve_user_model_path(username)}


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
        use_flash_attention_2=True
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
    username: str = Form(...),
    to_simple: int = Form(1),
    remove_pun: int = Form(0),
    num_beams: int = Form(1),
    model_name: str = Form("whisper-large-v3"),
    audio: UploadFile = File(...)
):
    # 1. 检查 GPU 状态 (如果是训练，直接拒绝)
    if GPU_STATE["status"] == GPUStatus.TRAINING:
        raise HTTPException(status_code=423, detail="GPU 正在进行训练任务，请稍后再试！")
    
    # 2. 确定目标模型路径（支持基础模型与用户微调模型）
    user_models = resolve_user_model_path(username)
    user_model_by_name = {m["model_name"]: m for m in user_models}
    base_models = resolve_base_model_paths()
    selected_model = None
    selected_type = None

    if model_name in user_model_by_name:
        selected_model = user_model_by_name[model_name]
        target_model_path = selected_model["model_path"]
        selected_type = "finetuned"
    elif model_name in base_models:
        target_model_path = base_models[model_name]
        selected_type = "base"
    else:
        raise HTTPException(status_code=400, detail="未找到指定模型，请重新选择")
    
    # 3. 动态热插拔：如果显存里的模型不是我们想要的，就重新加载
    if INFERENCE_CACHE["loaded_model_path"] != target_model_path:
        # 修改状态，防止加载时被别人打断
        GPU_STATE["status"] = GPUStatus.INFERENCING
        GPU_STATE["current_user"] = username
        try:
            load_model_to_gpu(target_model_path)
        except Exception as e:
            GPU_STATE["status"] = GPUStatus.IDLE
            GPU_STATE["current_user"] = None
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

    # 4. 执行推理
    GPU_STATE["status"] = GPUStatus.INFERENCING
    GPU_STATE["current_user"] = username
    try:
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
        # 推理结束后释放 GPU 锁（仅释放状态，不卸载模型）
        GPU_STATE["status"] = GPUStatus.IDLE
        GPU_STATE["current_user"] = None
