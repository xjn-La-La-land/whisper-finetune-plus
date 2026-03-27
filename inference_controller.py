# inference_controller.py
import os
import gc
import sqlite3
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from zhconv import convert
from utils.data_utils import remove_punctuation

from shared_state import GPUStatus, GPU_STATE, INFERENCE_CACHE

router = APIRouter()

# 基础模型路径
BASE_MODEL_PATH = "/home/featurize/whisper-large-v3"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tasks.db")

DEFAULT_BATCH_SIZE = 16


def resolve_user_model_path(username: str):
    """从数据库查询用户所有可用微调模型。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        '''
        SELECT model_name, model_path
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
        if os.path.exists(row["model_path"]):
            valid_models.append(row)
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

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        use_flash_attention_2=True # 如果显卡不支持可改为False
    )
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
    model_name: str = Form("BASE_MODEL"),
    audio: UploadFile = File(...)
):
    # 1. 检查 GPU 状态 (如果是训练，直接拒绝)
    if GPU_STATE["status"] == GPUStatus.TRAINING:
        raise HTTPException(status_code=423, detail="GPU 正在进行训练任务，请稍后再试！")
    
    # 2. 确定目标模型路径 (优先找该用户的微调模型，找不到用Base)
    user_models = resolve_user_model_path(username)
    selected_model = None
    if model_name != "BASE_MODEL":
        selected_model = next((m for m in user_models if m["model_name"] == model_name), None)
        if not selected_model:
            raise HTTPException(status_code=400, detail="未找到指定微调模型，请先重新选择模型")
        target_model_path = selected_model["model_path"]
    else:
        target_model_path = BASE_MODEL_PATH
    
    # 3. 动态热插拔：如果显存里的模型不是我们想要的，就重新加载
    if INFERENCE_CACHE["loaded_model_path"] != target_model_path:
        # 修改状态，防止加载时被别人打断
        GPU_STATE["status"] = GPUStatus.INFERENCING
        GPU_STATE["current_user"] = username
        try:
            load_model_to_gpu(target_model_path)
        except Exception as e:
            GPU_STATE["status"] = GPUStatus.IDLE
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

    # 4. 执行推理
    GPU_STATE["status"] = GPUStatus.INFERENCING
    try:
        data = await audio.read()
        generate_kwargs = {"task": "transcribe", "num_beams": num_beams, "language": "chinese"}
        
        result = INFERENCE_CACHE["pipeline"](data, return_timestamps=True, generate_kwargs=generate_kwargs)
        
        results = []
        for chunk in result.get("chunks", []):
            text = chunk['text']
            if to_simple == 1:
                text = convert(text, 'zh-cn')
            if remove_pun == 1:
                text = remove_punctuation(text)
            results.append({"text": text, "start": chunk['timestamp'][0], "end": chunk['timestamp'][1]})
            
        return {
            "code": 0,
            "results": results,
            "used_model": selected_model["model_name"] if selected_model else "BASE_MODEL"
        }
    finally:
        # 推理结束，状态保持 INFERENCING，不卸载模型，以便下次秒级响应
        pass
