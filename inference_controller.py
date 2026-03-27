# inference_controller.py
import os
import gc
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from zhconv import convert
from pydantic import BaseModel
from utils.data_utils import remove_punctuation

from shared_state import GPUStatus, GPU_STATE, INFERENCE_CACHE

router = APIRouter()

# 基础模型路径
BASE_MODEL_PATH = "/home/featurize/whisper-large-v3"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BATCH_SIZE = 16


def resolve_user_model_path(username: str):
    """兼容不同目录结构，查找用户的 checkpoint-final 目录。"""
    user_output_dir = os.path.join(PROJECT_ROOT, "output", username)

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
    model_type: str = Form("base"), # 接收模型选择 ("base" 或 "finetuned")
    audio: UploadFile = File(...)
):
    # 1. 检查 GPU 状态 (如果是训练，直接拒绝)
    if GPU_STATE["status"] == GPUStatus.TRAINING:
        raise HTTPException(status_code=423, detail="GPU 正在进行训练任务，请稍后再试！")
    
    # 2. 确定目标模型路径 (优先找该用户的微调模型，找不到用Base)
    user_model_dir = resolve_user_model_path(username)
    if model_type == "finetuned":
        if not user_model_dir:
            raise HTTPException(status_code=400, detail="未找到微调模型，请先进行微调")
        target_model_path = user_model_dir
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
            
        return {"code": 0, "results": results, "used_model": "Finetuned" if user_model_dir and target_model_path == user_model_dir else "Base"}
    finally:
        # 推理结束，状态保持 INFERENCING，不卸载模型，以便下次秒级响应
        pass
