# inference_controller.py
import os
import gc
import json
import time
import shutil
import asyncio
import sqlite3
import subprocess
import tempfile
import numpy as np
import librosa
import torch
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from zhconv import convert
from utils.data_utils import remove_punctuation
from utils.db import get_db
from utils.auth import get_current_user
from utils.path_safety import OUTPUT_BASE, safe_join, safe_resolve_under, is_valid_model_name

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
    # Flash-Attention 2 只支持 GPU。CPU 上即使装了 flash_attn 也不能用——
    # 显式传 attn_implementation="flash_attention_2" 时 transformers 不会自动回退，
    # 而是直接报 "Flash Attention 2 is not available on CPU"。所以必须先卡 CUDA。
    if not torch.cuda.is_available():
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# 模块加载时确定一次即可——切换需要重启服务
ATTN_IMPLEMENTATION = _detect_attn_implementation()
print(f"[inference] 使用 attention 实现: {ATTN_IMPLEMENTATION}")


def resolve_lora_base_model_path(model_path: str):
    """如果是 LoRA 目录，返回其 base_model（已重映射到本机路径）；否则返回 None。
    """
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return None

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        return None

    # 本机训练的模型：记录的绝对路径在本机存在 → 原样使用
    if os.path.isdir(base_model_path):
        return base_model_path

    # 跨机同步的模型：按 basename 重映射到本机 whisper-base-models/<name>
    local_base = os.path.join(BASE_MODELS_DIR, os.path.basename(base_model_path.rstrip("/")))
    if os.path.isdir(local_base):
        print(f"[lora] base_model 路径重映射: {base_model_path} -> {local_base}")
        return local_base

    return base_model_path


def resolve_base_model_paths():
    """扫描基础模型目录，返回 {model_name: model_path}。"""
    if not os.path.isdir(BASE_MODELS_DIR):
        return {}
    models = {}
    for entry in os.scandir(BASE_MODELS_DIR):
        # whisper-base-models/ggml/ 是导出的 ggml 产物目录（P1-6），不是基座模型，跳过
        if entry.is_dir() and entry.name != "ggml":
            models[entry.name] = entry.path
    return models


async def resolve_user_model_path(username: str):
    """从数据库查询用户所有可用微调模型，并返回发布状态。"""
    async with get_db(row_factory=sqlite3.Row) as conn:
        cursor = await conn.execute(
            '''
            SELECT model_name, is_published, version_tag, created_at, updated_at
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
        # 浏览器 WASM 实际加载的是量化版 q5_0；以它存在与否作为 has_ggml 判据
        # （与 has_tflite 一样按文件存在性现查，不进 DB 列）。
        ggml_q5_path = safe_join(model_dir, "ggml", "whisper_q5_0.bin")

        if os.path.exists(checkpoint_path):
            valid_models.append({
                "model_name": row["model_name"],
                "model_path": checkpoint_path,
                "is_published": bool(row["is_published"]),
                "version_tag": row["version_tag"],
                "has_tflite": os.path.exists(tflite_path),
                "has_ggml": os.path.exists(ggml_q5_path),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            })
    return valid_models


@router.get("/api/user_models")
async def get_user_models(current_user: str = Depends(get_current_user)):
    return {"models": await resolve_user_model_path(current_user)}


@router.post("/api/rescan_models")
async def rescan_models(current_user: str = Depends(get_current_user)):
    """扫描 output/<user>/ 下「已训练完成但还没登记进库」的模型并补登记，使在终端直接用
    finetune.py 训练的模型也能出现在「管理微调好的模型」列表（网页端发起的训练在
    finetune_controller 收尾时已自动登记，无需扫描——本接口面向终端训练）。

    判定口径与 resolve_user_model_path 完全一致：目录名合法 + 存在 checkpoint-final/
    才算一个可用模型（训练中途的模型还没写出 checkpoint-final，自然不会被误登记）。

    幂等：只新增库里缺失的行；已在库的模型一律不动（不刷新 updated_at，避免列表重排）。
    新模型默认未发布（is_published 取列默认 0），与网页端训练收尾的 _upsert_user_model 一致。
    """
    user_output_root = safe_join(OUTPUT_BASE, current_user)
    if not os.path.isdir(user_output_root):
        # 该用户还没有任何训练产物目录
        return {"added": [], "added_count": 0, "scanned": 0}

    # 1) 磁盘上「训练完成」的模型：output/<user>/<model_name>/checkpoint-final/ 存在
    on_disk = []
    for entry in os.scandir(user_output_root):
        if not entry.is_dir():
            continue
        model_name = entry.name
        # 与前端 / delete_model 同款白名单，非法目录名静默跳过
        # （不让一条脏目录名让整个接口 400）
        if not is_valid_model_name(model_name):
            continue
        checkpoint_final = safe_join(user_output_root, model_name, "checkpoint-final")
        if os.path.isdir(checkpoint_final):
            on_disk.append(model_name)

    # 2) 已在库的模型名
    async with get_db() as conn:
        cursor = await conn.execute(
            'SELECT model_name FROM models WHERE username = ?',
            (current_user,)
        )
        existing = {row[0] for row in await cursor.fetchall()}

        # 3) 只补登记库里缺失的（ON CONFLICT DO NOTHING 兜底并发下的重复插入）
        added = sorted(m for m in on_disk if m not in existing)
        if added:
            now_ts = int(time.time())
            for model_name in added:
                await conn.execute(
                    '''
                    INSERT INTO models (username, model_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(username, model_name) DO NOTHING
                    ''',
                    (current_user, model_name, now_ts, now_ts)
                )
            await conn.commit()

    return {"added": added, "added_count": len(added), "scanned": len(on_disk)}


class DeleteModelRequest(BaseModel):
    model_name: str


@router.post("/api/delete_model")
async def delete_model(
    req: DeleteModelRequest,
    current_user: str = Depends(get_current_user),
):
    """删除一个微调好的模型：清掉 output/{user}/{model_name}/ 整目录（LoRA + tflite +
    loss 日志）和 models 表记录。若该模型正驻留显存，先把它从显存踢掉。

    并发安全：复用全局 GPU_LOCK / GPU_STATE。训练或推理进行中一律拒绝（423），删除期间
    用 DELETING 状态占位，避免别人趁机加载我们正在 rmtree 的目录。
    """
    model_name = req.model_name.strip()
    # 与 start_finetune 同款白名单：挡住 `..` `/` 等会让 safe_join 之外的逻辑出错的输入
    if not is_valid_model_name(model_name):
        raise HTTPException(status_code=400, detail="模型名称非法")

    # 两层 safe_join：先到 user 根，再 join model_name，挡住 model_name=".." 上跳一层
    user_output_root = safe_join(OUTPUT_BASE, current_user)
    model_dir = safe_join(user_output_root, model_name)
    checkpoint_final = safe_join(model_dir, "checkpoint-final")

    # --- 原子化检查 + 占位 ---
    async with GPU_LOCK:
        if GPU_STATE["status"] != GPUStatus.IDLE:
            raise HTTPException(
                status_code=423,
                detail=f"GPU 正被用户 [{GPU_STATE['current_user']}] {GPU_STATE['status']} 占用，请稍后再删除"
            )
        GPU_STATE["status"] = GPUStatus.DELETING
        GPU_STATE["current_user"] = current_user

    # --- 锁外执行实际删除（仿 api_recognition 的 claim 模式，避免在锁内做长耗时 IO）---
    try:
        # 1. 若被删模型正驻留显存，先卸载，避免 pipeline 指向马上要删掉的权重文件
        loaded = INFERENCE_CACHE["loaded_model_path"]
        if loaded and (
            loaded == checkpoint_final
            or loaded == model_dir
            or loaded.startswith(model_dir + os.sep)
        ):
            unload_inference_cache()

        # 2. 删除磁盘目录（可能含上百 MB 的 tflite，放线程池里跑不阻塞 event loop）
        if os.path.isdir(model_dir):
            await asyncio.to_thread(shutil.rmtree, model_dir)

        # 3. 删除 DB 记录（目录已不存在但仍有记录的孤儿行也会被一并清掉）
        async with get_db() as conn:
            await conn.execute(
                'DELETE FROM models WHERE username = ? AND model_name = ?',
                (current_user, model_name)
            )
            await conn.commit()
    finally:
        async with GPU_LOCK:
            GPU_STATE["status"] = GPUStatus.IDLE
            GPU_STATE["current_user"] = None

    return {"message": "模型已删除", "model_name": model_name}


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


# ---------------------------------------------------------------------------
# ggml（q5_0）下载 / 信息接口 —— 供浏览器端 whisper.cpp WASM 推理加载模型（P1-5）。
# 与上面 tflite 的 latest_model_info / download_published_model 并列，但服务的是
# output/<user>/<model>/ggml/whisper_q5_0.bin（量化版，浏览器实际加载这个）。
# ---------------------------------------------------------------------------
def _ggml_etag(path: str) -> str:
    """基于文件 mtime + size 的 ETag（download 与 info 用同一函数，保证两端一致便于前端比对）。"""
    st = os.stat(path)
    return f'"{int(st.st_mtime)}-{st.st_size}"'


async def _resolve_published_ggml(current_user: str):
    """定位该用户【已发布】模型的 ggml(q5_0) 文件。
    返回 (model_name, version_tag, ggml_path)；未发布或文件缺失时 ggml_path 为 None。"""
    async with get_db() as conn:
        cursor = await conn.execute(
            'SELECT model_name, version_tag FROM models WHERE username = ? AND is_published = 1 LIMIT 1',
            (current_user,)
        )
        row = await cursor.fetchone()
    if not row:
        return None, None, None
    model_name, version_tag = row[0], row[1]
    user_output_root = safe_join(OUTPUT_BASE, current_user)
    ggml_path = safe_resolve_under(
        user_output_root, os.path.join(model_name, "ggml", "whisper_q5_0.bin")
    )
    if not ggml_path or not os.path.exists(ggml_path):
        return model_name, version_tag, None
    return model_name, version_tag, ggml_path


@router.get("/api/published_ggml_info")
async def get_published_ggml_info(current_user: str = Depends(get_current_user)):
    """供前端 WASM 流程轮询：当前发布模型的 ggml(q5_0) 信息，用于判断要不要重新下载。"""
    model_name, version_tag, ggml_path = await _resolve_published_ggml(current_user)
    if not ggml_path:
        return {"has_published": False}
    return {
        "has_published": True,
        "model_name": model_name,
        "version_tag": version_tag,
        "file_size": os.path.getsize(ggml_path),
        "etag": _ggml_etag(ggml_path),
    }


@router.get("/api/download_published_ggml")
async def download_published_ggml(current_user: str = Depends(get_current_user), v: str = ""):
    """供前端 WASM 加载流程 fetch：下载已发布模型的 ggml(q5_0) 二进制。

    缓存策略：浏览器端真正的缓存是 IndexedDB（见 P2-4），HTTP 缓存只是锦上添花。
    - 带 ?v=<version_tag>（前端从 /api/published_ggml_info 取）：URL 随版本唯一，可安全
      长缓存 immutable，重发布换了 version_tag 自然是新 URL，不会取到旧文件。
    - 不带 v：同一 URL 的内容可能随“发布了另一个模型”而变，故只用 no-cache 兜底，避免取旧。
    """
    model_name, version_tag, ggml_path = await _resolve_published_ggml(current_user)
    if not ggml_path:
        raise HTTPException(status_code=404, detail="该用户尚未发布带 ggml 的模型")
    cache_control = "public, max-age=31536000, immutable" if v else "no-cache"
    return FileResponse(
        path=ggml_path,
        filename=f"whisper_{model_name}_q5_0.bin",
        media_type="application/octet-stream",
        headers={"Cache-Control": cache_control, "ETag": _ggml_etag(ggml_path)},
    )


# 基座模型的 ggml（q5_0），供没微调过的用户也能跑 WASM 推理（P1-6）。由部署时
# `python ggml_export.py --base-models` 生成到 whisper-base-models/ggml/<name>_q5_0.bin。
BASE_GGML_DIR = os.path.join(BASE_MODELS_DIR, "ggml")


def _resolve_base_ggml(model: str):
    """校验 model 是真实基座目录名（白名单，排除派生的 ggml/）+ 定位其 q5_0；返回 path 或 None。"""
    if not os.path.isdir(BASE_MODELS_DIR):
        return None
    valid = {e.name for e in os.scandir(BASE_MODELS_DIR) if e.is_dir() and e.name != "ggml"}
    if model not in valid:
        return None
    path = safe_resolve_under(BASE_GGML_DIR, f"{model}_q5_0.bin")
    if not path or not os.path.exists(path):
        return None
    return path


@router.get("/api/download_base_ggml")
async def download_base_ggml(model: str, current_user: str = Depends(get_current_user)):
    """下载基座模型的 ggml(q5_0)，供未微调用户的浏览器 WASM 推理。

    基座内容固定不变（whisper-tiny 权重永远是那一份）→ 同一 URL 内容恒定，可直接 immutable
    长缓存，无需像用户模型那样带 ?v=。鉴权沿用 get_current_user（与全站一致）。
    """
    path = _resolve_base_ggml(model)
    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"基座 {model} 的 ggml 不存在（部署时需先跑 `python ggml_export.py --base-models` 生成）"
        )
    return FileResponse(
        path=path,
        filename=f"{model}_q5_0.bin",
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable", "ETag": _ggml_etag(path)},
    )


@router.get("/api/download_user_ggml")
async def download_user_ggml(model: str, current_user: str = Depends(get_current_user), v: str = ""):
    """下载【当前用户自己】某个微调模型的 ggml(q5_0)，供浏览器 WASM 本地推理。

    与 download_published_ggml 的区别：**不要求已发布**。发布（publish_model）是给
    Android / ModelScope 用的；浏览器本地推理只需要 ggml 文件本身。这样用户微调完、
    自动导出 ggml 后，无需走发布流程就能在浏览器里用上自己的模型。
    鉴权 + 路径都收口在 current_user 自己的 output 目录下。
    """
    if not is_valid_model_name(model):
        raise HTTPException(status_code=400, detail="模型名称非法")
    user_output_root = safe_join(OUTPUT_BASE, current_user)
    ggml_path = safe_resolve_under(user_output_root, os.path.join(model, "ggml", "whisper_q5_0.bin"))
    if not ggml_path or not os.path.exists(ggml_path):
        raise HTTPException(status_code=404, detail=f"模型 {model} 还没有 ggml 版本（训练后会自动生成，旧模型可用 ggml_export.py 补生成）")
    # 前端带 ?v=<updated_at> 做缓存隔离（重训会变）→ 可 immutable 长缓存；不带则 no-cache。
    cache_control = "public, max-age=31536000, immutable" if v else "no-cache"
    return FileResponse(
        path=ggml_path,
        filename=f"{model}_q5_0.bin",
        media_type="application/octet-stream",
        headers={"Cache-Control": cache_control, "ETag": _ggml_etag(ggml_path)},
    )


def unload_inference_cache():
    """彻底卸载显存中的推理模型并清空缓存。

    load_model_to_gpu 在加载新模型前调用它；delete_model 在删除一个正驻留显存的
    微调模型时也复用它，避免 pipeline 指向已被 rmtree 删掉的权重文件。
    """
    if INFERENCE_CACHE["pipeline"] is not None:
        del INFERENCE_CACHE["pipeline"].model
        del INFERENCE_CACHE["pipeline"].feature_extractor
        del INFERENCE_CACHE["pipeline"].tokenizer
        del INFERENCE_CACHE["pipeline"]
    INFERENCE_CACHE["pipeline"] = None
    # loaded_model_path 一并置空：下次推理时 api_recognition 会发现缓存为空而重新加载
    INFERENCE_CACHE["loaded_model_path"] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_to_gpu(model_path: str):
    """清理显存并加载新模型"""
    print(f"正在从显存卸载旧模型，准备加载: {model_path}")

    # 1. 彻底清空旧模型
    unload_inference_cache()

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

    # 不在这里设 forced_decoder_ids：语言/任务在 generate 时通过 generate_kwargs 传
    # （task=transcribe, language=chinese）。两边都设会冲突 → transformers 告警并忽略 forced_decoder_ids，
    # 这里显式置 None 清掉旧值，消除告警、避免歧义。
    model.generation_config.forced_decoder_ids = None
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


def segment_audio(pcm: np.ndarray, sr: int = 16000, top_db: int = 30,
                  pad_s: float = 0.15, merge_gap_s: float = 1.0,
                  min_seg_s: float = 0.2, max_seg_s: float = 25.0):
    """按静音把长音频切成「句级」片段，返回 [(start_sample, end_sample), ...]。

    为什么需要：朗读整段课文这种长音频，直接喂给在「短单句」上微调的模型会严重退化——模型被
    特化成「短语音 + 大片静音」的 30s 窗口分布，塞满连续语音的 30s 块对它是分布外。先按停顿切成
    短句、逐句识别再拼接，能让每段都落回训练分布；切在自然停顿也避免 30s 硬切切碎词。
    用 librosa 能量门限，零额外依赖。短音频（无内部长停顿）会被切成 1 段，等价于原来的整段识别。

    注意「段数」不是越少越好：模型是在「单条短句」上微调的，理想是**每段 ≈ 一句话**才落回训练
    分布。merge_gap_s 用来把同句内的小停顿（朗读者读得慢、字间断顿）并回去，避免一句被切碎；
    max_seg_s 是安全上限，防止把连续语音并成超长块——那会退回「长音频分布外」的老问题。
    """
    # 空 / 整段静音先挡掉：librosa 对全零会反常地把整段当成"非静音"返回 [[0,N]]，
    # 若不挡，静音会被当一段送去识别，而 Whisper 在静音上会幻觉出文字。
    if pcm.size == 0 or float(np.max(np.abs(pcm))) < 1e-4:
        return []
    intervals = librosa.effects.split(pcm, top_db=top_db)  # 非静音区间 [[s,e],...]（样本下标）
    if len(intervals) == 0:
        return []
    # 间隔够短的相邻段（同一句内的小停顿）合并，避免把一句切碎；但合并后不得超过 max_seg_s。
    merge_gap = int(merge_gap_s * sr)
    max_len = int(max_seg_s * sr)
    merged = [[int(intervals[0][0]), int(intervals[0][1])]]
    for s, e in intervals[1:]:
        s, e = int(s), int(e)
        if s - merged[-1][1] <= merge_gap and e - merged[-1][0] <= max_len:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    # 每段前后留一点余量（防切掉字头字尾），丢掉过短的碎片（多半是噪声）
    pad = int(pad_s * sr)
    min_len = int(min_seg_s * sr)
    n = pcm.size
    segs = []
    for s, e in merged:
        if e - s < min_len:
            continue
        segs.append((max(0, s - pad), min(n, e + pad)))
    return segs


@router.post("/api/recognition")
async def api_recognition(
    to_simple: int = Form(1),
    remove_pun: int = Form(0),
    num_beams: int = Form(1),
    model_name: str = Form("whisper-large-v3"),
    top_db: int = Form(30),
    merge_gap_s: float = Form(1.2),
    max_seg_s: float = Form(25.0),
    punctuate: int = Form(1),
    audio: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    """服务端 GPU 推理 —— 现已是**降级路径**，不再是默认。

    默认推理在用户浏览器里用 whisper.cpp WASM 本地跑（见 static/js/wasm_whisper.js +
    inference_panel.js），音频不离开本机、也不占服务端 GPU，多用户可并发识别。只有当
    浏览器不支持本地推理（旧浏览器 / 页面未跨源隔离 / 该模型无 ggml 版）时，前端才回落到
    本接口。本接口仍受 GPU_LOCK 约束（与训练互斥），与新增的 WASM 路径互不影响。
    """
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

        raw = await audio.read()
        generate_kwargs = {"task": "transcribe", "num_beams": num_beams, "language": "chinese"}

        # 写到临时文件后用 ffmpeg 解码为 16kHz mono float32 PCM。
        # 直接把 bytes 传给 pipeline 会走 `ffmpeg -i pipe:0`，MP4/M4A 等
        # 需要 seek 读 moov atom 的格式在 pipe 里会失败。文件路径 + 管道输出绕开此限制。
        suffix = os.path.splitext(audio.filename or "upload.wav")[1].lower() or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                capture_output=True,
            )
            if not proc.stdout:
                raise HTTPException(
                    status_code=400,
                    detail=f"音频格式无法解析（{suffix}），请上传 wav/mp3/m4a/flac 等常见格式",
                )
            pcm = np.frombuffer(proc.stdout, dtype=np.float32)
        finally:
            os.unlink(tmp_path)

        # VAD 切句：长音频按停顿切成句级片段，逐句识别（每段落回"短句"训练分布），
        # 再用片段起点偏移时间戳后拼接。短音频会切成 1 段，等价于原来的整段识别。
        segments = segment_audio(pcm, 16000, top_db=top_db,
                                 merge_gap_s=merge_gap_s, max_seg_s=max_seg_s)
        print(f"[recognition] 时长 {pcm.size / 16000:.1f}s，VAD 切出 {len(segments)} 段 "
              f"(top_db={top_db} merge_gap_s={merge_gap_s} max_seg_s={max_seg_s})")

        results = []      # 逐 whisper-chunk 明细（带全局时间戳），前端可做高亮/定位
        seg_texts = []    # 每个 VAD 段的整段文本，用于拼全文 / 加标点
        if segments:
            # 批量识别：把所有片段作为输入列表一次性交给 pipeline（batch_size 控制并行），
            # 避免逐段串行（HF 会因串行调用告警），大幅提速。输出与 segments 顺序一一对应。
            inputs = [{"array": pcm[s:e], "sampling_rate": 16000} for s, e in segments]
            outputs = list(INFERENCE_CACHE["pipeline"](
                inputs,
                return_timestamps=True,
                batch_size=DEFAULT_BATCH_SIZE,
                generate_kwargs=generate_kwargs,
            ))
            # 各段时间戳是段内相对值，按段起点偏移后拼接成全局时间轴
            for (seg_start, _seg_end), out in zip(segments, outputs):
                offset = seg_start / 16000.0
                parts = []
                for chunk in out.get("chunks", []):
                    text = chunk["text"]
                    if to_simple == 1:
                        text = convert(text, "zh-cn")
                    if remove_pun == 1:
                        text = remove_punctuation(text)
                    ts = chunk.get("timestamp") or (None, None)
                    results.append({
                        "text": text,
                        "start": (ts[0] + offset) if ts[0] is not None else offset,
                        "end": (ts[1] + offset) if ts[1] is not None else None,
                    })
                    parts.append(text)
                seg_texts.append("".join(parts).strip())

        # 把各段识别文本按顺序拼成整段全文（朗读整篇课文场景要的是连续转写）。
        # 模型是在「去标点的短句」上微调的，自身基本不产标点，直接拼会是一长串没断句的字。
        # punctuate=1 时按 VAD 切出的停顿补一层「近似标点」：每个停顿（段边界）补「，」、
        # 全文末尾补「。」。这是基于停顿的近似，不是真正的句法标点——要更准需另接中文标点恢复
        # 模型做后处理。remove_pun=1（显式要求去标点）时不补。
        do_punct = punctuate == 1 and remove_pun != 1
        nonempty = [t for t in seg_texts if t]
        if do_punct:
            # 先削掉模型可能带出的零散尾标点，避免和补的标点重复
            cleaned = [t.rstrip("，。,.！？!?、；;：: ") for t in nonempty]
            cleaned = [t for t in cleaned if t]
            full_text = ("，".join(cleaned) + "。") if cleaned else ""
        else:
            full_text = "".join(nonempty)

        payload = {
            "code": 0,
            "text": full_text,           # 整段拼接后的全文（已按停顿补近似标点）
            "results": results,          # 分段明细（带全局时间戳），前端可做高亮/定位
            "used_model": model_name,
            "used_model_type": selected_type,
        }

        # 终端打印，便于调试：全文 + 逐段 + 完整 JSON
        print(f"[recognition] 全文：{full_text}")
        print("[recognition] 分段：" + " | ".join(seg_texts))
        print("[recognition] JSON：" + json.dumps(payload, ensure_ascii=False))

        return payload
    finally:
        # 推理结束 / 异常退出，都要释放 GPU 状态（不卸载模型，仅置回 IDLE）
        async with GPU_LOCK:
            GPU_STATE["status"] = GPUStatus.IDLE
            GPU_STATE["current_user"] = None
