# dataset_builder.py: 读取 SQLite 数据库，清洗数据，并为指定用户生成 train.json 和 test.json

import os
import json
import random
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from utils.db import get_db
from utils.auth import get_current_user
from utils.path_safety import DATASET_BASE, UPLOAD_BASE, safe_join, safe_resolve_under

# 实例化 Router
router = APIRouter()

# 配置目录
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "tasks.db")
DATASET_DIR = "dataset"

# 定义前端传入的数据模型
# username 从 JWT 取，body 里只剩 test_ratio + seed
class DatasetBuildRequest(BaseModel):
    test_ratio: float = Field(default=0.05, ge=0.0, le=0.5, description="测试集比例")
    # 固定 seed 让 train/test 划分可复现，方便做"调参 → 对比"实验
    seed: int = Field(default=42, description="train/test 划分使用的随机种子")


@router.get("/api/check_dataset")
async def check_dataset(current_user: str = Depends(get_current_user)):
    """检查用户是否已经生成 train/test 数据集文件，供前端刷新后恢复步骤状态。"""
    user_data_dir = safe_join(DATASET_BASE, current_user)
    train_path = safe_join(user_data_dir, "train.json")
    test_path = safe_join(user_data_dir, "test.json")

    has_train = os.path.exists(train_path)
    has_test = os.path.exists(test_path)
    return {"has_dataset": has_train and has_test}

@router.post("/api/build_dataset")
async def build_dataset(
    request: DatasetBuildRequest,
    current_user: str = Depends(get_current_user),
):
    username = current_user
    test_ratio = request.test_ratio

    # 1. 确定用户的专属数据集输出目录
    user_data_dir = safe_join(DATASET_BASE, username)
    os.makedirs(user_data_dir, exist_ok=True)
    
    # 2. 从数据库获取该用户已完成录音的有效数据
    async with get_db() as conn:
        cursor = await conn.execute(
            'SELECT audio_path, text_content FROM tasks WHERE username = ? AND is_completed = 1',
            (username,)
        )
        rows = await cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=400, detail="没有找到已完成的录音任务，请先录制音频！")

    data_list = []
    
    # 3. 遍历数据库记录，构造数据集格式
    for db_audio_path, text in rows:
        # DB 里的 web 路径 "/uploads/{u}/X.wav" 还原为 UPLOAD_BASE 下的物理路径,
        # 走 safe_resolve_under 兜历史脏数据里的 `../`
        clean = db_audio_path.split('?')[0].lstrip('/')
        rel = clean[len('uploads/'):] if clean.startswith('uploads/') else clean
        physical_audio_path = safe_resolve_under(UPLOAD_BASE, rel)
        if not physical_audio_path:
            print(f"警告: 非法 audio_path 跳过 {db_audio_path}")
            continue

        if not os.path.exists(physical_audio_path):
            print(f"警告: 物理音频文件丢失 {physical_audio_path}")
            continue

        # 获取音频时长
        try:
            info = sf.info(physical_audio_path)
            duration = info.duration
        except Exception as e:
            print(f"警告: 读取音频失败 {physical_audio_path}: {e}")
            continue

        # 构造符合 Whisper-finetune 要求的数据条目
        entry = {
            "audio": {
                "path": physical_audio_path 
            },
            "sentence": text,
            "language": "Chinese",
            "duration": round(duration, 2)
        }
        data_list.append(entry)

    if not data_list:
        raise HTTPException(status_code=400, detail="处理音频时全部失败，未能生成有效数据。")

    # 4. 打乱并划分训练/测试集（用独立 Random 实例避免污染全局 random 状态）
    random.Random(request.seed).shuffle(data_list)
    split_idx = max(1, int(len(data_list) * test_ratio)) # 至少保留1条做测试集
    
    test_data = data_list[:split_idx]
    train_data = data_list[split_idx:]

    # 5. 写入该用户专属的 JSON 文件（标准 JSON 数组，便于直接校验/查看）
    train_path = safe_join(user_data_dir, "train.json")
    test_path = safe_join(user_data_dir, "test.json")

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    return {
        "message": "数据集构建成功！",
        "total": len(data_list),
        "train_count": len(train_data),
        "test_count": len(test_data),
        "dataset_dir": user_data_dir
    }
