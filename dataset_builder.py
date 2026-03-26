# dataset_builder.py: 读取 SQLite 数据库，清洗数据，并为指定用户生成 train.json 和 test.json

import os
import json
import random
import sqlite3
import soundfile as sf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# 实例化 Router
router = APIRouter()

# 配置目录
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "tasks.db")
DATASET_DIR = "dataset"

# 定义前端传入的数据模型
class DatasetBuildRequest(BaseModel):
    username: str
    test_ratio: float = Field(default=0.05, ge=0.0, le=0.5, description="测试集比例")

@router.post("/api/build_dataset")
async def build_dataset(request: DatasetBuildRequest):
    username = request.username.strip()
    test_ratio = request.test_ratio
    
    # 1. 确定用户的专属数据集输出目录
    user_data_dir = os.path.join(DATASET_DIR, username)
    os.makedirs(user_data_dir, exist_ok=True)
    
    # 2. 从数据库获取该用户已完成录音的有效数据
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT audio_path, text_content FROM tasks WHERE username = ? AND is_completed = 1', (username,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=400, detail="没有找到已完成的录音任务，请先录制音频！")

    data_list = []
    
    # 3. 遍历数据库记录，构造数据集格式
    for db_audio_path, text in rows:
        # 【关键】：将数据库中的 Web 路径 (如 /uploads/xxx/1.wav) 转为服务器物理路径 (uploads/xxx/1.wav)
        physical_audio_path = db_audio_path.lstrip('/')
        
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

    # 4. 打乱并划分训练/测试集
    random.shuffle(data_list)
    split_idx = max(1, int(len(data_list) * test_ratio)) # 至少保留1条做测试集
    
    test_data = data_list[:split_idx]
    train_data = data_list[split_idx:]

    # 5. 写入该用户专属的 JSON 文件（标准 JSON 数组，便于直接校验/查看）
    train_path = os.path.join(user_data_dir, "train.json")
    test_path = os.path.join(user_data_dir, "test.json")

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
