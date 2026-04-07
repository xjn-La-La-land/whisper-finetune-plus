#!/bin/bash

# ================= 配置区 =================
REMOTE_USER="ubuntu"
# 腾讯云服务器的公网 IP
REMOTE_IP="43.143.17.185"
# 项目在腾讯云服务器上的绝对路径 (例如 /home/ubuntu/whisper-data-collector)
REMOTE_DIR="/home/ubuntu/whisper-finetune-plus"
# 4090 本地的项目绝对路径 (例如 /home/user/whisper_finetune)
LOCAL_DIR="/home/featurize/whisper-finetune-plus"
# ==========================================

echo "🚀 开始从腾讯云轻量服务器拉取最新数据..."

# 确保本地的接收目录存在
mkdir -p ${LOCAL_DIR}/uploads
mkdir -p ${LOCAL_DIR}/data

# 1. 增量同步 uploads 目录 (音频文件)
# -a: 归档模式，保留权限和目录结构(会自动包含小明、小蓝等子文件夹)
# -v: 显示详细过程
# -z: 传输时压缩，节省带宽
echo "📦 正在增量同步音频文件 (uploads/)..."
rsync -avz --progress ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/uploads/ ${LOCAL_DIR}/uploads/

# 2. 扫描 uploads，将新增音频主动写入本地数据库
# 注意：不再覆盖同步 data/tasks.db，避免覆盖训练机上已有的模型记录。
echo "🔍 正在扫描 uploads 并回填本地数据库 (data/tasks.db)..."
python3 ${LOCAL_DIR}/sync_uploads_to_db.py \
  --uploads-dir ${LOCAL_DIR}/uploads \
  --db-path ${LOCAL_DIR}/data/tasks.db

echo "✅ 同步完美结束！"
echo "📂 音频和数据库索引都已准备好，你可以开始运行 Whisper 的微调脚本了！"
