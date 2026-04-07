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

# 2. 同步 data 目录 (SQLite 数据库)
echo "🗄️ 正在同步数据库文件 (data/)..."
rsync -avz --progress ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/data/ ${LOCAL_DIR}/data/

echo "✅ 同步完美结束！"
echo "📂 最新数据已准备好，你可以开始运行 Whisper 的微调脚本了！"