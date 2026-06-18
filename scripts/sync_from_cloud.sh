#!/bin/bash
set -uo pipefail

# ================= 配置区 =================
REMOTE_USER="xiejianan"
# 采集服务器公网 IP
REMOTE_IP="40.83.91.242"
SSH_KEY="$HOME/.ssh/halo_key.pem"
# 项目在采集服务器上的绝对路径
REMOTE_DIR="/home/xiejianan/whisper-finetune-plus"
# 本地项目绝对路径
LOCAL_DIR="/home/xiejianan/whisper-finetune-plus"
SSH_OPTS="-i ${SSH_KEY} -p 22"
# 远端数据库副本的本地临时目录（用于同步登录凭据，用完即删）
REMOTE_DB_DIR="${LOCAL_DIR}/data/.remote_db_tmp"
# ==========================================

SCRIPT_NAME="$(basename "$0")"

print_help() {
  cat <<EOF
从采集服务器（halo）增量同步语音、用户数据与登录凭据到本地。

用法:
  ${SCRIPT_NAME}              交互模式：列出所有用户，终端选择（含「全部用户」选项）
  ${SCRIPT_NAME} all          全量同步：直接拉取所有用户（非交互，适合脚本/定时任务）
  ${SCRIPT_NAME} <用户名>      只同步指定用户（非交互）
  ${SCRIPT_NAME} -h, --help   显示本帮助信息

同步内容:
  • 音频与文本：rsync 拉取 uploads/<用户>/（task_*.wav + words.txt）
  • 登录凭据  ：拉取远端数据库副本，把用户名 + 密码哈希(bcrypt)合并进本地库
                → 用户可用原密码在本地登录，使用 GPU 推理/微调

说明:
  • 仅读取远端、本地只增不删（rsync 无 --delete），可反复安全运行
  • 本地的 models 表（微调记录）不会被覆盖：只合并 users 表的凭据
  • 密码以 bcrypt 哈希存储，跨机复制安全，不涉及明文

当前配置（可在脚本顶部「配置区」修改）:
  采集服务器 : ${REMOTE_USER}@${REMOTE_IP}
  远端目录   : ${REMOTE_DIR}
  本地目录   : ${LOCAL_DIR}
  SSH 私钥   : ${SSH_KEY}

示例:
  ${SCRIPT_NAME}              # 交互选择
  ${SCRIPT_NAME} all         # 全量同步
  ${SCRIPT_NAME} 小明         # 只同步「小明」
EOF
}

MODE_ARG="${1:-}"

# -h / --help: 打印帮助后退出
if [ "${MODE_ARG}" = "-h" ] || [ "${MODE_ARG}" = "--help" ]; then
  print_help
  exit 0
fi

# ---------------- 工具函数 ----------------

# 增量拉取单个用户的音频
sync_user_files() {
  local user="$1"
  mkdir -p "${LOCAL_DIR}/uploads/${user}"
  echo "📦 正在增量同步 ${user} 的音频文件..."
  rsync -avz --progress -e "ssh ${SSH_OPTS}" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/uploads/${user}/" \
    "${LOCAL_DIR}/uploads/${user}/"
}

# 增量拉取全部用户的音频
sync_all_files() {
  mkdir -p "${LOCAL_DIR}/uploads"
  echo "📦 正在增量同步全部用户的音频文件 (uploads/)..."
  rsync -avz --progress -e "ssh ${SSH_OPTS}" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/uploads/" \
    "${LOCAL_DIR}/uploads/"
}

# 拉取远端数据库副本（仅 tasks.db + WAL，用于读取最新登录凭据）。
# 失败不致命：python 端会优雅跳过凭据同步。
pull_remote_db() {
  echo "🔑 正在拉取远端数据库副本（用于同步登录凭据）..."
  rm -rf "${REMOTE_DB_DIR}"
  mkdir -p "${REMOTE_DB_DIR}"
  # WAL 模式下需连同 -wal 一起拷贝才能读到未 checkpoint 的最新写入；-shm 不拷（可重建）。
  rsync -az -e "ssh ${SSH_OPTS}" \
    --include="tasks.db" --include="tasks.db-wal" --exclude="*" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/data/" \
    "${REMOTE_DB_DIR}/" \
    || echo "⚠️ 远端数据库拉取失败，本轮将跳过登录凭据同步（音频不受影响）"
}

cleanup_remote_db() {
  rm -rf "${REMOTE_DB_DIR}"
}

# 回填本地数据库；传用户名 = 只回填该用户，不传 = 全部。
# 若远端库副本存在，则一并合并登录凭据。
backfill_db() {
  local user="${1:-}"
  local extra=()
  if [ -f "${REMOTE_DB_DIR}/tasks.db" ]; then
    extra+=(--remote-db "${REMOTE_DB_DIR}/tasks.db")
  fi
  echo "🔍 正在扫描 uploads 并回填本地数据库 (data/tasks.db)..."
  if [ -n "${user}" ]; then
    python3 "${LOCAL_DIR}/scripts/sync_uploads_to_db.py" \
      --uploads-dir "${LOCAL_DIR}/uploads" \
      --db-path "${LOCAL_DIR}/data/tasks.db" \
      --user "${user}" "${extra[@]}"
  else
    python3 "${LOCAL_DIR}/scripts/sync_uploads_to_db.py" \
      --uploads-dir "${LOCAL_DIR}/uploads" \
      --db-path "${LOCAL_DIR}/data/tasks.db" "${extra[@]}"
  fi
}

# 完整执行：全部用户
run_sync_all() {
  sync_all_files || { echo "❌ rsync 同步失败"; exit 1; }
  pull_remote_db
  backfill_db
  cleanup_remote_db
  echo "✅ 全部用户同步完成！"
}

# 完整执行：单个用户
run_sync_user() {
  local user="$1"
  sync_user_files "${user}" || { echo "❌ rsync 同步失败"; exit 1; }
  pull_remote_db
  backfill_db "${user}"
  cleanup_remote_db
  echo "✅ 用户 ${user} 同步完成！"
}

mkdir -p "${LOCAL_DIR}/data"

# ---------------- 模式分派 ----------------

# 模式 A: 全量（命令行传 all）
if [ "${MODE_ARG}" = "all" ]; then
  echo "🚀 全量同步：从采集服务器拉取所有用户数据..."
  run_sync_all
  exit 0
fi

# 模式 B: 命令行直接指定用户
if [ -n "${MODE_ARG}" ]; then
  echo "🚀 同步指定用户：${MODE_ARG}"
  run_sync_user "${MODE_ARG}"
  exit 0
fi

# 模式 C: 交互式（无参数）—— 先列用户，再选择
echo "🔍 正在从采集服务器获取用户列表..."

# 远端: 遍历 uploads/ 下每个子目录，输出 "用户名|音频条数"
RAW_LIST=$(ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_IP}" "
  cd ${REMOTE_DIR}/uploads 2>/dev/null || exit 1
  for d in */; do
    name=\"\${d%/}\"
    count=\$(ls \"\$name\"/task_*.wav 2>/dev/null | wc -l)
    echo \"\${name}|\${count}\"
  done
")

if [ -z "${RAW_LIST}" ]; then
  echo "❌ 远端没有发现任何用户目录（或 SSH 连接失败）"
  exit 1
fi

# 解析成两个并行数组: USERS=用户名, LABELS=带条数的展示文案
USERS=()
LABELS=()
while IFS='|' read -r name count; do
  [ -z "${name}" ] && continue
  USERS+=("${name}")
  LABELS+=("${name}  (${count} 条音频)")
done <<< "${RAW_LIST}"

# 菜单第一项放「全部用户」，其余为各用户
echo "📋 采集服务器上的用户："
MENU=("★ 全部用户" "${LABELS[@]}")
PS3="请输入编号（输入 q 退出）: "
select CHOICE in "${MENU[@]}"; do
  if [ "${REPLY}" = "q" ]; then
    echo "已取消"
    exit 0
  fi
  if [[ "${REPLY}" =~ ^[0-9]+$ ]] && [ "${REPLY}" -ge 1 ] && [ "${REPLY}" -le "${#MENU[@]}" ]; then
    if [ "${REPLY}" -eq 1 ]; then
      SELECTED="__ALL__"
    else
      SELECTED="${USERS[$((REPLY - 2))]}"
    fi
    break
  fi
  echo "⚠️ 无效选择，请重新输入"
done

# ---------------- 执行 ----------------
if [ "${SELECTED}" = "__ALL__" ]; then
  echo "✅ 已选择: 全部用户"
  run_sync_all
else
  echo "✅ 已选择用户: ${SELECTED}"
  run_sync_user "${SELECTED}"
fi
