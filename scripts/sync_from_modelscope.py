#!/usr/bin/env python3
"""从 ModelScope 下载 LoRA 权重，并注册到本地数据库。

把训练好、上传到 ModelScope 的 checkpoint 拉到当前机器的
output/<username>/<model_name>/checkpoint-final/，然后写入数据库，
使模型在网页端可见。

用法
----
  python scripts/sync_from_modelscope.py \\
      --username 小崔 \\
      --model-name cui-v1 \\
      --repo-id smellyCat99/whisper-base-lora-xiaocui

参数
----
  --username      网页端登录用户名（对应 output/<username>/）
  --model-name    模型名（对应 output/<username>/<model_name>/）
  --repo-id       ModelScope 仓库 ID，格式为 <owner>/<repo>
  --output-dir    可选，覆盖默认的 output/ 根目录
  --db-path       可选，覆盖默认的 data/tasks.db 路径
  --force         若 checkpoint-final 已存在则先删除再下载
  --no-register   只下载，不写数据库
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time
from contextlib import closing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 推理所需文件；跳过体积最大的 optimizer.pt（仅训练续训用）
INFERENCE_PATTERNS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "preprocessor_config.json",
    "trainer_state.json",
    "training_args.bin",
]


def is_safe_name(name: str) -> bool:
    return bool(name) and name.strip() == name and not any(c in name for c in ("/", "\\")) and ".." not in name


def ensure_models_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            is_published INTEGER DEFAULT 0,
            version_tag TEXT,
            UNIQUE(username, model_name)
        )
        '''
    )
    cur.execute("PRAGMA table_info(models)")
    cols = {row[1] for row in cur.fetchall()}
    if "is_published" not in cols:
        cur.execute("ALTER TABLE models ADD COLUMN is_published INTEGER DEFAULT 0")
    if "version_tag" not in cols:
        cur.execute("ALTER TABLE models ADD COLUMN version_tag TEXT")
    conn.commit()


def download_checkpoint(repo_id: str, target_dir: str) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "未找到 modelscope，请先激活对应 conda 环境：conda activate whisper"
        ) from exc

    print(f"📥 从 ModelScope 下载：{repo_id}")
    print(f"   目标目录：{target_dir}")
    snapshot_download(
        model_id=repo_id,
        local_dir=target_dir,
        ignore_file_pattern=["optimizer\\.pt", "rng_state\\.pth", "scaler\\.pt", "scheduler\\.pt"],
    )
    print("   下载完成")


def register_in_db(username: str, model_name: str, db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    now_ts = int(time.time())
    with closing(sqlite3.connect(db_path)) as conn:
        ensure_models_schema(conn)
        existed = conn.execute(
            "SELECT 1 FROM models WHERE username = ? AND model_name = ? LIMIT 1",
            (username, model_name),
        ).fetchone() is not None
        conn.execute(
            '''
            INSERT INTO models (username, model_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username, model_name)
            DO UPDATE SET updated_at = excluded.updated_at
            ''',
            (username, model_name, now_ts, now_ts),
        )
        conn.commit()
    action = "更新" if existed else "新增"
    print(f"✅ 数据库{action}记录：username={username}, model_name={model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 ModelScope 下载 LoRA 权重并注册到本地数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--username", required=True, help="网页端用户名（对应 output/<username>/）")
    parser.add_argument("--model-name", required=True, help="模型名（对应 output/<username>/<model_name>/）")
    parser.add_argument("--repo-id", required=True, help="ModelScope 仓库 ID，格式：<owner>/<repo>")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"), help="模型输出根目录")
    parser.add_argument("--db-path", default=os.path.join(PROJECT_ROOT, "data", "tasks.db"), help="SQLite 数据库路径")
    parser.add_argument("--force", action="store_true", help="若 checkpoint-final 已存在则先删除再重新下载")
    parser.add_argument("--no-register", action="store_true", help="只下载，不写数据库")
    args = parser.parse_args()

    if not is_safe_name(args.username) or not is_safe_name(args.model_name):
        raise SystemExit("❌ 用户名或模型名含非法字符（不能有 / \\ .. 或首尾空格）")

    checkpoint_dir = os.path.join(args.output_dir, args.username, args.model_name, "checkpoint-final")

    if args.force and os.path.isdir(checkpoint_dir):
        print(f"🗑️  --force：删除已有目录 {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    download_checkpoint(args.repo_id, checkpoint_dir)

    # 验证关键文件存在
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not os.path.isfile(os.path.join(checkpoint_dir, f))]
    if missing:
        raise SystemExit(f"❌ 下载后仍缺少关键文件：{missing}，请检查 ModelScope 仓库内容")

    if not args.no_register:
        register_in_db(args.username, args.model_name, args.db_path)

    print(f"\n模型目录：{checkpoint_dir}")
    print("刷新网页端即可看到该模型（默认未发布，可在页面上发布后用于识别）。")


if __name__ == "__main__":
    main()
