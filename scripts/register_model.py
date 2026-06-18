#!/usr/bin/env python3
"""把「在终端用 finetune.py 训练好的模型」补登记进数据库，使其出现在网页端。

背景
----
网页端发起的微调由 finetune_controller.run_finetune_process 收尾：训练成功后会调用
_upsert_user_model() 往 models 表写一条记录。直接在终端跑 finetune.py 绕过了这一步，
所以模型不会出现在前端列表里。本脚本复刻那条 upsert（见 finetune_controller.py 的
_upsert_user_model），把终端训练的模型补登记。

前端 inference_controller.resolve_user_model_path 列模型需要同时满足：
  1) models 表里有 (username, model_name) 行          ← 本脚本负责
  2) 磁盘存在 output/<username>/<model_name>/checkpoint-final/   ← 训练产物，需已就位

用法
----
  python scripts/register_model.py --username 小崔 --model-name cui-v1

注意：训练时 --output_dir 必须是 output/<username>/<model_name>/，前端才找得到模型。
"""
import argparse
import os
import sqlite3
import time
from contextlib import closing

# 本脚本在 scripts/ 下，仓库根是其上一级目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_models_schema(conn: sqlite3.Connection) -> None:
    """建/补 models 表，结构与 data_collector.py、sync_uploads_to_db.py 保持一致。

    upsert 依赖 UNIQUE(username, model_name) 约束（ON CONFLICT 的冲突目标）。
    """
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
    # 老库可能缺这两列，补上（只加列不删行）
    cur.execute("PRAGMA table_info(models)")
    cols = {row[1] for row in cur.fetchall()}
    if "is_published" not in cols:
        cur.execute("ALTER TABLE models ADD COLUMN is_published INTEGER DEFAULT 0")
    if "version_tag" not in cols:
        cur.execute("ALTER TABLE models ADD COLUMN version_tag TEXT")
    conn.commit()


def is_safe_name(name: str) -> bool:
    """与前端 is_valid_model_name 同精神：挡住会越权/破坏路径的字符。"""
    return bool(name) and name.strip() == name and not any(c in name for c in ("/", "\\")) and ".." not in name


def main() -> None:
    parser = argparse.ArgumentParser(description="把终端训练好的模型补登记进数据库（复刻网页端的 _upsert_user_model）")
    parser.add_argument("--username", required=True, help="模型所属用户名（对应 output/<username>/）")
    parser.add_argument("--model-name", required=True, help="模型名（对应 output/<username>/<model_name>/）")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"), help="模型输出根目录")
    parser.add_argument("--db-path", default=os.path.join(PROJECT_ROOT, "data", "tasks.db"), help="SQLite 数据库路径")
    args = parser.parse_args()

    if not is_safe_name(args.username) or not is_safe_name(args.model_name):
        raise SystemExit("❌ 用户名或模型名含非法字符（不能有 / \\ .. 或首尾空格）")

    model_dir = os.path.join(args.output_dir, args.username, args.model_name)
    checkpoint_final = os.path.join(model_dir, "checkpoint-final")

    # 前端列模型的硬性条件：checkpoint-final 必须存在，否则即便有 DB 记录也会被静默跳过
    if not os.path.isdir(checkpoint_final):
        raise SystemExit(
            f"❌ 找不到 checkpoint-final 目录：{checkpoint_final}\n"
            f"   前端列模型要求它存在。请确认训练时 --output_dir 设成了：\n"
            f"     {model_dir}\n"
            f"   （即 output/<用户名>/<模型名>/，finetune.py 会在其下生成 checkpoint-final）"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.db_path)), exist_ok=True)
    now_ts = int(time.time())

    with closing(sqlite3.connect(args.db_path)) as conn:
        ensure_models_schema(conn)
        existed = conn.execute(
            "SELECT 1 FROM models WHERE username = ? AND model_name = ? LIMIT 1",
            (args.username, args.model_name),
        ).fetchone() is not None
        # 与 finetune_controller._upsert_user_model 完全一致的 upsert：
        # 已存在则只刷新 updated_at（保留 created_at / is_published / version_tag）
        conn.execute(
            '''
            INSERT INTO models (username, model_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username, model_name)
            DO UPDATE SET updated_at = excluded.updated_at
            ''',
            (args.username, args.model_name, now_ts, now_ts),
        )
        conn.commit()

    action = "更新" if existed else "新增"
    print(f"✅ 已{action} models 记录：username={args.username}, model_name={args.model_name}")
    print(f"   模型目录：{model_dir}")
    print("   刷新网页端即可看到该模型（默认未发布，可在页面上发布后用于识别）。")


if __name__ == "__main__":
    main()
