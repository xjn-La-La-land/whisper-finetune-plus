#!/usr/bin/env python3
import argparse
import os
import re
import sqlite3
from pathlib import Path

AUDIO_PATTERN = re.compile(r"^task_(\d+)\.wav$")


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            text_content TEXT NOT NULL,
            audio_path TEXT,
            is_completed BOOLEAN DEFAULT 0,
            updated_at INTEGER
        )
        '''
    )
    cur.execute("PRAGMA table_info(tasks)")
    task_columns = {row[1] for row in cur.fetchall()}
    if "updated_at" not in task_columns:
        cur.execute("ALTER TABLE tasks ADD COLUMN updated_at INTEGER")

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            created_at INTEGER,
            last_login_at INTEGER
        )
        '''
    )
    # users 表结构与采集 app（data_collector.py）保持一致：缺列就补，只加列不删行。
    # 这样后续采集 app 启动做迁移时不会把同步来的用户当“旧版无密码表”清空。
    cur.execute("PRAGMA table_info(users)")
    user_columns = {row[1] for row in cur.fetchall()}
    if "password_hash" not in user_columns:
        cur.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "created_at" not in user_columns:
        cur.execute("ALTER TABLE users ADD COLUMN created_at INTEGER")
    if "last_login_at" not in user_columns:
        cur.execute("ALTER TABLE users ADD COLUMN last_login_at INTEGER")

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
    # 自动升级逻辑
    cur.execute("PRAGMA table_info(models)")
    model_columns = {row[1] for row in cur.fetchall()}
    if "is_published" not in model_columns:
        cur.execute('ALTER TABLE models ADD COLUMN is_published INTEGER DEFAULT 0')
    if "version_tag" not in model_columns:
        cur.execute('ALTER TABLE models ADD COLUMN version_tag TEXT')
    
    conn.commit()


def read_words(words_path: Path) -> list[str]:
    if not words_path.exists():
        return []
    with words_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def sync_one_user(conn: sqlite3.Connection, uploads_dir: Path, username: str) -> tuple[int, int]:
    user_dir = uploads_dir / username
    if not user_dir.is_dir():
        return 0, 0

    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))

    words = read_words(user_dir / "words.txt")

    audio_entries: list[tuple[int, str, int]] = []
    for item in user_dir.iterdir():
        if not item.is_file():
            continue
        m = AUDIO_PATTERN.match(item.name)
        if not m:
            continue
        task_num = int(m.group(1))
        updated_at = int(item.stat().st_mtime * 1000)
        audio_entries.append((task_num, item.name, updated_at))

    audio_entries.sort(key=lambda x: x[0])

    inserted = 0
    updated = 0
    for idx, (_task_num, filename, updated_at) in enumerate(audio_entries):
        audio_path = f"/uploads/{username}/{filename}"
        text_content = words[idx] if idx < len(words) else f"[AUTO-SYNC] {filename}"

        cur.execute(
            "SELECT id, text_content FROM tasks WHERE username = ? AND audio_path = ?",
            (username, audio_path),
        )
        row = cur.fetchone()

        if row:
            task_id, old_text = row
            if old_text and old_text.strip() and not old_text.startswith("[AUTO-SYNC]"):
                final_text = old_text
            else:
                final_text = text_content

            cur.execute(
                '''
                UPDATE tasks
                SET text_content = ?, is_completed = 1, updated_at = ?
                WHERE id = ?
                ''',
                (final_text, updated_at, task_id),
            )
            updated += 1
        else:
            cur.execute(
                '''
                INSERT INTO tasks (username, text_content, audio_path, is_completed, updated_at)
                VALUES (?, ?, ?, 1, ?)
                ''',
                (username, text_content, audio_path, updated_at),
            )
            inserted += 1

    conn.commit()
    return inserted, updated


def sync_users_from_remote(conn, remote_db_path, only_user=None):
    """从远端数据库副本把登录凭据合并进本地 users 表。

    搬运 username + password_hash + created_at + last_login_at。密码是 bcrypt 哈希
    （单向、含 salt），跨机复制安全；用户搬过来后可用原密码登录。

    只动 users 表，本地的 models / tasks 表完全不受影响。
    返回成功合并的用户数。
    """
    remote_db = Path(remote_db_path)
    if not remote_db.exists():
        print(f"⚠️ 远端数据库副本不存在，跳过登录凭据同步: {remote_db}")
        return 0

    # 以读写方式打开“临时副本”，让 SQLite 能回放可能存在的 WAL，确保读到最新注册的用户。
    src = sqlite3.connect(str(remote_db))
    try:
        src_cur = src.cursor()
        src_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not src_cur.fetchone():
            print("⚠️ 远端数据库没有 users 表，跳过登录凭据同步")
            return 0

        src_cur.execute("PRAGMA table_info(users)")
        remote_cols = {row[1] for row in src_cur.fetchall()}
        if "password_hash" not in remote_cols:
            print("⚠️ 远端 users 表没有 password_hash 列，跳过登录凭据同步")
            return 0

        select_parts = ["username", "password_hash"]
        select_parts.append("created_at" if "created_at" in remote_cols else "NULL")
        select_parts.append("last_login_at" if "last_login_at" in remote_cols else "NULL")
        query = f"SELECT {', '.join(select_parts)} FROM users"
        params: tuple = ()
        if only_user:
            query += " WHERE username = ?"
            params = (only_user,)
        rows = src_cur.execute(query, params).fetchall()
    finally:
        src.close()

    cur = conn.cursor()
    count = 0
    for username, password_hash, created_at, last_login_at in rows:
        # 采集服务器是凭据的权威来源：password_hash 以远端为准。
        # created_at 取已知的较早值（保留本地已有），last_login_at 取两台机器中较近的一次登录。
        cur.execute(
            '''
            INSERT INTO users (username, password_hash, created_at, last_login_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash = excluded.password_hash,
                created_at    = COALESCE(users.created_at, excluded.created_at),
                last_login_at = CASE
                    WHEN excluded.last_login_at IS NULL THEN users.last_login_at
                    WHEN users.last_login_at IS NULL THEN excluded.last_login_at
                    WHEN excluded.last_login_at > users.last_login_at THEN excluded.last_login_at
                    ELSE users.last_login_at
                END
            ''',
            (username, password_hash, created_at, last_login_at),
        )
        count += 1
    conn.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan uploads and sync new audio into local SQLite DB.")
    parser.add_argument("--uploads-dir", default="uploads", help="Path to uploads directory")
    parser.add_argument("--db-path", default="data/tasks.db", help="Path to SQLite DB")
    parser.add_argument("--user", default=None, help="只同步指定用户名（默认扫描全部用户）")
    parser.add_argument("--remote-db", default=None,
                        help="远端数据库副本路径；提供后会把其中的登录凭据（用户名+密码哈希）合并进本地库")
    args = parser.parse_args()

    uploads_dir = Path(args.uploads_dir)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not uploads_dir.exists():
        print(f"⚠️ uploads 目录不存在: {uploads_dir}")
        return

    # CLI 脚本没有 event loop，用同步版的 get_db_sync 保证连接 close()。
    from utils.db import get_db_sync

    total_inserted = 0
    total_updated = 0
    cred_synced = 0
    with get_db_sync(db_path=str(db_path)) as conn:
        ensure_schema(conn)

        if args.remote_db:
            cred_synced = sync_users_from_remote(conn, args.remote_db, args.user)

        if args.user:
            target = uploads_dir / args.user
            if not target.is_dir():
                print(f"⚠️ 指定的用户目录不存在: {target}")
                return
            users = [args.user]
        else:
            users = sorted([p.name for p in uploads_dir.iterdir() if p.is_dir()])
        for username in users:
            inserted, updated = sync_one_user(conn, uploads_dir, username)
            total_inserted += inserted
            total_updated += updated

    print("✅ 音频扫描同步完成")
    print(f"   - 用户数: {len(users)}")
    print(f"   - 新增任务: {total_inserted}")
    print(f"   - 更新任务: {total_updated}")
    if args.remote_db:
        print(f"   - 同步登录凭据(用户名+密码哈希): {cred_synced} 个用户")


if __name__ == "__main__":
    main()
