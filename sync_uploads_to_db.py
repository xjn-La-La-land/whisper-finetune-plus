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
            username TEXT PRIMARY KEY
        )
        '''
    )

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE(username, model_name)
        )
        '''
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan uploads and sync new audio into local SQLite DB.")
    parser.add_argument("--uploads-dir", default="uploads", help="Path to uploads directory")
    parser.add_argument("--db-path", default="data/tasks.db", help="Path to SQLite DB")
    args = parser.parse_args()

    uploads_dir = Path(args.uploads_dir)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not uploads_dir.exists():
        print(f"⚠️ uploads 目录不存在: {uploads_dir}")
        return

    conn = sqlite3.connect(db_path)
    ensure_schema(conn)

    total_inserted = 0
    total_updated = 0
    users = sorted([p.name for p in uploads_dir.iterdir() if p.is_dir()])
    for username in users:
        inserted, updated = sync_one_user(conn, uploads_dir, username)
        total_inserted += inserted
        total_updated += updated

    conn.close()

    print("✅ 音频扫描同步完成")
    print(f"   - 用户数: {len(users)}")
    print(f"   - 新增任务: {total_inserted}")
    print(f"   - 更新任务: {total_updated}")


if __name__ == "__main__":
    main()
