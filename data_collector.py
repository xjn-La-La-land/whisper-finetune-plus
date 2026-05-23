# data_collector.py
import os
import sqlite3
import shutil
import aiofiles
import subprocess
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Depends
from pydantic import BaseModel, Field

from utils.db import get_db, get_db_sync
from utils.auth import hash_password, verify_password, create_access_token, get_current_user

# 1. 实例化 Router
router = APIRouter()

# 2. 目录和数据库配置 (原样保留)
UPLOAD_DIR = "uploads"
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "tasks.db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 3. 数据库初始化和辅助函数
def init_db():
    # 在模块 import 时同步调用，没有 event loop，使用同步版
    with get_db_sync() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                text_content TEXT NOT NULL,
                audio_path TEXT,
                is_completed BOOLEAN DEFAULT 0,
                updated_at INTEGER
            )
        ''')
        c.execute("PRAGMA table_info(tasks)")
        task_columns = {row[1] for row in c.fetchall()}
        if "updated_at" not in task_columns:
            c.execute('ALTER TABLE tasks ADD COLUMN updated_at INTEGER')
        # users 表：username + bcrypt 哈希过的密码
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT,
                created_at INTEGER,
                last_login_at INTEGER
            )
        ''')

        # --- users 表迁移逻辑 ---
        # P0-1 引入密码认证。如果 users 表是旧 schema（无 password_hash 列），
        # 升级时把旧用户行全部删除：旧行没有密码，让原用户走注册流程重新设密码。
        # tasks 表里的 username 字段不动，重新注册同名后即可恢复对原 tasks 的访问。
        c.execute("PRAGMA table_info(users)")
        user_columns = {row[1] for row in c.fetchall()}
        if "password_hash" not in user_columns:
            # 这是从旧 schema 升级而来：先添加列，再清空旧数据
            c.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
            c.execute('ALTER TABLE users ADD COLUMN created_at INTEGER')
            c.execute('ALTER TABLE users ADD COLUMN last_login_at INTEGER')
            c.execute('DELETE FROM users')
            print("⚠️  [init_db] 检测到旧版无密码的 users 表，已清空。原用户需重新注册同名账号以恢复 tasks 访问。")
        else:
            # 已经是新 schema，可能后面有人手动建过 created_at 之类。逐字段补齐：
            for col in ("created_at", "last_login_at"):
                if col not in user_columns:
                    c.execute(f'ALTER TABLE users ADD COLUMN {col} INTEGER')

        # 用户微调模型表：仅记录用户名与模型名，路径统一约定为 output/{username}/{model_name}
        c.execute('''
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
        ''')
        # 自动升级逻辑
        c.execute("PRAGMA table_info(models)")
        model_columns = {row[1] for row in c.fetchall()}
        if "is_published" not in model_columns:
            c.execute('ALTER TABLE models ADD COLUMN is_published INTEGER DEFAULT 0')
        if "version_tag" not in model_columns:
            c.execute('ALTER TABLE models ADD COLUMN version_tag TEXT')

        conn.commit()

init_db()



async def sync_user_words_txt(username: str):
    """
    每次任务有增、删、改时调用。
    将该用户的所有文本按行同步到 /uploads/{username}/words.txt 中。

    DB 查询走异步（aiosqlite），文件写入仍是同步的小 IO（几 KB），不再单独 to_thread。
    """
    # 先把数据从 DB 取出来（async with 块退出时连接立即释放），
    # 后面的文件 IO 如果抛异常也不会牵连数据库连接。
    async with get_db() as conn:
        # 按照 id 排序，确保和界面的显示顺序一致
        cursor = await conn.execute(
            'SELECT text_content FROM tasks WHERE username = ? ORDER BY id ASC',
            (username,)
        )
        rows = await cursor.fetchall()

    user_upload_dir = os.path.join(UPLOAD_DIR, username)
    
    # 如果用户没有任何任务，且目录不存在，就直接返回
    if not rows and not os.path.exists(user_upload_dir):
        return
        
    # 确保文件夹存在
    os.makedirs(user_upload_dir, exist_ok=True)
    txt_path = os.path.join(user_upload_dir, 'words.txt')
    
    # 将文本逐行写入 words.txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(row[0] + '\n')



# 定义一个用于接收前端文本数据的数据模型
class TaskText(BaseModel):
    text: str


# 用户名 / 密码 字段规则
# - username: 1-20 字符（含中文），可后续接入 P0-2 的更严格白名单
# - password: 4-16 字符任意（允许 PIN 风格的纯数字，也允许长一些的密码）
class UserAuth(BaseModel):
    username: str = Field(..., min_length=1, max_length=20)
    password: str = Field(..., min_length=4, max_length=16, description="4-16 字符密码")


@router.post("/api/register")
async def register_user(user: UserAuth):
    """注册新代号 + 密码。

    返回 JWT token，前端直接持有即可登录态。
    """
    clean_name = user.username.strip()
    if not clean_name:
        return JSONResponse(status_code=400, content={"message": "⚠️ 代号不能为空。"})

    pw_hash = hash_password(user.password)
    now_ts = int(time.time())

    async with get_db() as conn:
        # 检查是否同名
        cursor = await conn.execute('SELECT username FROM users WHERE username = ?', (clean_name,))
        if await cursor.fetchone():
            return JSONResponse(
                status_code=400,
                content={"message": "⚠️ 该代号已被注册！如果是你本人，请点击「直接登录」。"}
            )

        # 插入新用户（用户名 + 密码哈希 + 注册时间）
        await conn.execute(
            'INSERT INTO users (username, password_hash, created_at, last_login_at) VALUES (?, ?, ?, ?)',
            (clean_name, pw_hash, now_ts, now_ts)
        )
        await conn.commit()

    token = create_access_token(clean_name)
    return {"message": "注册成功", "token": token, "username": clean_name}


@router.post("/api/login")
async def login_user(user: UserAuth):
    """用代号 + 密码登录，成功返回 JWT token。

    返回的 401/400 错误信息刻意保持笼统（"代号或密码错误"），不告诉攻击者哪个错了，
    避免泄露"哪些用户名存在"。
    """
    clean_name = user.username.strip()
    generic_error = JSONResponse(
        status_code=400,
        content={"message": "⚠️ 代号或密码错误，请重试。"}
    )

    async with get_db() as conn:
        cursor = await conn.execute(
            'SELECT password_hash FROM users WHERE username = ?',
            (clean_name,)
        )
        row = await cursor.fetchone()
        if not row:
            return generic_error
        if not verify_password(user.password, row[0]):
            return generic_error

        # 更新最后登录时间
        await conn.execute(
            'UPDATE users SET last_login_at = ? WHERE username = ?',
            (int(time.time()), clean_name)
        )
        await conn.commit()

    token = create_access_token(clean_name)
    return {"message": "登录成功", "token": token, "username": clean_name}


@router.get("/api/me")
async def get_me(current_user: str = Depends(get_current_user)):
    """前端 token 健康检查 / "我是谁" 查询。

    任何带有效 token 的请求都能调用，返回当前用户名。
    前端在页面加载时用它确认 token 还没过期，避免后续 API 全部 401。
    """
    return {"username": current_user}


@router.post("/api/upload_txt")
async def upload_txt(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    content = await file.read()
    lines = content.decode('utf-8').split('\n')

    added_count = 0
    async with get_db() as conn:
        for line in lines:
            line = line.strip()
            if line:
                await conn.execute('INSERT INTO tasks (username, text_content) VALUES (?, ?)', (current_user, line,))
                added_count += 1
        await conn.commit()

    await sync_user_words_txt(current_user)

    return {"message": f"成功导入 {added_count} 条文本任务！"}


@router.get("/api/tasks")
async def get_tasks(current_user: str = Depends(get_current_user)):
    async with get_db(row_factory=sqlite3.Row) as conn:
        cursor = await conn.execute('SELECT * FROM tasks WHERE username = ? ORDER BY id ASC', (current_user,))
        rows = await cursor.fetchall()
    tasks = [dict(row) for row in rows]

    for task in tasks:
        audio_path = task.get("audio_path")
        updated_at = task.get("updated_at")
        if audio_path and updated_at:
            task["audio_path"] = f"{audio_path}?v={updated_at}"

    return tasks


@router.post("/api/task")
async def add_single_task(
    task: TaskText,
    current_user: str = Depends(get_current_user),
):
    """添加单个任务"""
    async with get_db() as conn:
        await conn.execute('INSERT INTO tasks (username, text_content) VALUES (?, ?)', (current_user, task.text.strip()))
        await conn.commit()

    # 同步 words.txt
    await sync_user_words_txt(current_user)

    return {"message": "添加成功"}


@router.put("/api/task/{task_id}")
async def update_task_text(
    task_id: int,
    task: TaskText,
    current_user: str = Depends(get_current_user),
):
    """修改现有任务的文本。

    如果文本发生了变化，旧录音和新文本就对不上号了，此时必须：
      1. 清空 audio_path 字段、重置 is_completed = 0
      2. 物理删除旧的 .wav 文件（避免 dataset_builder 后续误用）

    如果文本没变化（用户点开编辑又原样保存），不动录音状态。
    """
    new_text = task.text.strip()

    async with get_db() as conn:
        cursor = await conn.execute(
            'SELECT text_content, audio_path FROM tasks WHERE id = ? AND username = ?',
            (task_id, current_user)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="任务不存在或不属于当前用户")
        old_text, old_audio_path = row[0], row[1]

        # 文本未变 → 保留已有录音，直接返回
        if old_text == new_text:
            return {"message": "文本未变更", "audio_cleared": False}

        # 文本已变 → 旧录音作废
        await conn.execute(
            '''UPDATE tasks
               SET text_content = ?, audio_path = NULL, is_completed = 0, updated_at = ?
               WHERE id = ? AND username = ?''',
            (new_text, int(time.time() * 1000), task_id, current_user)
        )
        await conn.commit()

    # DB 提交完才删物理文件：即使文件删除失败，DB 状态也已经一致
    # （留下一个孤儿 wav 不影响 dataset_builder，因为 is_completed 已经是 0）
    audio_cleared = False
    if old_audio_path:
        clean_path = old_audio_path.split('?')[0].lstrip('/')
        if os.path.exists(clean_path):
            try:
                os.remove(clean_path)
                audio_cleared = True
            except OSError as e:
                # 不让文件删除失败阻止任务文本更新
                print(f"[update_task_text] 删除旧音频失败 {clean_path}: {e}")

    # 同步 words.txt
    await sync_user_words_txt(current_user)

    return {"message": "修改成功", "audio_cleared": audio_cleared}


@router.delete("/api/task/{task_id}")
async def delete_task(
    task_id: int,
    current_user: str = Depends(get_current_user),
):
    """删除任务，并同时清理服务器上的音频文件"""
    async with get_db() as conn:
        # 关键安全点：DELETE 时同时校验 username = current_user，
        # 防止用户 A 通过 task_id 删除用户 B 的录音
        cursor = await conn.execute(
            'SELECT audio_path FROM tasks WHERE id = ? AND username = ?',
            (task_id, current_user)
        )
        row = await cursor.fetchone()

        if not row:
            return {"message": "任务不存在或不属于当前用户"}

        audio_path = row[0]

        # 文件删除失败也不应导致连接泄漏（async with 块保证 close）
        if audio_path:
            clean_path = audio_path.split('?')[0].lstrip('/')
            if os.path.exists(clean_path):
                os.remove(clean_path)

        await conn.execute(
            'DELETE FROM tasks WHERE id = ? AND username = ?',
            (task_id, current_user)
        )
        await conn.commit()

    await sync_user_words_txt(current_user)

    return {"message": "删除成功"}


@router.delete("/api/tasks")
async def clear_all_tasks(current_user: str = Depends(get_current_user)):
    """清空所有任务，并同时清理服务器上的所有音频文件"""
    async with get_db() as conn:
        # 只删当前用户的记录
        await conn.execute('DELETE FROM tasks WHERE username = ?', (current_user,))
        await conn.commit()

    # 清理该用户专属的文件夹
    user_upload_dir = os.path.join(UPLOAD_DIR, current_user)
    if os.path.exists(user_upload_dir):
        shutil.rmtree(user_upload_dir)

    return {"message": "当前用户的任务及音频已清空"}


@router.post("/api/upload_audio/{task_id}")
async def upload_audio(
    task_id: int,
    audio: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    """接收前端录音，并转换为 16kHz 单声道 wav 格式，更新数据库状态"""
    # 动态拼接用户文件夹并创建
    user_upload_dir = os.path.join(UPLOAD_DIR, current_user)
    os.makedirs(user_upload_dir, exist_ok=True)
    # 浏览器通常录制为 webm 格式
    file_extension = audio.filename.split('.')[-1] if '.' in audio.filename else 'webm'
    temp_file_name = f"temp_task_{task_id}.{file_extension}"
    final_wav_name = f"task_{task_id}.wav"

    temp_file_path = os.path.join(user_upload_dir, temp_file_name)
    final_wav_path = os.path.join(user_upload_dir, final_wav_name)
    
    # 保存音频文件
    async with aiofiles.open(temp_file_path, 'wb') as out_file:
        content = await audio.read()
        await out_file.write(content)

    # 使用 FFmpeg 转换为 Whisper 标准格式
    # 参数说明：
    # -y: 覆盖输出文件
    # -i: 输入文件
    # -ar 16000: 采样率 16kHz
    # -ac 1: 单声道 (mono)
    # -c:a pcm_s16le: 16-bit little-endian PCM 编码
    try:
        subprocess.run([
            'ffmpeg', '-y', 
            '-i', temp_file_path,
            '-ar', '16000', 
            '-ac', '1', 
            '-c:a', 'pcm_s16le',
            final_wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 转换成功后，删除临时文件保持目录整洁
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="音频转换失败，请检查服务器是否已安装 ffmpeg！")
        
    # 更新数据库状态
    updated_at = int(time.time() * 1000)
    db_audio_path = f"/uploads/{current_user}/{final_wav_name}"
    async with get_db() as conn:
        await conn.execute('''
            UPDATE tasks
            SET audio_path = ?, is_completed = 1, updated_at = ?
            WHERE id = ? AND username = ?
        ''', (db_audio_path, updated_at, task_id, current_user))
        await conn.commit()

    return {"message": "录音保存并转换成功", "audio_path": db_audio_path}
