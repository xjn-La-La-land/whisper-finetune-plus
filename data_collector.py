# data_collector.py
import os
import sqlite3
import shutil
import aiofiles
import subprocess
import time
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

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
    conn = sqlite3.connect(DB_PATH)
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
    # 新增：users 表，用于记录已注册的代号
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY
        )
    ''')
    # 用户微调模型表：仅记录用户名与模型名，路径统一约定为 output/{username}/{model_name}
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE(username, model_name)
        )
    ''')
    conn.commit()
    conn.close()

init_db()



def sync_user_words_txt(username: str):
    """
    每次任务有增、删、改时调用。
    将该用户的所有文本按行同步到 /uploads/{username}/words.txt 中。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 按照 id 排序，确保和界面的显示顺序一致
    c.execute('SELECT text_content FROM tasks WHERE username = ? ORDER BY id ASC', (username,))
    rows = c.fetchall()
    conn.close()

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


class UserAuth(BaseModel):
    username: str


@router.post("/api/register")
async def register_user(user: UserAuth):
    clean_name = user.username.strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 检查是否同名
    c.execute('SELECT username FROM users WHERE username = ?', (clean_name,))
    if c.fetchone():
        conn.close()
        return JSONResponse(status_code=400, content={"message": "⚠️ 该代号已被注册！如果是你本人，请点击「直接登录」。"})
    
    # 未被注册则插入新用户
    c.execute('INSERT INTO users (username) VALUES (?)', (clean_name,))
    conn.commit()
    conn.close()
    
    return {"message": "注册成功"}


@router.post("/api/login")
async def login_user(user: UserAuth):
    clean_name = user.username.strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 检查用户是否存在
    c.execute('SELECT username FROM users WHERE username = ?', (clean_name,))
    if not c.fetchone():
        conn.close()
        return JSONResponse(status_code=400, content={"message": "⚠️ 找不到该代号！请先点击「注册新代号」。"})
        
    conn.close()
    return {"message": "登录成功"}


@router.post("/api/upload_txt")
async def upload_txt(username: str, file: UploadFile = File(...)):
    content = await file.read()
    lines = content.decode('utf-8').split('\n')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    added_count = 0
    for line in lines:
        line = line.strip()
        if line:
            c.execute('INSERT INTO tasks (username, text_content) VALUES (?, ?)', (username, line,))
            added_count += 1
    conn.commit()
    conn.close()

    sync_user_words_txt(username)
    
    return {"message": f"成功导入 {added_count} 条文本任务！"}


@router.get("/api/tasks")
async def get_tasks(username: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM tasks WHERE username = ?', (username,))
    tasks = [dict(row) for row in c.fetchall()]
    conn.close()

    for task in tasks:
        audio_path = task.get("audio_path")
        updated_at = task.get("updated_at")
        if audio_path and updated_at:
            task["audio_path"] = f"{audio_path}?v={updated_at}"

    return tasks


@router.post("/api/task")
async def add_single_task(username: str, task: TaskText):
    """添加单个任务"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 注意这里改用 username.strip()，而不是 task.username
    c.execute('INSERT INTO tasks (username, text_content) VALUES (?, ?)', (username.strip(), task.text.strip()))
    conn.commit()
    conn.close()

    # 同步 words.txt
    sync_user_words_txt(username)

    return {"message": "添加成功"}


@router.put("/api/task/{task_id}")
async def update_task_text(task_id: int, username: str, task: TaskText):
    """修改现有任务的文本"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE tasks SET text_content = ? WHERE id = ? AND username = ?', (task.text.strip(), task_id, username))
    conn.commit()
    conn.close()

    # 同步 words.txt
    sync_user_words_txt(username)

    return {"message": "修改成功"}


@router.delete("/api/task/{task_id}")
async def delete_task(task_id: int):
    """删除任务，并同时清理服务器上的音频文件"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT username, audio_path FROM tasks WHERE id = ?', (task_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        return {"message": "任务不存在"}
        
    username, audio_path = row[0], row[1]
    
    if audio_path:
        clean_path = audio_path.split('?')[0].lstrip('/')
        if os.path.exists(clean_path):
            os.remove(clean_path)
            
    c.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    sync_user_words_txt(username)
    
    return {"message": "删除成功"}


@router.delete("/api/tasks")
async def clear_all_tasks(username: str):
    """清空所有任务，并同时清理服务器上的所有音频文件"""

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 只删当前用户的记录
    c.execute('DELETE FROM tasks WHERE username = ?', (username,))
    conn.commit()
    conn.close()
    
    # 清理该用户专属的文件夹
    user_upload_dir = os.path.join(UPLOAD_DIR, username)
    if os.path.exists(user_upload_dir):
        shutil.rmtree(user_upload_dir)
            
    return {"message": "当前用户的任务及音频已清空"}


@router.post("/api/upload_audio/{task_id}")
async def upload_audio(task_id: int, username: str, audio: UploadFile = File(...)):
    """接收前端录音，并转换为 16kHz 单声道 wav 格式，更新数据库状态"""
    # 动态拼接用户文件夹并创建
    user_upload_dir = os.path.join(UPLOAD_DIR, username)
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
            
    except subprocess.CalledProcessError as e:
        return {"error": "音频转换失败，请检查服务器是否已安装 ffmpeg！"}
        
    # 更新数据库状态
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    updated_at = int(time.time() * 1000)
    db_audio_path = f"/uploads/{username}/{final_wav_name}"
    c.execute('''
        UPDATE tasks 
        SET audio_path = ?, is_completed = 1, updated_at = ?
        WHERE id = ? AND username = ?
    ''', (db_audio_path, updated_at, task_id, username))
    conn.commit()
    conn.close()
    
    return {"message": "录音保存并转换成功", "audio_path": db_audio_path}
