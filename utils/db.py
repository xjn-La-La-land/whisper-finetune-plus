"""数据库连接工具（同步 + 异步）。

提供两个上下文管理器：

  * `get_db()`      —— **异步**，供 FastAPI handler / 其他 async 上下文使用。
                       底层是 aiosqlite，SQL 在后台线程执行，不会阻塞 event loop。
  * `get_db_sync()` —— **同步**，仅供模块 import 时的 init / CLI 脚本等
                       没有 event loop 的场景使用。

两者都保证退出 with 块时连接被 close()，包括异常路径。

为什么不直接用 `with sqlite3.connect(...) as conn:`?
    sqlite3.Connection 自带的 context manager 只负责 commit/rollback 事务，
    并 *不会* 调用 conn.close()。在异常路径下连接对象会被遗留到 GC 才回收，
    在高并发或长生命周期进程里可能耗尽 OS 文件描述符 / WAL 文件锁。

aiosqlite vs 直接同步 + asyncio.to_thread?
    早期方案是 `await asyncio.to_thread(c.execute, ...)`，但每条语句都开一个线程
    （或线程池任务）开销高，且 cursor 状态难维护。aiosqlite 内部用一个长期运行的
    worker thread 串行处理一个连接的所有 SQL，开销稳定且 API 自然。

使用方式（异步）：

    from utils.db import get_db

    async with get_db() as conn:
        cursor = await conn.execute('SELECT ...')
        rows = await cursor.fetchall()
        # 写操作仍需显式 commit
        await conn.execute('INSERT ...')
        await conn.commit()
    # 离开 async with 块后 conn 保证已 close

若需要 sqlite3.Row（行 → dict-like）:

    import sqlite3
    async with get_db(row_factory=sqlite3.Row) as conn:
        ...

使用方式（同步，仅 init / CLI）：

    from utils.db import get_db_sync

    with get_db_sync() as conn:
        conn.execute('CREATE TABLE ...')
        conn.commit()
"""
import os
import sqlite3
from contextlib import contextmanager, asynccontextmanager
from typing import AsyncGenerator, Callable, Generator, Optional

import aiosqlite

# DB_PATH 锚定在项目根目录，避免不同模块用相对路径解析出歧义。
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tasks.db")


@asynccontextmanager
async def get_db(
    row_factory: Optional[Callable] = None,
    db_path: str = DB_PATH,
) -> AsyncGenerator[aiosqlite.Connection, None]:
    """异步获取一个 SQLite 连接，保证退出 async with 块时 close()。

    Args:
        row_factory: 可选，如 `sqlite3.Row`，让 fetchall 返回 dict-like 对象。
        db_path:     可选，仅在测试场景 override。生产代码不要传。
    """
    conn = await aiosqlite.connect(db_path)
    if row_factory is not None:
        conn.row_factory = row_factory
    try:
        yield conn
    finally:
        await conn.close()


@contextmanager
def get_db_sync(
    row_factory: Optional[Callable] = None,
    db_path: str = DB_PATH,
) -> Generator[sqlite3.Connection, None, None]:
    """同步获取一个 SQLite 连接，保证退出 with 块时 close()。

    仅供没有 event loop 的场景（模块 import 时的 init / CLI 脚本）使用。
    FastAPI handler 等 async 上下文请使用 `get_db()` 异步版本。
    """
    conn = sqlite3.connect(db_path)
    if row_factory is not None:
        conn.row_factory = row_factory
    try:
        yield conn
    finally:
        conn.close()
