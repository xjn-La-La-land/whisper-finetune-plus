"""认证与授权工具。

设计要点
========
* 密码：bcrypt（含 salt，常量时间对比）
* Token：JWT HS256，有效期 7 天，sub=username；活跃请求会滑动续期（响应头 X-Refresh-Token）
* Secret：环境变量 `JWT_SECRET_KEY` > `data/.jwt_secret` 文件 > 自动生成
  - 自动生成会持久化到文件，使服务重启后已发的 token 仍然有效
  - 生产部署建议显式设置 JWT_SECRET_KEY 环境变量
* 两个依赖：
  - `get_current_user`            从 `Authorization: Bearer <token>` 头取（默认）
  - `get_current_user_from_query` 从 `?token=...` 查询参数取（仅 SSE / EventSource）
"""
import os
import secrets
import time
from typing import Optional

import bcrypt
from fastapi import Header, HTTPException, Query, Response, status
from jose import JWTError, jwt

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
SECRET_PATH = os.path.join(PROJECT_ROOT, "data", ".jwt_secret")

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 7
# 滑动续期：token 自签发起被用过这么多天后，下一次带它的请求会通过响应头 X-Refresh-Token
# 下发一个新 token，把 7 天有效期重新顶满。于是只要用户保持活跃就永不掉线，连续
# JWT_EXPIRY_DAYS 天不活跃才会真正过期、需要重新登录。设成 1 天是为了把续期限制到
# 每会话约一天一次，避免每个请求都重签。
JWT_RENEW_AFTER_DAYS = 1
# 续期 token 走这个响应头返回；前端 apiFetch 见到就替换本地 token。
# 跨域部署时需在 CORS 的 expose_headers 里放行它（见 main.py）。
REFRESH_HEADER = "X-Refresh-Token"


# ----------------------------------------------------------------------------
# JWT secret 加载
# ----------------------------------------------------------------------------
def _load_or_generate_secret() -> str:
    """加载 JWT secret。优先级：环境变量 > 持久化文件 > 自动生成。"""
    env_secret = os.environ.get("JWT_SECRET_KEY")
    if env_secret:
        return env_secret

    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH, encoding="utf-8") as f:
            secret = f.read().strip()
        if secret:
            return secret

    # 首次启动，生成 32 字节随机 secret 并持久化（gitignore 已覆盖 data/）
    new_secret = secrets.token_urlsafe(32)
    os.makedirs(os.path.dirname(SECRET_PATH), exist_ok=True)
    with open(SECRET_PATH, "w", encoding="utf-8") as f:
        f.write(new_secret)
    try:
        # 仅文件所有者可读写
        os.chmod(SECRET_PATH, 0o600)
    except OSError:
        # Windows / FAT 等不支持 chmod 的文件系统忽略
        pass
    print(f"⚠️  [auth] 已生成新的 JWT secret 并写入 {SECRET_PATH}")
    print(f"   生产部署建议显式设置环境变量 JWT_SECRET_KEY，以便容器重启后保留登录态")
    return new_secret


JWT_SECRET = _load_or_generate_secret()


# ----------------------------------------------------------------------------
# 密码哈希
# ----------------------------------------------------------------------------
def hash_password(plain: str) -> str:
    """bcrypt 哈希明文密码，返回可存储的字符串（含 salt 和 cost factor）。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: Optional[str]) -> bool:
    """常量时间验证明文密码 vs 哈希。哈希缺失或非法时返回 False，永不抛异常。"""
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------------------------
# JWT 编解码
# ----------------------------------------------------------------------------
def create_access_token(username: str) -> str:
    """生成一个 7 天过期的 JWT token。"""
    now = int(time.time())
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + JWT_EXPIRY_DAYS * 24 * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_payload(token: str) -> Optional[dict]:
    """解码 JWT，验证签名 + exp，返回完整 payload。失败（含过期）返回 None。"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


def _decode_token(token: str) -> Optional[str]:
    """解码 JWT，返回 sub (username)。失败返回 None。"""
    payload = _decode_payload(token)
    return payload.get("sub") if payload else None


def _maybe_refresh(response: Optional[Response], payload: dict) -> None:
    """滑动续期：token 已签发超过 JWT_RENEW_AFTER_DAYS 天时，签发新 token 经响应头下发。

    前端 apiFetch 见到 REFRESH_HEADER 就替换本地 token，于是活跃用户的有效期被不断顶满、
    永不掉线。续期被门限到每会话约一天一次（续期后 iat 归零，要再过一天才会触发），
    避免每个请求都重签。response 为 None（理论上不会发生）时静默跳过。
    """
    if response is None:
        return
    iat = payload.get("iat")
    sub = payload.get("sub")
    if not sub or not isinstance(iat, (int, float)):
        return
    if int(time.time()) - int(iat) >= JWT_RENEW_AFTER_DAYS * 24 * 3600:
        response.headers[REFRESH_HEADER] = create_access_token(sub)


# ----------------------------------------------------------------------------
# FastAPI 依赖
# ----------------------------------------------------------------------------
_AUTH_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="未授权或登录已过期，请重新登录",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    response: Response,
    authorization: Optional[str] = Header(None),
) -> str:
    """标准依赖：从 `Authorization: Bearer <token>` 头取出当前用户名。

    用在 99% 的 API handler 上。token 缺失/格式错/无效/过期 → 401。
    token 用过一段时间后会通过响应头 X-Refresh-Token 滑动续期（见 _maybe_refresh）。

    注：`response` 由 FastAPI 注入，设置的响应头会合并进最终响应——前提是 handler 返回
    普通数据（FastAPI 负责序列化）。若 handler 直接返回 FileResponse/StreamingResponse，
    该头不会带上，但用户的其它常规 JSON 请求（轮询、/api/me 等）足以触发续期，不影响效果。
    """
    if not authorization:
        raise _AUTH_ERROR
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise _AUTH_ERROR
    payload = _decode_payload(parts[1])
    if not payload or not payload.get("sub"):
        raise _AUTH_ERROR
    _maybe_refresh(response, payload)
    return payload["sub"]


async def get_current_user_from_query(token: Optional[str] = Query(None)) -> str:
    """SSE 专用依赖：从 `?token=...` 查询参数取用户名。

    EventSource (浏览器 SSE) 不能附加自定义 Header，所以 SSE 端点只能走 query。
    这是经过权衡的选择——token 会出现在服务器访问日志里，但 dev/cpolar 部署可接受。
    """
    if not token:
        raise _AUTH_ERROR
    username = _decode_token(token)
    if not username:
        raise _AUTH_ERROR
    return username
