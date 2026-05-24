"""路径穿越防护 + 用户输入白名单。

两层防御：
  1. 入口白名单 (`USERNAME_RE` / `MODEL_NAME_RE` / `ALLOWED_AUDIO_EXT`) 在
     register / start_finetune / upload_audio 处拒绝危险字符。
  2. `safe_join` 在拼路径时做 abspath + commonpath 检查，任何漏过白名单的输入
     仍会在写盘瞬间被拦下。

`USERNAME_RE` 允许 CJK 字符是因为目标用户（脑瘫儿童 + 家长）习惯用中文小名做代号。
"""
import os
import re
from typing import Optional

from fastapi import HTTPException


# 项目根目录 + 业务子目录的绝对路径
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
UPLOAD_BASE = os.path.join(PROJECT_ROOT, "uploads")
DATASET_BASE = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")


# Username: ASCII 字母/数字/下划线/连字符 + 常用汉字 (CJK Unified Ideographs U+4E00-U+9FA5)
# 1-20 字符，与 UserAuth.username 长度约束对齐。连字符放最后表示字面量。
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_一-龥-]{1,20}$")

# Model name: 仅 ASCII。首字符必须是字母/数字/下划线，否则 `..` / `.` / `.foo` /
# `-foo` 会通过简单字符类，让 output/{u}/{m}/ 实际指向上级目录。
MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,79}$")

# 浏览器录音输出 webm；桌面/移动端可能上传 wav/mp3/m4a；ogg 是 Firefox 默认。
ALLOWED_AUDIO_EXT = {"webm", "wav", "mp3", "ogg", "m4a"}


def is_valid_username(name: str) -> bool:
    return bool(USERNAME_RE.match(name))


def is_valid_model_name(name: str) -> bool:
    return bool(MODEL_NAME_RE.match(name))


def is_allowed_audio_ext(ext: str) -> bool:
    return ext.lower() in ALLOWED_AUDIO_EXT


def safe_join(base: str, *parts: str) -> str:
    """像 os.path.join，但拒绝逃出 base 的路径。

    返回拼接后的绝对路径；若结果不在 `base` 之下立即抛 HTTPException(400)。
    abspath 会折叠 `..`，commonpath 检查公共前缀必须等于 base。

    注意：`safe_join(OUTPUT_BASE, user, "..")` 会被放过——结果折叠到 OUTPUT_BASE
    本身，仍在 base 之下。要挡住 `model_name=".."` 这种"在 user 子目录里上跳一层"
    的攻击，必须用两层 safe_join：先到 user 根，再 join model_name。
    """
    base_abs = os.path.abspath(base)
    if not parts:
        return base_abs

    # os.path.join("uploads", "/etc/passwd") 在 POSIX 上会丢弃 "uploads",
    # 直接返回 "/etc/passwd"。abspath 不会改变它，后续 commonpath 检查会拦下来。
    target_abs = os.path.abspath(os.path.join(base_abs, *parts))

    try:
        common = os.path.commonpath([base_abs, target_abs])
    except ValueError:
        # 跨盘符 / 空字符串等情况
        raise HTTPException(status_code=400, detail="非法路径")

    if common != base_abs:
        raise HTTPException(status_code=400, detail="非法路径")

    return target_abs


def safe_resolve_under(base: str, candidate: str) -> Optional[str]:
    """`safe_join` 的软失败版本——越界返 None，不抛异常。

    用于处理 DB 里读出的脏路径：宁可"少删一个孤儿 wav"也不要因为一条脏数据
    把整个请求打成 400。
    """
    try:
        return safe_join(base, candidate)
    except HTTPException:
        return None
