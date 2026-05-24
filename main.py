# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 1. 始终导入轻量级的采集模块
from data_collector import router as collector_router

app = FastAPI()

# ---------------------------------------------------------------------------
# CORS：默认 deny-all 跨域；同源请求（页面是这个 app 自己 serve 的）不受影响。
#
# 关键约束：我们用 `Authorization: Bearer <JWT>` 鉴权而非 cookie，所以一个恶意页面
# (evil.com) 即便发 fetch 到本接口，也拿不到 JWT（隔在另一个 origin 的 localStorage 里）
# —— CSRF 已被这套机制根本性挡掉。配 CORS 中间件主要是显式表达"不接受任意 origin"
# 的卫生层 + 避免浏览器误报。
#
# 部署时把生产 / cpolar / dev 的 origin 用逗号分隔放进 ALLOWED_ORIGINS 即可，例如：
#   ALLOWED_ORIGINS="https://xxx.cpolar.io,http://localhost:8000"
# ---------------------------------------------------------------------------
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()] if _origins_env else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # 不用 cookie，关掉 credentials；并且和 allow_origins=["*"] 不冲突
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
if ALLOWED_ORIGINS:
    print(f"🌐 CORS 允许的跨域 origin: {ALLOWED_ORIGINS}")
else:
    print("🌐 CORS 未配置 ALLOWED_ORIGINS，所有跨域请求会被浏览器拦截（同源不受影响）")

# 确保基础目录存在
os.makedirs("uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# 注意：uploads/ 不再通过 StaticFiles 公开（StaticFiles 不走依赖系统，
# 任何人猜到 /uploads/{user}/task_X.wav 都能下载）。音频统一走受保护路由
# `GET /api/audio/{task_id}`，详见 data_collector.get_audio。
app.mount("/static", StaticFiles(directory="static"), name="static")

# 注册采集路由 (必须开启)
app.include_router(collector_router)

# 2. 读取环境变量开关 (默认是 0，即完整模式)
# 如果 COLLECT_ONLY 为 "1"，则只开启采集功能
COLLECT_ONLY = os.environ.get("COLLECT_ONLY", "0") == "1"

@app.get("/api/config")
async def get_system_config():
    """让前端知道当前是否处于'仅采集'模式"""
    return {"collect_only": COLLECT_ONLY}

if not COLLECT_ONLY:
    try:
        # 只有在非 collect-only 模式下，才去 import 那些包含 torch 的重型模块
        from dataset_builder import router as dataset_router
        from finetune_controller import router as finetune_router
        from inference_controller import router as inference_router

        # 注册重型路由
        app.include_router(dataset_router)
        app.include_router(finetune_router)
        app.include_router(inference_router)
        print("✅ 完整模式启动：已成功加载微调和语音识别模块。")
    except ImportError as e:
        print(f"⚠️ 警告: 缺少核心依赖 ({e})，为防止崩溃，系统将仅以采集模式运行。")
else:
    print("🚀 轻量采集模式启动 (COLLECT_ONLY=1)：已跳过所有 AI 模型依赖！")


# 前端页面入口
@app.get("/", response_class=FileResponse)
async def read_root():
    """直接返回纯前端 HTML，避开 Jinja2 渲染"""
    return "templates/index.html"