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
    # 滑动续期下发的 token 走这个响应头，跨域时浏览器默认读不到自定义响应头，需显式放行
    expose_headers=["X-Refresh-Token"],
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

# ---------------------------------------------------------------------------
# 2. 启动模式 = 显式开关 (COLLECT_ONLY) + 运行时能力探测 (torch / CUDA)
#
#   - COLLECT_ONLY=1：纯采集门户（公网给标注员用 / 纯 CPU 常驻机刻意不装 torch）。
#     语义不变：只挂采集路由，前端只显示「音频采集」一个 tab。
#   - COLLECT_ONLY=0：完整模式。这里不再假设一定有 GPU，而是探测本机能力：
#       · HAS_TORCH —— torch 能否 import（决定推理 / 微调路由是否注册）
#       · HAS_GPU   —— 有没有 CUDA（决定「微调」能否真正启动；推理 CPU/GPU 都行）
#     推理在 CPU 上也能跑（inference_controller 会自动退回 cpu + float32），
#     所以 CPU 机器照样显示三个 tab，只是「微调」在前端置灰、后端兜底拒绝。
# ---------------------------------------------------------------------------
COLLECT_ONLY = os.environ.get("COLLECT_ONLY", "0") == "1"

HAS_TORCH = False
HAS_GPU = False
if not COLLECT_ONLY:
    try:
        import torch
        HAS_TORCH = True
        HAS_GPU = bool(torch.cuda.is_available())
    except ImportError:
        # 非采集模式却没装 torch：不崩，按「推理 / 微调不可用」降级
        # （前端三个 tab 仍在，但相关功能禁用）。
        print("⚠️ 未检测到 torch：推理与微调将不可用（如需启用请安装 torch / transformers 等依赖）。")

# 微调：必须有 GPU。
FINETUNE_ENABLED = HAS_GPU

# 服务端推理路由 /api/recognition 是否注册（顶层 import torch，需装了 torch 才有）。
# 注意：推理的「主路径」规划是浏览器端 whisper.cpp WASM（见 TODO_WHISPER_CPP_WASM.md），
# WASM 跑在用户浏览器、不依赖服务端 torch/GPU。所以这个标志只代表「服务端 GPU 兜底
# 路径是否可用」，**不要**用它来决定前端是否显示推理 tab —— WASM 能力由前端自己探测
# (WebAssembly / SIMD / IndexedDB，见 P2-5)。
SERVER_INFERENCE_FALLBACK = HAS_TORCH

@app.get("/api/config")
async def get_system_config():
    """
    把启动能力暴露给前端：
      - collect_only        纯采集门户（只显示采集 tab）
      - has_gpu             本机是否有可用 GPU
      - finetune_enabled    微调是否可用（需 GPU）
      - server_inference    服务端推理兜底 /api/recognition 是否可用（装了 torch 即可）。
                            推理主路径是浏览器 WASM，能否用由前端自行探测，与此无关。
    """
    return {
        "collect_only": COLLECT_ONLY,
        "has_gpu": HAS_GPU,
        "finetune_enabled": FINETUNE_ENABLED,
        "server_inference": SERVER_INFERENCE_FALLBACK,
    }

if COLLECT_ONLY:
    print("🚀 轻量采集模式启动 (COLLECT_ONLY=1)：已跳过所有 AI 模型依赖！")
else:
    # 微调 / 数据集路由顶层不依赖 torch（训练在子进程里跑），有没有 torch 都先挂上；
    # 真正能不能开训由 FINETUNE_ENABLED 决定（前端置灰 + 后端 start_finetune 兜底）。
    try:
        from dataset_builder import router as dataset_router
        from finetune_controller import router as finetune_router
        app.include_router(dataset_router)
        app.include_router(finetune_router)
    except ImportError as e:
        FINETUNE_ENABLED = False
        print(f"⚠️ 警告: 微调 / 数据集模块依赖缺失 ({e})，已跳过这两个模块。")

    # 推理模块顶层 import torch，只有装了 torch 才注册（即「服务端 GPU 兜底」路径）。
    # 推理主路径是浏览器端 WASM，不经过这里。
    if SERVER_INFERENCE_FALLBACK:
        try:
            from inference_controller import router as inference_router
            app.include_router(inference_router)
        except ImportError as e:
            SERVER_INFERENCE_FALLBACK = False
            print(f"⚠️ 警告: 服务端推理模块依赖缺失 ({e})，已跳过（不影响浏览器 WASM 推理）。")

    # 启动概要
    if HAS_GPU:
        print("✅ 完整模式启动（GPU）：微调 + 语音识别均可用。")
    elif HAS_TORCH:
        print("✅ CPU 模式启动：语音识别可用（CPU 推理）；微调需 GPU，已禁用。")
    else:
        print("⚠️ 降级启动：未装 torch，仅采集相关接口可用；推理 / 微调均不可用。")


# 前端页面入口
@app.get("/", response_class=FileResponse)
async def read_root():
    """直接返回纯前端 HTML，避开 Jinja2 渲染"""
    return "templates/index.html"