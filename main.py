# main.py
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 1. 始终导入轻量级的采集模块
from data_collector import router as collector_router

app = FastAPI()

# 确保基础目录存在
os.makedirs("uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# 挂载静态文件目录
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
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