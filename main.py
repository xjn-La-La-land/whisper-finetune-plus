# main.py
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 导入拆分后的路由模块
from data_collector import router as collector_router
from dataset_builder import router as dataset_router
from finetune_controller import router as finetune_router
from inference_controller import router as inference_router

app = FastAPI()
# 确保基础目录存在
os.makedirs("uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# 挂载静态文件目录 (用于前端音频播放)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
# 允许浏览器访问 static 文件夹里的所有前端 JS 文件！
app.mount("/static", StaticFiles(directory="static"), name="static")

# 将子模块的路由注册到主应用中
app.include_router(collector_router)
app.include_router(dataset_router)
app.include_router(finetune_router)
app.include_router(inference_router)


# 前端页面入口
@app.get("/", response_class=FileResponse)
async def read_root():
    """直接返回纯前端 HTML，避开 Jinja2 渲染"""
    return "templates/index.html"