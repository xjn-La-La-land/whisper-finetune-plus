# Whisper-Finetune-Plus

> 端到端的 Whisper 语音识别**定制化微调与 Web 部署平台**。

## 项目简介

针对通用语音模型在特定发音特征（如儿童语音）上识别率不足的问题，本项目打通了从「个性化数据采集」到「模型微调」，再到「Web 端推理验证」的完整闭环。

通过开箱即用的 Web 交互界面 + 基于单 GPU 卡的轻量级训练 / 推理平台，用户能以极低门槛录制专属音频、自动构建数据集、对 Whisper 模型做个性化微调，并直接在线验证识别效果。微调核心训练逻辑基于开源项目 whisper-finetune 实现，并在其上构建了完整的工程化全栈流水线。

> 仓库仍在开发调试中：<https://github.com/xjn-La-La-land/whisper-finetune-plus.git>

## 核心特点

1. **多用户支持**：用户名登录，多用户隔离的 Web 界面，数据在后端按用户分开存储。
2. **GPU 状态管理（模型热插拔）**：为在单卡 GPU 上多用户复用且不爆显存（OOM），按需清空 / 加载用户模型——
   - GPU 正在微调或评估时，拒绝推理请求；
   - GPU 空闲或正在推理时，检查显存中加载的是谁的模型：是当前用户则直接复用；否则**清空显存**，加载该用户的微调模型（若无微调模型则加载 Base 模型）。
3. **可视化模型微调**：Web 端一键启动并实时监控微调（Loss 曲线实时绘制、微调后可在测试集上评估字错率）。
4. **浏览器端本地推理（默认推理路径）**：识别默认在用户浏览器里用 whisper.cpp 的 WASM 跑——音频不离开本机（隐私更好），且**不占用服务端 GPU**，多用户可并发识别。模型（基座或微调）导出为 ggml q5_0 量化版后由前端下载并缓存在 IndexedDB（见「快速开始」第 4b 步）。浏览器不支持时（旧浏览器 / 页面未跨源隔离 / 该模型无 ggml 版）自动降级到服务端 GPU 推理（`/api/recognition`，受第 2 点的 GPU 状态管理约束）。
5. **端侧部署**：微调后可导出 tflite，配合安卓 IME app 离线使用（见下方文档导航）。

## 项目结构

```
.
├── README.md ----------------------------- 项目门户（本文件）
├── main.py ------------------------------- 后端入口（FastAPI app + 路由挂载）
├── data_collector.py --------------------- 音频采集 / 登录注册 API
├── dataset_builder.py -------------------- 数据集生成 API
├── finetune.py --------------------------- 模型微调核心训练逻辑
├── finetune_controller.py ---------------- 微调控制 API（启动 / 监控 / 评估）
├── inference_controller.py --------------- 推理控制 API（含 GPU 模型热插拔）
├── shared_state.py ----------------------- GPU 状态 / 显存占用管理
├── download_whisper_models.py ------------ 下载 Whisper 基座模型
├── tflite_export.py ----------------------- tflite 导出（安卓 app 用；本地测试见 scripts/test_tflite.py）
├── ggml_export.py --------------------------- LoRA→ggml 导出 + q5_0 量化（whisper.cpp WASM 推理用）
├── requirements.txt ---------------------- Web 端依赖（pip / 环境名 whisper）
├── env.yaml ------------------------------ 安卓 App 端环境（conda / 环境名 whisper-app）
├── data/
│   └── tasks.db -------------------------- 后端数据库（用户 / 任务 / 模型记录）
├── static/
│   ├── js/ ------------------------------- 前端逻辑（app / 采集 / 微调 / 推理面板等）
│   ├── vendor/ --------------------------- 本地化的前端依赖库
│   └── wasm/ ----------------------------- 浏览器端 whisper.cpp WASM 推理产物（libmain.* / helpers.js）
├── templates/
│   └── index.html ------------------------ 前端单页
├── uploads/ ------------------------------ 各用户的音频与文本（按用户名分目录）
├── dataset/ ------------------------------ 各用户的数据集与微调模型
├── output/ ------------------------------- tflite 等导出产物
├── whisper-base-models/ ------------------ 下载的基座模型
├── utils/ -------------------------------- 工具模块（auth / db / 音频处理等）
├── scripts/ ------------------------------ 构建与运维脚本（build_whisper_cpp.sh 编 native+WASM；数据同步 sync_*；test_tflite.py 本地测试）
├── vendor/ ------------------------------- 第三方子模块 whisper.cpp + 转换资产（mel_filters）
├── bin/ ---------------------------------- whisper.cpp native 二进制（编译生成，未入库）
├── deploy/ ------------------------------- 公网部署（Cloudflare 隧道配置 + 运行手册）
├── doc/ ---------------------------------- 文档（Web.md / Adroid_app.md / TODO_*.md 路线图）
└── assets/ ------------------------------- 文档图片与示例音频
```

## 快速开始

以下是 **Web 端**的最小启动流程（依赖用 `requirements.txt`，conda 环境名 `whisper`）。需要一张支持 CUDA 的 NVIDIA 显卡（用于微调与推理加速）。

> 安卓 App 开发用的是另一套环境（`env.yaml`，环境名 `whisper-app`），见 [`doc/Adroid_app.md`](doc/Adroid_app.md)。

```bash
# 1. 克隆代码（--recursive 一并拉取 whisper.cpp 子模块；
#    已 clone 过没带的话补一句： git submodule update --init --recursive）
git clone --recursive https://github.com/xjn-La-La-land/whisper-finetune-plus.git
cd whisper-finetune-plus

# 2. 建环境（Python 3.11）并装依赖
conda create -n whisper python=3.11 -y && conda activate whisper
sudo apt update && sudo apt install ffmpeg        # 系统依赖：音频流转码
pip install -r requirements.txt

# 3. 下载基座模型到 ./whisper-base-models（默认 ModelScope 源，境内直连免代理）
python download_whisper_models.py whisper-small

# 4. 编译 whisper.cpp native 二进制
scripts/build_whisper_cpp.sh --native-only

# 4b.（可选）转换基座模型的 ggml
python ggml_export.py --base-models        # 默认 tiny/base/small（medium 太大、浏览器跑不动）

# 5. 启动后端
uvicorn main:app --host 127.0.0.1 --port 8000
```

本机浏览器打开 <http://localhost:8000> 即可使用。要让外网访问（cpolar 快速验证 / Cloudflare 固定域名公网部署）及完整使用教程，见 [`doc/Web.md`](doc/Web.md)。

## 文档导航

| 文档 | 内容 |
| ---- | ---- |
| [`doc/Web.md`](doc/Web.md) | 网页端部署（cpolar 快速验证 / Cloudflare 公网两条路径）+ 完整使用教程（采集 / 微调 / 识别） |
| [`deploy/README.md`](deploy/README.md) | 公网部署运行手册（Cloudflare 具名隧道 + 固定域名，含 `JWT_SECRET_KEY`、开机 / 重启恢复流程） |
| [`doc/Adroid_app.md`](doc/Adroid_app.md) | 安卓 IME app 接入（tflite 导出、ModelScope 发布、app 与服务器连接配置） |
| [`doc/TODO_*.md`](doc/) | 路线图（代码评审 / 公网部署 / whisper.cpp WASM） |
