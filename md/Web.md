# Whisper语音识别项目网页端部署文档

## 一、项目结构

我们做了一个端到端的 Whisper 语音识别定制化微调与部署平台。针对通用语音模型在特定发音特征（如儿童语音）上识别率不足的问题，**本项目打通了从“个性化数据采集”到“模型微调”，再到“Web 端推理验证”的完整闭环。**

通过提供开箱即用的 Web 交互界面，基于单GPU卡的轻量级训练和推理平台，系统允许用户以极低的门槛录制专属音频、自动构建数据集，并对 Whisper 模型进行个性化微调，最后使用微调的模型进行在线语音识别。本项目微调核心训练逻辑基于开源项目 [whisper-finetune](https://www.google.com/search?q=这里可以填入该项目的GitHub链接) 实现，并在此基础上构建了完整的工程化全栈流水线。

项目代码已放在github中：https://github.com/xjn-La-La-land/whisper-finetune-plus.git（仍在开发调试中）

下面是整体的项目框架：

```
.
├── README.md
├── data_collector.py -------------------- 音频采集相关代码
├── dataset_builder.py ------------------- 数据集生成相关代码
├── evaluation.py ------------------------ 测试集评估
├── finetune.py -------------------------- 模型微调
├── finetune_controller.py --------------- 微调控制相关代码
├── inference_controller.py -------------- 推理控制相关代码
├── main.py ------------------------------ 后端入口
├── shared_state.py ---------------------- GPU 状态管理
├── data
│   └── tasks.db ------------------------- 后端数据库
├── static
│   └── js ------------------------------- 前端 javascript
│       ├── app.js
│       ├── audio_collector.js
│       ├── custom_audio.js
│       ├── finetune_panel.js
│       └── inference_panel.js
├── templates
│   └── index.html ----------------------- 前端 html
├── uploads ------------------------------ 存储不同用户的音频和文本数据
│   ├──  小明
│   └──  小蓝 
├── dataset ------------------------------ 存储不同用户的数据集和微调模型
│   ├──  小明
│   └──  小蓝
└── utils 
    ├── ...
```

## 二、核心特点

1. **多用户支持**：使用用户名登录，提供多用户隔离的Web界面，并且用户数据在后端分开存储；
2. **GPU状态管理**：为了实现单卡 GPU 的多用户复用，且保证显存不溢出（OOM），我们需要实现一个**“模型动态热插拔”**机制：
   * 如果 GPU 正在微调或评估，拒绝推理请求。
   * 如果 GPU 空闲或正在推理，检查当前显存里加载的是谁的模型。
   * 如果是当前请求用户的模型，直接复用；如果不是（或者是初次调用），则**清空显存**，加载该用户的微调模型（若无微调模型，则加载 Base 模型）。

3. **可视化模型微调**：支持在 Web 端直接启动并监控模型的微调进程。

## 三、一键部署

1. GPU 硬件：一张支持 CUDA 的 NVIDIA 显卡（用于模型微调与推理加速）

2. 克隆项目代码

   ```bash
   git clone https://github.com/xjn-La-La-land/whisper-finetune-plus.git
   ```

3. 安装与配置（建议使用 `conda` 或 `venv` 创建纯净的虚拟环境后进行安装）

   ```bash
   # 1. 安装 ffmpeg（用于处理和转换前端采集的音频流）
   sudo apt update && sudo apt install ffmpeg
   # 2. 安装 PyTorch (带 CUDA 加速的版本)
   # 请务必根据本机的 CUDA 版本去 PyTorch 官网 ([https://pytorch.org/](https://pytorch.org/)) 获取对应的安装命令。例如对于 CUDA 13.0：
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
   # 3. 安装项目依赖
   pip install -r requirements.txt
   ```

4. 启动服务：在项目根目录下，启动 FastAPI 后端服务

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. 使用 cpolar 内网穿透工具，将本地的 `8000` 端口映射到一个公网域名上。

   在 [cpolar 官网](https://dashboard.cpolar.com/get-started) 注册一个账号，然后安装 cpolar

   ```bash
   curl -L https://www.cpolar.com/static/downloads/install-release-cpolar.sh | sudo bash
   ```

   配置认证

   ```bash
   # token 可以在官网查看到
   cpolar authtoken YOUR_TOKEN
   ```

   启动 HTTP 穿透(在另一个终端)

   ```bash
   cpolar http 8000
   ```

   等待生成公网链接，随后在浏览器中访问这个链接即可进入 Web 交互界面。
   ![image-20260304113051222](G:\其他计算机\dummy\whisper-finetune\assets\image-20260304113051222.png)

## 四、网页端操作实例

### 登录&注册

首先是登录界面，可以输入用户名注册账号，或登录已有账号。用户名是每个用户唯一的标识哦！

<img src="G:\其他计算机\dummy\whisper-finetune\assets\image-20260304113456616.png" alt="image-20260304113456616" style="zoom:50%;" />

### 语音采集

登录后，我们进入音频采集界面。

![image-20260304113653856](G:\其他计算机\dummy\whisper-finetune\assets\image-20260304113653856.png)



音频采集首先需要上传需要录音的文本。上传的文本是一个txt文件，**其格式是每行一句录音内容**（用换行符来分割语句）。“选择文件”选择txt文件后，点击“导入”，在下方就会出现录音的每一条文本。

还可以通过“添加”按钮来手动输入录音文本。

同时，对于已经导入的文本，可以修改、删除。鼠标悬浮在对应文本的框上就会出现“修改”/“删除”按钮。

点击录音，就可以对照文本开始录音。录音的音频会自动转换为微调的格式上传到服务器上。

最后，录音界面还做了一个“沉浸模式”，点击录音任务的框即可显示悬浮模糊效果：

<img src="G:\其他计算机\dummy\whisper-finetune\assets\image-20260304114451694.png" alt="image-20260304114451694" style="zoom:50%;" />

### 微调训练

录音完成之后，我们有了用户个人定制的训练数据，下面就可以进入模型微调环节了。

<img src="G:\其他计算机\dummy\whisper-finetune\assets\image-20260304115112581.png" alt="image-20260304115112581" style="zoom:70%;" />

这里可以控制模型微调过程中的一些参数。点击“启动微调训练”即可一键开始微调。微调过程中的Loss曲线会在监控面板实时绘制。并且微调完成后，可以在测试集上评估微调效果，计算字错率。

### 语音识别

有了微调好的模型，就可以使用它来进行语音识别的应用了！

<img src="G:\其他计算机\dummy\whisper-finetune\assets\image-20260304115400690.png" alt="image-20260304115400690" style="zoom:80%;" />

在网页端可以直接录制音频，进行实时语音识别；也可以上传音频文件（.wav格式），对其进行识别。识别的文字会显示在上面这个“终端”文本框中。

可以选择使用的模型，如果没有微调的模型，默认使用Base模型。
