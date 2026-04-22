## 环境配置
1. 下载远程服务器代码
```bash
git clone https://github.com/xjn-La-La-land/whisper-finetune-plus.git
cd whisper-finetune-plus
```

```bash
conda env create -f env.yaml
conda activate asr
# 配置远程 modelscope 仓库
export MODELSCOPE_TOKEN=ms-3a7e679e-994b-4a73-a5ad-c7df48405f9a
export MODELSCOPE_REPO=hanxiaofeng218/CareSpeech-ASR
```

2. 下载app应用

## 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

在网页端启动微调指令，微调完成后自动发布tflite文件，目前导出功能只支持whisper-small模型，因此微调的基座模型请固定为这个模型。

## 测试tflite模型进行语音识别

```bash
python test_tflite.py \
    --tflite_path output/<username>/whisper-small/whisper_model.tflite \
    --audio_path uploads/<username>/task_101.wav \
    --base_model ./whisper-base-models/whisper-small
```

## 设置app与服务器之间的连接

1. 启动 HTTP 穿透(在服务器端的另一个终端)

   ```bash
   cpolar http 8000
   ```

2. 打开移动端 app 中的 “开发者配置”，将“服务器配置”中的网址修改为 cpolar 输出的网址。
