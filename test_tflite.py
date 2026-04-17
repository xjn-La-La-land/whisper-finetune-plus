import os
import argparse
# 同样屏蔽测试时的冗余日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tflite_export import test_tflite_model

def main():
    parser = argparse.ArgumentParser(description="单独测试 TFLite 模型的推理效果")
    parser.add_argument("--tflite_path", type=str, required=True, help="TFLite 模型文件的路径")
    parser.add_argument("--audio_path", type=str, required=True, help="待测试的音频文件路径 (.wav)")
    parser.add_argument("--base_model", type=str, default="./whisper-base-models/whisper-small", 
                        help="基础模型路径或名称 (用于加载对应的分词器)")
    
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.tflite_path):
        print(f"❌ 错误: 找不到 TFLite 模型文件: {args.tflite_path}")
        return
    if not os.path.exists(args.audio_path):
        print(f"❌ 错误: 找不到音频文件: {args.audio_path}")
        return

    try:
        # 调用 tflite_export.py 中的测试模块
        test_tflite_model(
            tflite_path=args.tflite_path,
            audio_path=args.audio_path,
            base_model_name=args.base_model
        )
    except Exception as e:
        print(f"❌ 推理过程中发生异常: {e}")

if __name__ == "__main__":
    main()
