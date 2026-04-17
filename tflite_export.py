import os
# 屏蔽 TensorFlow 的详细警告和信息 (必须在 import tensorflow 之前)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import shutil
import argparse
import tempfile
import numpy as np
import torch
import tensorflow as tf
# 进一步屏蔽 Python 端的日志
tf.get_logger().setLevel('ERROR')

import librosa
from typing import List, Optional
from peft import PeftModel
from transformers import (
    WhisperProcessor, 
    WhisperFeatureExtractor, 
    TFWhisperForConditionalGeneration, 
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    TFForceTokensLogitsProcessor
)

# -----------------------------------------------------------------------------
# 1. TFLite 补丁
# -----------------------------------------------------------------------------
def apply_tflite_patches():
    def my__init__(self, force_token_map: List[List[int]]):
        force_token_map = dict(force_token_map)
        force_token_array = np.ones((max(force_token_map.keys()) + 1), dtype=np.int32) * -1
        for index, token in force_token_map.items():
            if token is not None:
                force_token_array[index] = token
        self.force_token_array = tf.convert_to_tensor(force_token_array, dtype=tf.int32)

    def my__call__(self, input_ids: tf.Tensor, scores: tf.Tensor, cur_len: int) -> tf.Tensor:
        def _force_token(generation_idx):
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]
            new_scores = tf.ones_like(scores, dtype=scores.dtype) * -float(1)
            indices = tf.stack((tf.range(batch_size), tf.tile([current_token], [batch_size])), axis=1)
            updates = tf.zeros((batch_size,), dtype=scores.dtype)
            new_scores = tf.tensor_scatter_nd_update(new_scores, indices, updates)
            return new_scores

        scores = tf.cond(
            tf.greater_equal(cur_len, tf.shape(self.force_token_array)[0]),
            lambda: tf.identity(scores),
            lambda: tf.cond(
                tf.greater_equal(self.force_token_array[cur_len], 0),
                lambda: _force_token(cur_len),
                lambda: scores,
            ),
        )
        return scores

    TFForceTokensLogitsProcessor.__init__ = my__init__
    TFForceTokensLogitsProcessor.__call__ = my__call__

# -----------------------------------------------------------------------------
# 2. 推理模块
# -----------------------------------------------------------------------------
class WhisperGenerateModule(tf.Module):
    def __init__(self, model, forced_decoder_ids):
        super(WhisperGenerateModule, self).__init__()
        self.model = model
        self.forced_decoder_ids = forced_decoder_ids

    @tf.function(
        input_signature=[
            tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
        ],
    )
    def serving(self, input_features):
        outputs = self.model.generate(
            input_features,
            max_new_tokens=448,
            return_dict_in_generate=True,
            forced_decoder_ids=self.forced_decoder_ids,
        )
        return {"sequences": outputs["sequences"]}

# -----------------------------------------------------------------------------
# 3. 核心导出函数
# -----------------------------------------------------------------------------
def run_tflite_export(base_model_path: str, checkpoint_path: Optional[str] = None, output_tflite_path: str = "output/whisper_model.tflite"):
    """
    包装好的 TFLite 导出功能函数。
    """
    print(f"🚀 开始 TFLite 导出流程...")
    print(f"   基础模型: {base_model_path}")
    print(f"   检查点: {checkpoint_path if checkpoint_path else '未提供 (导出基座)'}")
    print(f"   输出路径: {output_tflite_path}")

    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    apply_tflite_patches()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # A. 确定合并模型
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"   正在合并 LoRA 权重...")
                merged_pt_path = os.path.join(tmp_dir, "merged_pt")
                base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path)
                peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
                merged_model = peft_model.merge_and_unload()
                merged_model.save_pretrained(merged_pt_path)
                model_to_convert = merged_pt_path
            else:
                model_to_convert = base_model_path

            # B. 转换 SavedModel
            print("   正在构建 TensorFlow SavedModel...")
            forced_decoder_ids = [[1, 50260], [2, 50359], [3, 50363]] # zh, transcribe
            tf_model = TFWhisperForConditionalGeneration.from_pretrained(model_to_convert, from_pt=True)
            generate_module = WhisperGenerateModule(model=tf_model, forced_decoder_ids=forced_decoder_ids)
            
            saved_model_path = os.path.join(tmp_dir, "saved_model")
            tf.saved_model.save(generate_module, saved_model_path, signatures={"serving_default": generate_module.serving})

            # C. 转换为 TFLite 并量化
            print("   正在执行 TFLite 转换与量化...")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            # D. 保存
            os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
            with open(output_tflite_path, "wb") as f:
                f.write(tflite_model)
            
            print(f"✅ TFLite 导出成功！大小: {os.path.getsize(output_tflite_path) / (1024*1024):.2f} MB")
            return True, "Success"

    except Exception as e:
        error_msg = f"❌ TFLite 导出失败: {str(e)}"
        print(error_msg)
        return False, error_msg

# -----------------------------------------------------------------------------
# 4. 推理测试 (保留但不默认执行)
# -----------------------------------------------------------------------------
def test_tflite_model(tflite_path, audio_path, base_model_name):
    print(f"\n🔍 运行推理测试...")
    speech, _ = librosa.load(audio_path, sr=16000)
    processor = WhisperProcessor.from_pretrained(base_model_name)
    input_features = processor(speech, sampling_rate=16000, return_tensors="tf").input_features
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    tflite_generate = interpreter.get_signature_runner()
    output = tflite_generate(input_features=input_features)
    transcription = processor.batch_decode(output["sequences"], skip_special_tokens=True)[0]
    print(f"   识别文字: {transcription}\n")

# -----------------------------------------------------------------------------
# 5. CLI 支持
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/whisper_model.tflite")
    parser.add_argument("--test_audio", type=str, help="测试音频路径")
    args = parser.parse_args()

    success, _ = run_tflite_export(args.base_model, args.checkpoint, args.output)
    if success and args.test_audio:
        test_tflite_model(args.output, args.test_audio, args.base_model)
