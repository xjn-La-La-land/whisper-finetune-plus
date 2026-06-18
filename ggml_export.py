# ggml_export.py
# LoRA 微调产物 → 合并 → 转 ggml(fp16) → 可选量化(q5_0)，供浏览器端 whisper.cpp WASM 推理。
#
# 结构对齐 tflite_export.py（同样返回 (success, error_msg) 元组，前端可透传）。
# 转换链路复用 whisper.cpp 自带的 models/convert-h5-to-ggml.py + build/bin/whisper-quantize。
#
# CLI:
#   python ggml_export.py --base_model whisper-base-models/whisper-small \
#       --checkpoint output/<user>/<model>/checkpoint-final \
#       --output output/<user>/<model>/ggml/whisper.bin --quantize q5_0 \
#       --test_audio uploads/admin/task_804.wav
#
# 见 doc/TODO_WHISPER_CPP_WASM.md 的 P1-2 / P0-3。
import os
import sys
import shutil
import argparse
import tempfile
import subprocess
from typing import Optional, Tuple

import torch  # noqa: F401  (peft/transformers 间接需要；显式 import 便于尽早暴露环境问题)
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- whisper.cpp 位置（均可用环境变量覆盖，便于部署到别的机器）---
# 优先用项目内 vendor/whisper.cpp（P1-1 的 submodule）+ bin/（build_whisper_cpp.sh 拷贝的
# native 二进制），回退到老的 ~/whisper.cpp，保证「在项目内编译之前/之后」都能用。
def _first_existing(candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[0]  # 都不存在则返回首选路径，留给后续检查报出清晰错误

WHISPER_CPP_DIR = os.environ.get("WHISPER_CPP_DIR") or _first_existing([
    os.path.join(PROJECT_ROOT, "vendor", "whisper.cpp"),
    os.path.expanduser("~/whisper.cpp"),
])
CONVERT_SCRIPT = os.environ.get("WHISPER_CONVERT_SCRIPT") or os.path.join(
    WHISPER_CPP_DIR, "models", "convert-h5-to-ggml.py")
QUANTIZE_BIN = os.environ.get("WHISPER_QUANTIZE_BIN") or _first_existing([
    os.path.join(PROJECT_ROOT, "bin", "whisper-quantize"),
    os.path.join(WHISPER_CPP_DIR, "build", "bin", "whisper-quantize"),
    os.path.expanduser("~/whisper.cpp/build/bin/whisper-quantize"),
])
WHISPER_CLI_BIN = os.environ.get("WHISPER_CLI_BIN") or _first_existing([
    os.path.join(PROJECT_ROOT, "bin", "whisper-cli"),
    os.path.join(WHISPER_CPP_DIR, "build", "bin", "whisper-cli"),
    os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"),
])

# convert-h5-to-ggml.py 第 102 行硬依赖 <dir_whisper>/whisper/assets/mel_filters.npz
# （来自 openai/whisper 仓库，仅 4.3KB）。vendored 进项目保证脚本自包含，不依赖外部 repo。
MEL_ASSETS_ROOT = os.path.join(PROJECT_ROOT, "vendor", "whisper_convert_assets")

# --- 基座模型批量导出（P1-6）：没微调过的用户也能用 base 模型的 ggml 跑 WASM 推理 ---
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, "whisper-base-models")
BASE_GGML_DIR = os.path.join(BASE_MODELS_DIR, "ggml")
# 默认导出的基座：medium 在浏览器 WASM 里会 OOM（fp16 ~1.5GB / q5_0 ~500MB），故默认不含。
DEFAULT_BASE_MODELS = ["whisper-tiny", "whisper-base", "whisper-small"]


def _ensure_prereqs() -> Optional[str]:
    """检查转换链路三件套是否就位。返回错误字符串；None 表示 OK。"""
    if not os.path.exists(CONVERT_SCRIPT):
        return (f"找不到 whisper.cpp 转换脚本: {CONVERT_SCRIPT}\n"
                "请设 WHISPER_CPP_DIR 指向 whisper.cpp 仓库根目录。")
    npz = os.path.join(MEL_ASSETS_ROOT, "whisper", "assets", "mel_filters.npz")
    if not os.path.exists(npz):
        return (f"缺少 mel_filters.npz: {npz}\n"
                "请从 openai/whisper 仓库的 whisper/assets/mel_filters.npz 拷到该位置"
                "（仅 4.3KB；见 doc/TODO_WHISPER_CPP_WASM.md P0-1）。")
    return None


def run_ggml_export(
    base_model_path: str,
    checkpoint_path: Optional[str] = None,
    output_path: str = "output/whisper_model.bin",
    quantize: Optional[str] = "q5_0",
) -> Tuple[bool, str]:
    """
    导出 ggml 模型。

    参数:
      base_model_path  HF 基座模型目录（如 whisper-base-models/whisper-small）
      checkpoint_path  LoRA 检查点目录；None / 不存在则直接导出基座
      output_path      fp16 产物落地路径（量化产物为同名 + _<quantize> 后缀）
      quantize         "q5_0" / "q4_0" / None（None 只出 fp16）

    返回: (success, message)
    """
    print("🚀 开始 ggml 导出流程...")
    print(f"   基础模型: {base_model_path}")
    print(f"   检查点:   {checkpoint_path if checkpoint_path else '未提供 (导出基座)'}")
    print(f"   输出路径: {output_path}")
    print(f"   量化:     {quantize or '不量化 (仅 fp16)'}")

    err = _ensure_prereqs()
    if err:
        print(f"❌ {err}")
        return False, err

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # A. 确定要转换的 HF 模型目录（合并 LoRA 或直接用基座）
            if checkpoint_path and os.path.exists(checkpoint_path):
                print("   正在合并 LoRA 权重 (merge_and_unload)...")
                merged_dir = os.path.join(tmp_dir, "merged_hf")
                base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path)
                peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
                merged_model = peft_model.merge_and_unload()
                merged_model.save_pretrained(merged_dir)
                # convert 脚本要读 vocab.json / added_tokens.json，而 save_pretrained 只存
                # 权重 + config，所以把 processor（tokenizer + feature extractor）一并存进去。
                WhisperProcessor.from_pretrained(base_model_path).save_pretrained(merged_dir)
                model_dir = merged_dir
            else:
                model_dir = base_model_path

            # B. 调 whisper.cpp 的 convert-h5-to-ggml.py → ggml-model.bin (fp16)
            ggml_out_dir = os.path.join(tmp_dir, "ggml_out")
            os.makedirs(ggml_out_dir, exist_ok=True)
            print("   正在转换为 ggml (fp16)...")
            cmd = [sys.executable, CONVERT_SCRIPT, model_dir, MEL_ASSETS_ROOT, ggml_out_dir]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return False, f"convert-h5-to-ggml 失败:\n{proc.stderr[-2000:]}"
            produced = os.path.join(ggml_out_dir, "ggml-model.bin")
            if not os.path.exists(produced):
                return False, f"转换未产出 ggml-model.bin\nstdout 尾部:\n{proc.stdout[-1000:]}"

            # C. 落地 fp16 产物
            out_abs = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(out_abs), exist_ok=True)
            shutil.copyfile(produced, out_abs)
            fp16_mb = os.path.getsize(out_abs) / (1024 * 1024)
            print(f"✅ fp16 ggml 导出成功: {output_path} ({fp16_mb:.1f} MB)")

            # D. 可选量化
            if quantize:
                if not os.path.exists(QUANTIZE_BIN):
                    return False, f"找不到 whisper-quantize: {QUANTIZE_BIN}（先编 whisper.cpp）"
                root, ext = os.path.splitext(out_abs)
                q_abs = f"{root}_{quantize}{ext}"
                print(f"   正在量化 → {quantize} ...")
                qproc = subprocess.run([QUANTIZE_BIN, out_abs, q_abs, quantize],
                                       capture_output=True, text=True)
                if qproc.returncode != 0 or not os.path.exists(q_abs):
                    return False, f"量化失败:\n{qproc.stderr[-2000:]}"
                q_mb = os.path.getsize(q_abs) / (1024 * 1024)
                print(f"✅ 量化成功: {os.path.relpath(q_abs, PROJECT_ROOT)} ({q_mb:.1f} MB)")

            return True, "Success"

    except Exception as e:
        msg = f"❌ ggml 导出失败: {e}"
        print(msg)
        return False, msg


def run_base_ggml_export(names=None, quantize: Optional[str] = "q5_0"):
    """把基座模型批量转 ggml + 量化，输出到 whisper-base-models/ggml/<name>(_<quantize>).bin。

    复用 run_ggml_export，checkpoint=None 即「纯基座、不融合 LoRA」（见 run_ggml_export 的分支）。
    names: 基座目录名列表（如 ['whisper-tiny']）；None → DEFAULT_BASE_MODELS。
    返回 [(name, ok, msg), ...]。
    """
    names = names or DEFAULT_BASE_MODELS
    results = []
    for name in names:
        src = os.path.join(BASE_MODELS_DIR, name)
        if not (os.path.isdir(src) and os.path.exists(os.path.join(src, "config.json"))):
            results.append((name, False, f"基座目录不存在或缺 config.json: {src}"))
            continue
        print(f"\n===== 基座 {name} → ggml =====")
        ok, msg = run_ggml_export(
            base_model_path=src,
            checkpoint_path=None,
            output_path=os.path.join(BASE_GGML_DIR, f"{name}.bin"),
            quantize=quantize,
        )
        results.append((name, ok, msg))
    return results


def test_with_whisper_cli(ggml_bin: str, audio_path: str, lang: str = "zh") -> None:
    """用 whisper-cli 跑一遍产物，肉眼对比转录质量（验证 LoRA 效果是否保留）。"""
    if not os.path.exists(WHISPER_CLI_BIN):
        print(f"⚠️ 跳过推理自测：找不到 whisper-cli ({WHISPER_CLI_BIN})")
        return
    print(f"\n🔍 whisper-cli 推理自测: {os.path.basename(ggml_bin)}  ←  {audio_path}")
    proc = subprocess.run(
        [WHISPER_CLI_BIN, "-m", ggml_bin, "-f", audio_path, "-l", lang, "-nt"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"   ❌ 推理失败:\n{proc.stderr[-1000:]}")
        return
    print(f"   识别结果: {proc.stdout.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=None, help="单模型导出：基座目录（如 whisper-base-models/whisper-small）")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="output/whisper_model.bin")
    parser.add_argument("--quantize", default="q5_0", help="q5_0 / q4_0 / none")
    parser.add_argument("--test_audio", default=None, help="导出后用 whisper-cli 跑这段音频")
    # 批量基座模式（P1-6）：--base-models 不带值=默认 tiny/base/small；带值=指定，如 --base-models whisper-tiny
    parser.add_argument("--base-models", dest="base_models", nargs="*", default=None, metavar="NAME",
                        help="批量把基座转 ggml → whisper-base-models/ggml/（默认 tiny/base/small）")
    args = parser.parse_args()

    q = None if args.quantize.lower() in ("none", "no", "") else args.quantize

    # --- 批量基座模式 ---
    if args.base_models is not None:
        results = run_base_ggml_export(args.base_models or None, q)
        print("\n=== 基座 ggml 导出结果 ===")
        all_ok = True
        for name, ok, msg in results:
            print(f"  {'✅' if ok else '❌'} {name}{'' if ok else ': ' + msg}")
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    # --- 单模型模式 ---
    if not args.base_model:
        parser.error("需要 --base_model（单模型）或 --base-models（批量基座）")
    ok, msg = run_ggml_export(args.base_model, args.checkpoint, args.output, q)
    if ok and args.test_audio:
        test_with_whisper_cli(args.output, args.test_audio)
        if q:
            root, ext = os.path.splitext(args.output)
            test_with_whisper_cli(f"{root}_{q}{ext}", args.test_audio)
    sys.exit(0 if ok else 1)
