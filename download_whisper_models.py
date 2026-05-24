#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "whisper-base-models")
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_SOURCE = "modelscope"

# 只拉 safetensors + 配置/分词文件，跳过 pytorch_model.bin / flax_model.msgpack /
# tf_model.h5 等重复格式权重（whisper repo 里同一份权重存了 4 份，会多下 ~9GB）
DEFAULT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
]

# 每个模型在两个源上的 repo_id；ModelScope 这边用 openai-mirror 组织（原生权重完整镜像）
SUPPORTED_MODELS = {
    "whisper-tiny":           {"hf": "openai/whisper-tiny",           "ms": "openai-mirror/whisper-tiny"},
    "whisper-base":           {"hf": "openai/whisper-base",           "ms": "openai-mirror/whisper-base"},
    "whisper-small":          {"hf": "openai/whisper-small",          "ms": "openai-mirror/whisper-small"},
    "whisper-medium":         {"hf": "openai/whisper-medium",         "ms": "openai-mirror/whisper-medium"},
    "whisper-large-v3":       {"hf": "openai/whisper-large-v3",       "ms": "openai-mirror/whisper-large-v3"},
    "whisper-large-v3-turbo": {"hf": "openai/whisper-large-v3-turbo", "ms": "openai-mirror/whisper-large-v3-turbo"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载 OpenAI Whisper 模型到本地 whisper-base-models 目录。",
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(SUPPORTED_MODELS.keys()),
        help="要下载的模型名；不传则下载全部支持的模型。",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default=DEFAULT_SOURCE,
        help="下载源：modelscope（默认，境内直连无需代理）或 huggingface（需可访问 huggingface.co）。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="模型保存目录，默认是项目下的 ./whisper-base-models",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help="仅 --source huggingface 时生效，覆盖 HF_ENDPOINT。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="可选的 Hugging Face token，用于需要鉴权的场景（仅 HF 源）。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载：开始前清空目标目录。",
    )
    return parser.parse_args()


def download_from_huggingface(repo_id, output_dir, token, endpoint):
    # HF_ENDPOINT 必须在 import huggingface_hub 之前设置（库在 import 时把
    # 它缓存到 constants.ENDPOINT，之后改环境变量没用）。所以这里 lazy import。
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "未找到 huggingface_hub，请先：pip install huggingface_hub"
        ) from exc
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        token=token,
        max_workers=1,
        etag_timeout=30,
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
    )


def download_from_modelscope(repo_id, output_dir):
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "未找到 modelscope，请先：pip install modelscope"
        ) from exc
    snapshot_download(
        model_id=repo_id,
        local_dir=output_dir,
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
        max_workers=1,
    )


def main():
    args = parse_args()
    selected_models = args.models or list(SUPPORTED_MODELS.keys())
    output_root = os.path.abspath(args.output_dir)
    source_key = "ms" if args.source == "modelscope" else "hf"

    extra = f" ({args.endpoint})" if args.source == "huggingface" else ""
    print(f"下载目录: {output_root}")
    print(f"下载源: {args.source}{extra}")
    print(f"模型列表: {', '.join(selected_models)}")

    failed = []
    for model_name in selected_models:
        repo_id = SUPPORTED_MODELS[model_name][source_key]
        target_dir = os.path.join(output_root, model_name)
        print(f"\n[{model_name}] 开始下载 ({repo_id}) -> {target_dir}")

        if args.force and os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        try:
            if args.source == "huggingface":
                download_from_huggingface(repo_id, target_dir, args.token, args.endpoint)
            else:
                download_from_modelscope(repo_id, target_dir)
            print(f"[{model_name}] 下载完成")
        except Exception as exc:
            failed.append(model_name)
            print(f"[{model_name}] 下载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

    if failed:
        print(f"\n以下模型下载失败: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("\n所有模型下载完成。")


if __name__ == "__main__":
    main()
