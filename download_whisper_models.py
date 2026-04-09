#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "未找到 huggingface_hub，请先安装依赖后再运行：pip install huggingface_hub"
    ) from exc


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "whisper-base-models")
DEFAULT_ENDPOINT = "https://hf-mirror.com"
SUPPORTED_MODELS = {
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "whisper-base": "openai/whisper-base",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载 OpenAI Whisper 基础模型到本地 whisper-base-models 目录。"
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(SUPPORTED_MODELS.keys()),
        help="要下载的模型名；不传则下载全部支持的模型。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="模型保存目录，默认是项目下的 ./whisper-base-models",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Hugging Face endpoint，默认使用 hf-mirror。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="可选的 Hugging Face token，用于需要鉴权的场景。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载，即使本地目录已存在。",
    )
    return parser.parse_args()


def download_model(
    repo_id: str,
    output_dir: str,
    endpoint: str,
    token: Optional[str],
    force: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=not force,
        force_download=force,
        token=token,
        endpoint=endpoint,
    )


def main():
    args = parse_args()
    selected_models = args.models or list(SUPPORTED_MODELS.keys())
    output_root = os.path.abspath(args.output_dir)

    print(f"下载目录: {output_root}")
    print(f"下载源: {args.endpoint}")
    print(f"模型列表: {', '.join(selected_models)}")

    failed = []
    for model_name in selected_models:
        repo_id = SUPPORTED_MODELS[model_name]
        target_dir = os.path.join(output_root, model_name)
        print(f"\n[{model_name}] 开始下载 -> {target_dir}")
        try:
            download_model(
                repo_id=repo_id,
                output_dir=target_dir,
                endpoint=args.endpoint,
                token=args.token,
                force=args.force,
            )
            print(f"[{model_name}] 下载完成")
        except Exception as exc:
            failed.append(model_name)
            print(f"[{model_name}] 下载失败: {exc}", file=sys.stderr)

    if failed:
        print(f"\n以下模型下载失败: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("\n所有模型下载完成。")


if __name__ == "__main__":
    main()
