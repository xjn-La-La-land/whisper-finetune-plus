#!/usr/bin/env python3
"""从 ModelScope 下载 LoRA 权重（网页端「上传到 ModelScope」的逆操作）。

把上传到 ModelScope 的 LoRA 拉到当前机器的 output/<username>/<model_name>/checkpoint-final/。
本脚本**只负责下载**；让模型出现在网页端的「注册到数据库」一步，下载完到网页端
「管理微调好的模型 → 刷新」即可完成——/api/rescan_models 会扫 output/ 把新模型补登记。
（注意：刷新只登记当前登录用户自己的模型，且需 web 服务跑在同一台机、用同一个 output/ 与 data/tasks.db。）

仓库组织约定（与网页端 /api/upload_lora 对称：所有用户共用一个仓库）
----------------------------------------------------------------
  所有用户共用一个仓库（环境变量 MODELSCOPE_LORA_REPO），每个模型放在 <用户名>/<模型名>/
  子目录下。下载默认就按这个布局拉 <用户名>/<模型名>/ 并摊平进 checkpoint-final/。
  · --repo-id 不传时默认取环境变量 MODELSCOPE_LORA_REPO（可先 source .env 让它生效）。
  · 要拉别的布局：用 --subdir 显式指定仓库内子目录；--subdir "" 表示仓库根
    （兼容早期“文件直接放仓库根”的旧仓库）。

用法
----
  # 默认（共享仓库，从 <MODELSCOPE_LORA_REPO>/<用户名>/<模型名>/ 拉回）：
  set -a && source .env && set +a     # 让 MODELSCOPE_LORA_REPO 生效
  python scripts/sync_from_modelscope.py --username 小崔 --model-name whisper-base-lora-v2
  # 下载完，到网页端「管理微调好的模型 → 刷新」即可看到该模型

  # 指定仓库 / 旧布局（文件在仓库根）：
  python scripts/sync_from_modelscope.py --username 小崔 --model-name cui-v1 \\
      --repo-id smellyCat99/whisper-base-lora-xiaocui --subdir ""

参数
----
  --username      网页端登录用户名（对应 output/<username>/）
  --model-name    模型名（对应 output/<username>/<model_name>/）
  --repo-id       ModelScope 仓库 ID <owner>/<repo>；默认取环境变量 MODELSCOPE_LORA_REPO
  --subdir        仓库内子目录；默认 <用户名>/<模型名>；传 "" 表示仓库根
  --output-dir    可选，覆盖默认的 output/ 根目录
  --force         若 checkpoint-final 已存在则先删除再下载
"""
import argparse
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_safe_name(name: str) -> bool:
    return bool(name) and name.strip() == name and not any(c in name for c in ("/", "\\")) and ".." not in name


def normalize_subdir(raw, username: str, model_name: str) -> str:
    """计算仓库内子目录：--subdir 未给时默认 <用户名>/<模型名>（与网页端上传布局一致）；
    给空串则为仓库根（返回 ""）。"""
    sub = f"{username}/{model_name}" if raw is None else raw
    sub = sub.strip().strip("/")
    if "\\" in sub or ".." in sub.split("/"):
        raise SystemExit("❌ --subdir 含非法路径（不能有 \\ 或 ..）")
    return sub


# 续训状态文件（体积大、仅续训用），下载时一律跳过（上传时默认也不传）
_IGNORE_FILE_PATTERN = ["optimizer\\.pt", "rng_state\\.pth", "scaler\\.pt", "scheduler\\.pt"]


def download_checkpoint(repo_id: str, target_dir: str, subdir: str = "") -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "未找到 modelscope，请先激活对应 conda 环境：conda activate whisper"
        ) from exc

    src_label = f"{repo_id}/{subdir}/" if subdir else repo_id
    print(f"📥 从 ModelScope 下载：{src_label}")
    print(f"   目标目录：{target_dir}")

    if not subdir:
        # 子目录为空 = 仓库根：直接下到 checkpoint-final（兼容旧的“文件放仓库根”布局）
        snapshot_download(
            model_id=repo_id,
            local_dir=target_dir,
            ignore_file_pattern=_IGNORE_FILE_PATTERN,
        )
        print("   下载完成")
        return

    # 指定了子目录：只拉 <subdir>/* 到临时 staging，再把该子目录内容摊平进 checkpoint-final
    staging = os.path.join(os.path.dirname(target_dir), ".ms_download_tmp")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    try:
        snapshot_download(
            model_id=repo_id,
            local_dir=staging,
            allow_patterns=[f"{subdir}/*"],
            ignore_file_pattern=_IGNORE_FILE_PATTERN,
        )
        src = os.path.join(staging, subdir)
        if not os.path.isdir(src) or not os.listdir(src):
            raise SystemExit(
                f"❌ 仓库 {repo_id} 里没有子目录 {subdir}/（或为空）。\n"
                f"   确认上传时用的子目录名，或旧布局请加 --subdir \"\" 拉仓库根。"
            )
        # 摊平：staging/<subdir>/* → checkpoint-final/*（已存在则覆盖）
        shutil.copytree(src, target_dir, dirs_exist_ok=True)
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
    print("   下载完成")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 ModelScope 下载 LoRA 权重（注册由网页端「刷新」完成）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--username", required=True, help="网页端用户名（对应 output/<username>/）")
    parser.add_argument("--model-name", required=True, help="模型名（对应 output/<username>/<model_name>/）")
    parser.add_argument("--repo-id", default=None,
                        help="ModelScope 仓库 ID <owner>/<repo>；不传则默认取环境变量 MODELSCOPE_LORA_REPO")
    parser.add_argument("--subdir", default=None,
                        help='仓库内子目录；默认 <用户名>/<模型名>（与网页端上传一致）；传 --subdir "" 表示仓库根')
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"), help="模型输出根目录")
    parser.add_argument("--force", action="store_true", help="若 checkpoint-final 已存在则先删除再重新下载")
    args = parser.parse_args()

    if not is_safe_name(args.username) or not is_safe_name(args.model_name):
        raise SystemExit("❌ 用户名或模型名含非法字符（不能有 / \\ .. 或首尾空格）")

    repo_id = args.repo_id or os.environ.get("MODELSCOPE_LORA_REPO")
    if not repo_id:
        raise SystemExit(
            "❌ 未指定仓库：请用 --repo-id 传入，或设置环境变量 MODELSCOPE_LORA_REPO\n"
            "   （例如先执行  set -a && source .env && set +a）"
        )

    subdir = normalize_subdir(args.subdir, args.username, args.model_name)

    checkpoint_dir = os.path.join(args.output_dir, args.username, args.model_name, "checkpoint-final")

    if args.force and os.path.isdir(checkpoint_dir):
        print(f"🗑️  --force：删除已有目录 {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    download_checkpoint(repo_id, checkpoint_dir, subdir)

    # 验证关键文件存在
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not os.path.isfile(os.path.join(checkpoint_dir, f))]
    if missing:
        raise SystemExit(f"❌ 下载后仍缺少关键文件：{missing}，请检查 ModelScope 仓库内容")

    print(f"\n✅ 下载完成，模型目录：{checkpoint_dir}")
    print("   到网页端「管理微调好的模型 → 刷新」即可扫描登记并看到该模型。")


if __name__ == "__main__":
    main()
