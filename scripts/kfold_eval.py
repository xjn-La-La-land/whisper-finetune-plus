#!/usr/bin/env python3
"""k 折交叉验证评估一套微调「配方」的中文 CER —— 在「只能用现有数据」时拿到可信的 CER 估计。

为什么需要它
------------
单次 train/test 划分的测试集太小（小崔只有 14 条），CER 噪声大到无法判断「干净标注 / 数据增强 /
换基座」这些改动到底有没有用（实测 cui-v1 26.0% vs cui-v2 27.4% 完全在噪声内）。k 折把每条数据
都轮流当一次测试样本（有效测试量 14→全量 298），报 mean±std CER：std 就是噪声地板，之后任何改动
「提升 > std」才算真提升。

它解决什么、不解决什么
----------------------
✓ 解决「抽样噪声」——可靠地比较不同配方的好坏。
✗ 不解决「同源乐观」——所有数据仍来自同一次录音 session，绝对 CER 仍偏乐观；那只有录一份
  新 session 才能修。所以这里的数字用于【配方对比】，不要当成线上真实 CER。

方法学要点
----------
每折用「固定轮数、关早停」训练（--epochs，默认 4，依 cui-v2 在 ~2.8 轮触底而定），然后在
【完全没参与训练的那一折】上评 CER。固定轮数是为了避免「用测试折做早停/选优」造成的信息泄漏
（那会让 CER 偏乐观且各配方不可比）。要比较配方，就让两次 k 折除了被测改动外其它参数完全一致。

用法（在 Featurize 上跑，需要 GPU 训练；whisper-base 单折约 20~90 秒）
----
  conda activate whisper
  # 基线配方
  python scripts/kfold_eval.py --username 小崔 \\
      --base-model whisper-base-models/whisper-base --k 5 --epochs 4

  # 对比：加数据增强（建好 augment 配置后）——只改这一个变量
  python scripts/kfold_eval.py --username 小崔 \\
      --base-model whisper-base-models/whisper-base --k 5 --epochs 4 \\
      --augment-config configs/augment.json

  # 对比：换 small 基座
  python scripts/kfold_eval.py --username 小崔 \\
      --base-model whisper-base-models/whisper-small --k 5 --epochs 4

  --dry-run 只打印折切分、不训练。
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE = os.path.join(PROJECT_ROOT, "finetune.py")
EVAL = os.path.join(PROJECT_ROOT, "scripts", "eval_model.py")


def load_all_data(dataset_dir: str, username: str) -> list:
    """合并 dataset/<user>/{train,test}.json 得到全量样本（已是修好标注的口径）。"""
    user_dir = os.path.join(dataset_dir, username)
    items = []
    for name in ("train.json", "test.json"):
        p = os.path.join(user_dir, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                items.extend(json.load(f))
    if not items:
        raise SystemExit(f"❌ {user_dir} 下没有 train.json / test.json，先在网页端或 dataset_builder 生成数据集")
    return items


def make_folds(n: int, k: int, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [sorted(idx[i::k]) for i in range(k)]


def run(cmd: list) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="k 折交叉验证评估微调配方的中文 CER",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--username", required=True)
    p.add_argument("--base-model", required=True, help="基座模型目录，如 whisper-base-models/whisper-base")
    p.add_argument("--k", type=int, default=5, help="折数")
    p.add_argument("--epochs", type=int, default=4, help="每折固定训练轮数（关早停，避免测试折信息泄漏）")
    p.add_argument("--seed", type=int, default=42, help="折切分随机种子（固定以便复现/对比）")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数；大模型显存吃紧时调小 batch、用它补回有效 batch")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA 秩 r")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA 缩放 alpha（有效缩放=alpha/r）")
    p.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--augment-config", default=None, help="数据增强配置（透传给 finetune.py 的 --augment_config_path）")
    p.add_argument("--dataset-dir", default=os.path.join(PROJECT_ROOT, "dataset"))
    p.add_argument("--work-root", default=os.path.join(PROJECT_ROOT, "output", "_kfold"),
                   help="每折的临时训练/数据产物目录")
    p.add_argument("--keep", action="store_true", help="保留每折的模型与数据（默认评完即删，省磁盘）")
    p.add_argument("--dry-run", action="store_true", help="只打印折切分，不训练")
    args = p.parse_args()

    data = load_all_data(args.dataset_dir, args.username)
    folds = make_folds(len(data), args.k, args.seed)
    print(f"全量 {len(data)} 条 → {args.k} 折，各折测试集大小 {[len(f) for f in folds]}")
    print(f"配方：base={args.base_model}  epochs={args.epochs}  lr={args.learning_rate}  "
          f"batch={args.batch_size}x{args.grad_accum}  lora(r={args.lora_r},α={args.lora_alpha},do={args.lora_dropout})  "
          f"augment={args.augment_config or '无'}\n")
    if args.dry_run:
        return

    work = os.path.join(args.work_root, args.username)
    os.makedirs(work, exist_ok=True)

    fold_cers, fold_saccs = [], []
    for i, test_idx in enumerate(folds):
        test_set = set(test_idx)
        fold_train = [d for j, d in enumerate(data) if j not in test_set]
        fold_test = [data[j] for j in test_idx]

        fdir = os.path.join(work, f"fold{i}")
        os.makedirs(fdir, exist_ok=True)
        train_path = os.path.join(fdir, "train.json")
        test_path = os.path.join(fdir, "test.json")
        out_dir = os.path.join(fdir, "model")
        result_path = os.path.join(fdir, "eval.json")
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(fold_train, f, ensure_ascii=False)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(fold_test, f, ensure_ascii=False)

        print(f"===== 折 {i + 1}/{args.k}：训练 {len(fold_train)} / 测试 {len(fold_test)} =====")
        # 固定轮数 + 关早停 + 不按测试折选最优 checkpoint（checkpoint-final = 末轮模型）。
        # 三者缺一不可：只关早停而留 load_best_model_at_end=True，仍会用测试折从若干 checkpoint
        # 里挑 CER 最低的那个 → 测试折泄漏、分数偏乐观。这里全部堵死，得到诚实的留出估计。
        run([
            sys.executable, FINETUNE,
            f"--base_model={args.base_model}",
            f"--train_data={train_path}",
            f"--test_data={test_path}",
            f"--output_dir={out_dir}",
            f"--num_train_epochs={args.epochs}",
            "--early_stopping_patience=0",
            "--load_best_model_at_end=False",
            f"--learning_rate={args.learning_rate}",
            f"--per_device_train_batch_size={args.batch_size}",
            f"--gradient_accumulation_steps={args.grad_accum}",
            f"--lora_r={args.lora_r}",
            f"--lora_alpha={args.lora_alpha}",
            f"--lora_dropout={args.lora_dropout}",
            f"--num_workers={args.num_workers}",
        ] + ([f"--augment_config_path={args.augment_config}"] if args.augment_config else []))

        ckpt = os.path.join(out_dir, "checkpoint-final")
        run([sys.executable, EVAL, ckpt, "--data", test_path, "--output", result_path])

        with open(result_path, "r", encoding="utf-8") as f:
            res = json.load(f)[0]
        fold_cers.append(res["cer_norm"])
        fold_saccs.append(res["sent_acc"])
        print(f"  → 折 {i + 1} CER={res['cer_norm']:.1%}  整句准确={res['sent_acc']:.1%}\n")

        if not args.keep:
            shutil.rmtree(out_dir, ignore_errors=True)

    # 汇总
    mean_cer = statistics.mean(fold_cers)
    std_cer = statistics.pstdev(fold_cers) if len(fold_cers) > 1 else 0.0
    mean_sacc = statistics.mean(fold_saccs)
    print("=" * 60)
    print(f"{args.k} 折 CER：" + "  ".join(f"{c:.1%}" for c in fold_cers))
    print(f"平均 CER = {mean_cer:.1%}  ±{std_cer:.1%}（std=噪声地板，改动提升需 > 此值才算真）")
    print(f"平均整句准确 = {mean_sacc:.1%}")
    print("=" * 60)
    print("⚠️ 同源乐观仍在：全部数据来自同一次录音 → 绝对值偏乐观，此数字仅用于【配方对比】。")
    if not args.keep:
        print(f"（每折模型已清理；数据/结果留在 {work}，加 --keep 可保留模型）")


if __name__ == "__main__":
    main()
