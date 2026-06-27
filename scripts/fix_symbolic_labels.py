#!/usr/bin/env python3
"""把标注里的符号化文本（算式/阿拉伯数字/范围号）改成口语汉字形式。

为什么要改
----------
ASR 是「语音 → 文字」。"9÷3=3" 这种符号标注对不上口语发音——人嘴里说的是
「九除以三等于三」，而 ÷ = 这些符号永远不会被读成某个字。这类样本既学不会、数量又少，
反而成了训练噪声和评测里的最差样本（拉高 CER）。改成口语汉字后训练和指标都更干净。

改哪里（三处必须一致）
----------------------
1. DB tasks.text_content  —— dataset_builder 真正读取的地方。**必须改这里。**
   注意：sync_uploads_to_db.py 对已有非空 text_content 不覆盖（保留旧值），
   所以只改 words.txt 不会生效。
2. uploads/<user>/words.txt —— 标注源头，保持一致，避免日后从零重建 DB 时又退回符号。
3. dataset/<user>/{train,test}.json —— 已生成的数据集，就地替换文本、**保留原切分**
   （不重新随机划分），这样微调前后的评测结果可直接对比。

幂等 & 跨机
-----------
按「精确旧串」匹配替换，改完旧串不再存在 → 重跑是 no-op。训练在 Featurize，把本脚本
scp 过去用同一份 CORRECTIONS 跑一遍即可同样修好云端数据。

用法
----
  conda activate whisper
  python scripts/fix_symbolic_labels.py --username 小崔            # 真正写入
  python scripts/fix_symbolic_labels.py --username 小崔 --dry-run  # 只预览不写
"""
import argparse
import json
import os
import shutil
import sqlite3
import time
from contextlib import closing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# {旧符号串: 新口语汉字}。键必须与 DB/words.txt 里的精确字符串一致（含全/半角符号、空格）。
# 字母行保留字母本身（口语说「A」转写就是「A」），只清掉多余的全角逗号+空格。
CORRECTIONS = {
    "1+2=3": "一加二等于三",
    "4×5=20": "四乘五等于二十",
    "7-6=1": "七减六等于一",
    "9÷3=3": "九除以三等于三",
    "8-5=3": "八减五等于三",
    "我的铅笔长18厘米": "我的铅笔长十八厘米",
    "201–300": "二百零一到三百",
    "答案是A， B， C， D": "答案是ABCD",  # 全角逗号 U+FF0C + 普通空格 U+0020
}


def backup(path: str) -> None:
    if os.path.exists(path):
        bak = f"{path}.bak.{int(time.time())}"
        shutil.copy2(path, bak)
        print(f"   备份: {bak}")


def patch_words_txt(path: str, dry_run: bool) -> int:
    if not os.path.isfile(path):
        print(f"⚠️ words.txt 不存在，跳过: {path}")
        return 0
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    changed = 0
    new_lines = []
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped in CORRECTIONS:
            print(f"   words.txt: {stripped!r} -> {CORRECTIONS[stripped]!r}")
            new_lines.append(CORRECTIONS[stripped] + "\n")
            changed += 1
        else:
            new_lines.append(line)
    if changed and not dry_run:
        backup(path)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed


def patch_db(db_path: str, username: str, dry_run: bool) -> int:
    if not os.path.isfile(db_path):
        print(f"⚠️ DB 不存在，跳过: {db_path}")
        return 0
    changed = 0
    with closing(sqlite3.connect(db_path)) as conn:
        for old, new in CORRECTIONS.items():
            cur = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE username = ? AND text_content = ?",
                (username, old),
            )
            n = cur.fetchone()[0]
            if n:
                print(f"   DB: {old!r} -> {new!r}  ({n} 行)")
                changed += n
                if not dry_run:
                    conn.execute(
                        "UPDATE tasks SET text_content = ? WHERE username = ? AND text_content = ?",
                        (new, username, old),
                    )
        if changed and not dry_run:
            conn.commit()
    return changed


def patch_dataset_json(path: str, dry_run: bool) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    for entry in data:
        if entry.get("sentence") in CORRECTIONS:
            old = entry["sentence"]
            entry["sentence"] = CORRECTIONS[old]
            print(f"   {os.path.basename(path)}: {old!r} -> {entry['sentence']!r}")
            changed += 1
    if changed and not dry_run:
        backup(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return changed


def main() -> None:
    p = argparse.ArgumentParser(description="把标注中的符号化文本改成口语汉字（DB + words.txt + 数据集三处一致）")
    p.add_argument("--username", required=True, help="用户名")
    p.add_argument("--db-path", default=os.path.join(PROJECT_ROOT, "data", "tasks.db"))
    p.add_argument("--uploads-dir", default=os.path.join(PROJECT_ROOT, "uploads"))
    p.add_argument("--dataset-dir", default=os.path.join(PROJECT_ROOT, "dataset"))
    p.add_argument("--dry-run", action="store_true", help="只预览不写入")
    args = p.parse_args()

    mode = "（dry-run 预览）" if args.dry_run else ""
    print(f"修正符号化标注 {mode}  username={args.username}\n")

    print("① words.txt")
    c1 = patch_words_txt(os.path.join(args.uploads_dir, args.username, "words.txt"), args.dry_run)
    print(f"  小计: {c1} 处\n")

    print("② DB tasks.text_content")
    c2 = patch_db(args.db_path, args.username, args.dry_run)
    print(f"  小计: {c2} 行\n")

    print("③ dataset train/test.json")
    c3 = 0
    for name in ("train.json", "test.json"):
        c3 += patch_dataset_json(os.path.join(args.dataset_dir, args.username, name), args.dry_run)
    print(f"  小计: {c3} 处\n")

    if c1 == c2 == 0 and c3 == 0:
        print("✅ 没有需要修正的符号化标注（可能已经改过）。")
    else:
        verb = "将修正" if args.dry_run else "已修正"
        print(f"✅ {verb}: words.txt {c1} 处 / DB {c2} 行 / dataset {c3} 处")
        if not args.dry_run:
            print("   后续：训练在 Featurize → 把本脚本与改好的数据同步过去，或在云端用同一份 CORRECTIONS 重跑。")


if __name__ == "__main__":
    main()
