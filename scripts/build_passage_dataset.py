#!/usr/bin/env python3
"""把「按句录」的课文朗读，构建成微调用的训练/测试集——让模型学会识别长段课文朗读。

为什么需要它（见 doc/passage_reading.md）
----------------------------------------
用 ~300 条孤立短句微调出的模型，识别整段课文朗读几乎是乱码，根因是「课文朗读」与训练数据在
**风格(连读) / 词汇(文学) / 格式(长窗口)** 三个维度都不同分布。本脚本把补录的课文朗读（按句、
干净标注）构建成数据集，并：
  1) 整篇 hold out 做测试（绝不参与训练）—— 量「没见过的课文」上的泛化，这才是目标场景指标；
  2) 把同篇连续句**拼成 10–25s 多句长片段**当训练样本 —— 补「长窗口/连续解码」这一格式缺口；
  3) 把现有 300 短句并入训练 —— 同人同分布，留着有益。

它不解决什么：跨句的协同发音（按句录天生缺）—— 若到瓶颈再升级「整段连续录 + 强制对齐」。

数据来源
--------
- 录音：data/tasks.db 的 tasks 表（采集页录入），按 text_content 匹配清单句子到已录 wav。
- 清单：passages.json，一篇一组、句子有序（既是「读什么」也是构建的分组依据）。见下方 --manifest。

用法
----
  conda activate whisper
  # 0)（可选）从清单导出待读文本，去采集页上传建任务
  python scripts/build_passage_dataset.py --username 小崔 --manifest passages.json \
         --emit-prompts prompts.txt

  # 1) 录完后构建数据集（非破坏性，写到 dataset/<user>/passage_v1/）
  python scripts/build_passage_dataset.py --username 小崔 --manifest passages.json \
         --holdout 猫 --existing dataset/小崔/all.json
"""
import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "tasks.db")
UPLOADS = os.path.join(PROJECT_ROOT, "uploads")
TARGET_SR = 16000


def norm_key(s: str) -> str:
    """匹配用的归一化 key：去首尾空白 + 去内部空白（标点保留，标签本就带标点）。"""
    return "".join(s.split())


def load_recorded(username: str) -> dict:
    """读 DB 里该用户「已完成」的 task，返回 {归一化文本: [物理wav路径, ...]}（可能一文多录）。"""
    if not os.path.isfile(DB_PATH):
        raise SystemExit(f"❌ 找不到数据库 {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT text_content, audio_path FROM tasks WHERE username = ? AND is_completed = 1",
        (username,),
    ).fetchall()
    conn.close()
    table = {}
    for text, audio_path in rows:
        if not audio_path:
            continue
        clean = audio_path.split("?")[0].lstrip("/")          # "/uploads/小崔/task_X.wav?v=1" -> "uploads/小崔/task_X.wav"
        physical = os.path.join(PROJECT_ROOT, clean)
        if not os.path.exists(physical):
            continue
        table.setdefault(norm_key(text), []).append(physical)
    return table


def read_16k_mono(path: str) -> np.ndarray:
    """读 wav → float32、单声道、16kHz。"""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import librosa  # 仅在确需重采样时引入
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(data, dtype=np.float32)


def to_record(path: str, sentence: str, duration: float) -> dict:
    return {"audio": {"path": path}, "sentence": sentence,
            "language": "Chinese", "duration": round(float(duration), 2)}


def main() -> None:
    p = argparse.ArgumentParser(description="构建课文朗读训练/测试集（按句录 → 句对 + 合成长片段 + 整篇hold out）",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--username", required=True)
    p.add_argument("--manifest", required=True, help="passages.json：[{passage, sentences:[...有序...]}, ...]")
    p.add_argument("--holdout", default="", help="留作测试的篇名（逗号分隔，整篇不参与训练）")
    p.add_argument("--existing", default=None, help="并入训练的现有数据集 json（如 dataset/<user>/all.json）")
    p.add_argument("--out-subdir", default="passage_v1", help="输出到 dataset/<user>/<out-subdir>/")
    p.add_argument("--synth-max-s", type=float, default=25.0, help="合成长片段时长上限（<30）")
    p.add_argument("--synth-gap-s", type=float, default=0.3, help="合成时句间插入的静音时长")
    p.add_argument("--synth-min-sentences", type=int, default=2, help="少于这么多句不单独合成（避免与原句重复）")
    p.add_argument("--emit-prompts", default=None, help="只把清单句子按序导出到该 txt（供采集页上传），不构建")
    args = p.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # --- 模式 0：导出待读清单 ---
    if args.emit_prompts:
        with open(args.emit_prompts, "w", encoding="utf-8") as f:
            for pas in manifest:
                for s in pas["sentences"]:
                    f.write(s.strip() + "\n")
        n = sum(len(pas["sentences"]) for pas in manifest)
        print(f"✅ 已导出 {n} 句到 {args.emit_prompts}（{len(manifest)} 篇）。去采集页『上传待读文本』建任务。")
        return

    holdout = {h.strip() for h in args.holdout.split(",") if h.strip()}
    recorded = load_recorded(args.username)
    print(f"DB 中 {args.username} 已录 {sum(len(v) for v in recorded.values())} 条；清单 {len(manifest)} 篇\n")

    out_dir = os.path.join(PROJECT_ROOT, "dataset", args.username, args.out_subdir)
    synth_dir = os.path.join(UPLOADS, args.username, "synth")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)

    train, test = [], []
    n_synth = 0
    missing = []          # 清单里有、但没录到音频的句子
    used = set()          # 已消费的 (key, path)，避免一条音频被两句复用

    for pas in manifest:
        name = pas["passage"]
        is_test = name in holdout
        seq = []          # 本篇匹配到的 (path, sentence, duration, audio)
        for sent in pas["sentences"]:
            sent = sent.strip()
            cands = recorded.get(norm_key(sent), [])
            path = next((c for c in cands if (norm_key(sent), c) not in used), None)
            if path is None:
                missing.append((name, sent))
                continue
            used.add((norm_key(sent), path))
            audio = read_16k_mono(path)
            dur = len(audio) / TARGET_SR
            seq.append((path, sent, dur, audio))

        # 句级样本：hold-out 篇 → test，其余 → train
        bucket = test if is_test else train
        for path, sent, dur, _audio in seq:
            bucket.append(to_record(path, sent, dur))

        # 合成长片段：只对训练篇做（test 要纯净的真实句级），同篇连续句贪心拼到 ≤ synth-max-s
        if not is_test and len(seq) >= args.synth_min_sentences:
            gap = np.zeros(int(args.synth_gap_s * TARGET_SR), dtype=np.float32)
            cur_audio, cur_texts, cur_dur, ci = [], [], 0.0, 0
            def flush():
                nonlocal cur_audio, cur_texts, cur_dur, ci, n_synth
                if len(cur_texts) >= args.synth_min_sentences:
                    wav = np.concatenate(cur_audio)
                    sp = os.path.join(synth_dir, f"{name}_{ci:03d}.wav")
                    sf.write(sp, wav, TARGET_SR)
                    train.append(to_record(sp, "".join(cur_texts), len(wav) / TARGET_SR))
                    n_synth += 1
                    ci += 1
                cur_audio, cur_texts, cur_dur = [], [], 0.0
            for _path, sent, dur, audio in seq:
                add = dur + (args.synth_gap_s if cur_audio else 0.0)
                if cur_audio and cur_dur + add > args.synth_max_s:
                    flush()
                    add = dur
                if cur_audio:
                    cur_audio.append(gap)
                cur_audio.append(audio)
                cur_texts.append(sent)
                cur_dur += add
            flush()

    # 并入现有数据（全部进 train —— 同人同分布的短句，留着有益）
    n_existing = 0
    if args.existing:
        with open(args.existing, "r", encoding="utf-8") as f:
            existing = json.load(f)
        train.extend(existing)
        n_existing = len(existing)

    all_data = train + test
    for name, data in (("train.json", train), ("test.json", test), ("all.json", all_data)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    # 汇总
    print("=" * 64)
    print(f"训练集 train.json：{len(train)} 条"
          f"（课文句 {len(train) - n_synth - n_existing} + 合成长片段 {n_synth} + 现有短句 {n_existing}）")
    print(f"测试集 test.json ：{len(test)} 条（hold-out 篇：{', '.join(holdout) or '无'}）")
    print(f"输出目录：{out_dir}")
    print("=" * 64)
    if missing:
        print(f"⚠️ 清单里有 {len(missing)} 句还没匹配到录音（漏录/文本对不上），需补录或核对文本：")
        for name, sent in missing[:30]:
            print(f"    [{name}] {sent}")
        if len(missing) > 30:
            print(f"    …… 还有 {len(missing) - 30} 句")
    if not holdout:
        print("⚠️ 没指定 --holdout：没有 hold-out 篇，test.json 为空，无法量化泛化。建议留 1 篇出来。")


if __name__ == "__main__":
    main()
