#!/usr/bin/env python3
"""评估 Whisper（含 LoRA 微调）模型的中文识别质量 —— 字错率 CER 为主指标。

为什么要有这个脚本
------------------
finetune.py 全程没有 compute_metrics，load_best_model_at_end 是按 **eval loss** 选的，
不是按识别错误率。loss 最低 ≠ CER 最低。本脚本把「真正的识别准确率」量化出来，
并支持一次评多个模型做对照（基座 vs 微调），回答「微调到底有没有用」。

用法
----
  conda activate whisper

  # 单模型
  python scripts/eval_model.py output/小崔/cui-v1/checkpoint-final \\
      --data dataset/小崔/test.json

  # 对照：基座 vs 微调（强烈推荐，delta 才是结论）
  python scripts/eval_model.py \\
      whisper-base-models/whisper-base \\
      output/小崔/cui-v1/checkpoint-final \\
      --data dataset/小崔/test.json

模型路径可以是：
  - 基座目录（whisper-base-models/whisper-base）           → 直接评
  - LoRA checkpoint 目录（output/<u>/<m>/checkpoint-final）→ 自动合并 base 后评

⚠️ 两个必须记住的局限
  1) 同源高估：test.json 与训练数据同一次录音 session，这里的 CER 会系统性偏乐观，
     真实「换天/换麦」场景会更差。要诚实评估请用 scripts/kfold_eval（如已建）或单独录测试集。
  2) 这里评的是 PyTorch 模型；用户实际跑的是浏览器 WASM 的 q5_0 ggml，存在量化差异。
     关注绝对值时需另测 ggml；但 base-vs-微调的相对结论不受影响。
"""
import argparse
import json
import os
import sys
from collections import Counter

# 仓库根 = 本脚本上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, "whisper-base-models")
sys.path.insert(0, PROJECT_ROOT)

# CER / 归一化与训练时的 compute_metrics 共用同一套（utils/cer.py），确保两边一把尺子
from utils.cer import normalize_text, align, cer_stats  # noqa: E402


# ----------------------------------------------------------------------------
# 模型加载（独立于 web 栈，便于离线评测）
# ----------------------------------------------------------------------------
def resolve_lora_base(model_path: str):
    """若是 LoRA 目录返回(已重映射到本机的)base 路径，否则 None。

    与 inference_controller.resolve_lora_base_model_path 保持同一逻辑——adapter_config.json
    里的 base_model_name_or_path 是训练机的绝对路径，本机不存在时按 basename 重映射到
    whisper-base-models/<name>。改这里也要改那边。
    """
    cfg_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        base = json.load(f).get("base_model_name_or_path")
    if not base:
        return None
    if os.path.isdir(base):
        return base
    local = os.path.join(BASE_MODELS_DIR, os.path.basename(base.rstrip("/")))
    return local if os.path.isdir(local) else base


def load_pipeline(model_path: str, num_beams: int):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_path = resolve_lora_base(model_path)
    is_lora = base_path is not None
    load_from = base_path if is_lora else model_path

    processor = AutoProcessor.from_pretrained(load_from)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        load_from, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
    )
    if is_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30, torch_dtype=dtype, device=device,
    )
    return pipe, ("LoRA→" + os.path.basename(base_path) if is_lora else "base")


# ----------------------------------------------------------------------------
# 评测主流程
# ----------------------------------------------------------------------------
def evaluate_model(model_path: str, data: list, num_beams: int, max_utt: int):
    pipe, kind = load_pipeline(model_path, num_beams)
    gen_kwargs = {"language": "chinese", "task": "transcribe", "num_beams": num_beams}

    items = data[:max_utt] if max_utt else data
    per_utt = []
    agg_raw_dist = agg_raw_N = 0
    agg_norm_dist = agg_norm_N = 0
    sent_correct = 0
    sub_pairs, del_chars, ins_chars = Counter(), Counter(), Counter()

    for idx, entry in enumerate(items):
        audio_path = entry["audio"]["path"]
        ref = entry["sentence"]
        try:
            hyp = pipe(audio_path, generate_kwargs=gen_kwargs)["text"]
        except Exception as e:
            print(f"  [warn] 推理失败 {audio_path}: {e}", file=sys.stderr)
            hyp = ""

        raw = cer_stats(ref, hyp)
        ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
        norm = cer_stats(ref_n, hyp_n)

        agg_raw_dist += raw["dist"]; agg_raw_N += raw["N"]
        agg_norm_dist += norm["dist"]; agg_norm_N += norm["N"]
        if ref_n == hyp_n:
            sent_correct += 1
        for op, rc, hc in norm["ops"]:
            if op == "sub":
                sub_pairs[(rc, hc)] += 1
            elif op == "del":
                del_chars[rc] += 1
            elif op == "ins":
                ins_chars[hc] += 1

        per_utt.append({"audio": os.path.basename(audio_path), "ref": ref, "hyp": hyp,
                        "ref_norm": ref_n, "hyp_norm": hyp_n, "cer_norm": norm["cer"]})
        print(f"  [{idx + 1}/{len(items)}] CER={norm['cer']:.3f}  ref={ref_n!r}  hyp={hyp_n!r}")

    return {
        "model": model_path, "kind": kind, "n_utt": len(items),
        "cer_raw": agg_raw_dist / agg_raw_N if agg_raw_N else 0.0,
        "cer_norm": agg_norm_dist / agg_norm_N if agg_norm_N else 0.0,
        "cer_norm_macro": sum(u["cer_norm"] for u in per_utt) / len(per_utt) if per_utt else 0.0,
        "sent_acc": sent_correct / len(items) if items else 0.0,
        "per_utt": per_utt,
        "top_subs": sub_pairs.most_common(15),
        "top_dels": del_chars.most_common(10),
        "top_ins": ins_chars.most_common(10),
    }


def main():
    p = argparse.ArgumentParser(description="评估 Whisper/LoRA 模型的中文 CER",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="一个或多个模型路径（base 目录或 LoRA checkpoint 目录）")
    p.add_argument("--data", required=True, help="评测集 json（dataset_builder 的 test.json 格式）")
    p.add_argument("--num-beams", type=int, default=1, help="beam 数；默认 1，与生产 api_recognition 一致")
    p.add_argument("--max-utt", type=int, default=0, help="只评前 N 条（冒烟测试用），0=全量")
    p.add_argument("--output", default=None, help="把完整结果写到这个 json 文件")
    args = p.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(f"❌ 找不到评测集：{args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"评测集：{args.data}（{len(data)} 条）  num_beams={args.num_beams}\n")

    results = []
    for mp in args.models:
        if not os.path.isdir(mp):
            print(f"⚠️ 跳过不存在的模型路径：{mp}", file=sys.stderr)
            continue
        print(f"=== 评测模型：{mp} ===")
        results.append(evaluate_model(mp, data, args.num_beams, args.max_utt))
        print()

    # 汇总对照表
    print("=" * 78)
    print(f"{'模型':<40}{'类型':<10}{'CER(norm)':>10}{'整句准确':>10}")
    print("-" * 78)
    for r in results:
        name = (r["model"][-37:]) if len(r["model"]) > 37 else r["model"]
        print(f"{name:<40}{r['kind']:<10}{r['cer_norm']:>9.1%}{r['sent_acc']:>10.1%}")
    print("=" * 78)
    if len(results) >= 2:
        base_cer, ft_cer = results[0]["cer_norm"], results[-1]["cer_norm"]
        delta = base_cer - ft_cer
        rel = delta / base_cer if base_cer else 0.0
        print(f"Δ CER（第1个 → 最后1个）：{base_cer:.1%} → {ft_cer:.1%}  "
              f"绝对降 {delta:+.1%}，相对降 {rel:+.1%}")

    # 每个模型的错误画像
    for r in results:
        print(f"\n----- {r['model']} 错误画像 -----")
        print(f"CER(raw含标点)={r['cer_raw']:.1%}  CER(norm)={r['cer_norm']:.1%}  "
              f"CER(norm逐句平均)={r['cer_norm_macro']:.1%}")
        if r["top_subs"]:
            print("  高频替换字对 (ref→hyp, 次数):",
                  "  ".join(f"{a}→{b}×{c}" for (a, b), c in r["top_subs"][:10]))
        if r["top_dels"]:
            print("  高频漏字 (次数):", "  ".join(f"{c}×{n}" for c, n in r["top_dels"]))
        if r["top_ins"]:
            print("  高频多字 (次数):", "  ".join(f"{c}×{n}" for c, n in r["top_ins"]))
        worst = sorted(r["per_utt"], key=lambda u: u["cer_norm"], reverse=True)[:5]
        print("  最差 5 句：")
        for u in worst:
            print(f"    CER={u['cer_norm']:.2f}  {u['audio']}  ref={u['ref_norm']!r}  hyp={u['hyp_norm']!r}")

    print("\n⚠️ 提醒：test.json 与训练同源 → 此 CER 偏乐观；且评的是 PyTorch 模型而非线上 q5_0 ggml。")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已写入：{args.output}")


if __name__ == "__main__":
    main()
