"""中文 CER（字错率）计算 —— 训练(finetune.py 的 compute_metrics)与评估(scripts/eval_model.py)共用同一套口径。

为什么要共用一个模块：训练时按 CER 选 best checkpoint / 早停，事后又用 eval_model.py 报 CER，
两边必须是同一把尺子，否则「训练选出来的最优」对不上「评估看到的最优」。归一化口径也统一在这里：
繁→简 + 去标点 + 去空白（复用 utils.data_utils 里推理/训练同款的 to_simple / remove_punctuation）。
"""
import re

from utils.data_utils import remove_punctuation, to_simple

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """繁→简 + 去标点 + 去所有空白。中文按字符比对，空格无意义。"""
    return _WS_RE.sub("", remove_punctuation(to_simple(text)))


def align(ref: str, hyp: str):
    """字符级 Levenshtein + 回溯，返回 [(op, ref_char, hyp_char), ...]，op ∈ {eq, sub, del, ins}。"""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("eq", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1])); j -= 1
    ops.reverse()
    return ops


def cer_stats(ref: str, hyp: str) -> dict:
    """单句 CER：返回 dict(S, D, I, N, dist, cer, ops)。"""
    ops = align(ref, hyp)
    s = sum(1 for o in ops if o[0] == "sub")
    d = sum(1 for o in ops if o[0] == "del")
    ins = sum(1 for o in ops if o[0] == "ins")
    n = len(ref)
    dist = s + d + ins
    return {"S": s, "D": d, "I": ins, "N": n, "dist": dist,
            "cer": dist / n if n else (0.0 if dist == 0 else 1.0), "ops": ops}


def corpus_cer(refs, hyps, normalize: bool = True) -> float:
    """语料级（micro）CER = Σ编辑距离 / Σ参考字符数。这是 ASR 报告 CER 的标准口径。"""
    tot_dist = tot_n = 0
    for ref, hyp in zip(refs, hyps):
        if normalize:
            ref, hyp = normalize_text(ref), normalize_text(hyp)
        st = cer_stats(ref, hyp)
        tot_dist += st["dist"]
        tot_n += st["N"]
    return tot_dist / tot_n if tot_n else 0.0
