"""ROC / AUC 分析（neg_score.py 的输出）。分数 = log P(答有隐私) − log P(答无隐私)：
  yn                      logsumexp(Yes, yes) − logsumexp(No, no)
  level / level_explicit  logsumexp(A, B, C) − D
  dfirst                  logsumexp(B, C, D) − A
sham 为正、masked 为负，每个模型 × 格式报：
  AUC        全部 sham 对全部 masked 的 Mann–Whitney 统计量；与阈值无关，0.5 = 不能分辨
  配对 AUC   同一截图内 sham 分数高于 masked 的比例（平局记 0.5），即同屏二选一的正确率
  Φ(d′/√2)   由贪心解码那一个工作点的 d′ 推出的 AUC（等方差假设下），与上面对照
  zROC 斜率  在所有阈值上 z(H) 对 z(F) 的回归斜率；等方差假设成立时为 1
  一致率     一次前向的 argmax 与 neg_eval.py 贪心回答第一个 token 相同的比例，核对两者是同一个模型行为
CI 为按近重复簇整簇重采样的 bootstrap 95% CI（2000 次，种子 0），与 analyze.py 相同。
用法：python roc.py                  结果写 artifacts/summary_roc.json
"""
import json, os
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, pearsonr, rankdata
from analyze import ALL_MODELS, DETECT, clusters
from paths import RUNS, SUMMARY

POS = {"yn": (["Yes", "yes"], ["No", "no"]), "level": ("ABC", "D"),
       "level_explicit": ("ABC", "D"), "dfirst": ("BCD", "A")}
FT = ("3b+pair", "3b+pair-s1", "3b+pair-s2", "3b+pos", "3b+shuffle")


def score(lp, fmt):
    y, n = POS[fmt]
    return logsumexp([lp[w] for w in y]) - logsumexp([lp[w] for w in n])


def auc(p, q):
    """P(p > q) + 0.5 P(p = q)，p 为正样本分数，q 为负样本分数。"""
    r = rankdata(np.concatenate([p, q]))  # 平局取平均秩
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q))


def zroc_slope(p, q):
    th = np.unique(np.concatenate([p, q]))
    H = np.array([(p >= t).mean() for t in th])
    F = np.array([(q >= t).mean() for t in th])
    k = (H > .01) & (H < .99) & (F > .01) & (F < .99)
    if k.sum() < 3:
        return float("nan")
    return float(np.polyfit(norm.ppf(F[k]), norm.ppf(H[k]), 1)[0])


def load(model, fmt, prefix="score"):
    p = os.path.join(RUNS, f"{prefix}_{fmt}_{model}.jsonl")
    if not os.path.exists(p):
        return None
    S = {(r["idx"], r["img"]): r for r in map(json.loads, open(p, encoding="utf-8"))}
    ids = sorted({i for i, _ in S if (i, "sham") in S and (i, "masked") in S})
    return S, ids


def row(model, fmt, B=2000):
    got = load(model, fmt)
    if got is None or len(got[1]) < 196:
        return None
    S, ids = got
    p = np.array([score(S[(i, "sham")]["lp"], fmt) for i in ids])
    q = np.array([score(S[(i, "masked")]["lp"], fmt) for i in ids])
    pair = (p > q) + 0.5 * (p == q)
    # 核对：一次前向的 argmax 与 neg_eval.py 贪心回答的第一个 token 相同。
    # 注意不能用"分数 > 0"核对：多选题的分数比较 P(A)+P(B)+P(C) 与 P(D)，贪心只取单个最大选项，两者工作点不同
    resp = {(r["idx"], r["img"]): r["resp"]
            for r in map(json.loads, open(os.path.join(RUNS, f"neg_{fmt}_{model}.jsonl"), encoding="utf-8"))}
    agree = np.mean([resp[k].strip().startswith(S[k]["top"].strip()) for k in S if k in resp])
    g = {k: DETECT[fmt](v) for k, v in resp.items()}
    t, f = np.array([g[(i, "sham")] for i in ids], float), np.array([g[(i, "masked")] for i in ids], float)
    dp = norm.ppf((t.sum() + .5) / (len(t) + 1)) - norm.ppf((f.sum() + .5) / (len(f) + 1))
    # 按近重复簇整簇重采样
    find = clusters()
    cid = np.unique([find(i) for i in ids], return_inverse=True)[1]
    groups = [np.flatnonzero(cid == k) for k in range(cid.max() + 1)]
    rng = np.random.default_rng(0)
    A, P = [], []
    for _ in range(B):
        ix = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        A.append(auc(p[ix], q[ix]))
        P.append(pair[ix].mean())
    return dict(model=model, fmt=fmt, n=len(ids), auc=float(auc(p, q)),
                auc_lo=float(np.percentile(A, 2.5)), auc_hi=float(np.percentile(A, 97.5)),
                pair_auc=float(pair.mean()), pair_lo=float(np.percentile(P, 2.5)), pair_hi=float(np.percentile(P, 97.5)),
                dprime=float(dp), auc_from_dprime=float(norm.cdf(dp / np.sqrt(2))),
                slope=zroc_slope(p, q), agree=float(agree),
                mean_sham=float(p.mean()), mean_masked=float(q.mean()))


def table(models, title):
    print(f"\n== {title}")
    print(f"  {'模型':11s} {'格式':15s} {'AUC [95% CI]':>21s} {'配对AUC [95% CI]':>21s} {'Φ(d′/√2)':>9s} {'zROC斜率':>8s} {'一致率':>6s}")
    rows = []
    for m in models:
        for f in POS:
            r = row(m, f)
            if r is None:
                continue
            rows.append(r)
            print(f"  {m:11s} {f:15s} {r['auc']:.3f} [{r['auc_lo']:.3f},{r['auc_hi']:.3f}] "
                  f"{r['pair_auc']:.3f} [{r['pair_lo']:.3f},{r['pair_hi']:.3f}] {r['auc_from_dprime']:9.3f} "
                  f"{r['slope']:8.2f} {r['agree'] * 100:5.1f}%")
    return rows


def main():
    rows = table(ALL_MODELS, "配对样本 ROC：原模型（sham = 正，masked = 负）")
    rows += table(("3b",) + FT, "配对样本 ROC：微调模型（与原 3B 对照）")
    json.dump(rows, open(SUMMARY.replace("summary.json", "summary_roc.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # 区分效度：SAPA 式检出率（summary.json）与与阈值无关的 AUC 的相关
    det = {(r["model"], r["fmt"]): r["detect"] for r in json.load(open(SUMMARY, encoding="utf-8"))}
    base = [r for r in rows if r["model"] in ALL_MODELS and (r["model"], r["fmt"]) in det]
    base = list({(r["model"], r["fmt"]): r for r in base}.values())
    if len(base) >= 4:
        x = [det[(r["model"], r["fmt"])] for r in base]
        for k in ("auc", "pair_auc"):
            rr, pv = pearsonr(x, [r[k] for r in base])
            print(f"  corr(检出率, {k}) = {rr:+.2f}（p = {pv:.2g}，n = {len(base)}）")
        print(f"  AUC 范围 {min(r['auc'] for r in base):.3f}–{max(r['auc'] for r in base):.3f}；"
              f"检出率范围 {min(x):.1f}–{max(x):.1f}")


if __name__ == "__main__":
    main()
