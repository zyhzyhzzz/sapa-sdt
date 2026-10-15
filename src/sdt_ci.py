"""信号检测论指标的置信区间：d′、c，以及 16 个（模型 × 格式）组合上 corr(SAPA 式检出率, c)、corr(检出率, d′)。

评测集内部有近重复截图，所以全部按近重复簇整簇重采样（簇定义同 analyze.clusters，来自 artifacts/test_dups.json）。
每次重采样里 sham/masked 两张图跟着同一个场景走（配对），所有组合共用同一组重采样下标，
这样相关系数的 CI 同时反映了两类数据的抽样波动。B = 2000，种子 0。
用法：python sdt_ci.py  →  打印表格并写 artifacts/sdt_ci.json
"""
import json, os
import numpy as np
from scipy.stats import norm
from analyze import ALL_MODELS, DETECT, clusters
from paths import RUNS, SUMMARY

SRC = {"level": "level", "level_explicit": "level_explicit", "dfirst": "fmt_dfirst", "yn": "fmt_yn"}
B = 2000


def sdt(t, q):
    """t/q: (..., n) 的 0/1 数组；用 (k + 0.5) / (n + 1) 校正，返回 d′、c。"""
    n = t.shape[-1]
    zH, zF = norm.ppf((t.sum(-1) + .5) / (n + 1)), norm.ppf((q.sum(-1) + .5) / (n + 1))
    return zH - zF, -(zH + zF) / 2


def cluster_index(ids, rng):
    find = clusters()
    cid = np.unique([find(i) for i in ids], return_inverse=True)[1]
    groups = [np.flatnonzero(cid == k) for k in range(cid.max() + 1)]
    pick = rng.integers(0, len(groups), (B, len(groups)))
    return [np.concatenate([groups[g] for g in row]) for row in pick], len(groups)


def main():
    rng = np.random.default_rng(0)
    combos = []
    for m in ALL_MODELS:
        for f, fn in DETECT.items():
            p1, p2 = os.path.join(RUNS, f"{SRC[f]}_{m}.jsonl"), os.path.join(RUNS, f"neg_{f}_{m}.jsonl")
            if not (os.path.exists(p1) and os.path.exists(p2)):
                continue
            V = {r["idx"]: fn(r["resp"]) for r in map(json.loads, open(p1, encoding="utf-8")) if (r["instr"], r["img"]) == ("orig", "real")}
            R = {(r["idx"], r["img"]): fn(r["resp"]) for r in map(json.loads, open(p2, encoding="utf-8"))}
            combos.append((m, f, V, R))
    # 所有组合共用同一批场景，取交集保证下标对齐
    vid = sorted(set.intersection(*[set(V) for _, _, V, _ in combos]))
    pid = sorted(set.intersection(*[{k[0] for k in R if (k[0], "sham") in R and (k[0], "masked") in R} for *_, R in combos]))
    vix, nv = cluster_index(vid, rng)
    pix, npc = cluster_index(pid, rng)
    print(f"组合 {len(combos)}；SAPA 式检出 n={len(vid)}（{nv} 簇）；配对 n={len(pid)}（{npc} 簇）；B={B}，按簇重采样")
    print(f"  {'模型':6s} {'格式':15s} {'检出率 [95% CI]':>20s} {'d′ [95% CI]':>22s} {'c [95% CI]':>22s}")
    rows, DET, DP, C = [], [], [], []
    for m, f, V, R in combos:
        v = np.array([V[i] for i in vid], float)
        t = np.array([R[(i, "sham")] for i in pid], float)
        q = np.array([R[(i, "masked")] for i in pid], float)
        det_b = np.array([v[ix].mean() * 100 for ix in vix])
        dp_b, c_b = map(np.array, zip(*[sdt(t[ix], q[ix]) for ix in pix]))
        dp, c = sdt(t, q)
        ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
        r = dict(model=m, fmt=f, detect=float(v.mean() * 100), detect_ci=ci(det_b),
                 dprime=float(dp), dprime_ci=ci(dp_b), c=float(c), c_ci=ci(c_b))
        rows.append(r); DET.append(det_b); DP.append(dp_b); C.append(c_b)
        print(f"  {m:6s} {f:15s} {r['detect']:6.1f} [{r['detect_ci'][0]:5.1f},{r['detect_ci'][1]:5.1f}] "
              f"{dp:+7.2f} [{r['dprime_ci'][0]:+5.2f},{r['dprime_ci'][1]:+5.2f}] "
              f"{c:+7.2f} [{r['c_ci'][0]:+5.2f},{r['c_ci'][1]:+5.2f}]")
    DET, DP, C = map(np.array, (DET, DP, C))  # (组合, B)
    det0 = np.array([r["detect"] for r in rows]); dp0 = np.array([r["dprime"] for r in rows]); c0 = np.array([r["c"] for r in rows])
    corr = lambda a, b: np.array([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(B)])
    out = dict(rows=rows, n_detect=len(vid), n_detect_clusters=nv, n_pairs=len(pid), n_pair_clusters=npc, B=B)
    for name, a0, ab in (("c", c0, C), ("dprime", dp0, DP)):
        r0, rb = np.corrcoef(det0, a0)[0, 1], corr(DET, ab)
        out[f"corr_detect_{name}"] = dict(r=float(r0), ci=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))])
        print(f"  corr(检出率, {name}) = {r0:+.2f} [{np.percentile(rb, 2.5):+.2f},{np.percentile(rb, 97.5):+.2f}]")
    hi = max(r["dprime_ci"][1] for r in rows)
    out["dprime_max_upper"] = hi
    print(f"  d′ 点估计最大 {dp0.max():.2f}；所有组合 95% CI 上界最大 {hi:.2f}")
    json.dump(out, open(os.path.join(os.path.dirname(SUMMARY), "sdt_ci.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
