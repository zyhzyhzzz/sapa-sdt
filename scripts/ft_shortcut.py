"""微调捷径检验：3b+pair 是否靠“抹掉的面积”判 No（按面积分箱、面积匹配子集、逻辑回归），以及按隐私类别的判别力。用法：python scripts/ft_shortcut.py（在仓库根目录）"""
import json, sys, collections
import numpy as np
meta = {r["idx"]: r for r in json.load(open("artifacts/mask_verify.json", encoding="utf-8"))}
def load(m, f):
    d = collections.defaultdict(dict)
    for r in map(json.loads, open(f"results/runs/neg_{f}_{m}.jsonl", encoding="utf-8")):
        d[r["idx"]][r["img"]] = r["resp"].strip().lower().startswith("yes")
    return d
for m in ["3b", "3b+pair"]:
    d = load(m, "yn")
    rows = [(meta[i]["priv_area"], meta[i]["sham_area"], v["sham"], v["masked"], meta[i]["category"]) for i, v in d.items() if i in meta]
    pa, sa, ys, ym, cat = map(np.array, zip(*rows))
    print(f"== {m} yn  n={len(rows)}")
    # 面积捷径：若模型按"抹掉面积"判 No，则 sham 面积大时也会更多答 No
    allarea = np.concatenate([sa, pa]); allyes = np.concatenate([ys, ym]); ismask = np.r_[np.zeros(len(sa)), np.ones(len(pa))]
    q = np.quantile(allarea, [0, .25, .5, .75, 1])
    for lo, hi in zip(q[:-1], q[1:]):
        for k, nm in [(0, "sham"), (1, "masked")]:
            s = (allarea >= lo) & (allarea <= hi) & (ismask == k)
            print(f"  area[{lo:7.0f},{hi:7.0f}] {nm:6s} n={s.sum():3d}  P(Yes)={allyes[s].mean():.2f}")
    # 面积匹配子集：sham 面积 >= 隐私面积 的配对
    s = sa >= pa
    print(f"  sham面积>=隐私面积 的配对 n={s.sum()}: P(Yes|sham)={ys[s].mean():.2f} P(Yes|masked)={ym[s].mean():.2f} 判别力={ys[s].mean()-ym[s].mean():+.2f}")
    print(f"  其余配对 n={(~s).sum()}: 判别力={ys[~s].mean()-ym[~s].mean():+.2f}")
    # 逻辑回归：Yes ~ 是否遮隐私 + log面积
    try:
        from sklearn.linear_model import LogisticRegression
        X = np.c_[ismask, np.log(allarea)]; X = (X - X.mean(0)) / X.std(0)
        lr = LogisticRegression().fit(X, allyes); print("  标准化系数 [遮隐私, log面积] =", lr.coef_.round(2))
    except Exception as e: print(e)
    print("  按类别判别力:")
    for c in sorted(set(cat)):
        s = cat == c
        print(f"    {c[:38]:38s} n={s.sum():3d} {ys[s].mean()-ym[s].mean():+.2f}")
