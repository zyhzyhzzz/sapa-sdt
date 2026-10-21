"""原模型配对 AUC 的面积捷径检验（连续分数版，对应 ft_shortcut.py 对微调模型做的检验）。
若模型只是"抹掉的面积越大越倾向答无隐私"，同屏分数差 d = s(sham) − s(masked) 应随 log(隐私面积 / sham 面积) 增大，
且在 sham 面积 ≥ 隐私面积 的配对上消失甚至反向。回归 d ~ log面积比，截距 = 两者面积相等时的同屏分数差。
用法：python scripts/roc_shortcut.py（在仓库根目录）
"""
import json, sys
import numpy as np
from scipy.stats import linregress
sys.path.insert(0, "src")
from roc import load, score

meta = {r["idx"]: r for r in json.load(open("artifacts/mask_verify.json", encoding="utf-8"))}
print(f"{'模型':10s} {'格式':15s} {'配对AUC':>7s} {'sham≥隐私面积':>14s} {'其余':>10s} {'截距 [p]':>16s} {'斜率 [p]':>16s}")
for m in ["3b", "7b", "ivl2b", "ivl8b", "3b+pair"]:
    for f in ["yn", "level", "level_explicit", "dfirst"]:
        S, ids = load(m, f)
        d = np.array([score(S[(i, "sham")]["lp"], f) - score(S[(i, "masked")]["lp"], f) for i in ids])
        x = np.log(np.array([meta[i]["priv_area"] / meta[i]["sham_area"] for i in ids]))
        big = x <= 0  # sham 面积 ≥ 隐私面积：面积捷径预测 sham 分数不高于 masked
        pa = lambda v: ((v > 0) + 0.5 * (v == 0)).mean()
        r = linregress(x, d)
        # 截距的 t 检验
        n = len(x); res = d - (r.intercept + r.slope * x)
        se_b0 = np.sqrt(res.var(ddof=2) * (1 / n + x.mean() ** 2 / ((x - x.mean()) ** 2).sum()))
        from scipy.stats import t as tdist
        p0 = 2 * tdist.sf(abs(r.intercept / se_b0), n - 2)
        print(f"{m:10s} {f:15s} {pa(d):7.3f} {pa(d[big]):8.3f} (n={big.sum():3d}) {pa(d[~big]):6.3f} "
              f"{r.intercept:+7.3f} [{p0:.1g}] {r.slope:+7.3f} [{r.pvalue:.1g}]")
