"""微调泄露敏感性检验：放宽近重复判据（dHash ≤ 10 或像素差 ≤ 10，判重阈值为 6 / 5），去掉与某张训练截图较接近的评测对后，各模型的 d′ 是否变化。用法：python scripts/ft_near_dup.py（在仓库根目录）"""
import json, os, sys
import numpy as np
from scipy.stats import norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "finetune"))
os.chdir(os.path.join(REPO, "src"))
from paths import RUNS  # noqa: E402
from check_leak import feats  # noqa: E402
import analyze  # noqa: E402

ART = os.path.join(REPO, "artifacts")
ld = lambda p: json.load(open(p, encoding="utf-8"))
test = {x["idx"]: x for x in ld(os.path.join(ART, "subset320.json"))}
tsub = {x["idx"]: x for x in ld(os.path.join(ART, "train", "subset320.json"))}
train = [tsub[r["idx"]] for r in ld(os.path.join(ART, "train", "mask_verify.json")) if not r["residual"] and r["idx"] in tsub]
TF = [feats(x["image"]) for x in train]
H, P = np.stack([f[0] for f in TF]), np.stack([f[1] for f in TF])
near = {}
for i, x in test.items():
    h, p = feats(x["image"])
    ham = np.unpackbits(H ^ h, axis=1).sum(1).min()
    pix = np.abs(P - p).mean((1, 2)).min()
    near[i] = (ham <= 10) or (pix <= 10)


def stats(R, ids):
    t = np.array([R[(i, "sham")] for i in ids], float)
    q = np.array([R[(i, "masked")] for i in ids], float)
    zH, zF = norm.ppf((t.sum() + .5) / (len(t) + 1)), norm.ppf((q.sum() + .5) / (len(q) + 1))
    return len(ids), zH - zF, (t.mean() + 1 - q.mean()) / 2 * 100


print("宽松近邻判据：dHash ≤ 10 或像素差 ≤ 10（判重阈值是 6 / 5）")
print(f"{'模型':14s} {'格式':15s} {'全部 n/d′/BA':>20s} {'去掉近邻 n/d′/BA':>22s} {'仅近邻 n/d′/BA':>20s}")
for m in ["3b", "3b+pair", "3b+pair-s1", "3b+pair-s2", "3b+pos", "3b+shuffle"]:
    for f, fn in analyze.DETECT.items():
        p = os.path.join(RUNS, f"neg_{f}_{m}.jsonl")
        if not os.path.exists(p):
            continue
        R = {(r["idx"], r["img"]): fn(r["resp"]) for r in map(json.loads, open(p, encoding="utf-8"))}
        ids = sorted({k[0] for k in R if (k[0], "sham") in R and (k[0], "masked") in R})
        out = []
        for sub in (ids, [i for i in ids if not near[i]], [i for i in ids if near[i]]):
            n, d, ba = stats(R, sub)
            out.append(f"{n:4d} {d:+5.2f} {ba:5.1f}")
        print(f"{m:14s} {f:15s} {out[0]:>20s} {out[1]:>22s} {out[2]:>20s}")
