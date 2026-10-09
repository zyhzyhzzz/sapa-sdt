"""训练/评测截图近重复检查。SAPA 会在不同 episode 里复用几乎相同的截图，所以按文件名隔离 episode 不够。
判定为近重复（任一即可）：64 位差值哈希（dHash）汉明距离 ≤ 6，或缩到 60x135 灰度后平均像素差 ≤ 5（0–255）。
命中的训练场景写入 artifacts/train/leak.json，构造训练数据时剔除。另报告评测集内部的近重复对。
用法：python finetune/check_leak.py
"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import IMG_DIR, ROOT  # noqa: E402


def feats(name):
    im = Image.open(os.path.join(IMG_DIR, name)).convert("L")
    a = np.asarray(im.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel()), np.asarray(im.resize((60, 135), Image.BILINEAR), np.float32)


def near(h, p, H, P):
    ham = np.unpackbits(H ^ h, axis=1).sum(1)
    pix = np.abs(P - p).mean((1, 2))
    return (ham <= 6) | (pix <= 5), ham, pix


def main():
    ART = os.path.join(ROOT, "artifacts")
    test = json.load(open(os.path.join(ART, "subset320.json"), encoding="utf-8"))
    raw = os.path.join(ART, "train", "subset_raw.json")
    train = json.load(open(raw if os.path.exists(raw) else os.path.join(ART, "train", "subset320.json"), encoding="utf-8"))
    TF = [feats(x["image"]) for x in test]
    H, P = np.stack([f[0] for f in TF]), np.stack([f[1] for f in TF])
    leak = []
    for x in train:
        hit, ham, pix = near(*feats(x["image"]), H, P)
        if hit.any():
            j = int(np.argmin(np.where(hit, pix, np.inf)))
            leak.append(dict(idx=x["idx"], image=x["image"], test_image=test[j]["image"], dhash=int(ham[j]),
                             pixdiff=round(float(pix[j]), 2)))
    json.dump(leak, open(os.path.join(ART, "train", "leak.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 原始抽样保留为 subset_raw.json；剔除近重复后的场景写回 subset320.json，供后续 neutralize / mask 直接使用
    raw = os.path.join(ART, "train", "subset_raw.json")
    if not os.path.exists(raw):
        json.dump(train, open(raw, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    bad = {l["idx"] for l in leak}
    keep = [x for x in json.load(open(raw, encoding="utf-8")) if x["idx"] not in bad]
    json.dump(keep, open(os.path.join(ART, "train", "subset320.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"与评测集近重复的训练截图 {len(leak)}/{len(train)}（剔除），保留 {len(keep)}")
    dup = []
    for i in range(len(test)):
        hit, _, _ = near(H[i], P[i], H[i + 1:], P[i + 1:])
        dup += [(test[i]["idx"], test[i + 1 + j]["idx"]) for j in np.flatnonzero(hit)]
    json.dump(dup, open(os.path.join(ART, "test_dups.json"), "w"), indent=0)
    print(f"评测集内部近重复对 {len(dup)}，涉及 {len({i for p in dup for i in p})} 个场景")


if __name__ == "__main__":
    main()
