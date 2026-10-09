"""从 SAPA 中抽取微调用的训练场景，与评测用的 320 个场景严格隔离。

隔离规则：
  (1) 同一 episode（文件名中间段）的截图只要有一张在评测集里，整个 episode 都不进训练集；
  (2) 与任一评测指令 TF-IDF 余弦相似度 ≥ 0.8 的指令不进训练集（防止模板化指令泄漏）。
其余做法与 make_subset.py 相同：只取 "Both" 位置，按 8 个隐私类别分层，每类 100 个，种子 0。
近重复截图在图片下载后由 finetune/check_leak.py 再查一遍（感知哈希）。
用法：python make_train_split.py   → artifacts/train/subset320.json（文件名沿用评测集，内容是训练场景）
"""
import collections, json, os, random, re
import numpy as np
from paths import ANNOT, ROOT

PER_CAT = 100


def episode(img):
    """SAPA 截图文件名有三种：<a>_<episode>_<step>.png（占大多数）、<...>_<episode>-<step>.png、
    以及无 episode 信息的 UUID.png（这类只能靠后面的感知哈希查重）。"""
    stem = os.path.splitext(img)[0]
    m = re.fullmatch(r"\d+_(\d+)_\d+", stem)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(.+_[0-9a-f]+)-\d+", stem)
    return m.group(1) if m else stem


def main():
    random.seed(0)
    d = json.load(open(ANNOT, encoding="utf-8"))
    test = json.load(open(os.path.join(ROOT, "artifacts", "subset320.json"), encoding="utf-8"))
    test_ep = {episode(x["image"]) for x in test}
    from sklearn.feature_extraction.text import TfidfVectorizer
    pool = [dict(idx=i, image=x["images"], category=x["image_type"].strip().lower(),
                 level=x["conversations1"]["security_type"].strip().capitalize(),
                 instruction=x["conversations1"]["instructions"].strip())
            for i, x in enumerate(d) if x["Privacy-exposed locations"].startswith("Both")]
    n0 = len(pool)
    pool = [x for x in pool if episode(x["image"]) not in test_ep]
    n1 = len(pool)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([x["instruction"] for x in pool + test])
    A = vec.transform([x["instruction"] for x in pool])
    B = vec.transform([x["instruction"] for x in test])
    sim = (A @ B.T).max(1).toarray().ravel()
    pool = [x for x, s in zip(pool, sim) if s < 0.8]
    print(f"候选 {n0} → 排除评测 episode 后 {n1} → 排除近重复指令后 {len(pool)}")
    by = collections.defaultdict(list)
    for x in pool:
        by[x["category"]].append(x)
    sub = []
    for k in sorted(by):
        sub += random.sample(by[k], min(PER_CAT, len(by[k])))
    out = os.path.join(ROOT, "artifacts", "train", "subset320.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(sub, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(sub), collections.Counter(x["category"] for x in sub))


if __name__ == "__main__":
    main()
