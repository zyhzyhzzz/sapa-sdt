"""论文正文里不在 analyze.py 表格中的数字，全部从原始回答重新计算。
每行标出对应的论文位置，便于逐项核对。用法：python paper_numbers.py
"""
import json, os, re
import numpy as np
from analyze import DETECT, ALL_MODELS, letter, CATS
from mask import norm
from paths import RUNS, SUMMARY, SUBSET, OCR_PAIRS, MASK_VERIFY


def load(name):
    return list(map(json.loads, open(os.path.join(RUNS, f"{name}.jsonl"), encoding="utf-8")))


def disc(prefix, m, f):
    fn = DETECT[f]
    R = {(r["idx"], r["img"]): fn(r["resp"]) for r in load(f"{prefix}_{f}_{m}")}
    ids = sorted({k[0] for k in R if (k[0], "sham") in R and (k[0], "masked") in R})
    t = np.array([R[(i, "sham")] for i in ids], float)
    q = np.array([R[(i, "masked")] for i in ids], float)
    return (t.mean() - q.mean()) * 100, (t.mean() + 1 - q.mean()) / 2 * 100, t.mean() * 100


def main():
    rows = json.load(open(SUMMARY, encoding="utf-8"))
    det = np.array([r["detect"] for r in rows])
    print(f"[5.2] 16 个组合：corr(SAPA 式识别率, c) = {np.corrcoef(det, [r['c'] for r in rows])[0, 1]:.2f}，"
          f"corr(识别率, d') = {np.corrcoef(det, [r['dprime'] for r in rows])[0, 1]:.2f}；"
          f"max d' = {max(r['dprime'] for r in rows):.2f}，max BA = {max(r['bacc'] for r in rows):.1f}")

    # 5.1 排名翻转的原因：3B 的首选项偏好
    for name, opt in (("level_3b", "官方格式选 A(=Low)"), ("fmt_dfirst_3b", "无隐私放首位后选 A(=无隐私)")):
        v = [letter(r["resp"], "ABCD") for r in load(name) if (r["instr"], r["img"]) == ("orig", "real")]
        print(f"[5.1] Qwen-3B {opt}：{sum(x == 'A' for x in v)}/{len(v)}")
    v = [DETECT["yn"](r["resp"]) for r in load("fmt_yn_ivl2b") if (r["instr"], r["img"]) == ("neutral", "none")]
    print(f"[5.1] InternVL3-2B 是非题，中性化指令 + 不给图答 Yes：{np.mean(v) * 100:.1f}%")

    # 5.3 类别任务里"截图覆盖文本"
    sub = {x["idx"]: x for x in json.load(open(SUBSET, encoding="utf-8"))}
    R = {(r["idx"], r["instr"], r["img"]): r["resp"] for r in load("category_7b")}

    def ok(i, c):
        l = letter(R[(i, *c)], "ABCDEFGHI")
        return l is not None and l != "I" and CATS[ord(l) - 65] == sub[i]["category"]
    ids = [i for i in sub if all((i, *c) in R for c in (("orig", "real"), ("orig", "blank"), ("neutral", "real")))]
    txt = [i for i in ids if ok(i, ("orig", "blank"))]
    flip = [i for i in txt if not ok(i, ("orig", "real"))]
    img = [i for i in ids if ok(i, ("neutral", "real"))]
    keep = [i for i in img if ok(i, ("orig", "real"))]
    print(f"[5.3] 类别题：仅凭原始指令答对 {len(txt)}，加截图后答错 {len(flip)}（{len(flip) / len(txt) * 100:.0f}%）；"
          f"仅凭截图答对 {len(img)}，加原始指令后仍对 {len(keep) / len(img) * 100:.0f}%")

    # 5.4 RA：固定警告回答的分项
    par = [r for r in load("ra_agent_7b_judged-7b") if r["idx"] == "paranoid"][0]["score"]
    print(f"[5.4] 固定警告回答 RA：{ {k: v for k, v in par.items() if 'score' in k} }")

    # 5.5 表 tab:ocr 与读屏文字保留的隐私片段
    print("[5.5] 表 tab:ocr：模型 | 看截图 平均Δ 最高BA | 读OCR 平均Δ 最高BA | 看截图最高 H | 读OCR最高 H")
    for m in ALL_MODELS:
        a = [disc("neg", m, f) for f in DETECT]
        b = [disc("ocrneg", m, f) for f in DETECT]
        print(f"      {m:6s} {np.mean([x[0] for x in a]):+5.1f} {max(x[1] for x in a):5.1f} | "
              f"{np.mean([x[0] for x in b]):+5.1f} {max(x[1] for x in b):5.1f} | {max(x[2] for x in a):5.1f} | {max(x[2] for x in b):5.1f}")
    cache = json.load(open(OCR_PAIRS, encoding="utf-8"))
    hit = tot = 0
    for r in json.load(open(MASK_VERIFY, encoding="utf-8")):
        if r["residual"]:
            continue
        s = norm(cache[os.path.splitext(r["image"])[0] + "_sham"])
        for p in r["priv_text"]:
            if len(norm(p)) < 4:  # 与 mask.py verify 的残留判定口径一致：过短的片段不计
                continue
            tot += 1
            hit += norm(p) in s
    print(f"[5.5] sham 读屏文字保留的目标隐私片段：{hit}/{tot}（{hit / tot * 100:.1f}%）")


if __name__ == "__main__":
    main()
