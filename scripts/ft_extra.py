"""微调结论的补充汇总（在 analyze.py 之后跑，读 summary_ft.json 与 category_*.jsonl）：
1. 多种子：3b+pair（种子 0）/ 3b+pair-s1 / 3b+pair-s2 的 d'、c、判别力，报告均值 ± 标准差（n-1）与范围。
2. 副作用：配对微调是否损害 SAPA 原有的隐私类别识别（9 选 1，原始指令 + 真实截图）。
   与原模型逐条配对比较类别正确率（精确 McNemar）。
用法：python scripts/ft_extra.py > logs/ft_extra.txt
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from analyze import RUNS, SUBSET, SUMMARY, CATS, letter, boot, mcnemar

FMTS = ("level", "level_explicit", "dfirst", "yn")


def seeds():
    S = {(r["model"], r["fmt"]): r for r in json.load(open(SUMMARY.replace("summary.json", "summary_ft.json"), encoding="utf-8"))}
    ms = [m for m in ("3b+pair", "3b+pair-s1", "3b+pair-s2") if all((m, f) in S for f in FMTS)]
    print(f"== 多种子（{', '.join(ms)}）：均值 ± 标准差 [最小, 最大]")
    for f in FMTS:
        line = f"  {f:15s}"
        for k in ("dprime", "c", "disc", "bacc"):
            v = np.array([S[(m, f)][k] for m in ms])
            line += f" | {k} {v.mean():+6.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:4.2f} [{v.min():+.2f},{v.max():+.2f}]"
        print(line)
        print("    各种子 d':", " ".join(f"{S[(m, f)]['dprime']:+.2f}" for m in ms))


def sidefx(models=("3b", "3b+pair", "3b+pos", "3b+shuffle")):
    sub = {x["idx"]: x for x in json.load(open(SUBSET, encoding="utf-8"))}
    R = {}
    for m in models:
        p = os.path.join(RUNS, f"category_{m}.jsonl")
        if os.path.exists(p):
            R[m] = {(r["idx"], r["instr"]): r["resp"] for r in map(json.loads, open(p, encoding="utf-8")) if r["img"] == "real"}
    if "3b" not in R:
        return
    for ins in ("orig", "neutral"):
        ids = sorted(i for i in sub if all((i, ins) in R[m] for m in R))
        print(f"\n== 副作用 · 类别识别（指令={ins}，真实截图，n={len(ids)}）")
        base = None
        for m in R:
            L = [letter(R[m][(i, ins)], "ABCDEFGHI") for i in ids]
            det = [l is not None and l != "I" for l in L]
            cor = np.array([l is not None and l != "I" and CATS[ord(l) - 65] == sub[i]["category"] for l, i in zip(L, ids)])
            d, k = boot(det, ids=ids), boot(cor, ids=ids)
            line = (f"  {m:11s} 检出 {d[0]:5.1f} [{d[1]:4.1f},{d[2]:4.1f}] | 类别正确 {k[0]:5.1f} [{k[1]:4.1f},{k[2]:4.1f}]"
                    f" | 无效 {sum(l is None for l in L)}")
            if base is None:
                base = cor
            else:
                n10, n01, pv = mcnemar(base, cor)
                line += f" | vs 3b: 仅原模型对 {n10} / 仅微调对 {n01}, McNemar p={pv:.2g}"
            print(line)


if __name__ == "__main__":
    seeds()
    sidefx()
