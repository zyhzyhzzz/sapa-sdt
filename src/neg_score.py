"""配对样本的连续分数（ROC / AUC 用）：输入与 neg_eval.py 完全相同，但不生成，只做一次前向，
记下第一个生成位置上候选答案的对数概率。分析时把"答有隐私"与"答无隐私"的对数概率之差当作分数，
在所有阈值上扫一遍得到 ROC，不再只看贪心解码对应的那一个工作点。
top = 贪心会选的 token，用来核对与 neg_eval.py 的回答一致。
用法：python neg_score.py --model 7b --fmt yn
"""
import argparse, json, os, time
from PIL import Image
from common import load, next_logprobs
from neg_eval import FMTS
from paths import FILL, MASKED, RUNS, NEUTRAL, MASK_VERIFY

WORDS = {"yn": ["Yes", "No", "yes", "no"]}  # 其余三种格式都是 A–D 单选


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b")
    ap.add_argument("--fmt", choices=list(FMTS), required=True)
    a = ap.parse_args()
    words = WORDS.get(a.fmt, ["A", "B", "C", "D"])
    meta = [r for r in json.load(open(MASK_VERIFY, encoding="utf-8")) if not r["residual"]]
    neu = {r["idx"]: r["neutral"] for r in map(json.loads, open(NEUTRAL, encoding="utf-8"))}
    out = os.path.join(RUNS, f"score{'' if FILL == 'median' else '-' + FILL}_{a.fmt}_{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["idx"], r["img"]) for r in map(json.loads, open(out, encoding="utf-8"))}
    if all((r["idx"], k) in done for r in meta for k in ("sham", "masked")):
        return
    model, proc = load(a.model)
    for w in words:  # 候选词必须是单个 token，否则首 token 概率不代表整个词
        assert len(proc.tokenizer.encode(w, add_special_tokens=False)) == 1, w
    t0, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for r in meta:
            stem = os.path.splitext(r["image"])[0]
            for kind in ("sham", "masked"):
                if (r["idx"], kind) in done:
                    continue
                im = Image.open(os.path.join(MASKED, f"{stem}_{kind}.png")).convert("RGB")
                lp, top = next_logprobs(model, proc, f"{FMTS[a.fmt]}\nInstruction:\n{neu[r['idx']]}", words, image=im)
                f.write(json.dumps({"idx": r["idx"], "img": kind, "lp": lp, "top": top}, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 60 == 1:
                    print(f"[{a.model} {a.fmt}] {n} {time.time()-t0:.0f}s | {kind} {top!r}", flush=True)


if __name__ == "__main__":
    main()
