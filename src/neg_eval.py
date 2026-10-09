"""在配对负样本上重测：中性化指令 × {sham 假遮挡（正样本）, masked 真遮挡（负样本）}。
四种提问格式与前面相同；判别力 = P(检出|sham) - P(检出|masked)。总选同一个选项的模型，判别力为 0。
用法：python neg_eval.py --model 7b --fmt level
"""
import argparse, json, os, time
from PIL import Image
from common import load, generate
from run_cf import LEVEL, LEVEL_EXPLICIT
from format_probe import PROMPTS
from paths import MASKED, RUNS, NEUTRAL, MASK_VERIFY

FMTS = {"level": LEVEL, "level_explicit": LEVEL_EXPLICIT, "dfirst": PROMPTS["dfirst"], "yn": PROMPTS["yn"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b")
    ap.add_argument("--fmt", choices=FMTS, required=True)
    a = ap.parse_args()
    # 只用重新 OCR 后读不到被遮隐私文字的样本（mask.py verify 的结果）
    meta = [r for r in json.load(open(MASK_VERIFY, encoding="utf-8")) if not r["residual"]]
    neu = {r["idx"]: r["neutral"] for r in map(json.loads, open(NEUTRAL, encoding="utf-8"))}
    out = os.path.join(RUNS, f"neg_{a.fmt}_{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["idx"], r["img"]) for r in map(json.loads, open(out, encoding="utf-8"))}
    model, proc = load(a.model)
    t0, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for r in meta:
            stem = os.path.splitext(r["image"])[0]
            for kind in ("sham", "masked"):
                if (r["idx"], kind) in done:
                    continue
                im = Image.open(os.path.join(MASKED, f"{stem}_{kind}.png")).convert("RGB")
                resp = generate(model, proc, f"{FMTS[a.fmt]}\nInstruction:\n{neu[r['idx']]}", image=im, max_new_tokens=6)
                f.write(json.dumps({"idx": r["idx"], "img": kind, "resp": resp}, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 60 == 1:
                    print(f"[{a.model} {a.fmt}] {n} {time.time()-t0:.0f}s | {kind} {resp!r}", flush=True)


if __name__ == "__main__":
    main()
