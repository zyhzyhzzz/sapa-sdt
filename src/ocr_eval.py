"""感知 vs 推理诊断：把截图换成它的 OCR 文字（按从上到下、从左到右排序），其余与 neg_eval 相同。
若判别力明显上升 → 瓶颈在"看清屏幕上的字"（感知）；若仍接近 0 → 瓶颈在"认出这是隐私"（推理）。
OCR 文字只来自图像本身（masked 图的 OCR 自然不含被遮内容），相当于一个"完美读屏"的模型。

用法：python ocr_eval.py ocr                    # CPU，先对 196 对图做 OCR 并缓存
      python ocr_eval.py run --model 7b --fmt yn
"""
import argparse, json, os, time
from paths import MASKED, RUNS, NEUTRAL, MASK_VERIFY, OCR_PAIRS

CACHE = OCR_PAIRS


def pairs():
    return [r for r in json.load(open(MASK_VERIFY, encoding="utf-8")) if not r["residual"]]


def screen_text(res):
    # rapidocr 框：按左上角 y（粗分行，20px 一档）再按 x 排序
    items = sorted(res, key=lambda b: (round(b[0][0][1] / 20), b[0][0][0]))
    return "\n".join(t for _, t, c in items if c >= 0.5)


def cmd_ocr():
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    for i, r in enumerate(pairs()):
        stem = os.path.splitext(r["image"])[0]
        for kind in ("sham", "masked"):
            k = f"{stem}_{kind}"
            if k in cache:
                continue
            res, _ = ocr(os.path.join(MASKED, k + ".png"))
            cache[k] = screen_text([[b, t, float(c)] for b, t, c in (res or [])])
        if i % 20 == 0:
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
            print(i, flush=True)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)


def cmd_run(a):
    from common import load, generate
    from neg_eval import FMTS
    cache = json.load(open(CACHE, encoding="utf-8"))
    neu = {r["idx"]: r["neutral"] for r in map(json.loads, open(NEUTRAL, encoding="utf-8"))}
    out = os.path.join(RUNS, f"ocrneg_{a.fmt}_{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["idx"], r["img"]) for r in map(json.loads, open(out, encoding="utf-8"))}
    model, proc = load(a.model)
    # 提示词里把 "Observe the image" 换成读屏文字的说法，其余选项与格式不变
    fmt = FMTS[a.fmt].replace("Observe the image (if any) and", "Read the screen text and") \
                     .replace("Observe the image and", "Read the screen text and") \
                     .replace("on the shown screen", "on the screen described by the screen text")
    t0, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for r in pairs():
            stem = os.path.splitext(r["image"])[0]
            for kind in ("sham", "masked"):
                if (r["idx"], kind) in done:
                    continue
                txt = f"{fmt}\nScreen text (OCR, top to bottom):\n{cache[f'{stem}_{kind}']}\nInstruction:\n{neu[r['idx']]}"
                resp = generate(model, proc, txt, image=None, max_new_tokens=6)
                f.write(json.dumps({"idx": r["idx"], "img": kind, "resp": resp}, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 60 == 1:
                    print(f"[ocr {a.model} {a.fmt}] {n} {time.time()-t0:.0f}s | {kind} {resp!r}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ocr", "run"])
    ap.add_argument("--model", default="7b")
    ap.add_argument("--fmt", default="yn")
    a = ap.parse_args()
    cmd_ocr() if a.cmd == "ocr" else cmd_run(a)
