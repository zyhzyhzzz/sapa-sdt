"""配对样本示意图：原图 / sham / masked 三联图，裁到截图上半部分（隐私所在区域）。
用法：python make_fig_mask.py [样本 idx，默认 1520]  → fig_mask.pdf
"""
import json, os, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import IMG_DIR, MASKED, MASK_VERIFY  # noqa: E402

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1520
meta = {r["idx"]: r for r in json.load(open(MASK_VERIFY, encoding="utf-8"))}
r = meta[idx]
stem = os.path.splitext(r["image"])[0]
ims = [Image.open(os.path.join(IMG_DIR, r["image"])).convert("RGB"),
       Image.open(os.path.join(MASKED, stem + "_sham.png")).convert("RGB"),
       Image.open(os.path.join(MASKED, stem + "_masked.png")).convert("RGB")]
w, h = ims[0].size
crop = (0, int(0.13 * h), w, int(0.47 * h))  # 去掉状态栏，保留邮件头部
ims = [im.crop(crop) for im in ims]
gap = 24
out = Image.new("RGB", (3 * ims[0].width + 2 * gap, ims[0].height), "white")
for k, im in enumerate(ims):
    out.paste(im, (k * (im.width + gap), 0))
out.save(os.path.join(HERE, "fig_mask.png"))
out.save(os.path.join(HERE, "fig_mask.pdf"))
print(r["priv_text"], r["sham_text"], out.size)
