"""用遮挡隐私区域的方法给 SAPA 造配对负样本。

SAPA 只有正样本，"总选某个选项"和"真看出隐私"无法区分。这里对每张截图生成两个版本：
  masked  遮掉隐私文字框 → 负样本（配中性化指令）
  sham    遮掉同样数量、面积相近的非隐私文字框 → 正样本对照（配中性化指令）
两者被修改的程度相当，只差遮掉的是不是隐私，从而排除"图被改过"这一混淆。

隐私文字框的认定（OCR 用 rapidocr）。负样本的含义是"目标隐私被移除"，截图中可能仍有附带隐私（头像、其他联系人等）：
  (a) 与指令中的隐私片段匹配：引号内容、邮箱、电话、IP、≥4 位数字串、字母数字混合串、首字母大写的人名对；
  (b) 文字本身符合通用隐私格式：邮箱、电话、IP、金额、≥6 位数字串、≥8 位字母数字混合随机串（可含 _ -）、街道地址。
遮挡方式：用文字框外圈像素的中位数颜色填平（不用黑框，避免"黑框 = 这里有隐私"成为新捷径）。
验证：对 masked 图重新 OCR，检查隐私片段是否还残留。

用法：python mask.py ocr | build | verify
"""
import json, os, re, sys, random
import numpy as np
from PIL import Image
from paths import IMG_DIR, MASKED, SUBSET, OCR, MASK_META, MASK_VERIFY

IMG, OUT = IMG_DIR, MASKED

GENERIC = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                      # 邮箱
    re.compile(r"\+?\d[\d ()-]{7,}\d"),                           # 电话/长号码
    re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b"),                     # IP
    re.compile(r"[$€£¥]\s?\d[\d,]*(\.\d+)?"),                     # 金额
    re.compile(r"\b\d{6,}\b"),                                    # ≥6 位数字
    re.compile(r"(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{8,}"),  # 随机串（含 _ -）
    re.compile(r"\b\d+ [A-Z][a-z]+ (St|Dr|Way|Ave|Rd|Blvd|Ln|Street|Drive|Avenue|Road)\b"),  # 街道地址
]
# 语义上敏感的词：假遮挡不能选含这些词的文字框（否则对照组本身也遮掉了隐私）
SENSITIVE = re.compile(r"\b(privacy|private|sensitive|personal|password|passcode|pin|credential|account|login|"
                       r"location|address|contact|phone|email|message|payment|bank|card|credit|financ|balance|"
                       r"medical|health|insurance|emergency|identity|ssn|passport|license|salary|token|key|"
                       r"secret|confidential|security|verification|otp|birth|gallery|vault)\w*", re.I)
UI_WORDS = {"Tap", "Open", "Click", "Select", "Enter", "Send", "Share", "Save", "Add", "Search", "Log", "Turn",
            "Access", "Check", "Copy", "Play", "Post", "Store", "Write", "View", "Google", "Gmail", "Settings"}


def instr_tokens(s):
    toks = set()
    toks |= {m.strip() for m in re.findall(r"'([^']{3,})'|\"([^\"]{3,})\"", s) for m in m if m}
    for rx in GENERIC[:3]:
        toks |= {m.group(0) for m in rx.finditer(s)}
    toks |= set(re.findall(r"\b\d{4,}\b", s))
    toks |= set(re.findall(r"\b(?=[A-Za-z0-9_]*\d)(?=[A-Za-z0-9_]*[A-Za-z])[A-Za-z0-9_]{6,}\b", s))
    for a, b in re.findall(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b", s):
        if a not in UI_WORDS and b not in UI_WORDS:
            toks.add(f"{a} {b}")
    return {t for t in toks if len(t) >= 4}


def norm(t):
    return re.sub(r"[^a-z0-9@.]", "", t.lower())


def is_private(text, toks):
    nt = norm(text)
    if len(nt) >= 4:
        for t in toks:
            n = norm(t)
            # 截图里文字常被截断（如 "2023_04_22_fa..."），所以双向子串匹配，且要求重叠 ≥4 字符
            if len(n) >= 4 and (n in nt or (nt.rstrip(".") in n and len(nt.rstrip(".")) >= 4)):
                return True
    return any(rx.search(text) for rx in GENERIC)


def sham_ok(text, toks, priv_texts):
    """假遮挡只选"显然不是隐私"的纯界面文字：无数字/@/$，且与任何隐私片段都不相似。"""
    from difflib import SequenceMatcher
    if re.search(r"[\d@$€£¥]", text) or any(rx.search(text) for rx in GENERIC):
        return False
    if re.search(r"https?:|www\.|\.com|\.org|\.net|/", text, re.I) or SENSITIVE.search(text):
        return False
    # 像人名（Victor James / KATHLEEN KELLY）或聊天内容（问句）的也排除
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b[A-Z]{2,} [A-Z]{2,}\b|\?\s*$", text.strip()):
        return False
    nt = norm(text)
    if len(nt) < 2:
        return False
    for t in list(toks) + list(priv_texts):
        n = norm(t)
        if not n:
            continue
        if SequenceMatcher(None, nt, n).ratio() >= 0.6:
            return False
        if len(nt) >= 5 and (nt in n or n in nt):
            return False
        # 共享 ≥5 字符的连续片段
        m = SequenceMatcher(None, nt, n).find_longest_match(0, len(nt), 0, len(n))
        if m.size >= 5:
            return False
    return True


def fill(im, box, pad=4):
    a = np.asarray(im).copy()
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    x0, x1 = max(int(min(xs)) - pad, 0), min(int(max(xs)) + pad, a.shape[1])
    y0, y1 = max(int(min(ys)) - pad, 0), min(int(max(ys)) + pad, a.shape[0])
    ring = np.concatenate([a[max(y0 - 3, 0):y0, x0:x1].reshape(-1, 3), a[y1:y1 + 3, x0:x1].reshape(-1, 3),
                           a[y0:y1, max(x0 - 3, 0):x0].reshape(-1, 3), a[y0:y1, x1:x1 + 3].reshape(-1, 3)])
    a[y0:y1, x0:x1] = np.median(ring, 0) if len(ring) else 255
    return Image.fromarray(a)


def area(box):
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def cmd_ocr():
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    p = OCR
    cache = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    for i, x in enumerate(json.load(open(SUBSET, encoding="utf-8"))):
        if x["image"] in cache:
            continue
        res, _ = ocr(os.path.join(IMG, x["image"]))
        cache[x["image"]] = [[[list(map(float, q)) for q in b], t, float(c)] for b, t, c in (res or [])]
        if i % 20 == 0:
            json.dump(cache, open(p, "w", encoding="utf-8"), ensure_ascii=False)
            print(i, flush=True)
    json.dump(cache, open(p, "w", encoding="utf-8"), ensure_ascii=False)


def cmd_build():
    random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    ocr = json.load(open(OCR, encoding="utf-8"))
    rows = []
    for x in json.load(open(SUBSET, encoding="utf-8")):
        boxes = [b for b in ocr[x["image"]] if b[2] >= 0.5]
        toks = instr_tokens(x["instruction"])
        priv = [b for b in boxes if is_private(b[1], toks)]
        # 状态栏 / 调试叠加层（顶部 ~7% 高度）不算隐私也不拿来做假遮挡
        im = Image.open(os.path.join(IMG, x["image"])).convert("RGB")
        H = im.size[1]
        priv = [b for b in priv if min(p[1] for p in b[0]) > 0.07 * H]
        other = [b for b in boxes if b not in priv and min(p[1] for p in b[0]) > 0.07 * H
                 and sham_ok(b[1], toks, [q[1] for q in priv])]
        row = dict(idx=x["idx"], image=x["image"], category=x["category"], n_priv=len(priv),
                   priv_text=[b[1] for b in priv], usable=False)
        if priv and len(other) >= len(priv):
            # 假遮挡：为每个隐私框挑一个面积最接近、未被选过的非隐私框
            pool, sham = list(other), []
            for b in priv:
                j = min(range(len(pool)), key=lambda k: abs(area(pool[k][0]) - area(b[0])))
                sham.append(pool.pop(j))
            mi, si = im, im
            for b in priv:
                mi = fill(mi, b[0])
            for b in sham:
                si = fill(si, b[0])
            stem = os.path.splitext(x["image"])[0]
            mi.save(os.path.join(OUT, stem + "_masked.png"))
            si.save(os.path.join(OUT, stem + "_sham.png"))
            row.update(usable=True, sham_text=[b[1] for b in sham],
                       priv_area=float(sum(area(b[0]) for b in priv)), sham_area=float(sum(area(b[0]) for b in sham)))
        rows.append(row)
    json.dump(rows, open(MASK_META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    u = [r for r in rows if r["usable"]]
    print(f"可用 {len(u)}/{len(rows)}")
    cats = sorted({r["category"] for r in rows})
    for c in cats:
        print(f"  {c[:45]:45s} {sum(r['usable'] for r in rows if r['category'] == c)}/40")
    pa, sa = np.array([r["priv_area"] for r in u]), np.array([r["sham_area"] for r in u])
    print(f"遮挡面积 中位数：隐私 {np.median(pa):.0f} px²，假遮挡 {np.median(sa):.0f} px²，比值中位数 {np.median(sa / pa):.2f}")


def cmd_verify():
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    meta = [r for r in json.load(open(MASK_META, encoding="utf-8")) if r["usable"]]
    left = 0
    for r in meta:
        stem = os.path.splitext(r["image"])[0]
        res, _ = ocr(os.path.join(OUT, stem + "_masked.png"))
        txt = [norm(t) for _, t, _ in (res or [])]
        rem = [p for p in r["priv_text"] if len(norm(p)) >= 4 and any(norm(p) in t for t in txt)]
        r["residual"] = rem
        left += bool(rem)
    json.dump(meta, open(MASK_VERIFY, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"重新 OCR 后仍能读到被遮隐私文字的样本：{left}/{len(meta)}")


if __name__ == "__main__":
    {"ocr": cmd_ocr, "build": cmd_build, "verify": cmd_verify}[sys.argv[1]]()
