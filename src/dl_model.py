"""下载 HF 模型仓库全部文件到 models/<名称>，按元数据校验大小，支持续传（跳过已完整的文件）。
用法：python dl_model.py OpenGVLab/InternVL3-2B-hf
版本固定到 REVS 中的提交（2026-10-08 核对）。"""
import os, sys, requests
from huggingface_hub import HfApi
from paths import MODEL_DIR

REVS = {"Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "OpenGVLab/InternVL3-2B-hf": "cb57a075cb75a2e6d1b668b128d48bb00ae321d2",
        "OpenGVLab/InternVL3-8B-hf": "259a3b64a14623c0ec91a045cb43f7c5af5fa6af"}
repo = sys.argv[1]
rev = REVS.get(repo, "main")
ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
out = os.path.join(MODEL_DIR, repo.split("/")[1])
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
for s in info.siblings:
    p = os.path.join(out, s.rfilename)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and os.path.getsize(p) == s.size:
        continue
    tmp = p + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    h = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(f"{ep}/{repo}/resolve/{rev}/{s.rfilename}", headers=h, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "ab" if have and r.status_code == 206 else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if os.path.getsize(tmp) != s.size:
        raise RuntimeError(f"size mismatch: {s.rfilename}")
    os.replace(tmp, p)
    print("ok", s.rfilename, s.size, flush=True)
print("DONE", repo, flush=True)
