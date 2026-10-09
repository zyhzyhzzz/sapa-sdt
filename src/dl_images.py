"""下载 SAPA 标注（all_annot.json）与子集截图；版本固定到 REV，按 HF 元数据校验文件大小，不符则重下。
用法：python dl_images.py [子集 JSON，默认 artifacts/subset320.json]"""
import json, os, sys, time, requests, concurrent.futures as cf
from huggingface_hub import HfApi
from paths import ANNOT, IMG_DIR, SUBSET

ep = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
REV = '3bde8ec32ef889ab2d254ee3d7a8796a4d4504c2'  # OmniQuest/SAPA-Bench，2026-10-08 核对
out = IMG_DIR; os.makedirs(out, exist_ok=True)
names = [x['image'] for x in json.load(open(sys.argv[1] if len(sys.argv) > 1 else SUBSET, encoding='utf-8'))]
info = HfApi().dataset_info('OmniQuest/SAPA-Bench', revision=REV, files_metadata=True)
size = {s.rfilename.split('/', 1)[1]: s.size for s in info.siblings if s.rfilename.startswith('images/')}
annot_size = [s.size for s in info.siblings if s.rfilename == 'all_annot.json'][0]


def get(n):
    p = f'{out}/{n}'
    if os.path.exists(p) and os.path.getsize(p) == size[n]:
        return 0
    for k in range(5):
        try:
            r = requests.get(f'{ep}/datasets/OmniQuest/SAPA-Bench/resolve/{REV}/images/{n}', timeout=60)
            r.raise_for_status()
        except requests.RequestException:  # 偶发 SSL/连接中断，退避后重试
            time.sleep(2 ** k)
            continue
        if len(r.content) == size[n]:
            open(p, 'wb').write(r.content)
            return 1
    raise RuntimeError(f'size mismatch after retries: {n}')


if not (os.path.exists(ANNOT) and os.path.getsize(ANNOT) == annot_size):
    r = requests.get(f'{ep}/datasets/OmniQuest/SAPA-Bench/resolve/{REV}/all_annot.json', timeout=120)
    r.raise_for_status()
    open(ANNOT, 'wb').write(r.content)
    print('all_annot.json downloaded')
with cf.ThreadPoolExecutor(8) as ex:
    print(sum(ex.map(get, names)), 'downloaded')
