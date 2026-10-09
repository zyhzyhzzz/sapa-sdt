#!/bin/bash
# 复现抽查：把仓库里的若干结果文件截掉末尾几十行，放到临时目录，用当前代码重新生成被截掉的部分，
# 再与仓库原版逐行比较。覆盖 Qwen 3B/7B、InternVL3-8B、读 OCR 文字条件与指令中性化。
# 用法：PY=... SAPA_DATA=... SAPA_MODELS=... bash scripts/verify_spot.sh
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY=${PY:-python}; export PYTHONIOENCODING=utf-8
V=${V:-$(mktemp -d)}
mkdir -p "$V/runs"; cp -r artifacts "$V/art"
cut() { head -n -$2 "results/runs/$1.jsonl" > "$V/runs/$1.jsonl"; }
cut neg_yn_3b 40; cut level_7b 24; cut neg_dfirst_ivl8b 40; cut ocrneg_yn_7b 40
head -n -5 artifacts/neutral_7b.jsonl > "$V/art/neutral_7b.jsonl"
export SAPA_ART="$V/art" SAPA_RUNS="$V/runs"
cd src
"$PY" neutralize.py --model 7b
"$PY" run_cf.py --model 7b --task level --imgs real,blank,none
"$PY" neg_eval.py --model 3b --fmt yn
"$PY" neg_eval.py --model ivl8b --fmt dfirst
"$PY" ocr_eval.py run --model 7b --fmt yn
cd "$ROOT"
"$PY" - "$V" <<'PY'
import json, sys
V = sys.argv[1]
def rows(p, key):
    return {tuple(r[k] for k in key): r for r in map(json.loads, open(p, encoding="utf-8"))}
bad = 0
checks = [("art/neutral_7b.jsonl", "artifacts/neutral_7b.jsonl", ("idx",), "neutral")]
for n in ("level_7b",):
    checks.append((f"runs/{n}.jsonl", f"results/runs/{n}.jsonl", ("idx", "instr", "img"), "resp"))
for n in ("neg_yn_3b", "neg_dfirst_ivl8b", "ocrneg_yn_7b"):
    checks.append((f"runs/{n}.jsonl", f"results/runs/{n}.jsonl", ("idx", "img"), "resp"))
for new, old, key, field in checks:
    a, b = rows(f"{V}/{new}", key), rows(old, key)
    diff = [k for k in b if k not in a or a[k][field] != b[k][field]]
    bad += len(diff)
    print(f"{old:42s} {len(b) - len(diff)}/{len(b)} 一致", ("不一致示例: " + str(diff[:3])) if diff else "")
print("全部一致" if bad == 0 else f"共 {bad} 处不一致")
PY
