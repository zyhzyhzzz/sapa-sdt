#!/bin/bash
# 微调副作用检查：配对微调是否损害了 SAPA 原有的隐私类别识别（9 选 1，原始指令 + 真实截图）。
# 等 run_deepen.sh 结束（DEEPEN_DONE）后再跑，避免两个大进程同时占显存。
#   nohup bash scripts/run_sidefx.sh >> logs/sidefx_status.txt 2>&1 &
set -e
cd "$(dirname "$0")/.."
[ -f env.local.sh ] && . ./env.local.sh
PY=${PY:-python}; export PYTHONIOENCODING=utf-8
until grep -q "DEEPEN_DONE" logs/deepen_status.txt 2>/dev/null; do sleep 60; done
cd src
for m in 3b 3b+pair 3b+pos 3b+shuffle; do
  for t in 1 2 3; do "$PY" run_cf.py --model $m --task category --imgs real >> ../logs/sidefx_eval.txt 2>&1 && break; echo "retry $m ($t)"; done
  echo "$m done $(date)"
done
"$PY" analyze.py 7b > ../logs/analyze.txt 2>&1
echo SIDEFX_DONE
