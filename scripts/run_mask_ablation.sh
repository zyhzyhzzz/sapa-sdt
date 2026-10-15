#!/bin/bash
# 遮挡方式消融：同样的遮挡位置，填充改为高斯模糊 / 纯黑块，重测 4 个模型 × 4 种格式的配对判别力。
# 依赖 mask.py build/verify 已在 SAPA_FILL=blur|black 下跑过（产物带后缀，不覆盖主实验）。
# WAIT_FT=1 时先等微调流水线（finetune/run_ft.sh）结束，避免两个大进程同时占显存。
set -e
cd "$(dirname "$0")/.."
[ -f env.local.sh ] && . ./env.local.sh
PY=${PY:-python}; export PYTHONIOENCODING=utf-8
if [ "${WAIT_FT:-0}" = 1 ]; then
  until grep -q "FT_ALL_DONE" logs/ft_status.txt; do sleep 60; done
fi
cd src
for fill in blur black; do
  for m in 3b 7b ivl2b ivl8b; do
    for f in level level_explicit dfirst yn; do
      SAPA_FILL=$fill "$PY" neg_eval.py --model $m --fmt $f >> ../logs/mask_ablation_eval.txt 2>&1
    done
  done
  echo "$fill done $(date)" >> ../logs/mask_ablation_eval.txt
done
"$PY" analyze.py 7b > ../logs/analyze.txt 2>&1
echo MASK_ABLATION_DONE >> ../logs/mask_ablation_eval.txt
