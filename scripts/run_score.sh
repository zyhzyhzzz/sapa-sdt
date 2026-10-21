#!/bin/bash
# ROC / AUC：配对样本上的连续分数（neg_score.py，每张图一次前向）。原 3B 与 3b+pair 放在最后，用最终版代码重跑（第一版结果在 results/score_v1/ 作对照）。
#   nohup bash scripts/run_score.sh >> logs/score_status.txt 2>&1 &
set -e
cd "$(dirname "$0")/.."
[ -f env.local.sh ] && . ./env.local.sh
PY=${PY:-python}; export PYTHONIOENCODING=utf-8
cd src
for m in ${MODELS:-7b ivl8b ivl2b 3b+pos 3b+shuffle 3b+pair-s1 3b+pair-s2 3b 3b+pair}; do
  for f in yn level level_explicit dfirst; do
    for t in 1 2 3; do "$PY" neg_score.py --model $m --fmt $f >> ../logs/score_eval.txt 2>&1 && break; echo "retry $m $f ($t)"; done
  done
  echo "$m done $(date)"
done
echo SCORE_DONE
