#!/bin/bash
# 智能体模式的配对评测：执行任务时会不会主动提醒隐私（官方智能体提示词 / 自拟提示版），4 个模型含 3b+pair（检验微调能否迁移到智能体模式）。
# 等 run_sidefx.sh 结束（SIDEFX_DONE）后再跑，避免两个大进程同时占显存。
#   nohup bash scripts/run_agent.sh >> logs/agent_status.txt 2>&1 &
set -e
cd "$(dirname "$0")/.."
[ -f env.local.sh ] && . ./env.local.sh
PY=${PY:-python}; export PYTHONIOENCODING=utf-8
until grep -q "SIDEFX_DONE" logs/sidefx_status.txt 2>/dev/null; do sleep 60; done
cd src
for m in ${MODELS:-3b 3b+pair 7b ivl8b}; do
  for f in agent agent_hint; do
    for t in 1 2 3; do "$PY" neg_eval.py --model $m --fmt $f >> ../logs/agent_eval.txt 2>&1 && break; echo "retry $m $f ($t)"; done
  done
  echo "$m done $(date)"
done
"$PY" analyze.py 7b > ../logs/analyze.txt 2>&1
echo AGENT_DONE
