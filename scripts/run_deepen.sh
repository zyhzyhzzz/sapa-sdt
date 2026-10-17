#!/bin/bash
# 微调结论的稳健性：等 run_ft.sh 结束 → 遮挡方式消融（含 3b+pair 跨填充迁移）→ pair 条件再训两个种子（种子 1、2）并跑配对评测。
#   nohup bash scripts/run_deepen.sh >> logs/deepen_status.txt 2>&1 &
set -e
cd "$(dirname "$0")/.."
[ -f env.local.sh ] && . ./env.local.sh
PY=${PY:-python}; export PYTHONIOENCODING=utf-8
until grep -q "FT_ALL_DONE" logs/ft_status.txt; do sleep 60; done
echo "ft done, start ablation $(date)"  # 重启时 neg_eval 会跳过已完成样本
bash scripts/run_mask_ablation.sh
echo "ablation done $(date)"
for s in ${SEEDS:-1 2}; do
  m="3b+pair-s$s"
  [ -f "artifacts/adapters/$m/adapter_model.safetensors" ] || "$PY" finetune/train_lora.py --cond pair --model 3b --seed $s >> logs/ft_train_pair-s$s.txt 2>&1
  (cd src && for f in level level_explicit dfirst yn; do for t in 1 2 3; do "$PY" neg_eval.py --model $m --fmt $f >> ../logs/ft_eval_pair-s$s.txt 2>&1 && break; done; done)
  echo "seed $s done $(date)"
done
"$PY" src/analyze.py 7b > logs/analyze.txt 2>&1
echo DEEPEN_DONE
