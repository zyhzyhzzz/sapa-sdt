#!/bin/bash
# 微调实验全流程：三种训练条件 × 训练 + 在评测集上跑与论文主表完全相同的评测
#   CONDS="pair" bash finetune/run_ft.sh     # 只跑一个条件
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY=${PY:-python}; export PYTHONIOENCODING=utf-8
CONDS=${CONDS:-"pair pos shuffle"}; BASE=${BASE:-3b}
mkdir -p logs
for c in $CONDS; do
  m="$BASE+$c"
  [ -f "artifacts/adapters/$m/adapter_model.safetensors" ] || "$PY" finetune/train_lora.py --cond $c --model $BASE > logs/ft_train_$c.txt 2>&1
  cd src
  # SAPA 式识别率（原始指令 + 原图）：两种多选格式只需真实截图；yn / dfirst 由 format_probe 跑（含不给图）
  "$PY" run_cf.py --model $m --task level --imgs real >> ../logs/ft_eval_$c.txt 2>&1
  "$PY" run_cf.py --model $m --task level_explicit --imgs real >> ../logs/ft_eval_$c.txt 2>&1
  for v in yn dfirst; do "$PY" format_probe.py --model $m --variant $v >> ../logs/ft_eval_$c.txt 2>&1; done
  for f in level level_explicit dfirst yn; do "$PY" neg_eval.py --model $m --fmt $f >> ../logs/ft_eval_$c.txt 2>&1; done
  cd "$ROOT"
  echo "$c done $(date)"
done
"$PY" src/analyze.py 7b > logs/analyze.txt 2>&1
echo FT_ALL_DONE
