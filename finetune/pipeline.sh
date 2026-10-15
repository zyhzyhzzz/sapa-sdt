#!/bin/bash
# 串行跑完微调实验（避免多个大进程同时占内存）：train 划分的 OCR → 配对图 → 残留检查 → 训练 + 评测
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}; export PYTHONIOENCODING=utf-8
(cd src && SAPA_SPLIT=train "$PY" mask.py ocr >> ../logs/train_mask.txt 2>&1 \
  && SAPA_SPLIT=train "$PY" mask.py build >> ../logs/train_mask.txt 2>&1 \
  && SAPA_SPLIT=train "$PY" mask.py verify >> ../logs/train_mask.txt 2>&1)
echo "prep done $(date)" >> logs/ft_status.txt
bash finetune/run_ft.sh >> logs/ft_status.txt 2>&1
