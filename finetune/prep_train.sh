#!/bin/bash
# 准备微调训练数据（train 划分）：中性化指令（GPU）与 OCR → 配对图 → 残留检查（CPU）并行
set -e
cd "$(dirname "$0")/../src"
PY=${PY:-python}; export PYTHONIOENCODING=utf-8 SAPA_SPLIT=train
"$PY" neutralize.py --model 7b > ../logs/train_neutralize.txt 2>&1 &
G=$!
"$PY" mask.py ocr > ../logs/train_mask.txt 2>&1 && "$PY" mask.py build >> ../logs/train_mask.txt 2>&1 && "$PY" mask.py verify >> ../logs/train_mask.txt 2>&1
wait $G
echo PREP_DONE
