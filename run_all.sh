#!/bin/bash
# 一键复现。每一步都支持断点续跑（已有结果的样本会跳过），可以用 STAGES 只跑其中几步：
#   bash run_all.sh                  # 全部（单张 8GB 显卡上 GPU 步骤合计十余小时）
#   STAGES="analyze figs" bash run_all.sh   # 只用仓库自带的原始回答重算全部数字与图（几秒，无需 GPU）
# 步骤：setup data neutralize mask eval ocr judge analyze figs
# 环境变量：PY（解释器，默认 python），SAPA_DATA / SAPA_MODELS / SAPA_CODE（见 src/paths.py）
set -e
cd "$(dirname "$0")"
[ -f env.local.sh ] && . ./env.local.sh  # 本机路径与解释器（见 README）
ROOT=$(pwd)
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
STAGES=${STAGES:-"setup data neutralize mask eval ocr judge analyze figs"}
MODELS=${MODELS:-"3b 7b ivl2b ivl8b"}
FMTS="level level_explicit dfirst yn"
mkdir -p logs
has() { [[ " $STAGES " == *" $1 "* ]]; }
src() { (cd "$ROOT/src" && "$PY" "$@"); }

if has setup; then
  # SAPA 官方代码只用到 RA 评分提示词；固定到我们核对过的提交
  SAPA_CODE=${SAPA_CODE:-third_party/SAPA-Bench}
  if [ ! -d "$SAPA_CODE/.git" ]; then
    git clone https://github.com/Zhixin-L/SAPA-Bench "$SAPA_CODE"
  fi
  git -C "$SAPA_CODE" checkout -q 8cd896b4f3647487b15b90cce3d7cb9b43c337b3
  for r in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct OpenGVLab/InternVL3-2B-hf OpenGVLab/InternVL3-8B-hf; do
    src dl_model.py "$r" | tail -1
  done
fi

if has data; then
  # 先下标注再按标注抽子集（种子 0，结果应与仓库里的 artifacts/subset320.json 一致），再下子集截图
  src dl_images.py
  src make_subset.py
  git diff --quiet -- artifacts/subset320.json 2>/dev/null || echo "警告：subset320.json 与仓库版本不同"
fi

if has neutralize; then
  # 用 7B 把指令改写成中性化版本（只看文本，不看截图）
  src neutralize.py --model 7b
fi

if has mask; then
  # OCR → 构造 masked / sham 配对图 → 重新 OCR 检查隐私残留
  src mask.py ocr && src mask.py build && src mask.py verify
  src ocr_eval.py ocr
fi

if has eval; then
  for m in $MODELS; do
    # 官方格式与明确提示：指令 {原始, 中性化} × 截图 {真实, 不给图}；7B 另跑灰图与类别、智能体任务
    imgs="none,real"; [ "$m" = 7b ] && imgs="real,blank,none"
    src run_cf.py --model $m --task level --imgs $imgs >> logs/eval_$m.txt 2>&1
    src run_cf.py --model $m --task level_explicit --imgs $imgs >> logs/eval_$m.txt 2>&1
    for v in yn dfirst; do src format_probe.py --model $m --variant $v >> logs/eval_$m.txt 2>&1; done
    for f in $FMTS; do src neg_eval.py --model $m --fmt $f >> logs/eval_$m.txt 2>&1; done
    if [ "$m" = 7b ]; then
      src run_cf.py --model 7b --task category >> logs/eval_7b.txt 2>&1
      src run_cf.py --model 7b --task agent >> logs/eval_7b.txt 2>&1
    fi
    echo "eval $m done $(date)"
  done
fi

if has ocr; then
  # 感知 vs 推理：把截图换成 OCR 文字
  for m in $MODELS; do for f in $FMTS; do
    src ocr_eval.py run --model $m --fmt $f >> logs/ocr_$m.txt 2>&1
  done; done
fi

if has judge; then
  # 官方 RA 评分提示词；评委为本地 7B（官方用 GPT-4o）
  src judge_ra.py --model 7b --src agent_7b >> logs/judge.txt 2>&1
fi

if has analyze; then
  src analyze.py 7b | tee logs/analyze.txt
  src paper_numbers.py | tee logs/paper_numbers.txt
  src sdt_ci.py | tee logs/sdt_ci.txt
fi

if has figs; then
  "$PY" figs/make_fig_main.py
  [ -d artifacts/masked_images ] && "$PY" figs/make_fig_mask.py || echo "跳过 fig_mask（需要先跑 mask 步骤生成配对图）"
fi
