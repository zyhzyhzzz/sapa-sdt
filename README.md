# 第三只眼睁开了吗？——用信号检测论重新解读 SAPA-Bench 的隐私意识分数

本仓库是课程论文的全部代码、派生数据与原始模型回答。

**主要发现**：[SAPA-Bench](https://huggingface.co/datasets/OmniQuest/SAPA-Bench)（AAAI 2026）只有正样本，模型“总回答有隐私”就能拿高分。我们用遮挡造了 196 对配对样本：masked 抹去目标隐私，sham 遮掉等量非隐私文字作对照。在 4 个模型 × 4 种提问格式共 16 个组合上：

- SAPA 式识别率随提问格式在 0%–97% 之间变化，排名会翻转；
- 配对辨别力 d′ ≤ 0.39，平衡准确率 ≤ 57.4%；
- 识别率与判定阈值 c 的相关系数为 −0.89，与辨别力 d′ 只有 0.22；
- 一条与输入无关的固定警告回答，按官方 RA 评分提示词能得 90/100 分。

## 目录

```
run_all.sh            一键复现入口（分步骤、可断点续跑）
src/
  paths.py            全部路径；大文件位置可用环境变量改
  common.py           模型加载（4-bit NF4）与贪心生成
  make_subset.py      从 SAPA 分层抽取 320 个场景（8 类 × 40，种子 0）
  neutralize.py       用 7B 把指令改写成不含隐私线索的中性化版本
  mask.py             OCR → 构造 masked / sham 配对图 → 重新 OCR 检查残留
  run_cf.py           指令 × 截图反事实实验（官方级别题、明确提示、类别题、智能体）
  format_probe.py     提问格式检验（是非题、“无隐私”放首位）
  neg_eval.py         配对样本评测
  ocr_eval.py         感知 vs 推理：把截图换成 OCR 文字
  judge_ra.py         官方 RA 评分提示词 + 本地评委
  analyze.py          全部统计表（bootstrap CI、McNemar、d′、c）
  paper_numbers.py    论文正文中不在表格里的数字
  dl_images.py / dl_model.py   下载数据与模型（版本固定）
scripts/verify_spot.sh   复现抽查：截掉部分结果后重新生成并逐行比对
figs/                 论文插图脚本
artifacts/            派生数据：子集、中性化指令、OCR 缓存、遮挡元数据、summary.json
results/runs/         全部原始模型回答（jsonl，51 个文件）
```

## 快速复现：不需要 GPU，几秒钟

仓库自带全部原始回答，论文里的所有数字和主图都能直接重算：

```bash
pip install numpy scipy scikit-learn matplotlib pillow
STAGES="analyze figs" bash run_all.sh
```

`analyze.py` 的中性化质量检查要读 SAPA 标注文件（3.9 MB）。请先运行 `python src/dl_images.py` 下载，或用 `SAPA_DATA` 指向已有目录。

## 完整复现：从数据下载到出图

```bash
pip install -r requirements.txt      # torch 请按 CUDA 版本单独安装
bash run_all.sh                      # 全部步骤
STAGES="eval" MODELS="7b" bash run_all.sh   # 只跑某一步、某个模型
```

| 步骤 | 内容 | 需要 |
|---|---|---|
| `setup` | 克隆 SAPA 官方代码（固定提交 `8cd896b`），下载 4 个模型（固定 HF 版本） | 约 42 GB 磁盘 |
| `data` | 下载标注与 320 张截图（固定数据集版本 `3bde8ec`），重新抽子集 | |
| `neutralize` | 生成中性化指令 | GPU |
| `mask` | OCR、构造配对图、残留检查 | CPU |
| `eval` | 4 个模型的全部评测 | GPU，耗时最长 |
| `ocr` | 读屏文字诊断 | GPU |
| `judge` | RA 评分 | GPU |
| `analyze` / `figs` | 统计与出图 | CPU |

在单张 RTX 4070 Laptop（8 GB）上，GPU 步骤合计十余小时。所有模型都以 4-bit NF4 量化加载。7B/8B 模型要在系统内存里逐分片加载，建议内存 ≥ 16 GB。

如果数据或模型已经在别处，可以用环境变量指过去，不必重复下载：

```bash
export SAPA_DATA=/path/to/SAPA-data SAPA_MODELS=/path/to/models SAPA_CODE=/path/to/SAPA-Bench
```

## 论文结果与命令对照

| 论文位置 | 内容 | 来源 |
|---|---|---|
| 图 1 | 识别率 vs 判别力 | `figs/make_fig_main.py`（读 `artifacts/summary.json`） |
| 图 2 | 原图 / sham / masked 示例 | `figs/make_fig_mask.py` |
| 表 1 | 格式敏感性与配对辨别力 | `analyze.py` 的“总表”和“配对负样本 · 看截图”；“原始/无图”列见“格式检验”与各任务表 |
| 表 2 | 模态消融（7B） | `analyze.py` 的 `level（n=320）` |
| 表 3 | RA 评委探针 | `analyze.py` 的 RA 段 |
| 表 4 | 感知 vs 推理 | `paper_numbers.py` 的 `[5.5]` 行，明细见 `analyze.py` 的“读 OCR 文字” |
| §5.1–5.5 正文数字 | 首选项计数、相关系数、类别题翻转、RA 分项、OCR 保留率 | `paper_numbers.py`（每行标有章节号） |
| §4 中性化质量 | 只看指令的分类器准确率 70.3% → 30.3% | `analyze.py` 的“中性化质量检查” |

## 可复现性

- **确定性**：全部生成都用贪心解码（`do_sample=False`），加载模型时另设 `set_seed(0)`。子集抽样、假遮挡选择和 bootstrap 都固定种子 0。
- **版本固定**：数据集、4 个模型和 SAPA 官方代码都固定到具体提交，见 `src/dl_images.py`、`src/dl_model.py`、`run_all.sh`。依赖版本见 `requirements.txt`。
- **已核对**：
  - 用本仓库代码从 `ocr.json` 重建 410 张配对图，与论文所用版本逐字节相同。
  - 重算的 `summary.json` 与原版逐字节相同。
  - `scripts/verify_spot.sh` 抽查 Qwen-3B/7B、InternVL3-8B、读屏文字和中性化指令，重新生成的回答与仓库原版逐条比对。
- **已知限制**：4-bit 量化的计算核依赖 GPU 架构与 bitsandbytes 版本，换显卡后个别回答可能不同，但不影响统计结论。RA 的评委是本地 Qwen2.5-VL-7B，不是官方用的 GPT-4o。

## 数据与许可

SAPA-Bench 的截图和标注版权归原作者所有，本仓库不分发截图。`artifacts/` 中的子集文件只包含样本索引、类别与指令文本，用于定位原数据。配对图可由 `mask` 步骤在本地重建。
