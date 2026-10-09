"""全部路径集中在这里。大文件（数据集、模型权重）默认放在仓库内的 data/ 与 models/，
也可以用环境变量指到别处，避免重复下载：
  SAPA_DATA    SAPA-Bench 数据目录（含 all_annot.json 与 images/）
  SAPA_MODELS  模型权重目录（含 Qwen2.5-VL-3B-Instruct 等子目录）
  SAPA_CODE    SAPA-Bench 官方代码（只用到 RA 评分提示词）
  SAPA_ART / SAPA_RUNS  派生产物与原始回答的目录（默认仓库内；复现核对时指到临时目录，避免覆盖）
  SAPA_SPLIT   test（默认，论文评测用的 320 个场景）或 train（微调用，见 finetune/）。
               train 的全部派生产物放在 artifacts/train/，文件名与 test 相同，所以 neutralize / mask 等脚本可直接复用。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SAPA_DATA", os.path.join(ROOT, "data", "SAPA-data"))
IMG_DIR = os.path.join(DATA, "images")
ANNOT = os.path.join(DATA, "all_annot.json")
MODEL_DIR = os.environ.get("SAPA_MODELS", os.path.join(ROOT, "models"))
SAPA_CODE = os.environ.get("SAPA_CODE", os.path.join(ROOT, "third_party", "SAPA-Bench"))

# 实验派生产物（子集、中性化指令、OCR 缓存、遮挡元数据）与原始回答
SPLIT = os.environ.get("SAPA_SPLIT", "test")
ART = os.environ.get("SAPA_ART", os.path.join(ROOT, "artifacts"))
if SPLIT == "train":
    ART = os.path.join(ART, "train")
ADAPTERS = os.path.join(ROOT, "artifacts", "adapters")
MASKED = os.path.join(ART, "masked_images")
RUNS = os.environ.get("SAPA_RUNS", os.path.join(ROOT, "results", "runs"))

SUBSET = os.path.join(ART, "subset320.json")
NEUTRAL = os.path.join(ART, "neutral_7b.jsonl")
OCR = os.path.join(ART, "ocr.json")
MASK_META = os.path.join(ART, "mask_meta.json")
MASK_VERIFY = os.path.join(ART, "mask_verify.json")
OCR_PAIRS = os.path.join(ART, "ocr_pairs.json")
SUMMARY = os.path.join(ART, "summary.json")
