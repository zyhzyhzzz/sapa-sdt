"""SAPA 反事实实验的公共部分：模型加载、单轮生成。支持 Qwen2.5-VL 与 InternVL3（transformers 原生 -hf 版本）。"""
import os
import torch
from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration, set_seed)
from paths import MODEL_DIR, IMG_DIR, ADAPTERS

MODELS = {"3b": "Qwen2.5-VL-3B-Instruct", "7b": "Qwen2.5-VL-7B-Instruct",
          "ivl2b": "InternVL3-2B-hf", "ivl8b": "InternVL3-8B-hf"}


def load(name):
    """name 为 MODELS 的键；"3b+pair" 形式表示在 3b 上加载 artifacts/adapters/3b+pair 下的 LoRA 适配器。"""
    set_seed(0)  # 解码是贪心的；这里只为把任何潜在随机性也固定住
    base, _, tag = name.partition("+")
    if tag:
        from peft import PeftModel
        model, proc = load(base)
        model = PeftModel.from_pretrained(model, os.path.join(ADAPTERS, name))
        model.eval()
        return model, proc
    path = os.path.join(MODEL_DIR, MODELS[name])
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    # low_cpu_mem_usage：逐分片加载，避免 7B/8B 在系统内存里整份展开（9/22 那次就是因此被杀）
    kw = dict(quantization_config=q, device_map="cuda:0", low_cpu_mem_usage=True)
    if name.startswith("ivl"):
        # InternVL 的视觉编码器量化后精度损失大，只量化语言模型部分
        q.llm_int8_skip_modules = ["vision_tower", "multi_modal_projector"]
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, **kw)
        # InternVL 按 448 像素切块；手机长截图最多 6 块 + 1 张缩略图，控制显存
        proc = AutoProcessor.from_pretrained(path)
        proc.image_processor.max_patches = 6
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kw)
        # 截图 1080x2400，限制像素数以控制显存（约 1024*28*28 ≈ 80 万像素）
        proc = AutoProcessor.from_pretrained(path, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    return model, proc


@torch.no_grad()
def generate(model, proc, text, image=None, system=None, max_new_tokens=64):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": text}]
    msgs.append({"role": "user", "content": content})
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[prompt], images=[image] if image is not None else None, return_tensors="pt").to("cuda:0")
    # 只对 InternVL 转 dtype；Qwen 保持原行为，保证已有结果可逐位复现
    if "pixel_values" in inp and model.config.model_type.startswith("internvl"):
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    o = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
