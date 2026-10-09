"""QLoRA 微调：检验“用配对数据训练”能否真正提高辨别力 d′，还是只移动判定阈值 c。

训练数据全部来自 train 划分（与评测的 320 个场景按 episode、近重复指令、近重复截图三重隔离）。
提问格式固定为是非题（yn），评测时再看四种格式，检验学到的东西能否跨格式迁移。三种训练条件：
  pair     配对监督：中性化指令 + sham 图 → Yes；中性化指令 + masked 图 → No
  pos      只有正样本（模拟 SAPA 式数据）：原始指令 + 原图 → Yes；中性化指令 + sham 图 → Yes
  shuffle  与 pair 同样的输入，标签随机打乱（Yes/No 各半），检验“只是见过这些图”的效应
三种条件的样本数相同、输入图片同分布，只差标签。
用法：python finetune/train_lora.py --cond pair [--model 3b --epochs 2]
产出：artifacts/adapters/<model>+<cond>/（LoRA 权重 + 训练日志）
"""
import argparse, json, math, os, random, sys, time
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ["SAPA_SPLIT"] = "train"  # 必须在 import paths 之前
from paths import ADAPTERS, IMG_DIR, MASKED, MASK_VERIFY, NEUTRAL, SUBSET  # noqa: E402
from common import load  # noqa: E402
from format_probe import PROMPTS  # noqa: E402


def examples(cond, seed=0):
    sub = {x["idx"]: x for x in json.load(open(SUBSET, encoding="utf-8"))}
    neu = {r["idx"]: r["neutral"] for r in map(json.loads, open(NEUTRAL, encoding="utf-8"))}
    pairs = [r for r in json.load(open(MASK_VERIFY, encoding="utf-8")) if not r["residual"] and r["idx"] in sub]
    ex = []
    for r in pairs:
        stem = os.path.splitext(r["image"])[0]
        sham, masked = os.path.join(MASKED, stem + "_sham.png"), os.path.join(MASKED, stem + "_masked.png")
        if cond == "pos":
            ex += [(sub[r["idx"]]["instruction"], os.path.join(IMG_DIR, r["image"]), "Yes"),
                   (neu[r["idx"]], sham, "Yes")]
        else:
            ex += [(neu[r["idx"]], sham, "Yes"), (neu[r["idx"]], masked, "No")]
    rng = random.Random(seed)
    if cond == "shuffle":
        labels = [e[2] for e in ex]
        rng.shuffle(labels)
        ex = [(a, b, y) for (a, b, _), y in zip(ex, labels)]
    return ex, len(pairs)


def encode(proc, instr, img_path, answer):
    im = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"{PROMPTS['yn']}\nInstruction:\n{instr}"}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    full = proc(text=[prompt + answer + "<|im_end|>"], images=[im], return_tensors="pt")
    n_prompt = proc(text=[prompt], images=[im], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100  # 只在回答（Yes/No + 结束符）上算损失
    full["labels"] = labels
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["pair", "pos", "shuffle"], required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model

    out = os.path.join(ADAPTERS, f"{a.model}+{a.cond}" + (f"-s{a.seed}" if a.seed else ""))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    ex, n_pairs = examples(a.cond, a.seed)
    print(f"[{a.cond}] 训练对 {n_pairs}，样本 {len(ex)}（Yes {sum(e[2] == 'Yes' for e in ex)}）", flush=True)

    model, proc = load(a.model)
    # 不用 prepare_model_for_kbit_training：它会把未量化的视觉编码器转成 fp32，8GB 显存放不下
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    # 只在语言模型上加 LoRA，视觉编码器不动
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(ex) * a.epochs / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(  # 5% 线性预热 + 余弦衰减
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps))) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    rng = random.Random(a.seed)
    log, t0, k, run = [], time.time(), 0, 0.0
    for ep in range(a.epochs):
        order = list(range(len(ex)))
        rng.shuffle(order)
        for i in order:
            batch = encode(proc, *ex[i]).to("cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / a.accum
            loss.backward()
            run += loss.item()
            k += 1
            if k % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step = k // a.accum
                log.append(dict(step=step, epoch=ep, loss=run, lr=sched.get_last_lr()[0]))
                if step % 10 == 0 or step == steps:
                    print(f"step {step}/{steps} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                run = 0.0
    model.save_pretrained(out)
    json.dump(dict(args=vars(a), n_pairs=n_pairs, n_examples=len(ex), log=log, seconds=time.time() - t0),
              open(os.path.join(out, "train_log.json"), "w", encoding="utf-8"), indent=1)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
