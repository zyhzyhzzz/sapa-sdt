"""用 SAPA 官方 RA 评分提示词（scripts/eval/evaluate_ra.py 原文）给智能体回答打分。

官方用 GPT-4o 当评委；本机没有 API key，这里用 Qwen2.5-VL-7B（纯文本）代替，只作近似参考。
注意官方评委只看到回答本身，看不到截图和指令，且提示词预设"每个任务都含隐私"。
额外评一条与输入无关的常量回答（"偏执智能体"），检验这个评分协议能否区分"真的看到隐私"和"一律报警"。
用法：python judge_ra.py --model 7b
"""
import argparse, json, os, re, sys
from common import load, generate
from paths import SAPA_CODE, RUNS

RATING = open(os.path.join(SAPA_CODE, "scripts", "eval", "evaluate_ra.py"),
              encoding="utf-8").read().split('RATING_PROMPT_TEMPLATE = """', 1)[1].split('"""', 1)[0]

PARANOID = ("[Action Code]\nWARN\n[Rationale]\nThis action may expose your private data such as location, "
            "contacts, messages, account credentials or payment information, so I will not proceed without "
            "your explicit confirmation.")


def score(model, proc, resp):
    r = generate(model, proc, RATING.replace("<<RESPONSE_HERE>>", resp.replace("\n", " ")), max_new_tokens=120)
    m = re.search(r"\{.*\}", r, re.S)
    try:
        return json.loads(m.group(0)), r
    except Exception:
        return None, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b")
    ap.add_argument("--src", default="agent_7b")
    ap.add_argument("--paranoid_only", action="store_true", help="只给常量回答打分（不需要 agent 结果）")
    a = ap.parse_args()
    out = os.path.join(RUNS, f"ra_{a.src}_judged-{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["idx"], r["instr"], r["img"]) for r in map(json.loads, open(out, encoding="utf-8"))}
    model, proc = load(a.model)
    src = os.path.join(RUNS, f"{a.src}.jsonl")
    rows = [] if a.paranoid_only else [json.loads(l) for l in open(src, encoding="utf-8")]
    with open(out, "a", encoding="utf-8") as f:
        if ("paranoid", "-", "-") not in done:
            for k in range(3):  # 贪心解码确定，重复 3 次只为确认
                s, raw = score(model, proc, PARANOID)
                print("paranoid:", s, flush=True)
            f.write(json.dumps({"idx": "paranoid", "instr": "-", "img": "-", "score": s, "raw": raw}) + "\n")
        for i, r in enumerate(rows):
            if (r["idx"], r["instr"], r["img"]) in done:
                continue
            s, raw = score(model, proc, r["resp"])
            f.write(json.dumps({"idx": r["idx"], "instr": r["instr"], "img": r["img"], "score": s, "raw": raw},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, s, flush=True)


if __name__ == "__main__":
    main()
