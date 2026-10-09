"""主图：SAPA 式检出率（横轴）vs 配对负样本判别力（纵轴）。
颜色 = 模型（固定顺序的分类色，已用 dataviz 校验脚本验证），形状 = 提问格式（第二编码，黑白打印也可区分）。
数据来自 artifacts/summary.json（src/analyze.py 生成）。
用法：python make_fig_main.py  → fig_main.pdf
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import SUMMARY  # noqa: E402
rows = json.load(open(SUMMARY, encoding="utf-8"))

MODELS = [("3b", "Qwen2.5-VL-3B", "#2a78d6"), ("7b", "Qwen2.5-VL-7B", "#eb6834"),
          ("ivl2b", "InternVL3-2B", "#1baf7a"), ("ivl8b", "InternVL3-8B", "#eda100")]
FMTS = [("level", "Official MC", "o"), ("level_explicit", "Explicit MC", "s"),
        ("dfirst", "'None' first", "^"), ("yn", "Yes / No", "D")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(3.35, 2.45))  # CVPR 单栏宽度
ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
ax.grid(True, color=GRID, lw=0.6, zorder=0)
for key, name, col in MODELS:
    for f, _, mk in FMTS:
        r = [x for x in rows if x["model"] == key and x["fmt"] == f]
        if not r:
            continue
        r = r[0]
        ax.errorbar(r["detect"], r["disc"], yerr=[[r["disc"] - r["lo"]], [r["hi"] - r["disc"]]], fmt="none",
                    ecolor=col, elinewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(r["detect"], r["disc"], s=34, marker=mk, color=col, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_xlim(-3, 103)
ax.set_ylim(-8, 30)
ax.set_xlabel("SAPA-style detection rate (%)")
ax.set_ylabel("Paired discrimination (pp)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# 图例：模型（颜色）+ 格式（形状，灰色）
hm = [plt.Line2D([], [], ls="", marker="o", ms=5, color=c, label=n) for k, n, c in MODELS if any(x["model"] == k for x in rows)]
hf = [plt.Line2D([], [], ls="", marker=m, ms=5, color=MUTED, label=n) for _, n, m in FMTS]
l1 = ax.legend(handles=hm, loc="upper left", frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)
ax.add_artist(l1)
ax.legend(handles=hf, loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(HERE, "fig_main.pdf"))
fig.savefig(os.path.join(HERE, "fig_main.png"), dpi=200)
print("saved", len(rows), "points")
