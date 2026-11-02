"""Draws the chart images from numbers.json (which comes from the saved call files).

    ../../agentic-rag-eval/.venv/bin/python make_charts.py

figures/featured_chart.png     Haiku, padded policies: direct answers against hand off to Prolog (little text, for the portal)
figures/figure1_haiku.png      Haiku accuracy by depth, short and padded policies, arms A, A0, B
figures/figure2_qwen.png       Qwen3 4B accuracy by depth, short and padded policies, arms A, A0 and B (short only)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False, "font.size": 15})
NAVY, RED, GRAY, INK = "#102a43", "#d64545", "#8794a6", "#1c2430"
N = json.load(open(HERE / "numbers.json"))
D = [2, 6, 10, 14, 18, 22]
STYLE = {"A": (NAVY, "-", "o", "Model answers, reasoning allowed"), "B": (RED, "-", "s", "Hand off to Prolog"),
         "A0": (GRAY, "--", "^", "Model answers, one word only")}


def series(m, arm, pad):
    c = N["cell"][m][f"{arm}_{pad}"]
    return [c[str(d)]["mean"] for d in D], [c[str(d)]["lo"] for d in D], [c[str(d)]["hi"] for d in D]


def panel(ax, m, pad, arms, title):
    ax.axhline(50, color="#cbd5e1", lw=1.5, zorder=0)
    for arm in arms:
        col, ls, mk, lab = STYLE[arm]
        y, lo, hi = series(m, arm, pad)
        ax.fill_between(D, lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(D, y, color=col, ls=ls, marker=mk, ms=6, lw=2.4, label=lab)
    ax.set_title(title, fontsize=16, color=INK, loc="left")
    ax.set_xlabel("Rule steps from the facts to the question", color=INK)
    ax.set_xticks(D)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#e2e8f0", lw=1)
    ax.text(2, 52, "chance", color="#64748b", fontsize=12)


def fig(m, name, arms_short, arms_padded, t_short, t_padded):
    f, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    panel(axes[0], m, "short", arms_short, t_short)
    panel(axes[1], m, "padded", arms_padded, t_padded)
    axes[0].set_ylabel("Correct answers per 100 questions", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.02))
    f.tight_layout(rect=(0, 0.08, 1, 1))
    f.savefig(FIG / name, dpi=120)
    plt.close(f)


def featured():
    f, ax = plt.subplots(figsize=(16, 9))
    for arm in ("A", "B"):
        col, ls, mk, lab = STYLE[arm]
        y, lo, hi = series("haiku", arm, "padded")
        ax.fill_between(D, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(D, y, color=col, lw=6, marker=mk, ms=12)
    ya, yb = series("haiku", "A", "padded")[0], series("haiku", "B", "padded")[0]
    ax.text(21.7, ya[-1] - 6, "model answers", color=NAVY, fontsize=30, ha="right", va="top", fontweight="bold")
    ax.text(21.7, yb[-1] + 6, "hand off to Prolog", color=RED, fontsize=30, ha="right", va="bottom", fontweight="bold")
    ax.set_ylim(48, 104); ax.set_xlim(1, 23)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    f.tight_layout()
    f.savefig(FIG / "featured_chart.png", dpi=100)
    plt.close(f)


if __name__ == "__main__":
    fig("haiku", "figure1_haiku.png", ("A", "B", "A0"), ("A", "B", "A0"), "Short policies", "Policies padded with irrelevant rules")
    fig("qwen", "figure2_qwen.png", ("A", "B", "A0"), ("A", "A0"), "Short policies", "Policies padded with irrelevant rules (no Prolog arm)")
    featured()
    print("charts written to", FIG)
