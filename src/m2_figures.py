"""M2 figures from the AGGREGATE results only (no dates, no per-cycle values).

    python src/m2_figures.py      (after attribution.py)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"

COLORS = {"formula": "#2a78d6", "xgboost": "#eb6834"}            # categorical slots 1-2
LABELS = {"formula": "Formula (logit ridge)", "xgboost": "XGBoost"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def paired_bars(df, value, title, xlabel, fname, lo=None, hi=None, pct=True):
    order = (df[df.model == "formula"].sort_values(value)["group"]).tolist()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    h = 0.38
    for k, model in enumerate(["formula", "xgboost"]):
        d = df[df.model == model].set_index("group").loc[order]
        y = [i + (k - 0.5) * (h + 0.04) for i in range(len(order))]
        scale = 100 if pct else 1
        ax.barh(y, d[value] * scale, height=h, color=COLORS[model], label=LABELS[model])
        if lo and hi and lo in d:
            ax.errorbar(d[value] * scale, y, xerr=[(d[value] - d[lo]).clip(lower=0) * scale,
                                                   (d[hi] - d[value]).clip(lower=0) * scale],
                        fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=160)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RESULTS / "m2_attribution_summary.csv")
    paired_bars(s, "share_of_attribution", "What moves my recovery score (share of attribution)",
                "% of mean |contribution| (whiskers: 5-95%, 25 refits on 44-89% of training data)",
                "m2_attribution_share.png", "share_of_attribution_p5", "share_of_attribution_p95")
    paired_bars(s, "share_of_daily_moves", "What drives day-to-day changes in predicted recovery",
                "average % of each day-over-day move (whiskers: 5-95%, 25 refits on 44-89% of training data)",
                "m2_daily_move_share.png", "share_of_daily_moves_p5", "share_of_daily_moves_p95")

    inter = pd.read_csv(RESULTS / "m2_interaction.csv")
    inter = inter[inter.hrv_tercile != "all"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    terc = ["low HRV", "mid HRV", "high HRV"]
    for k, model in enumerate(["formula", "xgboost"]):
        d = inter[inter.model == model].set_index("hrv_tercile").loc[terc]
        x = [i + (k - 0.5) * 0.4 for i in range(3)]
        ax.bar(x, d["sleep_points_per_hour"], width=0.36, color=COLORS[model], label=LABELS[model])
    ax.set_xticks(range(3), [f"{t}\n(vs own baseline)" for t in terc])
    ax.set_ylabel("recovery points per extra hour of sleep")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Does sleep matter less when HRV is high?", loc="left", color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m2_hrv_sleep_interaction.png", dpi=160)
    plt.close(fig)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
