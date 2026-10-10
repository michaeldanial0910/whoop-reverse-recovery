"""M3 figures from the AGGREGATE results only (no dates, no per-cycle values).

    python src/m3_figures.py      (after m3_anomalies.py, m3_irregularity.py, m3_archetypes.py)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m2_figures import COLORS, GRID, INK, INK_2, SURFACE  # same style as the M2 figures
import m3_common as mc

BLUE, ORANGE = COLORS["formula"], COLORS["xgboost"]


def style(ax, title=None):
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.axhline(0, color=INK_2, linewidth=1)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)


def bin_label(b, first, last):
    """'(4.0, 5.0]' -> '4 to 5'; the outer bins are open-ended -> '< 4' / '> 9'."""
    lo, hi = (float(v) for v in b.strip("(]").split(", "))
    fmt = lambda v: f"{v:g}".replace("-", "\u2212")
    if first:
        return f"< {fmt(hi)}"
    if last and hi in (24, 1000):
        return f"> {fmt(lo)}"
    return f"{fmt(lo)}\nto {fmt(hi)}"


def residual_by_input():
    d = pd.read_csv(mc.RESULTS_DIR / "m3_anomaly_residual_by_input.csv")
    s = pd.read_csv(mc.RESULTS_DIR / "m3_anomaly_summary.csv", index_col=0)["value"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    panels = [("total_sleep_hours", "By hours slept", "hours slept"),
              ("hrv_pct_vs_baseline", "By HRV vs own 30-day baseline", "HRV, % above/below baseline")]
    for ax, (col, title, xl) in zip(axes, panels):
        g = d[d.input == col].reset_index(drop=True)
        labels = [bin_label(b, i == 0, i == len(g) - 1) for i, b in enumerate(g["bin"])]
        ax.bar(range(len(g)), g["mean_residual"], width=0.7, color=BLUE)
        for i, r in g.iterrows():
            y = r.mean_residual
            ax.text(i, y + (0.6 if y >= 0 else -0.6), f"n={int(r.n)}\n{r.share_flagged:.0%} flagged",
                    ha="center", va="bottom" if y >= 0 else "top", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(g)), labels)
        ax.set_xlabel(xl)
        style(ax, title)
    axes[0].set_ylabel("mean residual, points (actual - predicted)")
    axes[0].set_ylim(-17, 9)
    fig.suptitle(f"Where the reconstructed formula is wrong: {int(s.n_flagged)} of {int(s.n_cycles)} cycles "
                 f"flagged, {int(s.n_worse_than_expected)} worse than expected",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(mc.FIG_DIR / "m3_residual_by_input.png", dpi=160)
    plt.close(fig)


def irregularity():
    r = pd.read_csv(mc.RESULTS_DIR / "m3_irregularity_results.csv")
    names = {"onset_sd7_hours": "Onset variability\n(7-night SD, per +1 h)\nPRIMARY",
             "onset_shift_abs_hours": "Shift from last night\n(per +1 h, either way)",
             "onset_hours": "Onset clock time\n(per 1 h later)"}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k, (model, color, lab) in enumerate([("total", BLUE, "Total effect (HRV/RHR not controlled)"),
                                             ("direct", ORANGE, "Direct effect (HRV/RHR controlled)")]):
        g = r[r.model == model].set_index("term").loc[list(names)]
        x = np.arange(len(g)) + (k - 0.5) * 0.25
        ax.errorbar(x, g.coef_points_per_unit, yerr=[g.coef_points_per_unit - g.ci95_lo, g.ci95_hi - g.coef_points_per_unit],
                    fmt="o", color=color, ms=7, elinewidth=2, capsize=0, label=lab)
    ax.axhline(-2, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(2.45, -2.15, "pass line for the primary: -2", ha="right", va="top", fontsize=8, color=INK_2)
    ax.set_xticks(range(3), list(names.values()))
    ax.set_ylabel("recovery points (95% block-bootstrap CI)")
    style(ax, "Sleep-schedule irregularity: pre-registered test FAILED (no detectable effect)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(mc.FIG_DIR / "m3_irregularity.png", dpi=160)
    plt.close(fig)


def archetypes():
    a = pd.read_csv(mc.RESULTS_DIR / "m3_archetypes.csv")
    labels = [f"{l}\n({n} days)" for l, n in zip(a.label.str.replace(", ", "\n"), a.n_days)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (col, title, yl) in zip(axes, [
            ("next_recovery", "Next-day recovery", "next cycle's recovery (mean, 95% CI)"),
            ("next_residual", "Next-day residual: what the model's inputs don't explain",
             "actual - predicted, points (mean, 95% CI)")]):
        m, lo, hi = a[f"{col}_mean"], a[f"{col}_ci_lo"], a[f"{col}_ci_hi"]
        ax.errorbar(range(len(a)), m, yerr=[m - lo, hi - m], fmt="o", color=BLUE, ms=8, elinewidth=2, capsize=0)
        for i, v in enumerate(m):
            ax.text(i + 0.12, v, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(a)), labels, fontsize=8)
        ax.set_ylabel(yl)
        style(ax, title)
    axes[0].set_ylim(40, 75)
    axes[1].set_ylim(-8, 4)
    fig.suptitle("Day archetypes: busy days are followed by lower recovery; the model's inputs account for the gap",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(mc.FIG_DIR / "m3_archetypes.png", dpi=160)
    plt.close(fig)


def main():
    mc.FIG_DIR.mkdir(parents=True, exist_ok=True)
    residual_by_input()
    irregularity()
    archetypes()
    print("Saved figures to", mc.FIG_DIR)


if __name__ == "__main__":
    main()
