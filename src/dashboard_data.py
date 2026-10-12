"""Data layer for the local dashboard (app.py). Pure Python, no Streamlit imports.

Everything here reuses the M2/M3 code paths so the dashboard can never drift from the
published numbers:
  - the formula model = logit ridge on Set B, fitted on the TRAINING period only (frozen)
  - attribution = the same exact grouped Shapley explainer and the same fixed 100-cycle
    background as src/attribution.py, so the reference B is identical (~60.7 points)
  - archetypes / anomalies / irregularity are read from the M3 output files
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import attribution as at
import m3_common as mc
import train_baselines as tb

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
COLS = at.COLS
GROUPS = list(at.GROUPS)

REQUIRED = {
    "features": PROC / "features.csv",
    "oof residuals": PROC / "m3_oof_residuals.csv",
    "archetype days": PROC / "m3_archetype_days.csv",
    "attribution summary": RES / "m2_attribution_summary.csv",
    "archetypes": RES / "m3_archetypes.csv",
    "irregularity": RES / "m3_irregularity_results.csv",
    "anomaly bins": RES / "m3_anomaly_residual_by_input.csv",
}
RUN_ORDER = "python run.py --skip-sync"

# where the formula is known to be miscalibrated: input ranges whose average miss (from the M3
# anomaly bins of THIS person's data) is at least this many points, over at least this many mornings
GUARDRAIL_MIN_MISS, GUARDRAIL_MIN_N = 4.0, 8
INPUT_LABELS = {"total_sleep_hours": ("sleep", "h"), "hrv_pct_vs_baseline": ("HRV vs baseline", "%")}


def _parse_bin(label):
    lo, hi = label.strip("(]").split(",")
    return float(lo), float(hi)


def miscalibrated_bins(bins=None):
    b = read("anomaly bins") if bins is None else bins
    b = b[(b["n"] >= GUARDRAIL_MIN_N) & (b["mean_residual"].abs() >= GUARDRAIL_MIN_MISS)].copy()
    return b.reindex(b["mean_residual"].abs().sort_values(ascending=False).index)


def _bin_phrase(r):
    name, unit = INPUT_LABELS.get(r["input"], (r["input"], ""))
    lo, hi = _parse_bin(r["bin"])
    return f"{name} {lo:+g} to {hi:+g}{unit}" if unit == "%" else f"{name} {lo:g} to {hi:g} {unit}".strip()


def anomaly_pattern_text(bins):
    m = miscalibrated_bins(bins)
    if m.empty:
        return "The surprises show no clear pattern by sleep length or HRV: the formula's misses are spread evenly."
    parts = [f"{_bin_phrase(r)}: WHOOP scores you about {abs(r['mean_residual']):.0f} points "
             f"{'lower' if r['mean_residual'] < 0 else 'higher'} than the formula ({int(r['n'])} mornings)"
             for _, r in m.head(3).iterrows()]
    return ("The surprises aren't random. " + "; ".join(parts) + ". A smooth formula can't bend like that, "
            "which suggests WHOOP caps or steepens credit at the extremes.")


def guardrail_warnings(row):
    out = []
    for _, r in miscalibrated_bins().iterrows():
        v = row.get(r["input"])
        lo, hi = _parse_bin(r["bin"])
        if pd.notna(v) and lo < v <= hi:
            out.append(f"In this range ({_bin_phrase(r)}) the formula has {'over' if r['mean_residual'] < 0 else 'under'}"
                       f"-predicted by ~{abs(r['mean_residual']):.0f} points on average "
                       f"({r['share_flagged']:.0%} of such mornings were surprises).")
    return out


def nap_rule_check(cycles):
    """How often WHOOP's 'sleep need from recent nap' equals minus the previous cycle's nap sleep."""
    d = cycles[["need_from_nap_hours", "prior_nap_hours"]].dropna()
    if d.empty:
        return "no mornings with nap data yet"
    hit = (d["need_from_nap_hours"] + d["prior_nap_hours"]).abs() < 1e-3
    return (f"every one of {len(d)} mornings checked" if hit.all()
            else f"{hit.sum()} of {len(d)} mornings checked ({hit.mean():.0%})")


def missing_files():
    return [name for name, p in REQUIRED.items() if not p.exists()]


@dataclass
class Bundle:
    cycles: pd.DataFrame          # every usable cycle (train, test, forward), time-ordered
    model: object                 # frozen formula model
    explainer: at.GroupShapley
    B: float                      # fixed reference (points)
    train: pd.DataFrame


def load_bundle():
    df = mc.load_features()
    usable = df[df["usable"]].reset_index(drop=True)
    train, _ = tb.load_data()
    model = tb.make_model("logit_ridge").fit(train[COLS], train[tb.TARGET])
    rng = np.random.default_rng(tb.SEED)            # identical draw to attribution.py
    bg = train[COLS].iloc[rng.choice(len(train), min(100, len(train)), replace=False)].to_numpy(float)
    explainer = at.GroupShapley(COLS, at.GROUPS, bg)
    B = float(model.predict(pd.DataFrame(bg, columns=COLS)).mean())
    usable["predicted"] = model.predict(usable[COLS])
    usable["split"] = np.where(usable["is_forward"], "forward", np.where(usable["is_test"], "test", "train"))
    usable["day"] = usable["sleep_end_local"].dt.normalize()   # the morning the score appeared
    return Bundle(usable, model, explainer, B, train)


def explain_rows(bundle, X):
    """Exact grouped Shapley contributions (points) for each row of X against the fixed B."""
    phi, V = bundle.explainer.explain(bundle.model.predict, X[COLS])
    return pd.DataFrame(phi, columns=GROUPS, index=X.index)


def what_if(row, sleep_h=None, need_h=None, hrv_ms_delta=0.0, rhr_delta=0.0, prior_strain=None):
    """Return a modified copy of one cycle's inputs, keeping derived columns consistent."""
    r = row.copy()
    if sleep_h is not None:
        r["total_sleep_hours"] = sleep_h
    if need_h is not None:
        r["sleep_need_total_hours"] = need_h
    if hrv_ms_delta:
        r["hrv_rmssd_milli"] += hrv_ms_delta
        r["hrv_vs_baseline"] += hrv_ms_delta
        r["hrv_pct_vs_baseline"] = r["hrv_vs_baseline"] / r["hrv_baseline"] * 100
    if rhr_delta:
        r["resting_heart_rate"] += rhr_delta
        r["rhr_vs_baseline"] += rhr_delta
    if prior_strain is not None:
        r["prior_strain"] = prior_strain
    return r


def read(name, **kw):
    return pd.read_csv(REQUIRED[name], **kw)


def residuals():
    r = read("oof residuals")
    r["cycle_start_local"] = pd.to_datetime(r["cycle_start_local"], format="ISO8601")
    return r


def archetype_days():
    d = read("archetype days")
    d["cycle_start_local"] = pd.to_datetime(d["cycle_start_local"], format="ISO8601")
    return d


ARCHETYPE_NAMES = ["Rest day", "Short-sleep day", "Active day", "Many-activities day"]


def archetype_table():
    """Archetypes ordered by centroid strain, named by role. Labels are matched by centroid,
    not by k-means label number, which can change between refits."""
    a = read("archetypes").sort_values("centroid_strain").reset_index(drop=True)
    if len(a) == 4:
        # the two low-strain clusters differ by opening sleep, the two high-strain by activities
        low, high = a.iloc[:2], a.iloc[2:]
        names = {}
        names[low["centroid_sleep_hours"].idxmax()] = "Rest day"
        names[low["centroid_sleep_hours"].idxmin()] = "Short-sleep day"
        names[high["centroid_activities"].idxmin()] = "Active day"
        names[high["centroid_activities"].idxmax()] = "Many-activities day"
        a["name"] = a.index.map(names)
    else:
        a["name"] = [f"Archetype {i + 1}" for i in range(len(a))]
    return a


def classify_day(a, strain, n_act, sleep_h, days):
    """Nearest centroid in the same z-scored space the clustering used."""
    feats = ["strain", "n_activities", "total_sleep_hours"]
    mu, sd = days[feats].mean(), days[feats].std(ddof=0)
    c = (a[["centroid_strain", "centroid_activities", "centroid_sleep_hours"]].to_numpy() - mu.to_numpy()) / sd.to_numpy()
    x = (np.array([strain, n_act, sleep_h]) - mu.to_numpy()) / sd.to_numpy()
    return a.iloc[int(np.argmin(((c - x) ** 2).sum(1)))]
