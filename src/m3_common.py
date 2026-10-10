"""Shared helpers for the M3 scripts (anomalies, irregularity, archetypes)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import train_baselines as tb

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"            # aggregates only -> safe to commit
FIG_DIR = RESULTS_DIR / "figures"
PRIVATE_DIR = ROOT / "data" / "processed"  # per-cycle tables -> git-ignored
SEED = tb.SEED


def load_features():
    """Every cycle (excluded ones too), time-ordered, with the one-hot day-of-week columns."""
    df = pd.read_csv(tb.FEATURES_PATH)
    for col in ["cycle_start_utc", "cycle_end_utc"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    for col in ["cycle_start_local", "sleep_start_local", "sleep_end_local"]:
        df[col] = pd.to_datetime(df[col], format="ISO8601")
    df["exclude_reason"] = df["exclude_reason"].fillna("")
    df["usable"] = df["exclude_reason"] == ""
    dow = pd.get_dummies(df["wake_day_of_week"].astype("Int64"), prefix="dow").astype(float)
    df = pd.concat([df, dow.reindex(columns=tb.CALENDAR, fill_value=0.0)], axis=1)
    return df.sort_values("cycle_start_utc").reset_index(drop=True)


def analysis_rows(df):
    """Usable cycles in train + frozen test. The forward set is never used in M3."""
    return df[df["usable"] & ~df["is_forward"]].reset_index(drop=True)


def moving_block_indices(n, block, rng):
    """One moving-block bootstrap resample of row positions 0..n-1 (time order kept within blocks)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap_p(draws):
    """Two-sided bootstrap p-value for 'coefficient = 0'."""
    draws = np.asarray(draws)
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


def holm(pvals):
    """Holm step-down adjusted p-values, same order as the input."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def check_committed(path):
    """Refuse to run a pre-registered analysis unless its pre-registration is committed and clean."""
    path = Path(path)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path.name], cwd=path.parent,
                             capture_output=True, text=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", path.name], cwd=path.parent,
                           capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        sys.exit(f"Refusing to run: {path.name} must be committed (and unchanged) before this analysis.")
    return json.loads(path.read_text())
