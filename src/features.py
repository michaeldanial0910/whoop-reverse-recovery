"""Build the M1 modelling table from data/processed/cycles_dataset.csv.

One row per cycle. Every feature is computed from information available when
WHOOP computes that cycle's recovery score (i.e. at wake-up):
  - same-cycle physiology and sleep (the inputs WHOOP itself uses)
  - PREVIOUS cycle's strain and naps (this cycle's strain happens after the score)
  - personal baselines from PREVIOUS cycles only (closed="left" -> today excluded)

Run normalize_data.py first.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "cycles_dataset.csv"
OUT_PATH = ROOT / "data" / "processed" / "features.csv"

CONFIG = {
    # personal baseline: calendar window over previous cycles (gap-safe: a 13-day
    # device gap just means fewer observations in the window, not wrong ones)
    "baseline_window": "30D",
    "baseline_min_obs": 14,
    # a previous cycle only counts as "previous" if it ended within this tolerance
    # of the current cycle's start (otherwise there was a gap in between)
    "contiguity_tolerance": pd.Timedelta("1h"),
    # first cycle of the held-out test period (post device-gap stretch, local time)
    "test_start_local": "2026-07-30",
    # recovery scores treated as unusable (Michael's decision 2026-10-03)
    "drop_scores": [1.0],
}

BASELINE_COLS = {
    "hrv_rmssd_milli": "hrv",
    "resting_heart_rate": "rhr",
    "respiratory_rate": "resp_rate",
    "skin_temp_celsius": "skin_temp",
}


def load_cycles():
    df = pd.read_csv(IN_PATH)
    for col in ["cycle_start_utc", "cycle_end_utc"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    for col in ["cycle_start_local", "sleep_start_local", "sleep_end_local"]:
        df[col] = pd.to_datetime(df[col])
    return df.sort_values("cycle_start_utc").reset_index(drop=True)


def add_previous_cycle_features(df):
    prev_end = df["cycle_end_utc"].shift(1)
    contiguous = (df["cycle_start_utc"] - prev_end).abs() <= CONFIG["contiguity_tolerance"]
    df["prev_cycle_contiguous"] = contiguous
    df["prior_strain"] = df["strain"].shift(1).where(contiguous)
    df["prior_nap_hours"] = df["nap_sleep_hours"].shift(1).where(contiguous)
    # NOT a model feature -- only used by the "yesterday's score" naive baseline
    df["prev_recovery_score"] = df["recovery_score"].shift(1).where(contiguous)
    return df


def add_personal_baselines(df):
    """Mean of the previous `baseline_window` of cycles, excluding the current one.

    Uses every measured value (including cycles later excluded from modelling),
    because those are still real physiology for the baseline.
    """
    indexed = df.set_index("cycle_start_utc")
    for col, short in BASELINE_COLS.items():
        base = indexed[col].rolling(CONFIG["baseline_window"], closed="left",
                                    min_periods=CONFIG["baseline_min_obs"]).mean()
        df[f"{short}_baseline"] = base.to_numpy()
        df[f"{short}_vs_baseline"] = df[col] - df[f"{short}_baseline"]
    # HRV is right-skewed and usually compared in relative terms: also keep % deviation
    df["hrv_pct_vs_baseline"] = df["hrv_vs_baseline"] / df["hrv_baseline"] * 100
    return df


def add_sleep_features(df):
    total = df["total_sleep_hours"].replace(0, np.nan)
    for stage in ["light", "deep", "rem"]:
        df[f"{stage}_sleep_pct"] = df[f"{stage}_sleep_hours"] / total * 100
    df["disturbances_per_hour"] = df["disturbance_count"] / df["in_bed_hours"].replace(0, np.nan)
    df["sleep_need_total_hours"] = df[["need_baseline_hours", "need_from_debt_hours",
                                       "need_from_strain_hours", "need_from_nap_hours"]].sum(axis=1, min_count=4)
    return df


def add_calendar_features(df):
    # the score is shown on waking, so the "day" a score belongs to is the wake-up day
    df["wake_day_of_week"] = df["sleep_end_local"].dt.dayofweek  # 0 = Monday
    return df


def add_exclusion_flags(df):
    reason = pd.Series("", index=df.index)
    reason[df["recovery_score"].isna()] = "no_recovery"
    reason[(reason == "") & (df["user_calibrating"] == True)] = "calibrating"
    reason[(reason == "") & df["recovery_score"].isin(CONFIG["drop_scores"])] = "dropped_score"
    reason[(reason == "") & df["hrv_baseline"].isna()] = "no_baseline_yet"
    df["exclude_reason"] = reason
    df["is_test"] = df["cycle_start_local"] >= pd.Timestamp(CONFIG["test_start_local"])
    return df


def main():
    df = load_cycles()
    df = add_previous_cycle_features(df)
    df = add_personal_baselines(df)
    df = add_sleep_features(df)
    df = add_calendar_features(df)
    df = add_exclusion_flags(df)
    df.to_csv(OUT_PATH, index=False)

    usable = df[df["exclude_reason"] == ""]
    print(f"Saved {len(df)} cycles to {OUT_PATH}")
    print("Excluded:", df.loc[df["exclude_reason"] != "", "exclude_reason"].value_counts().to_dict())
    print(f"Usable: {len(usable)}  (train {(~usable['is_test']).sum()}, test {usable['is_test'].sum()})")
    missing = usable.isna().sum()
    print("Missing values in usable rows:", missing[missing > 0].to_dict())


if __name__ == "__main__":
    main()
