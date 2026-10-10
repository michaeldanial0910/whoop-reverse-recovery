"""Build the M1 modelling table from data/processed/cycles_dataset.csv.

One row per cycle. Every feature is computed from information available when
WHOOP computes that cycle's recovery score (i.e. at wake-up):
  - same-cycle physiology and sleep (the inputs WHOOP itself uses)
  - PREVIOUS cycle's strain and naps (this cycle's strain happens after the score)
  - personal baselines from PREVIOUS cycles only (closed="left" -> today excluded)

Run normalize_data.py first.
"""
import json

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "cycles_dataset.csv"
OUT_PATH = ROOT / "data" / "processed" / "features.csv"
WORKOUTS_PATH = ROOT / "data" / "raw" / "workouts.json"

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
    # end (exclusive) of the held-out test period, frozen in M3 so the evaluated test set stays
    # the original 54 cycles. Later cycles form the FORWARD set: neither trained on nor part of
    # the evaluated test (reserved for forward testing with a frozen model).
    "test_end_local": "2026-09-27",
    # recovery scores treated as unusable (Michael's decision 2026-10-03)
    "drop_scores": [1.0],
    # a "nap" longer than this is almost certainly a main sleep WHOOP misclassified
    # (seen 2026-03-30: an 8.7 h "nap" left that cycle without a recovery score and
    # gave the NEXT cycle an 8.7 h nap credit, i.e. a sleep need of 1.2 h).
    # WHOOP's nap credit equals the previous cycle's nap sleep exactly, so the next
    # cycle's sleep-need inputs are corrupted -> exclude that next cycle.
    "max_plausible_nap_hours": 6.0,
    # --- M3 sleep-timing features (decided 2026-10-10) ---
    # onset clock time is written as hours after this hour of the previous day, so a 05:00
    # onset = 29 and there is no midnight wrap. 18:00 sits in the empty part of the clock:
    # no main sleep in the data starts between 14:00 and 20:00.
    "onset_day_cut_hour": 18,
    # onset variability = circular SD of onsets in the previous 7 calendar days (tonight
    # excluded), only when at least 5 nights fall in that window
    "onset_window": "7D",
    "onset_min_nights": 5,
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


def add_timing_features(df):
    """M3 schedule-irregularity inputs, from the main sleep that opens each cycle.

    onset_hours        clock time of sleep onset, hours after `onset_day_cut_hour` of the
                       previous day (05:00 -> 29). Linear, no midnight wrap.
    onset_sd7_hours    circular SD of onset over the previous 7 calendar days (tonight
                       excluded), in hours. Circular so 23:30 and 00:30 are 1 h apart, not 23.
    onset_shift_hours  tonight's onset minus the previous cycle's onset (back-to-back cycles
                       only); positive = later than last night.
    Uses every recorded sleep, including cycles later excluded from modelling.
    """
    start = df["sleep_start_local"]
    clock = start.dt.hour + start.dt.minute / 60 + start.dt.second / 3600
    cut = CONFIG["onset_day_cut_hour"]
    df["onset_hours"] = np.where(clock < cut, clock + 24, clock)
    angle = clock / 24 * 2 * np.pi
    trig = pd.DataFrame({"c": np.cos(angle).to_numpy(), "s": np.sin(angle).to_numpy()},
                        index=pd.DatetimeIndex(start))
    trig = trig[trig.index.notna()].sort_index()
    roll = trig.rolling(CONFIG["onset_window"], closed="left", min_periods=CONFIG["onset_min_nights"])
    R = np.hypot(roll["c"].mean(), roll["s"].mean()).clip(upper=1.0)
    sd = np.sqrt(-2 * np.log(R)) * 24 / (2 * np.pi)       # circular SD, radians -> hours
    sd = sd[~sd.index.duplicated()]
    df["onset_sd7_hours"] = sd.reindex(start).to_numpy()
    df["onset_shift_hours"] = (df["onset_hours"] - df["onset_hours"].shift(1)).where(df["prev_cycle_contiguous"])
    return df


def add_activity_features(df):
    """M3: logged activities (WHOOP workouts) whose start falls inside each cycle.

    All sport types count, including auto-detected-looking 'walking' and generic 'activity'
    (decided 2026-10-10: leaving the house is the signal; strain already carries intensity).
    These describe the DAY of the cycle, i.e. they happen after that cycle's recovery score.
    """
    w = pd.DataFrame(json.loads(WORKOUTS_PATH.read_text()))
    w_start = pd.to_datetime(w["start"], utc=True)
    w_minutes = (pd.to_datetime(w["end"], utc=True) - w_start).dt.total_seconds() / 60
    starts = df["cycle_start_utc"].to_numpy()
    ends = df["cycle_end_utc"].fillna(pd.Timestamp.max.tz_localize("UTC")).to_numpy()
    idx = np.searchsorted(starts, w_start.to_numpy(), side="right") - 1
    inside = (idx >= 0) & (w_start.to_numpy() < ends[idx.clip(0)])
    df["n_activities"] = np.bincount(idx[inside], minlength=len(df))
    df["activity_minutes"] = np.bincount(idx[inside], weights=w_minutes[inside], minlength=len(df))
    return df


def add_exclusion_flags(df):
    reason = pd.Series("", index=df.index)
    reason[df["recovery_score"].isna()] = "no_recovery"
    reason[(reason == "") & (df["user_calibrating"] == True)] = "calibrating"
    reason[(reason == "") & df["recovery_score"].isin(CONFIG["drop_scores"])] = "dropped_score"
    misclassified = df["prior_nap_hours"] > CONFIG["max_plausible_nap_hours"]
    reason[(reason == "") & misclassified] = "prior_nap_misclassified"
    reason[(reason == "") & df["hrv_baseline"].isna()] = "no_baseline_yet"
    df["exclude_reason"] = reason
    start = df["cycle_start_local"]
    df["is_forward"] = start >= pd.Timestamp(CONFIG["test_end_local"])
    df["is_test"] = (start >= pd.Timestamp(CONFIG["test_start_local"])) & ~df["is_forward"]
    return df


def main():
    df = load_cycles()
    df = add_previous_cycle_features(df)
    df = add_personal_baselines(df)
    df = add_sleep_features(df)
    df = add_calendar_features(df)
    df = add_timing_features(df)
    df = add_activity_features(df)
    df = add_exclusion_flags(df)
    df.to_csv(OUT_PATH, index=False)

    usable = df[df["exclude_reason"] == ""]
    print(f"Saved {len(df)} cycles to {OUT_PATH}")
    print("Excluded:", df.loc[df["exclude_reason"] != "", "exclude_reason"].value_counts().to_dict())
    n_train = (~usable["is_test"] & ~usable["is_forward"]).sum()
    print(f"Usable: {len(usable)}  (train {n_train}, test {usable['is_test'].sum()}, "
          f"forward {usable['is_forward'].sum()})")
    missing = usable.isna().sum()
    print("Missing values in usable rows:", missing[missing > 0].to_dict())


if __name__ == "__main__":
    main()
