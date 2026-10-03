"""Flatten raw WHOOP JSON into one row per physiological cycle.

Observation unit: cycle_id (NOT calendar date). A WHOOP cycle starts when you
fall asleep and ends when you next fall asleep, so cycle length varies (~11-48h)
and calendar dates are not unique. The recovery score belongs to the cycle it
opens; that cycle's strain accumulates AFTER the score is computed.

Local time: WHOOP timestamps are UTC; timezone_offset gives local time. All
date-like fields are derived from local time, never from the UTC date.
"""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
raw_dir = ROOT / "data" / "raw"
processed_dir = ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

MS_PER_HOUR = 1000 * 60 * 60


def load_raw(name):
    with open(raw_dir / f"{name}.json") as f:
        return json.load(f)


def to_local(utc_series, offset_series):
    """UTC timestamps + '+02:00'-style offsets -> naive local timestamps."""
    utc = pd.to_datetime(utc_series, utc=True)
    offsets = offset_series.fillna("Z").replace("Z", "+00:00")
    sign = offsets.str[0].map({"+": 1, "-": -1})
    hours = offsets.str[1:3].astype(int)
    minutes = offsets.str[4:6].astype(int)
    delta = pd.to_timedelta(sign * (hours * 60 + minutes), unit="m")
    return (utc + delta).dt.tz_localize(None)


# --- Cycles ---
cycles_df = pd.json_normalize(load_raw("cycles"))
cycles_df = cycles_df.rename(columns={
    "id": "cycle_id",
    "start": "cycle_start_utc",
    "end": "cycle_end_utc",
    "score.strain": "strain",
    "score.kilojoule": "kilojoule",
    "score.average_heart_rate": "avg_heart_rate",
    "score.max_heart_rate": "max_heart_rate",
})
cycles_df["cycle_start_local"] = to_local(cycles_df["cycle_start_utc"], cycles_df["timezone_offset"])
cycles_df["cycle_complete"] = cycles_df["cycle_end_utc"].notna()  # the in-progress cycle has partial strain
cycles_df = cycles_df[["cycle_id", "cycle_start_utc", "cycle_end_utc", "cycle_start_local",
                       "timezone_offset", "cycle_complete", "score_state", "strain",
                       "kilojoule", "avg_heart_rate", "max_heart_rate"]]

# --- Recovery ---
recovery_df = pd.json_normalize(load_raw("recovery"))
recovery_df = recovery_df.rename(columns={
    "score.recovery_score": "recovery_score",
    "score.resting_heart_rate": "resting_heart_rate",
    "score.hrv_rmssd_milli": "hrv_rmssd_milli",
    "score.spo2_percentage": "spo2_percentage",
    "score.skin_temp_celsius": "skin_temp_celsius",
    "score.user_calibrating": "user_calibrating",
    "score_state": "recovery_score_state",
})
recovery_df = recovery_df[["cycle_id", "sleep_id", "recovery_score_state", "recovery_score",
                           "resting_heart_rate", "hrv_rmssd_milli", "spo2_percentage",
                           "skin_temp_celsius", "user_calibrating"]]

# --- Sleep (main sleeps and naps come from the same endpoint) ---
sleep_all = pd.json_normalize(load_raw("sleep"))
sleep_all = sleep_all.rename(columns={
    "id": "sleep_id",
    "cycle_id": "sleep_cycle_id",
    "score.respiratory_rate": "respiratory_rate",
    "score.sleep_performance_percentage": "sleep_performance_pct",
    "score.sleep_consistency_percentage": "sleep_consistency_pct",
    "score.sleep_efficiency_percentage": "sleep_efficiency_pct",
    "score.stage_summary.total_in_bed_time_milli": "in_bed_ms",
    "score.stage_summary.total_light_sleep_time_milli": "light_sleep_ms",
    "score.stage_summary.total_slow_wave_sleep_time_milli": "deep_sleep_ms",
    "score.stage_summary.total_rem_sleep_time_milli": "rem_sleep_ms",
    "score.stage_summary.total_awake_time_milli": "awake_ms",
    "score.stage_summary.disturbance_count": "disturbance_count",
    "score.stage_summary.sleep_cycle_count": "sleep_cycle_count",
    "score.sleep_needed.baseline_milli": "need_baseline_ms",
    "score.sleep_needed.need_from_sleep_debt_milli": "need_from_debt_ms",
    "score.sleep_needed.need_from_recent_strain_milli": "need_from_strain_ms",
    "score.sleep_needed.need_from_recent_nap_milli": "need_from_nap_ms",
})
sleep_all["sleep_start_local"] = to_local(sleep_all["start"], sleep_all["timezone_offset"])
sleep_all["sleep_end_local"] = to_local(sleep_all["end"], sleep_all["timezone_offset"])

ms_cols = [c for c in sleep_all.columns if c.endswith("_ms")]
for col in ms_cols:
    sleep_all[col.replace("_ms", "_hours")] = sleep_all[col] / MS_PER_HOUR
sleep_all["total_sleep_hours"] = (sleep_all["light_sleep_hours"] + sleep_all["deep_sleep_hours"]
                                  + sleep_all["rem_sleep_hours"])

# Naps: summed per cycle they happened in. A nap in cycle N can only affect the
# recovery of cycle N+1 -- the lag is applied in features.py, not here.
naps = (sleep_all[sleep_all["nap"] == True]
        .groupby("sleep_cycle_id")
        .agg(nap_count=("sleep_id", "count"), nap_sleep_hours=("total_sleep_hours", "sum"))
        .reset_index()
        .rename(columns={"sleep_cycle_id": "cycle_id"}))

main_sleep = sleep_all[sleep_all["nap"] == False]
keep_cols = (["sleep_id", "sleep_start_local", "sleep_end_local", "respiratory_rate",
              "sleep_performance_pct", "sleep_consistency_pct", "sleep_efficiency_pct",
              "disturbance_count", "sleep_cycle_count", "total_sleep_hours"]
             + [c.replace("_ms", "_hours") for c in ms_cols])
main_sleep = main_sleep[keep_cols]

# --- Join into one row per cycle ---
# cast join keys to string first -- an int id and a string id that "look the same"
# will still silently fail to match in a merge otherwise
for df, col in [(cycles_df, "cycle_id"), (recovery_df, "cycle_id"), (recovery_df, "sleep_id"),
                (main_sleep, "sleep_id"), (naps, "cycle_id")]:
    df[col] = df[col].astype(str)

per_cycle = cycles_df.merge(recovery_df, on="cycle_id", how="left")
per_cycle = per_cycle.merge(main_sleep, on="sleep_id", how="left")
per_cycle = per_cycle.merge(naps, on="cycle_id", how="left")
per_cycle[["nap_count", "nap_sleep_hours"]] = per_cycle[["nap_count", "nap_sleep_hours"]].fillna(0)
per_cycle = per_cycle.sort_values("cycle_start_utc").reset_index(drop=True)

assert per_cycle["cycle_id"].is_unique, "cycle_id must be unique -- check the joins"

out_path = processed_dir / "cycles_dataset.csv"
per_cycle.to_csv(out_path, index=False)
print(f"Saved {len(per_cycle)} cycles x {len(per_cycle.columns)} columns to {out_path}")
