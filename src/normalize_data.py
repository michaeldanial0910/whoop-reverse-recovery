import json
import pandas as pd
from pathlib import Path

raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

def load_raw(name):
    with open(raw_dir / f"{name}.json") as f:
        return json.load(f)

# --- Cycles ---
cycles_df = pd.json_normalize(load_raw("cycles"))
cycles_df = cycles_df.rename(columns={
    "id": "cycle_id",
    "score.strain": "strain",
    "score.kilojoule": "kilojoule",
    "score.average_heart_rate": "avg_heart_rate",
    "score.max_heart_rate": "max_heart_rate",
})
cycles_df["date"] = pd.to_datetime(cycles_df["start"]).dt.date
cycles_df = cycles_df[["cycle_id", "date", "score_state", "strain", "kilojoule", "avg_heart_rate", "max_heart_rate"]]

# --- Recovery ---
recovery_df = pd.json_normalize(load_raw("recovery"))
recovery_df = recovery_df.rename(columns={
    "score.recovery_score": "recovery_score",
    "score.resting_heart_rate": "resting_heart_rate",
    "score.hrv_rmssd_milli": "hrv_rmssd_milli",
    "score.spo2_percentage": "spo2_percentage",
    "score.skin_temp_celsius": "skin_temp_celsius",
    "score.user_calibrating": "user_calibrating",
})
recovery_df = recovery_df[["cycle_id", "sleep_id", "recovery_score", "resting_heart_rate",
                            "hrv_rmssd_milli", "spo2_percentage", "skin_temp_celsius", "user_calibrating"]]

# --- Sleep ---
sleep_df = pd.json_normalize(load_raw("sleep"))
sleep_df = sleep_df.rename(columns={
    "id": "sleep_id",
    "score.respiratory_rate": "respiratory_rate",
    "score.sleep_performance_percentage": "sleep_performance_pct",
    "score.sleep_consistency_percentage": "sleep_consistency_pct",
    "score.sleep_efficiency_percentage": "sleep_efficiency_pct",
    "score.stage_summary.total_light_sleep_time_milli": "light_sleep_ms",
    "score.stage_summary.total_slow_wave_sleep_time_milli": "deep_sleep_ms",
    "score.stage_summary.total_rem_sleep_time_milli": "rem_sleep_ms",
    "score.stage_summary.total_awake_time_milli": "awake_ms",
    "score.stage_summary.disturbance_count": "disturbance_count",
})
keep_cols = ["sleep_id", "nap", "respiratory_rate", "sleep_performance_pct", "sleep_consistency_pct",
             "sleep_efficiency_pct", "light_sleep_ms", "deep_sleep_ms", "rem_sleep_ms",
             "awake_ms", "disturbance_count"]
sleep_df = sleep_df[[c for c in keep_cols if c in sleep_df.columns]]
for col in ["light_sleep_ms", "deep_sleep_ms", "rem_sleep_ms", "awake_ms"]:
    if col in sleep_df.columns:
        sleep_df[col.replace("_ms", "_hours")] = sleep_df[col] / 1000 / 60 / 60

# --- Join into one row per day ---
# cast join keys to string first -- an int id and a string id that "look the same"
# will still silently fail to match in a merge otherwise, same class of bug as the
# os.getenv() mix-up, just harder to notice because nothing throws an error
for df, col in [(cycles_df, "cycle_id"), (recovery_df, "cycle_id"),
                (recovery_df, "sleep_id"), (sleep_df, "sleep_id")]:
    df[col] = df[col].astype(str)

daily = cycles_df.merge(recovery_df, on="cycle_id", how="left")
daily = daily.merge(sleep_df, on="sleep_id", how="left")
daily = daily.sort_values("date").reset_index(drop=True)

out_path = processed_dir / "daily_recovery_dataset.csv"
daily.to_csv(out_path, index=False)
print(f"Saved {len(daily)} rows, {len(daily.columns)} columns to {out_path}")
print(daily.head())