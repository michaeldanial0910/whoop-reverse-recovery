"""Baseline models: naive baselines, ridge, logit-link ridge, XGBoost, plus tautology ablations.

M1 used ridge + XGBoost (results/m1_*, preregistration.json). M2 adds the
logit-link ridge as the "reconstructed formula" model: recovery is bounded
0-100, and plain ridge extrapolated to impossible values (-47) on held-out
nights with out-of-range HRV. Fitting on logit(recovery/100) keeps the model
exactly additive (in logit units) while saturating like the real score.

Two modes:
  python src/train_baselines.py                  -> time-series CV on the TRAINING period only
  python src/train_baselines.py --evaluate-test  -> one-off evaluation on the held-out post-gap
                                                    period; refuses to run until the thresholds in
                                                    preregistration_m2.json are filled in AND committed

The test period is never touched in the default mode, so CV results can be
studied freely without contaminating the held-out check.

Run normalize_data.py and features.py first.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
RUN_TAG = "m2"                                      # output prefix; M1 files are kept as history
PREREG_PATH = ROOT / f"preregistration_{RUN_TAG}.json"
RESULTS_DIR = ROOT / "results"                      # aggregate metrics only -> safe to commit
PRIVATE_DIR = ROOT / "data" / "processed"           # per-cycle predictions -> git-ignored
TARGET = "recovery_score"
SEED = 42

CV_CONFIG = {"n_splits": 5, "test_size": 40}        # 5 expanding folds, 40 cycles each
MODEL_KINDS = ["ridge", "logit_ridge", "xgboost"]
LOGIT_EPS = 0.5                                     # clip to [0.5, 99.5] so logit stays finite

# --- Feature sets ---------------------------------------------------------
HRV = ["hrv_rmssd_milli", "hrv_vs_baseline", "hrv_pct_vs_baseline"]
RHR = ["resting_heart_rate", "rhr_vs_baseline"]
OTHER_PHYSIO = ["respiratory_rate", "resp_rate_vs_baseline",
                "skin_temp_celsius", "skin_temp_vs_baseline", "spo2_percentage"]
SLEEP = ["total_sleep_hours", "light_sleep_pct", "deep_sleep_pct", "rem_sleep_pct",
         "sleep_efficiency_pct", "disturbances_per_hour", "sleep_consistency_pct",
         "sleep_need_total_hours"]
LOAD = ["prior_strain", "prior_nap_hours"]
CALENDAR = [f"dow_{d}" for d in range(7)]           # one-hot wake day-of-week

FEATURE_SETS = {
    # Set B (chosen 2026-10-08): raw sleep variables instead of WHOOP's sleep_performance_pct
    "set_b_full":   HRV + RHR + OTHER_PHYSIO + SLEEP + LOAD + CALENDAR,
    # ablations -- the tautology check
    "hrv_only":     HRV,
    "hrv_rhr":      HRV + RHR,
    "physiology":   HRV + RHR + OTHER_PHYSIO,
    "set_b_no_hrv": RHR + OTHER_PHYSIO + SLEEP + LOAD + CALENDAR,
}


def logit_pct(y):
    p = np.clip(np.asarray(y, dtype=float), LOGIT_EPS, 100 - LOGIT_EPS) / 100
    return np.log(p / (1 - p))


def inv_logit_pct(z):
    return 100 / (1 + np.exp(-np.asarray(z, dtype=float)))


def make_model(kind):
    if kind == "logit_ridge":
        # same ridge, fitted on logit(recovery/100); predictions mapped back to 0-100
        return TransformedTargetRegressor(regressor=make_model("ridge"), func=logit_pct,
                                          inverse_func=inv_logit_pct, check_inverse=False)
    if kind == "ridge":
        # scaled so the penalty treats all features equally; alpha picked by internal CV
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 30)))
    if kind == "xgboost":
        # deliberately small/shallow and untuned: ~300 training rows
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            random_state=SEED, n_jobs=1)
    raise ValueError(kind)


def load_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["cycle_start_local"])
    df = df[df["exclude_reason"].isna()].sort_values("cycle_start_local").reset_index(drop=True)
    dow = pd.get_dummies(df["wake_day_of_week"].astype(int), prefix="dow").astype(float)
    df = pd.concat([df, dow.reindex(columns=CALENDAR, fill_value=0.0)], axis=1)
    # forward-set cycles (after the frozen test window) are in neither train nor test
    df = df[~df["is_forward"]]
    return df[~df["is_test"]].reset_index(drop=True), df[df["is_test"]].reset_index(drop=True)


def predict_all(train, test):
    """Fit every (model, feature set) on `train`, predict `test`. Returns {name: predictions}."""
    preds = {
        "naive_mean": np.full(len(test), train[TARGET].mean()),
        # yesterday's score; falls back to the training mean when the previous cycle has none
        "naive_previous": test["prev_recovery_score"].fillna(train[TARGET].mean()).to_numpy(),
    }
    for set_name, cols in FEATURE_SETS.items():
        for kind in MODEL_KINDS:
            model = make_model(kind).fit(train[cols], train[TARGET])
            preds[f"{kind}__{set_name}"] = model.predict(test[cols])
    return preds


def score(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


def run_cv(train):
    splitter = TimeSeriesSplit(**CV_CONFIG)
    fold_rows, oof_rows = [], []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train)):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        for name, pred in predict_all(tr, va).items():
            fold_rows.append({"fold": fold, "model": name, "n_train": len(tr), "n_val": len(va),
                              **score(va[TARGET], pred)})
            oof_rows.append(pd.DataFrame({"cycle_id": va["cycle_id"], "fold": fold, "model": name,
                                          "actual": va[TARGET].to_numpy(), "predicted": pred}))
    folds = pd.DataFrame(fold_rows)
    summary = (folds.groupby("model")[["mae", "r2"]].agg(["mean", "std"]).round(3)
               .sort_values(("mae", "mean")))
    summary.columns = ["mae_mean", "mae_std", "r2_mean", "r2_std"]
    return folds, summary, pd.concat(oof_rows, ignore_index=True)


def check_preregistration():
    prereg = json.loads(PREREG_PATH.read_text())
    missing = [k for k, v in prereg.items() if v is None]
    if missing:
        sys.exit(f"Refusing to evaluate the test set: thresholds not set in preregistration.json: {missing}")
    dirty = subprocess.run(["git", "status", "--porcelain", str(PREREG_PATH)], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("Refusing to evaluate the test set: preregistration.json has uncommitted changes. Commit it first.")
    return prereg


def evaluate_test(train, test, prereg):
    preds = predict_all(train, test)
    results = pd.DataFrame([{"model": n, **score(test[TARGET], p)} for n, p in preds.items()])
    results = results.sort_values("mae").round(3)
    by_model = results.set_index("model")
    primary = by_model.loc[prereg["primary_model"]]
    flexible = by_model.loc[prereg["flexible_model"]]
    checks = {
        "mae_within_threshold": primary["mae"] <= prereg["max_test_mae_points"],
        "r2_above_threshold": primary["r2"] >= prereg["min_test_r2"],
        "close_to_flexible_model": primary["mae"] - flexible["mae"] <= prereg["max_mae_gap_to_flexible_points"],
        "no_leakage_alarm": primary["r2"] < prereg["leakage_alarm_r2"],
    }
    return results, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    train, test = load_data()
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Train cycles: {len(train)} | held-out test cycles: {len(test)}")

    if not args.evaluate_test:
        folds, summary, oof = run_cv(train)
        folds.round(3).to_csv(RESULTS_DIR / f"{RUN_TAG}_cv_folds.csv", index=False)
        summary.to_csv(RESULTS_DIR / f"{RUN_TAG}_cv_summary.csv")
        oof.to_csv(PRIVATE_DIR / f"{RUN_TAG}_cv_predictions.csv", index=False)
        print("\nTime-series CV on the training period (test set untouched):")
        print(summary.to_string())
        return

    prereg = check_preregistration()
    results, checks = evaluate_test(train, test, prereg)
    results.to_csv(RESULTS_DIR / f"{RUN_TAG}_test_results.csv", index=False)
    print("\nHeld-out test results:")
    print(results.to_string(index=False))
    print(f"\nPre-registered checks (primary = {prereg['primary_model']}):")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
