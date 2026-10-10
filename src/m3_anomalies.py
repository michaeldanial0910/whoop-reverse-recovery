"""M3 step 2: anomaly detector. Which cycles did the reconstructed formula get badly wrong?

  residual = actual recovery - the formula model's OUT-OF-FOLD prediction
  flagged  = |residual| > 2.5 x robust SD of the residuals          (decided 2026-10-10)

Out-of-fold predictions (decided 2026-10-10):
  - training cycles: 10 contiguous time blocks; each block is predicted by a model fitted on
    the other 9 (so it may see later data; fine for spotting anomalies, which is not forecasting)
  - test cycles: the model fitted on the whole training period (the one evaluated in M2)
  - forward-set cycles: not used

Robust SD = 1.4826 x median absolute deviation. Unlike RMSE it is not inflated by the very
outliers we are hunting.

For each cycle the same exact grouped Shapley attribution as M2 (src/attribution.py) shows
what the model relied on, using the model that made that cycle's prediction and the same
fixed 100-cycle training background.

Outputs: per-cycle tables -> data/processed (private); aggregates -> results/ (public).
  python src/m3_anomalies.py
"""
import numpy as np
import pandas as pd

import attribution as at
import m3_common as mc
import train_baselines as tb

CONFIG = {"n_blocks": 10, "threshold_robust_sd": 2.5, "background_size": 100,
          # bins for "where does the model go wrong": residual by input level (public aggregate)
          "sleep_bins_hours": [0, 4, 5, 6, 7, 8, 9, 24],
          "hrv_bins_pct_vs_baseline": [-100, -30, -15, 0, 15, 30, 45, 1000]}
COLS = at.COLS


def oof_predictions(train, test):
    """Returns per-row predictions plus the fitted model that produced each row."""
    rows = pd.concat([train, test], ignore_index=True)
    pred = np.empty(len(rows))
    fold = np.empty(len(rows), dtype=object)
    models = {}
    for b, idx in enumerate(np.array_split(np.arange(len(train)), CONFIG["n_blocks"])):
        fit_rows = train.drop(train.index[idx])
        models[f"block_{b}"] = tb.make_model("logit_ridge").fit(fit_rows[COLS], fit_rows[tb.TARGET])
        pred[idx] = models[f"block_{b}"].predict(train.iloc[idx][COLS])
        fold[idx] = f"block_{b}"
    models["full_train"] = tb.make_model("logit_ridge").fit(train[COLS], train[tb.TARGET])
    t_idx = np.arange(len(train), len(rows))
    pred[t_idx] = models["full_train"].predict(test[COLS])
    fold[t_idx] = "full_train"
    return rows, pred, fold, models


def main():
    train, test = tb.load_data()
    rows, pred, fold, models = oof_predictions(train, test)
    resid = rows[tb.TARGET].to_numpy() - pred
    mad = np.median(np.abs(resid - np.median(resid)))
    robust_sd = 1.4826 * mad
    thr = CONFIG["threshold_robust_sd"] * robust_sd
    flagged = np.abs(resid) > thr
    print(f"{len(rows)} cycles | OOF MAE {np.abs(resid).mean():.2f} | robust SD {robust_sd:.2f} | "
          f"threshold {thr:.1f} pts | flagged {flagged.sum()} ({flagged.mean():.1%})")

    # --- attribution with the model that made each prediction --------------
    rng = np.random.default_rng(tb.SEED)        # same draw as attribution.py -> same background
    bg = train[COLS].iloc[rng.choice(len(train), CONFIG["background_size"], replace=False)].to_numpy(float)
    explainer = at.GroupShapley(COLS, at.GROUPS, bg)
    names = explainer.names
    phi = np.zeros((len(rows), len(names)))
    B = np.zeros(len(rows))
    for name, model in models.items():
        m = fold == name
        phi[m], V = explainer.explain(model.predict, rows.loc[m, COLS])
        B[m] = V[:, 0]
        assert np.allclose(phi[m].sum(1), pred[m] - B[m], atol=1e-6), name

    # how unusual was each input vs the training distribution (robust z)
    med = train[COLS].median()
    scale = 1.4826 * (train[COLS] - med).abs().median()
    scale = scale.where(scale > 0, train[COLS].std())   # binary day-of-week columns
    z = ((rows[COLS] - med) / scale).abs()
    weekday = [c for c in COLS if not c.startswith("dow_")]

    per = pd.DataFrame({
        "cycle_id": rows["cycle_id"], "cycle_start_local": rows["cycle_start_local"],
        "is_test": rows["is_test"], "oof_model": fold, "actual": rows[tb.TARGET],
        "predicted": pred, "residual": resid, "flagged": flagged,
        "direction": np.where(~flagged, "", np.where(resid > 0, "better than expected",
                                                     "worse than expected")),
        "baseline_B": B})
    contrib = pd.DataFrame(phi, columns=[f"pts_{n}" for n in names])
    order = np.argsort(-np.abs(phi), axis=1)
    per["top_groups"] = [", ".join(f"{names[j]} {phi[i, j]:+.1f}" for j in order[i, :3])
                         for i in range(len(rows))]
    per["most_unusual_input"] = z[weekday].idxmax(axis=1).to_numpy()
    per["most_unusual_robust_z"] = z[weekday].max(axis=1).to_numpy()
    per = pd.concat([per, contrib], axis=1)
    per.round(3).to_csv(mc.PRIVATE_DIR / "m3_oof_residuals.csv", index=False)
    per[per["flagged"]].round(2).to_csv(mc.PRIVATE_DIR / "m3_anomalies.csv", index=False)

    # --- public aggregates (no dates, no per-cycle values) -----------------
    ctx_cols = ["prior_strain", "total_sleep_hours", "sleep_need_total_hours", "hrv_vs_baseline",
                "onset_shift_hours", "onset_sd7_hours"]
    ctx = rows[ctx_cols].copy()
    ctx["prev_day_activities"] = mc.load_features().set_index("cycle_id")["n_activities"] \
        .shift(1).reindex(rows["cycle_id"]).to_numpy()   # activities of the day before
    ctx["group"] = np.where(~flagged, "not flagged", per["direction"])
    ctx["max_input_robust_z"] = per["most_unusual_robust_z"]
    context = ctx.groupby("group").median().T
    context.insert(0, "metric", "median")
    counts = ctx["group"].value_counts()

    summary = pd.Series({
        "n_cycles": len(rows), "n_train": int((~rows["is_test"]).sum()), "n_test": int(rows["is_test"].sum()),
        "oof_mae": np.abs(resid).mean(), "oof_r2": 1 - (resid ** 2).sum() / ((rows[tb.TARGET] - rows[tb.TARGET].mean()) ** 2).sum(),
        "robust_sd": robust_sd, "threshold_points": thr, "n_flagged": int(flagged.sum()),
        "share_flagged": flagged.mean(),
        "n_worse_than_expected": int((flagged & (resid < 0)).sum()),
        "n_better_than_expected": int((flagged & (resid > 0)).sum()),
        "share_flagged_train": flagged[~rows["is_test"].to_numpy()].mean(),
        "share_flagged_test": flagged[rows["is_test"].to_numpy()].mean(),
        "median_abs_residual_flagged": np.median(np.abs(resid[flagged])),
        "expected_share_if_normal": 2 * (1 - 0.99379),   # P(|Z| > 2.5)
    })
    top = pd.DataFrame({
        "group": names,
        "top_group_freq_flagged": np.bincount(np.abs(phi[flagged]).argmax(1), minlength=len(names)) / flagged.sum(),
        "top_group_freq_all": np.bincount(np.abs(phi).argmax(1), minlength=len(names)) / len(rows),
        "mean_abs_pts_flagged": np.abs(phi[flagged]).mean(0),
        "mean_abs_pts_all": np.abs(phi).mean(0)})
    unusual = pd.DataFrame({"flagged": per.loc[flagged, "most_unusual_input"].value_counts(),
                            "all": per["most_unusual_input"].value_counts()}).fillna(0).astype(int)

    # residual by input level: a systematic shape the model misses shows up as a trend here
    by_bin = []
    for col, edges in [("total_sleep_hours", CONFIG["sleep_bins_hours"]),
                       ("hrv_pct_vs_baseline", CONFIG["hrv_bins_pct_vs_baseline"])]:
        g = pd.DataFrame({"bin": pd.cut(rows[col], edges).astype(str), "residual": resid, "flagged": flagged})
        agg = g.groupby("bin", sort=False).agg(n=("residual", "size"), mean_residual=("residual", "mean"),
                                               share_flagged=("flagged", "mean"))
        agg = agg.reindex([str(i) for i in pd.IntervalIndex.from_breaks(edges)]).dropna()
        by_bin.append(agg.reset_index().assign(input=col))
    by_bin = pd.concat(by_bin)[["input", "bin", "n", "mean_residual", "share_flagged"]]

    mc.RESULTS_DIR.mkdir(exist_ok=True)
    summary.round(4).to_csv(mc.RESULTS_DIR / "m3_anomaly_summary.csv", header=["value"])
    top.round(4).to_csv(mc.RESULTS_DIR / "m3_anomaly_groups.csv", index=False)
    context.round(3).to_csv(mc.RESULTS_DIR / "m3_anomaly_context.csv")
    by_bin.round(3).to_csv(mc.RESULTS_DIR / "m3_anomaly_residual_by_input.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print("\nGroup sizes:", counts.to_dict())
    print("\nContext medians:\n", context.round(2).to_string())
    print("\nLargest-contribution group:\n", top.round(3).to_string(index=False))
    print("\nResidual by input level:\n", by_bin.round(2).to_string(index=False))
    print("\nMost unusual input:\n", unusual.to_string())
    print("\nFlagged cycles (private):")
    print(per.loc[flagged, ["cycle_start_local", "is_test", "actual", "predicted", "residual",
                            "top_groups", "most_unusual_input", "most_unusual_robust_z"]]
          .round({"actual": 1, "predicted": 1, "residual": 1, "most_unusual_robust_z": 1}).to_string(index=False))


if __name__ == "__main__":
    main()
