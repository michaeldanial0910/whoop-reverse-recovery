"""M2 attribution: how many recovery points each input group accounts for.

Method (plan Appendix C-5): FIXED BASELINE + DIFFERENCING, exact grouped Shapley values.
  - Inputs are grouped (HRV's three columns are one group, etc.), so correlated
    columns that describe the same signal cannot split the credit arbitrarily.
  - With 11 groups there are 2^11 = 2048 coalitions, few enough to evaluate every
    one: the Shapley values are exact, not sampled.
  - "Removing" a group means replacing it with values from a FIXED background
    sample of training cycles (interventional Shapley). The reference B is the
    model's mean prediction over that background, identical for every cycle.
  - So for every cycle: sum of group contributions == prediction - B (exactly),
    in recovery points, for BOTH models. Same method, same units, comparable.
  - Day-over-day: today's contributions minus yesterday's. B cancels, so the
    differences sum exactly to pred_today - pred_yesterday.

Attribution describes the MODEL, not physiology: "HRV accounts for X points"
means the model's output moves X points with HRV, not that raising HRV causes it.

  python src/attribution.py              full run (models fit on the training period)
  python src/attribution.py --quick      skip the stability refits (full run: ~20 min)

Run features.py first. Per-cycle outputs -> data/processed (git-ignored);
aggregate outputs -> results/.
"""
import argparse
import time
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

import train_baselines as tb

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PRIVATE_DIR = ROOT / "data" / "processed"

GROUPS = {
    "HRV": ["hrv_rmssd_milli", "hrv_vs_baseline", "hrv_pct_vs_baseline"],
    "Resting HR": ["resting_heart_rate", "rhr_vs_baseline"],
    "Respiratory rate": ["respiratory_rate", "resp_rate_vs_baseline"],
    "Skin temp": ["skin_temp_celsius", "skin_temp_vs_baseline"],
    "SpO2": ["spo2_percentage"],
    "Sleep amount vs need": ["total_sleep_hours", "sleep_need_total_hours"],
    "Sleep stages": ["light_sleep_pct", "deep_sleep_pct", "rem_sleep_pct"],
    "Sleep quality": ["sleep_efficiency_pct", "disturbances_per_hour"],
    "Sleep consistency": ["sleep_consistency_pct"],
    "Prior load": ["prior_strain", "prior_nap_hours"],
    "Day of week": tb.CALENDAR,
}
MODELS = {"formula": "logit_ridge", "xgboost": "xgboost"}   # display name -> make_model kind

CONFIG = {
    "background_size": 100,     # fixed training cycles that define B
    # stability refits on random 80% subsamples of the training cycles. Subsampling, not the
    # classic bootstrap: duplicated rows fool RidgeCV's leave-one-out alpha search (a duplicate's
    # twin is always in the fit), which shrinks regularization and biases the shares.
    "n_subsamples": 20,
    "subsample_frac": 0.8,
    "stability_background_size": 25,  # smaller background for the 50 refits (runtime); B noise only
    "seed": tb.SEED,
}


class GroupShapley:
    """Exact interventional Shapley values over feature groups."""

    def __init__(self, columns, groups, background):
        self.columns = list(columns)
        self.names = list(groups)
        n = len(self.names)
        col_group = np.array([next(i for i, g in enumerate(groups.values()) if c in g) for c in self.columns])
        masks = np.arange(2 ** n)
        self.in_coalition = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)  # (2^n, n)
        self.col_on = self.in_coalition[:, col_group]                            # (2^n, n_cols)
        self.background = np.asarray(background, dtype=float)
        # Shapley weight matrix: phi = W @ v, where v[mask] = E_b f(x_S, b_rest)
        size = self.in_coalition.sum(1)
        w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
        self.W = np.zeros((n, 2 ** n))
        for i in range(n):
            without = ~self.in_coalition[:, i]
            S = masks[without]
            self.W[i, S | (1 << i)] += w[size[S]]
            self.W[i, S] -= w[size[S]]
        self.n = n

    def explain(self, predict, X, batch=16):
        """Several cycles per predict call: far fewer calls, same numbers."""
        X = np.asarray(X, dtype=float)
        V = []
        for start in range(0, len(X), batch):
            xb = X[start:start + batch]                                       # (k, n_cols)
            Z = np.where(self.col_on[None, :, None, :], xb[:, None, None, :],
                         self.background[None, None, :, :])                   # (k, 2^n, n_bg, n_cols)
            preds = predict(pd.DataFrame(Z.reshape(-1, len(self.columns)), columns=self.columns))
            V.append(preds.reshape(len(xb), 2 ** self.n, -1).mean(2))
        V = np.vstack(V)                                                      # (n_rows, 2^n)
        return V @ self.W.T, V                                                # phi (n_rows, n_groups)

    def pair_interaction(self, V, i, j):
        """Shapley interaction index between groups i and j (total, i.e. both halves), per row."""
        n, masks = self.n, np.arange(2 ** self.n)
        S = masks[~self.in_coalition[:, i] & ~self.in_coalition[:, j]]
        size = self.in_coalition[S].sum(1)
        w = np.array([factorial(s) * factorial(n - s - 2) / factorial(n - 1) for s in size])
        delta = V[:, S | (1 << i) | (1 << j)] - V[:, S | (1 << i)] - V[:, S | (1 << j)] + V[:, S]
        return delta @ w


def load_all():
    train, test = tb.load_data()
    return train, test, pd.concat([train, test], ignore_index=True)


def explain_model(kind, train, explain_rows, explainer):
    model = tb.make_model(kind).fit(train[COLS], train[tb.TARGET])
    if kind == "xgboost":
        model.set_params(n_jobs=-1)   # threads only speed up prediction; the fitted trees are unchanged
    phi, V = explainer.explain(model.predict, explain_rows[COLS])
    pred = model.predict(explain_rows[COLS])
    return model, phi, V, pred


def day_over_day(rows, phi):
    """Differences between consecutive table rows that are back-to-back cycles.

    A pair counts only if row t-1 is the cycle immediately before row t (no
    excluded cycle or device gap in between)."""
    ends = pd.to_datetime(rows["cycle_end_utc"], utc=True, format="ISO8601").to_numpy()[:-1]
    starts = pd.to_datetime(rows["cycle_start_utc"], utc=True, format="ISO8601").to_numpy()[1:]
    adjacent = np.abs(starts - ends) <= np.timedelta64(1, "h")
    d = phi[1:] - phi[:-1]
    return d[adjacent], np.flatnonzero(adjacent) + 1


def global_shares(phi):
    imp = np.abs(phi).mean(0)
    return imp / imp.sum()


def move_shares(dphi):
    absd = np.abs(dphi)
    return (absd / absd.sum(1, keepdims=True)).mean(0)


def top_driver_freq(dphi, n_groups):
    return np.bincount(np.abs(dphi).argmax(1), minlength=n_groups) / len(dphi)


# same column order as the evaluated M1/M2 models (XGBoost column sampling is order-sensitive)
COLS = tb.FEATURE_SETS["set_b_full"]
assert sorted(COLS) == sorted(c for g in GROUPS.values() for c in g), "GROUPS must cover Set B exactly"


def formula_table(model, train):
    """Readable effects of the logit ridge: points per realistic change, at a typical day."""
    pipe = model.regressor_
    imputer, scaler, ridge = pipe[0], pipe[1], pipe[-1]
    beta = ridge.coef_ / scaler.scale_                       # logit units per raw unit
    coef = pd.Series(beta, index=COLS)
    z0 = ridge.intercept_ + ((imputer.statistics_ - scaler.mean_) / scaler.scale_ * ridge.coef_).sum()
    # slope of the sigmoid at the median training recovery: d points / d logit
    p_med = np.median(train[tb.TARGET]) / 100
    pts_per_logit = 100 * p_med * (1 - p_med)
    hrv_base = train["hrv_baseline"].median()
    effects = {
        # a +10 ms HRV night, own baseline unchanged: moves all three HRV columns together
        "HRV +10 ms above baseline": 10 * (coef["hrv_rmssd_milli"] + coef["hrv_vs_baseline"]
                                           + coef["hrv_pct_vs_baseline"] * 100 / hrv_base),
        "Resting HR +1 bpm above baseline": coef["resting_heart_rate"] + coef["rhr_vs_baseline"],
        "Respiratory rate +1 rpm above baseline": coef["respiratory_rate"] + coef["resp_rate_vs_baseline"],
        "Skin temp +0.5 C above baseline": 0.5 * (coef["skin_temp_celsius"] + coef["skin_temp_vs_baseline"]),
        "SpO2 +1 %": coef["spo2_percentage"],
        "Total sleep +1 h (need unchanged)": coef["total_sleep_hours"],
        "Sleep need +1 h (sleep unchanged)": coef["sleep_need_total_hours"],
        "Deep sleep +5 % of sleep (from light)": 5 * (coef["deep_sleep_pct"] - coef["light_sleep_pct"]),
        "REM sleep +5 % of sleep (from light)": 5 * (coef["rem_sleep_pct"] - coef["light_sleep_pct"]),
        "Sleep efficiency +5 %": 5 * coef["sleep_efficiency_pct"],
        "Disturbances +1 per hour": coef["disturbances_per_hour"],
        "Sleep consistency +10 %": 10 * coef["sleep_consistency_pct"],
        "Prior-day strain +5": 5 * coef["prior_strain"],
    }
    table = pd.DataFrame({"change": list(effects), "logit_units": list(effects.values())})
    table["points_at_typical_day"] = table["logit_units"] * pts_per_logit
    # personal baseline levels (HRV/RHR medians) deliberately left out: results/ is public
    meta = {"median_recovery": p_med * 100, "points_per_logit_unit": pts_per_logit,
            "intercept_logit_at_medians": z0}
    return table, coef, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip stability refits")
    args = parser.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(CONFIG["seed"])

    train, test, rows = load_all()
    bg_idx = rng.choice(len(train), CONFIG["background_size"], replace=False)
    background = train[COLS].iloc[bg_idx].to_numpy(dtype=float)
    explainer = GroupShapley(COLS, GROUPS, background)
    names = explainer.names
    print(f"Explaining {len(rows)} cycles ({len(train)} train + {len(test)} test), "
          f"{len(names)} groups, {2 ** len(names)} coalitions, background {len(background)} training cycles")

    per_cycle, shares, moves, fitted = [], [], [], {}
    for label, kind in MODELS.items():
        model, phi, V, pred = explain_model(kind, train, rows, explainer)
        B = V[:, 0].mean()  # v(empty) is the same for every row: the fixed baseline
        gap = np.abs(phi.sum(1) - (pred - V[:, 0])).max()
        # tolerance: XGBoost predicts in float32
        assert gap < 1e-3, f"{label}: contributions do not sum to prediction - B (gap {gap})"
        fitted[label] = (model, phi, V, pred, B)
        print(f"  {label}: B = {B:.2f} points; max |sum(phi) - (pred - B)| = {gap:.1e}")

        df = pd.DataFrame(phi, columns=names)
        df.insert(0, "model", label)
        df.insert(1, "cycle_id", rows["cycle_id"].to_numpy())
        df.insert(2, "is_test", rows["is_test"].to_numpy())
        df["baseline_B"], df["predicted"], df["actual"] = B, pred, rows[tb.TARGET].to_numpy()
        per_cycle.append(df)

        dphi, idx = day_over_day(rows, phi)
        dpred = pred[idx] - pred[idx - 1]
        assert np.allclose(dphi.sum(1), dpred, atol=1e-3), "differenced contributions must sum to the prediction change"
        shares.append(pd.DataFrame({"model": label, "group": names,
                                    "mean_abs_points": np.abs(phi).mean(0),
                                    "share_of_attribution": global_shares(phi)}))
        big = np.abs(dpred) >= 10
        moves.append(pd.DataFrame({"model": label, "group": names,
                                   "share_of_daily_moves": move_shares(dphi),
                                   "share_of_moves_10pt_plus": move_shares(dphi[big]),
                                   "top_driver_freq": top_driver_freq(dphi, len(names)),
                                   "n_moves": len(dphi), "n_moves_10pt_plus": int(big.sum())}))

    per_cycle = pd.concat(per_cycle, ignore_index=True)
    shares, moves = pd.concat(shares), pd.concat(moves)

    # --- agreement between the two models ---------------------------------
    pf, px = fitted["formula"][1], fitted["xgboost"][1]
    agree = pd.DataFrame({"group": names,
                          "per_cycle_corr": [np.corrcoef(pf[:, g], px[:, g])[0, 1] for g in range(len(names))]})
    dF, idx = day_over_day(rows, pf)
    dX, _ = day_over_day(rows, px)
    top_agree = float((np.abs(dF).argmax(1) == np.abs(dX).argmax(1)).mean())

    # --- interaction: does sleep matter less when HRV is high? -------------
    hrv_i, sleep_i = names.index("HRV"), names.index("Sleep amount vs need")
    inter_rows = []
    hrv_pct = rows["hrv_pct_vs_baseline"].to_numpy()
    terciles = pd.qcut(hrv_pct, 3, labels=["low HRV", "mid HRV", "high HRV"])
    for label, (model, phi, V, pred, B) in fitted.items():
        inter = explainer.pair_interaction(V, hrv_i, sleep_i)
        sleep_phi = phi[:, sleep_i]
        for t in ["low HRV", "mid HRV", "high HRV"]:
            m = np.asarray(terciles == t)
            slope = np.polyfit(rows["total_sleep_hours"].to_numpy()[m], sleep_phi[m], 1)[0]
            inter_rows.append({"model": label, "hrv_tercile": t, "n": int(m.sum()),
                               "sleep_points_per_hour": slope,
                               "mean_abs_hrv_sleep_interaction": np.abs(inter[m]).mean()})
        inter_rows.append({"model": label, "hrv_tercile": "all", "n": len(rows),
                           "sleep_points_per_hour": np.polyfit(rows["total_sleep_hours"], sleep_phi, 1)[0],
                           "mean_abs_hrv_sleep_interaction": np.abs(inter).mean()})
    interaction = pd.DataFrame(inter_rows)

    # --- the reconstructed formula ----------------------------------------
    table, coef, meta = formula_table(fitted["formula"][0], train)

    # --- stability: refit on CV-fold training sets and 80% subsamples ------
    stab = pd.DataFrame()
    if not args.quick:
        fits = [("fold", i, train.iloc[tr_idx]) for i, (tr_idx, _) in
                enumerate(tb.TimeSeriesSplit(**tb.CV_CONFIG).split(train))]
        n_sub = int(CONFIG["subsample_frac"] * len(train))
        fits += [("subsample", b, train.iloc[np.sort(rng.choice(len(train), n_sub, replace=False))])
                 for b in range(CONFIG["n_subsamples"])]
        stab_explainer = GroupShapley(COLS, GROUPS, background[:CONFIG["stability_background_size"]])
        stab_rows = []
        for kind_label, kind in MODELS.items():
            for src, i, tr in fits:
                _, phi, _, _ = explain_model(kind, tr, rows, stab_explainer)
                dphi, _ = day_over_day(rows, phi)
                for g, name in enumerate(names):
                    stab_rows.append({"model": kind_label, "refit": f"{src}_{i}", "group": name,
                                      "share_of_attribution": global_shares(phi)[g],
                                      "share_of_daily_moves": move_shares(dphi)[g]})
            print(f"  stability refits done for {kind_label} ({time.time() - t0:.0f}s)")
        stab = pd.DataFrame(stab_rows)
        spread = (stab.groupby(["model", "group"])[["share_of_attribution", "share_of_daily_moves"]]
                  .quantile([0.05, 0.5, 0.95]).unstack())
        spread.columns = [f"{m}_p{int(q * 100)}" for m, q in spread.columns]
        shares = shares.merge(spread.reset_index(), on=["model", "group"], how="left")

    # --- save ----------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    per_cycle.to_csv(PRIVATE_DIR / "m2_contributions.csv", index=False)
    shares.merge(moves, on=["model", "group"]).merge(agree, on="group").round(4) \
        .sort_values(["model", "share_of_attribution"], ascending=[True, False]) \
        .to_csv(RESULTS_DIR / "m2_attribution_summary.csv", index=False)
    interaction.round(4).to_csv(RESULTS_DIR / "m2_interaction.csv", index=False)
    table.round(4).to_csv(RESULTS_DIR / "m2_formula_effects.csv", index=False)
    coef.rename("logit_units_per_raw_unit").round(6).to_csv(RESULTS_DIR / "m2_formula_coefficients.csv")
    if len(stab):
        stab.round(4).to_csv(RESULTS_DIR / "m2_stability_refits.csv", index=False)
    meta.update({"baseline_B_formula": fitted["formula"][4], "baseline_B_xgboost": fitted["xgboost"][4],
                 "top_driver_agreement_daily_moves": top_agree, "n_daily_moves": len(dF),
                 "background_size": CONFIG["background_size"], "n_explained": len(rows)})
    pd.Series(meta).round(4).to_csv(RESULTS_DIR / "m2_attribution_meta.csv", header=["value"])

    print("\nShare of attribution (mean |points|):")
    print(shares.pivot(index="group", columns="model", values="share_of_attribution")
          .sort_values("formula", ascending=False).round(3).to_string())
    print("\nShare of day-over-day moves:")
    print(moves.pivot(index="group", columns="model", values="share_of_daily_moves")
          .sort_values("formula", ascending=False).round(3).to_string())
    print(f"\nTop-driver agreement on {len(dF)} daily moves: {top_agree:.1%}")
    print("\nFormula effects at a typical day:")
    print(table.round(2).to_string(index=False))
    print("\nHRV x sleep:")
    print(interaction.round(3).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
