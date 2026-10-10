"""M3 step 3: does an irregular sleep schedule go with lower recovery?

Pre-registered in preregistration_m3.json, which must be committed before this runs.

  TOTAL effect (headline, pass/fail): recovery ~ timing + sleep amount & quality + prior load
                                      + day of week. HRV/RHR are NOT controlled: an irregular
                                      schedule may lower recovery THROUGH lower HRV, and that
                                      pathway counts.
  DIRECT effect (reported alongside): same model + HRV and RHR. What timing adds beyond the
                                      physiology WHOOP measures directly.

Ordinary least squares in recovery points, so a coefficient reads "points per unit".
Neighbouring days are not independent, so CIs and p-values come from a moving-block
bootstrap (7-row blocks), and Holm corrects for testing three timing terms.

  python src/m3_irregularity.py
"""
import numpy as np
import pandas as pd

import m3_common as mc
import train_baselines as tb

PREREG_PATH = mc.ROOT / "preregistration_m3.json"
TIMING = ["onset_sd7_hours", "onset_shift_abs_hours", "onset_hours"]
CONTROLS_TOTAL = ["total_sleep_hours", "sleep_need_total_hours", "sleep_efficiency_pct",
                  "disturbances_per_hour", "prior_strain", "prior_nap_hours"] + tb.CALENDAR[1:]
# Monday (dow_0) is the reference day: dropping it avoids collinearity with the intercept
CONTROLS_DIRECT_ADD = ["hrv_rmssd_milli", "hrv_vs_baseline", "hrv_pct_vs_baseline",
                       "resting_heart_rate", "rhr_vs_baseline"]
MODELS = {"total": TIMING + CONTROLS_TOTAL, "direct": TIMING + CONTROLS_TOTAL + CONTROLS_DIRECT_ADD}


def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0][1:]


def main():
    prereg = mc.check_committed(PREREG_PATH)
    inf = prereg["inference"]
    rule = prereg["pass_rule_primary"]

    df = mc.analysis_rows(mc.load_features())
    df["onset_shift_abs_hours"] = df["onset_shift_hours"].abs()
    needed = sorted(set(MODELS["direct"]) | {tb.TARGET})
    data = df.dropna(subset=needed).reset_index(drop=True)   # same rows for both models
    print(f"Rows: {len(data)} of {len(df)} usable train+test cycles (complete cases, time order)")

    rows = []
    for label, cols in MODELS.items():
        X, y = data[cols].to_numpy(float), data[tb.TARGET].to_numpy(float)
        coef = ols(X, y)
        rng = np.random.default_rng(inf["seed"])
        draws = np.array([ols(X[i], y[i]) for i in
                          (mc.moving_block_indices(len(y), inf["block_length_rows"], rng)
                           for _ in range(inf["n_resamples"]))])
        k = len(TIMING)
        p = [mc.bootstrap_p(draws[:, j]) for j in range(k)]
        p_holm = mc.holm(p)
        for j, term in enumerate(TIMING):
            lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
            rows.append({"model": label, "term": term, "coef_points_per_unit": coef[j],
                         "ci95_lo": lo, "ci95_hi": hi, "p_boot": p[j], "p_holm": p_holm[j],
                         "sd_of_term": data[term].std(), "n": len(y)})
    res = pd.DataFrame(rows)
    res["points_per_1sd_of_term"] = res["coef_points_per_unit"] * res["sd_of_term"]

    prim = res[(res.model == "total") & (res.term == "onset_sd7_hours")].iloc[0]
    passed = (prim.coef_points_per_unit <= rule["max_coef_points_per_hour_sd"]
              and prim.p_holm < rule["max_holm_adjusted_p"])
    res["primary_verdict"] = np.where((res.model == "total") & (res.term == "onset_sd7_hours"),
                                      "PASS" if passed else "FAIL", "")

    desc = data[TIMING].describe().T[["mean", "std", "25%", "50%", "75%"]]
    corr = data[TIMING + ["total_sleep_hours", "hrv_vs_baseline"]].corr().loc[TIMING]

    mc.RESULTS_DIR.mkdir(exist_ok=True)
    res.round(4).to_csv(mc.RESULTS_DIR / "m3_irregularity_results.csv", index=False)
    pd.concat([desc, corr], axis=1).round(3).to_csv(mc.RESULTS_DIR / "m3_irregularity_timing_describe.csv")

    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"\nPRIMARY (total effect of onset_sd7_hours): coef {prim.coef_points_per_unit:.2f} points per +1 h SD, "
          f"95% CI [{prim.ci95_lo:.2f}, {prim.ci95_hi:.2f}], Holm p {prim.p_holm:.3f} -> "
          f"{'PASS' if passed else 'FAIL'} (needs <= {rule['max_coef_points_per_hour_sd']} and Holm p < "
          f"{rule['max_holm_adjusted_p']})")


if __name__ == "__main__":
    main()
