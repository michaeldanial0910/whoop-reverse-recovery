"""M3 step 4: day archetypes. What kinds of days are there, and what follows each?

A "day" is one WHOOP cycle, described by three things (decided 2026-10-10):
  - strain                the day's strain (accrued AFTER that cycle's recovery score)
  - n_activities          logged activities that day, every sport type including walks
  - total_sleep_hours     the sleep that OPENED the day (not the night after: that night
                          drives the next score directly and would build the answer in)

Clustering: k-means on z-scored features, k = 2..6 chosen by silhouette score. Stability is
the adjusted Rand index between the full fit and refits on 80% subsamples.

Because a day's strain comes after its own score, each archetype is compared on the NEXT
cycle (decided 2026-10-10), descriptively and with no pass/fail:
  - raw next-cycle recovery
  - next-cycle residual from the formula model (out-of-fold, from m3_anomalies.py):
    does the archetype predict more than the sleep/HRV/prior strain it leads to?
  - pathway context: next night's sleep, next HRV vs baseline
CIs come from a moving-block bootstrap over time-ordered days (7-day blocks).

Run m3_anomalies.py first (for the residuals).
  python src/m3_archetypes.py
"""
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import m3_common as mc
import train_baselines as tb

FEATURES = ["strain", "n_activities", "total_sleep_hours"]
CONFIG = {"k_range": range(2, 7), "n_init": 50, "n_subsamples": 50, "subsample_frac": 0.8,
          "block_length_days": 7, "n_resamples": 2000,
          # the plan's "overreach to ~20 strain" case, checked separately because the clusters
          # need not isolate it
          "overreach_strain": 18.0}


def describe(centroid, overall):
    """Plain label from how a centroid sits against the overall medians."""
    s, a, h = centroid
    strain = "high strain" if s > overall["strain"] + 2 else "low strain" if s < overall["strain"] - 2 else "mid strain"
    acts = f"{a:.1f} activities"
    sleep = "long sleep" if h > overall["total_sleep_hours"] + 0.75 else \
        "short sleep" if h < overall["total_sleep_hours"] - 0.75 else "typical sleep"
    return f"{strain}, {acts}, {sleep}"


def block_ci(values, labels, k, rng):
    """95% CI of the per-archetype mean of `values`, moving-block bootstrap over days."""
    ok = ~np.isnan(values)
    draws = np.full((CONFIG["n_resamples"], k), np.nan)
    for r in range(CONFIG["n_resamples"]):
        i = mc.moving_block_indices(len(values), CONFIG["block_length_days"], rng)
        i = i[ok[i]]
        for c in range(k):
            m = labels[i] == c
            if m.any():
                draws[r, c] = values[i][m].mean()
    return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def main():
    df = mc.load_features()
    df = df[~df["is_forward"] & df["cycle_end_utc"].notna()].reset_index(drop=True)
    days = df.dropna(subset=FEATURES).reset_index(drop=True)
    X = StandardScaler().fit_transform(days[FEATURES])
    print(f"Days clustered: {len(days)} (train+test window, complete cycles)")

    sil = {}
    fits = {}
    for k in CONFIG["k_range"]:
        km = KMeans(k, n_init=CONFIG["n_init"], random_state=mc.SEED).fit(X)
        sil[k], fits[k] = silhouette_score(X, km.labels_), km
    k = max(sil, key=sil.get)
    labels = fits[k].labels_
    print("Silhouette by k:", {kk: round(v, 3) for kk, v in sil.items()}, "-> k =", k)

    rng = np.random.default_rng(mc.SEED)
    ari = []
    for _ in range(CONFIG["n_subsamples"]):
        idx = rng.choice(len(X), int(CONFIG["subsample_frac"] * len(X)), replace=False)
        sub = KMeans(k, n_init=10, random_state=mc.SEED).fit(X[idx])
        ari.append(adjusted_rand_score(labels[idx], sub.labels_))

    # --- next-cycle outcomes ------------------------------------------------
    allc = mc.load_features().set_index("cycle_id")
    res = pd.read_csv(mc.PRIVATE_DIR / "m3_oof_residuals.csv").set_index("cycle_id")["residual"]
    order = allc.sort_values("cycle_start_utc")
    next_of = pd.Series(order.index[1:], index=order.index[:-1])
    days["next_id"] = days["cycle_id"].map(next_of)
    nxt = allc.reindex(days["next_id"])
    valid = (nxt["prev_cycle_contiguous"].fillna(False).to_numpy().astype(bool)
             & nxt["usable"].fillna(False).to_numpy().astype(bool)
             & ~nxt["is_forward"].fillna(True).to_numpy().astype(bool))
    days["next_recovery"] = np.where(valid, nxt[tb.TARGET], np.nan)
    days["next_residual"] = np.where(valid, res.reindex(days["next_id"]).to_numpy(), np.nan)
    days["next_sleep_hours"] = np.where(valid, nxt["total_sleep_hours"], np.nan)
    days["next_hrv_pct_vs_baseline"] = np.where(valid, nxt["hrv_pct_vs_baseline"], np.nan)
    days["archetype"] = labels

    overall = days[FEATURES].median()
    centers = pd.DataFrame(StandardScaler().fit(days[FEATURES]).inverse_transform(fits[k].cluster_centers_),
                           columns=FEATURES)
    rows = []
    cis = {col: block_ci(days[col].to_numpy(float), labels, k, np.random.default_rng(mc.SEED))
           for col in ["next_recovery", "next_residual"]}
    for c in range(k):
        g = days[days["archetype"] == c]
        rows.append({"archetype": c, "label": describe(centers.loc[c].to_numpy(), overall),
                     "n_days": len(g), "share_of_days": len(g) / len(days),
                     "centroid_strain": centers.loc[c, "strain"],
                     "centroid_activities": centers.loc[c, "n_activities"],
                     "centroid_sleep_hours": centers.loc[c, "total_sleep_hours"],
                     "median_activity_minutes": g["activity_minutes"].median(),
                     "n_with_next": int(g["next_recovery"].notna().sum()),
                     "next_recovery_mean": g["next_recovery"].mean(),
                     "next_recovery_ci_lo": cis["next_recovery"][0][c], "next_recovery_ci_hi": cis["next_recovery"][1][c],
                     "next_residual_mean": g["next_residual"].mean(),
                     "next_residual_ci_lo": cis["next_residual"][0][c], "next_residual_ci_hi": cis["next_residual"][1][c],
                     "next_sleep_hours_mean": g["next_sleep_hours"].mean(),
                     "next_hrv_pct_vs_baseline_mean": g["next_hrv_pct_vs_baseline"].mean()})
    table = pd.DataFrame(rows).sort_values("centroid_strain").reset_index(drop=True)

    kw = {col: kruskal(*[days.loc[days.archetype == c, col].dropna() for c in range(k)]).pvalue
          for col in ["next_recovery", "next_residual"]}
    meta = pd.Series({"n_days": len(days), "k": k, **{f"silhouette_k{kk}": v for kk, v in sil.items()},
                      "ari_median": np.median(ari), "ari_p5": np.percentile(ari, 5),
                      "n_overreach_days": int((days["strain"] >= CONFIG["overreach_strain"]).sum()),
                      "overreach_next_recovery_mean": days.loc[days["strain"] >= CONFIG["overreach_strain"], "next_recovery"].mean(),
                      "overreach_next_residual_mean": days.loc[days["strain"] >= CONFIG["overreach_strain"], "next_residual"].mean(),
                      "overreach_n_archetypes_spanned": days.loc[days["strain"] >= CONFIG["overreach_strain"], "archetype"].nunique(),
                      "all_days_next_recovery_mean": days["next_recovery"].mean(),
                      "all_days_next_residual_mean": days["next_residual"].mean(),
                      "kruskal_p_next_recovery": kw["next_recovery"],
                      "kruskal_p_next_residual": kw["next_residual"]})

    mc.RESULTS_DIR.mkdir(exist_ok=True)
    table.round(3).to_csv(mc.RESULTS_DIR / "m3_archetypes.csv", index=False)
    meta.round(4).to_csv(mc.RESULTS_DIR / "m3_archetypes_meta.csv", header=["value"])
    days[["cycle_id", "cycle_start_local", *FEATURES, "activity_minutes", "archetype", "next_recovery",
          "next_residual", "next_sleep_hours", "next_hrv_pct_vs_baseline"]] \
        .to_csv(mc.PRIVATE_DIR / "m3_archetype_days.csv", index=False)

    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))
    print("\n", meta.round(3).to_string())


if __name__ == "__main__":
    main()
