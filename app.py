"""Recovery, reverse-engineered: a local dashboard over YOUR OWN processed WHOOP data.

    pip install -r requirements.txt
    streamlit run app.py

Reads only local files (data/processed, results/). Nothing is uploaded or hosted.
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="Recovery, reverse-engineered", page_icon="🫀", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- design tokens
INK, INK_2, MUTED, GRID, SURFACE, PAGE = "#16202b", "#4a5866", "#7d8a97", "#e3e8ee", "#ffffff", "#f3f6f8"
UP, DOWN, TOTAL = "#2a78d6", "#eb6834", "#9aa7b4"          # raises / lowers the score / totals
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       # validated categorical slots 1-4
ZONE = {"green": "#16a36b", "yellow": "#d99a00", "red": "#d9443f"}


def zone(score):
    return "green" if score >= 67 else "yellow" if score >= 34 else "red"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Instrument Sans', system-ui, sans-serif; }}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 3.4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ font-family: 'Instrument Sans', system-ui, sans-serif; color: {INK}; letter-spacing: -0.01em; }}
.app-title {{ font-size: 2.1rem; font-weight: 700; color: {INK}; line-height: 1.1; margin: 0; letter-spacing: -0.025em; }}
.app-sub {{ color: {INK_2}; font-size: 0.98rem; margin: .35rem 0 1.4rem 0; max-width: 70ch; }}
.note {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.16rem; line-height: 1.55; color: {INK}; max-width: 68ch; }}
.note p {{ margin: 0 0 .55rem 0; }}
.score-wrap {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 14px; padding: 1.3rem 1.4rem 1.2rem; height: 100%; }}
.score-date {{ color: {INK_2}; font-size: .95rem; margin-bottom: .2rem; }}
.score {{ font-size: 5.2rem; font-weight: 700; line-height: 1; letter-spacing: -0.04em; }}
.score-unit {{ font-size: 1.6rem; font-weight: 600; margin-left: .15rem; }}
.score-line {{ color: {INK_2}; font-size: 1rem; margin-top: .7rem; }}
.score-line b {{ color: {INK}; }}
.chip {{ display: inline-block; font-size: .8rem; padding: .18rem .6rem; border-radius: 999px; margin: .55rem .3rem 0 0;
        background: {PAGE}; color: {INK_2}; border: 1px solid {GRID}; }}
.chip.warn {{ background: #fff4e5; color: #8a4b00; border-color: #f3d3a3; }}
.panel {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 14px; padding: 1.1rem 1.3rem; }}
.kicker {{ color: {INK_2}; font-size: .92rem; margin: 0 0 .4rem 0; }}
.big {{ font-size: 2.2rem; font-weight: 700; color: {INK}; line-height: 1.05; }}
.verdict {{ font-size: 1.05rem; color: {INK}; }}
.small {{ color: {MUTED}; font-size: .85rem; }}
div[data-testid="stTabs"] button p {{ font-size: 1rem; font-weight: 500; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {GRID}; }}
</style>
""", unsafe_allow_html=True)


@alt.theme.register("recovery", enable=True)
def chart_theme():
    return alt.theme.ThemeConfig({"config": {
        "background": "transparent",
        "font": "Instrument Sans, Helvetica Neue, Arial, sans-serif",
        "view": {"stroke": None},
        "axis": {"labelColor": INK_2, "titleColor": INK_2, "gridColor": GRID, "domain": False,
                 "tickColor": GRID, "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500},
        "legend": {"labelColor": INK_2, "titleColor": INK_2, "labelFontSize": 12, "titleFontSize": 12,
                   "orient": "top", "labelLimit": 320},
        "title": {"color": INK, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
    }})


def show(chart):
    st.altair_chart(chart, width="stretch")


def padded(values, pad=0.3):
    """x-scale with room on both sides for value labels outside the bars."""
    v = np.asarray(values, dtype=float)
    lo, hi = min(0.0, v.min()), max(0.0, v.max())
    span = max(hi - lo, 1.0)
    return alt.Scale(domain=[lo - (pad * span if lo < 0 else 0), hi + (pad * span if hi > 0 else 0)], nice=False)


def signed_labels(bars, field, fmt="+.1f"):
    """Value labels outside the bar end: right of positive bars, left of negative ones, in ink."""
    common = dict(color=INK_2, fontSize=12, fontWeight=500)
    text = alt.Text(f"{field}:Q", format=fmt)
    pos = bars.transform_filter(f"datum['{field}'] >= 0").mark_text(align="left", dx=5, **common).encode(
        text=text, color=alt.value(INK_2))
    neg = bars.transform_filter(f"datum['{field}'] < 0").mark_text(align="right", dx=-5, **common).encode(
        text=text, color=alt.value(INK_2))
    return pos + neg


# ---------------------------------------------------------------- data
missing = dd.missing_files()
if missing:
    st.markdown('<p class="app-title">Recovery, reverse-engineered</p>', unsafe_allow_html=True)
    st.error(f"Some processed files are missing: {', '.join(missing)}. Build them from your own WHOOP data first:")
    st.code(dd.RUN_ORDER, language="bash")
    st.stop()


@st.cache_resource(show_spinner="Fitting the formula model on your training period...")
def bundle():
    return dd.load_bundle()


@st.cache_data
def tables():
    return {"res": dd.residuals(), "days": dd.archetype_days(), "arche": dd.archetype_table(),
            "shares": dd.read("attribution summary"), "irr": dd.read("irregularity"),
            "bins": dd.read("anomaly bins"),
            "effects": pd.read_csv(dd.RES / "m2_formula_effects.csv"),
            "test": pd.read_csv(dd.RES / "m2_test_results.csv"),
            "anom": pd.read_csv(dd.RES / "m3_anomaly_summary.csv", index_col=0)["value"],
            "ameta": pd.read_csv(dd.RES / "m3_archetypes_meta.csv", index_col=0)["value"],
            "timing": pd.read_csv(dd.RES / "m3_irregularity_timing_describe.csv", index_col=0)}


@st.cache_data
def contributions(_b, idx_tuple):
    rows = _b.cycles.loc[list(idx_tuple)]
    return dd.explain_rows(_b, rows)


b = bundle()
T = tables()
cyc = b.cycles
by_day = cyc.drop_duplicates("day", keep="last").set_index("day")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Pick a morning")
    days = list(by_day.index)
    if "day" not in st.session_state:
        st.session_state.day = days[-1]
    c1, c2, c3 = st.columns([1, 1, 1])
    i = days.index(st.session_state.day)
    if c1.button("Earlier", width="stretch", disabled=i == 0):
        st.session_state.day = days[i - 1]
    if c2.button("Later", width="stretch", disabled=i == len(days) - 1):
        st.session_state.day = days[i + 1]
    if c3.button("Latest", width="stretch"):
        st.session_state.day = days[-1]
    st.select_slider("Morning", options=days, key="day", format_func=lambda d: d.strftime("%d %b %Y"),
                     label_visibility="collapsed")
    st.markdown(f'<p class="small">{len(cyc)} scored mornings, {cyc["day"].min():%b %Y} to '
                f'{cyc["day"].max():%d %b %Y}. The formula was fitted only on mornings before '
                f'{b.cycles.loc[b.cycles.split == "test", "day"].min():%d %b %Y}; everything after that is '
                f'new to it.</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Refresh")
    st.markdown('<p class="small">Run the daily sync, then rebuild the processed files. This app only reads '
                'files on your machine.</p>', unsafe_allow_html=True)
    st.code("bash scripts/refresh.sh", language="bash")

row = by_day.loc[st.session_state.day]
idx = cyc.index[(cyc["day"] == st.session_state.day)][-1]
phi = contributions(b, (idx,)).iloc[0]

# ---------------------------------------------------------------- header
st.markdown('<div class="app-title">Recovery, reverse-engineered</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-sub">WHOOP tells you your recovery and names the reasons in words. This rebuilds '
            f'the score from its inputs, so every morning splits into points you can check. Built on '
            f'{len(cyc)} of your mornings.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- waterfall
def waterfall(phi, B, pred, actual=None, top=6, height=330):
    order = phi.abs().sort_values(ascending=False).index
    keep, rest = list(order[:top]), list(order[top:])
    steps = [("Typical morning", B, "total")]
    steps += [(g, phi[g], "up" if phi[g] >= 0 else "down") for g in keep]
    if rest:
        v = phi[rest].sum()
        steps.append((f"{len(rest)} smaller inputs", v, "up" if v >= 0 else "down"))
    rows, run = [], 0.0
    for k, (name, v, kind) in enumerate(steps):
        if kind == "total":
            start, end = 0.0, v
        else:
            start, end = run, run + v
        run = end
        rows.append({"step": name, "start": start, "end": end, "kind": kind, "order": k,
                     "label": f"{v:+.1f}" if kind != "total" else f"{v:.1f}"})
    rows.append({"step": "Model's estimate", "start": 0.0, "end": pred, "kind": "total", "order": len(rows),
                 "label": f"{pred:.1f}"})
    d = pd.DataFrame(rows)
    # zoom the axis to where the action is; totals are drawn from the left edge
    pts = list(d.loc[d.kind != "total", ["start", "end"]].to_numpy().ravel()) + [B, pred]
    if actual is not None:
        pts.append(actual)
    x0 = max(0, np.floor((min(pts) - 6) / 5) * 5)
    x1 = min(100, np.ceil((max(pts) + 6) / 5) * 5)
    d.loc[d.kind == "total", "start"] = x0
    d["lo"], d["hi"] = d[["start", "end"]].min(axis=1), d[["start", "end"]].max(axis=1)
    sort = list(d["step"])
    color = alt.Color("kind:N", scale=alt.Scale(domain=["up", "down", "total"], range=[UP, DOWN, TOTAL]),
                      legend=None)
    base = alt.Chart(d).encode(y=alt.Y("step:N", sort=sort, title=None, axis=alt.Axis(labelLimit=220)))
    bars = base.mark_bar(height=18, cornerRadius=4).encode(
        x=alt.X("lo:Q", title="recovery points", scale=alt.Scale(domain=[x0, x1], clamp=True)), x2="hi:Q",
        color=color,
        tooltip=[alt.Tooltip("step:N", title="Input"), alt.Tooltip("label:N", title="Points")])
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=12, fontWeight=500).encode(
        x="hi:Q", text="label:N")
    layers = [bars, labels]
    if actual is not None:
        a = pd.DataFrame({"x": [actual], "t": [f"WHOOP said {actual:.0f}"]})
        layers.append(alt.Chart(a).mark_rule(color=ZONE[zone(actual)], strokeWidth=2.5, strokeDash=[5, 3])
                      .encode(x="x:Q"))
        layers.append(alt.Chart(a).mark_text(align="left", dx=5, dy=-8, color=ZONE[zone(actual)], fontWeight=600,
                                             fontSize=12).encode(x="x:Q", y=alt.value(0), text="t:N"))
    return alt.layer(*layers).properties(height=height)


# ---------------------------------------------------------------- hero
anom_row = T["res"].set_index("cycle_id").reindex([row["cycle_id"]]).iloc[0]
thr = T["anom"]["threshold_points"]
actual, pred = row["recovery_score"], row["predicted"]
miss = actual - pred
split_note = {"train": "a morning the model learned from",
              "test": "held out: the model never trained on it",
              "forward": "new since the model was frozen"}[row["split"]]

left, right = st.columns([1, 2.1], gap="large")
with left:
    chips = [f'<span class="chip">{split_note}</span>']
    if abs(miss) > thr:
        chips.append(f'<span class="chip warn">⚠ surprise: off by {abs(miss):.0f} pts</span>')
    if dd.guardrail_warnings(row):
        chips.append('<span class="chip warn">⚠ formula is weak in this range</span>')
    prev = T["days"].set_index("cycle_id")
    prev_idx = cyc.index.get_loc(idx) - 1
    if prev_idx >= 0:
        pid = cyc.iloc[prev_idx]["cycle_id"]
        if pid in prev.index:
            pa = prev.loc[pid]
            arche_name = dd.classify_day(T["arche"], pa["strain"], pa["n_activities"], pa["total_sleep_hours"],
                                         T["days"])["name"]
            chips.append(f'<span class="chip">yesterday: {arche_name.lower()}</span>')
    st.markdown(f"""
<div class="score-wrap">
  <div class="score-date">{row['sleep_end_local']:%A, %d %B %Y}</div>
  <div class="score" style="color:{ZONE[zone(actual)]}">{actual:.0f}<span class="score-unit">%</span></div>
  <div class="score-line">The formula expected <b>{pred:.0f}</b>, so it was
  <b>{'spot on' if abs(miss) < 1.5 else f'{abs(miss):.0f} points too ' + ('low' if miss > 0 else 'high')}</b>.</div>
  <div class="score-line">Slept <b>{row['total_sleep_hours']:.1f} h</b> of a <b>{row['sleep_need_total_hours']:.1f} h</b> need.
  HRV <b>{row['hrv_rmssd_milli']:.0f} ms</b>, {row['hrv_pct_vs_baseline']:+.0f}% vs your 30-day baseline.</div>
  {''.join(chips)}
</div>""", unsafe_allow_html=True)
with right:
    st.markdown(f'<p class="kicker">How the estimate was built: start from a typical morning ({b.B:.0f}), '
                f'then each input pushes it up or down</p>', unsafe_allow_html=True)
    show(waterfall(phi, b.B, pred, actual))

# ---------------------------------------------------------------- since the previous morning
pos = cyc.index.get_loc(idx)
prev_ok = pos > 0 and bool(row["prev_cycle_contiguous"]) and \
    abs((cyc.iloc[pos]["cycle_start_utc"] - cyc.iloc[pos - 1]["cycle_end_utc"]).total_seconds()) < 3600
if prev_ok:
    pidx = cyc.index[pos - 1]
    both = contributions(b, (pidx, idx))
    dphi = (both.loc[idx] - both.loc[pidx])
    prow = cyc.loc[pidx]
    d_actual = actual - prow["recovery_score"]
    d_pred = pred - prow["predicted"]
    top = dphi.reindex(dphi.abs().sort_values(ascending=False).index)
    shown = top.iloc[:5]
    other = top.iloc[5:].sum()
    dd_ = shown.rename("points").reset_index().rename(columns={"index": "group"})
    if abs(other) >= 0.05:
        dd_ = pd.concat([dd_, pd.DataFrame({"group": ["Everything else"], "points": [other]})])
    dd_["kind"] = np.where(dd_["points"] >= 0, "up", "down")
    lead = top.index[0]
    rest = d_pred - top.iloc[0]
    if abs(d_pred) < 1 or np.sign(top.iloc[0]) != np.sign(d_pred):
        move_text = (f"The formula barely moved ({d_pred:+.1f}): <b>{lead.lower()}</b> pulled it "
                     f"{'up' if top.iloc[0] > 0 else 'down'} {abs(top.iloc[0]):.1f}, and the other inputs "
                     f"pushed back {rest:+.1f}.")
    else:
        move_text = (f"The formula moved {d_pred:+.1f}, led by <b>{lead.lower()}</b> ({top.iloc[0]:+.1f}); "
                     f"everything else added {rest:+.1f}.")
    if abs(d_actual - d_pred) > 8:
        move_text += f" WHOOP moved {abs(d_actual - d_pred):.0f} points more than the formula can account for."
    st.write("")
    c1, c2 = st.columns([1, 2.1], gap="large")
    with c1:
        st.markdown(f"""<div class="panel"><p class="kicker">Since the previous morning</p>
<div class="big">{d_actual:+.0f} pts</div>
<div class="verdict" style="margin-top:.45rem">WHOOP went {prow['recovery_score']:.0f} → {actual:.0f}. {move_text}</div>
<p class="small" style="margin-top:.6rem">This is the move the in-app coach explains in words, split into points.
The reference cancels out, so these numbers don't depend on it.</p></div>""", unsafe_allow_html=True)
    with c2:
        ch = alt.Chart(dd_).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("group:N", sort=list(dd_["group"]), title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("points:Q", title="change in points since the previous morning", scale=padded(dd_["points"])),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["up", "down"], range=[UP, DOWN]), legend=None),
            tooltip=[alt.Tooltip("group:N", title="Input"), alt.Tooltip("points:Q", title="Points", format="+.1f")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_2).encode(x="x:Q")
        show((zero + ch + signed_labels(ch, "points")).properties(height=alt.Step(30),
                                                                   title="What changed, in points"))

# ---------------------------------------------------------------- one-minute summary
sh = T["shares"][T["shares"].model == "formula"].set_index("group")
top2 = sh["share_of_attribution"].sort_values(ascending=False).head(2)
irr = T["irr"].set_index(["model", "term"])
prim = irr.loc[("total", "onset_sd7_hours")]
ar_sorted = T["arche"].sort_values("next_recovery_mean")
best_day, worst_day = ar_sorted.iloc[-1], ar_sorted.iloc[0]
gap = best_day.next_recovery_mean - worst_day.next_recovery_mean
busy_explained = float(T["ameta"]["kruskal_p_next_residual"]) >= 0.05
gap_tail = ("all of that runs through the night and HRV that follow it" if busy_explained
            else "part of that gap is still there after the formula sees the next night")
irr_lo, irr_hi, irr_coef = float(prim.ci95_lo), float(prim.ci95_hi), float(prim.coef_points_per_unit)
iqr = T["timing"].loc["onset_sd7_hours", "75%"] - T["timing"].loc["onset_sd7_hours", "25%"]
if irr_lo <= 0 <= irr_hi:
    irr_head = "No measurable cost"
    irr_line = (f"An irregular bedtime showed <b>no measurable cost</b> once sleep length and quality are counted: "
                f"at most ~{abs(min(irr_lo, 0)) * iqr:.1f} points for a typical swing.")
elif irr_hi < 0:
    irr_head = "A measurable cost"
    irr_line = (f"An irregular bedtime <b>costs points</b> even with sleep length and quality counted: about "
                f"{abs(irr_coef) * iqr:.1f} points for a typical swing.")
else:
    irr_head = "No cost, if anything a gain"
    irr_line = (f"A more irregular bedtime went with a <b>slightly higher</b> score once sleep length and quality "
                f"are counted (+{irr_coef * iqr:.1f} points for a typical swing).")
st.markdown("#### In one minute")
st.markdown(f"""<div class="note">
<p>Your score is mostly two things: <b>{top2.index[0]}</b> ({top2.iloc[0]:.0%} of the movement)
and <b>{top2.index[1]}</b> ({top2.iloc[1]:.0%}).
{"Everything else is small change." if top2.sum() >= 0.6 else "The rest is spread across several smaller inputs."}</p>
<p>A {worst_day['name'].lower()} is followed by a score about <b>{gap:.0f} points lower</b> than a
{best_day['name'].lower()}, and {gap_tail}.</p>
<p>{irr_line}</p>
</div>""", unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- tabs
tabs = st.tabs(["What moves the score", "Track record", "Kinds of days", "Surprises", "Bedtime",
                "What if", "How this works"])

# ---- 1. what moves the score
with tabs[0]:
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        s = T["shares"].copy()
        s["model"] = s["model"].map({"formula": "Formula (readable)", "xgboost": "XGBoost (flexible)"})
        order = list(s[s.model == "Formula (readable)"].sort_values("share_of_attribution", ascending=False)["group"])
        ch = alt.Chart(s).mark_bar(cornerRadiusEnd=4, height=11).encode(
            y=alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
            yOffset=alt.YOffset("model:N"),
            x=alt.X("share_of_attribution:Q", title="share of what moves the score", axis=alt.Axis(format="%")),
            color=alt.Color("model:N", scale=alt.Scale(range=[UP, DOWN]), title=None),
            tooltip=[alt.Tooltip("group:N", title="Input"), alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("share_of_attribution:Q", title="Share", format=".1%"),
                     alt.Tooltip("mean_abs_points:Q", title="Avg points", format=".1f")])
        show(ch.properties(height=420, title="Two independent models agree on the ranking"))
        st.markdown('<p class="small">Share of the average absolute contribution, against the same reference '
                    f'morning (B = {b.B:.1f}). Ranks were identical in all 25 refits. Shares under ~5% are minor, '
                    'not ranked.</p>', unsafe_allow_html=True)
    with c2:
        e = T["effects"].copy()
        e = e[e["change"].isin(["HRV +10 ms above baseline", "Total sleep +1 h (need unchanged)",
                                "Sleep need +1 h (sleep unchanged)", "Prior-day strain +5",
                                "Resting HR +1 bpm above baseline", "Respiratory rate +1 rpm above baseline",
                                "Skin temp +0.5 C above baseline"])]
        e["kind"] = np.where(e["points_at_typical_day"] >= 0, "up", "down")
        ch = alt.Chart(e).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("change:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=260)),
            x=alt.X("points_at_typical_day:Q", title="points on a typical morning",
                    scale=padded(e["points_at_typical_day"])),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["up", "down"], range=[UP, DOWN]), legend=None),
            tooltip=[alt.Tooltip("change:N", title="Change"),
                     alt.Tooltip("points_at_typical_day:Q", title="Points", format="+.1f")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_2).encode(x="x:Q")
        show((zero + ch + signed_labels(ch, "points_at_typical_day")).properties(height=alt.Step(34),
                                                                             title="What one change is worth"))
        st.markdown('<p class="small">From the readable formula, at a typical morning (~65). Effects shrink '
                    'near 0 and 100.</p>', unsafe_allow_html=True)
        st.markdown(f"""<div class="panel"><p class="kicker">Found exactly, not estimated</p>
<div class="verdict">WHOOP's sleep-need credit for a nap equals the previous day's nap length
on {dd.nap_rule_check(cyc)}.</div></div>""", unsafe_allow_html=True)

# ---- 2. track record
with tabs[1]:
    test = T["test"].set_index("model")
    fw = cyc[cyc.split == "forward"]
    fw_mae = (fw["recovery_score"] - fw["predicted"]).abs().mean() if len(fw) else np.nan
    oof = T["res"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Held-out error", f"{test.loc['logit_ridge__set_b_full', 'mae']:.1f} pts",
              help=f"Mean absolute error on the {int((cyc.split == 'test').sum())} held-out mornings.")
    k2.metric("Held-out R²", f"{test.loc['logit_ridge__set_b_full', 'r2']:.2f}")
    k3.metric("New mornings since freeze", f"{fw_mae:.1f} pts" if len(fw) else "none yet",
              help=f"Mean absolute error on the {len(fw)} mornings that arrived after the model was frozen. "
                   "Few mornings, so this number moves a lot; it is the most honest check there is.")
    k4.metric("Guessing your average", f"{test.loc['naive_mean', 'mae']:.1f} pts",
              help="What you'd get by always predicting your average score.")
    win = st.radio("Window", ["Last 60 mornings", "Last 6 months", "Everything"], horizontal=True,
                   label_visibility="collapsed")
    d = cyc[["day", "recovery_score", "predicted", "split", "cycle_id"]].copy()
    d = d.merge(oof[["cycle_id", "flagged"]], on="cycle_id", how="left")
    d["flagged"] = d["flagged"].astype("boolean").fillna(False).astype(bool)
    if win == "Last 60 mornings":
        d = d.tail(60)
    elif win == "Last 6 months":
        d = d[d.day >= d.day.max() - pd.Timedelta("183D")]
    d["miss"] = d["recovery_score"] - d["predicted"]
    spans = d.groupby("split")["day"].agg(["min", "max"]).reset_index()
    spans["label"] = spans["split"].map({"train": "learned from", "test": "held out", "forward": "new"})
    band = alt.Chart(spans[spans.split != "train"]).mark_rect(opacity=0.045, color=INK).encode(
        x="min:T", x2="max:T")
    bandtxt = alt.Chart(spans).mark_text(align="left", dx=4, dy=8, color=MUTED, fontSize=11).encode(
        x="min:T", y=alt.value(0), text="label:N")
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["day"], empty=False)
    xenc = alt.X("day:T", title=None)
    line = alt.Chart(d).mark_line(color=UP, strokeWidth=2).encode(x=xenc, y=alt.Y(
        "predicted:Q", title="recovery", scale=alt.Scale(domain=[0, 100])))
    dots = alt.Chart(d).mark_circle(size=34, opacity=0.9).encode(
        x=xenc, y="recovery_score:Q",
        color=alt.Color("recovery_score:Q", scale=alt.Scale(domain=[0, 33.5, 34, 66.5, 67, 100],
                        range=[ZONE["red"], ZONE["red"], ZONE["yellow"], ZONE["yellow"], ZONE["green"],
                               ZONE["green"]]), legend=None))
    ring = alt.Chart(d[d.flagged]).mark_point(size=190, shape="circle", color=INK, strokeWidth=1.6).encode(
        x=xenc, y="recovery_score:Q")
    hover = alt.Chart(d).mark_rule(color=MUTED).encode(
        x=xenc, opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("day:T", title="Morning", format="%a %d %b %Y"),
                 alt.Tooltip("recovery_score:Q", title="WHOOP", format=".0f"),
                 alt.Tooltip("predicted:Q", title="Formula", format=".0f"),
                 alt.Tooltip("miss:Q", title="Miss", format="+.0f")]).add_params(nearest)
    show(alt.layer(band, bandtxt, line, dots, ring, hover).properties(
        height=380, title="WHOOP's score (dots) against the formula (line); ringed mornings are surprises"))
    st.markdown('<p class="small">Mornings the model learned from are predicted out-of-sample in the '
                'Surprises tab; this line is the single frozen model, so it fits the learned stretch more '
                'closely than it would fit unseen days.</p>', unsafe_allow_html=True)

# ---- 3. kinds of days
with tabs[2]:
    a = T["arche"]
    dd_days = T["days"].copy()
    lab_to_name = dict(zip(a["archetype"], a["name"]))
    dd_days["kind"] = dd_days["archetype"].map(lab_to_name)
    names = list(a["name"])
    cscale = alt.Scale(domain=names, range=SERIES[:len(names)])
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        ch = alt.Chart(dd_days).mark_circle(opacity=0.75, stroke="white", strokeWidth=0.6).encode(
            x=alt.X("strain:Q", title="the day's strain"),
            y=alt.Y("total_sleep_hours:Q", title="hours slept the night before"),
            size=alt.Size("n_activities:Q", scale=alt.Scale(range=[30, 420]), legend=None),
            color=alt.Color("kind:N", scale=cscale, title=None),
            tooltip=[alt.Tooltip("cycle_start_local:T", title="Day", format="%a %d %b %Y"),
                     alt.Tooltip("kind:N", title="Kind"), alt.Tooltip("strain:Q", format=".1f"),
                     alt.Tooltip("n_activities:Q", title="Activities"),
                     alt.Tooltip("total_sleep_hours:Q", title="Slept (h)", format=".1f"),
                     alt.Tooltip("next_recovery:Q", title="Next morning", format=".0f")])
        show(ch.properties(height=420, title=f"{len(dd_days)} days fall into {len(a)} kinds (bigger dot = more activities)"))
        st.markdown(f'<p class="small">k-means on strain, activity count and the sleep that opened the day; '
                    f'k = {int(T["ameta"]["k"])} by silhouette ({T["ameta"]["silhouette_k4"]:.2f}, modest structure). '
                    f'Very stable: the same days group together in {T["ameta"]["ari_median"]:.0%} agreement '
                    f'across 50 refits.</p>', unsafe_allow_html=True)
    with c2:
        def ci_chart(mean, lo, hi, title, xtitle, ref=None):
            d = a.assign(mean=a[mean], lo=a[lo], hi=a[hi])
            base = alt.Chart(d).encode(y=alt.Y("name:N", sort=names, title=None, axis=alt.Axis(labelLimit=180)),
                                       color=alt.Color("name:N", scale=cscale, legend=None))
            layers = []
            if ref is not None:
                layers.append(alt.Chart(pd.DataFrame({"x": [ref]})).mark_rule(color=MUTED, strokeDash=[4, 3])
                              .encode(x="x:Q"))
            layers += [base.mark_rule(strokeWidth=3).encode(x=alt.X("lo:Q", title=xtitle,
                                                                    scale=alt.Scale(zero=False)), x2="hi:Q"),
                       base.mark_circle(size=140, opacity=1).encode(
                           x="mean:Q", tooltip=[alt.Tooltip("name:N", title="Kind"),
                                                alt.Tooltip("mean:Q", title="Average", format=".1f"),
                                                alt.Tooltip("lo:Q", title="95% low", format=".1f"),
                                                alt.Tooltip("hi:Q", title="95% high", format=".1f")]),
                       base.mark_text(dy=-14, color=INK_2, fontSize=11).encode(
                           x="mean:Q", text=alt.Text("mean:Q", format=".1f"))]
            return alt.layer(*layers).properties(height=250, title=title)
        show(ci_chart("next_recovery_mean", "next_recovery_ci_lo", "next_recovery_ci_hi",
                      "Next morning's score", "average with 95% interval"))
        show(ci_chart("next_residual_mean", "next_residual_ci_lo", "next_residual_ci_hi",
                      "Left over after the formula", "points the formula couldn't explain",
                      ref=float(T["ameta"]["all_days_next_residual_mean"])))
        st.markdown(f"""<div class="panel"><div class="verdict">The next morning after a {worst_day['name'].lower()} scores about
<b>{gap:.0f} points</b> below one after a {best_day['name'].lower()}.
{"Nothing is left over once the formula sees the next night's sleep and HRV" if busy_explained
 else "Some of it is left over even after the formula sees the next night's sleep and HRV"}
(p = {float(T['ameta']['kruskal_p_next_residual']):.2f}).</div></div>""",
                    unsafe_allow_html=True)
    show_t = a[["name", "n_days", "centroid_strain", "centroid_activities", "centroid_sleep_hours",
                "next_recovery_mean", "next_sleep_hours_mean", "next_hrv_pct_vs_baseline_mean"]].rename(columns={
        "name": "Kind", "n_days": "Days", "centroid_strain": "Strain", "centroid_activities": "Activities",
        "centroid_sleep_hours": "Slept before (h)", "next_recovery_mean": "Next score",
        "next_sleep_hours_mean": "Next night (h)", "next_hrv_pct_vs_baseline_mean": "Next HRV vs baseline (%)"})
    st.dataframe(show_t.round(1), hide_index=True, width="stretch")

# ---- 4. surprises
with tabs[3]:
    an = T["anom"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Surprising mornings", f"{int(an['n_flagged'])} of {int(an['n_cycles'])}",
              help=f"Off by more than {thr:.1f} points (2.5 × a robust estimate of the typical miss).")
    k2.metric("WHOOP scored lower than expected", f"{int(an['n_worse_than_expected'])} of {int(an['n_flagged'])}")
    k3.metric("Typical size of a surprise", f"{an['median_abs_residual_flagged']:.0f} pts")
    st.write("")
    bins = T["bins"].copy()

    def binlab(s, first, last):
        lo, hi = (float(v) for v in s.strip("(]").split(", "))
        if first:
            return f"< {hi:g}"
        if last and hi in (24, 1000):
            return f"> {lo:g}"
        return f"{lo:g} to {hi:g}"
    bins["label"] = [binlab(v, i == 0 or bins["input"].iloc[i] != bins["input"].iloc[i - 1],
                            i == len(bins) - 1 or bins["input"].iloc[i] != bins["input"].iloc[i + 1])
                     for i, v in enumerate(bins["bin"])]
    bins["kind"] = np.where(bins["mean_residual"] >= 0, "up", "down")
    c1, c2 = st.columns(2, gap="large")
    for col, inp, title, xt in [(c1, "total_sleep_hours", "By hours slept", "hours slept"),
                                (c2, "hrv_pct_vs_baseline", "By HRV against your baseline", "HRV, % vs baseline")]:
        g = bins[bins.input == inp]
        ch = alt.Chart(g).mark_bar(cornerRadius=4, width={"band": 0.7}).encode(
            x=alt.X("label:N", sort=list(g["label"]), title=xt, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean_residual:Q", title="WHOOP minus formula (points)"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["up", "down"], range=[UP, DOWN]), legend=None),
            tooltip=[alt.Tooltip("label:N", title="Range"), alt.Tooltip("n:Q", title="Mornings"),
                     alt.Tooltip("mean_residual:Q", title="Avg miss", format="+.1f"),
                     alt.Tooltip("share_flagged:Q", title="Surprises", format=".0%")])
        col.altair_chart(ch.properties(height=260, title=title), width="stretch")
    st.markdown(f'<div class="note"><p>{dd.anomaly_pattern_text(T["bins"])}</p></div>', unsafe_allow_html=True)
    flagged = oof[oof.flagged].sort_values("cycle_start_local", ascending=False)
    if len(flagged):
        opts = {f"{r.cycle_start_local:%a %d %b %Y}: WHOOP {r.actual:.0f}, formula {r.predicted:.0f}": r.cycle_id
                for r in flagged.itertuples()}
        pick = st.selectbox("Look inside a surprising morning", list(opts))
        fid = opts[pick]
        fr = flagged.set_index("cycle_id").loc[fid]
        fphi = pd.Series({g: fr[f"pts_{g}"] for g in dd.GROUPS})
        st.markdown(f'<p class="kicker">Out-of-sample estimate from the model that never saw this stretch. '
                    f'Most unusual input: {fr["most_unusual_input"].replace("_", " ")} '
                    f'({fr["most_unusual_robust_z"]:.1f}× its usual spread).</p>', unsafe_allow_html=True)
        show(waterfall(fphi, fr["baseline_B"], fr["predicted"], fr["actual"], height=300))

# ---- 5. bedtime
with tabs[4]:
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        on = cyc["sleep_start_local"]
        hrs = (on.dt.hour + on.dt.minute / 60).dropna()
        clock = pd.DataFrame({"hour": range(24)})
        clock["n"] = clock["hour"].map(np.floor(hrs).astype(int).value_counts()).fillna(0)
        clock["label"] = clock["hour"].map(lambda h: f"{h:02d}:00")
        clock["t0"] = clock["hour"] / 24 * 2 * np.pi + 0.012
        clock["t1"] = (clock["hour"] + 1) / 24 * 2 * np.pi - 0.012
        clock["tm"] = (clock["hour"] + 0.5) / 24 * 2 * np.pi
        r_in, r_out = 46, 168
        clock["r1"] = r_in + (r_out - r_in) * np.sqrt(clock["n"] / clock["n"].max())
        ring = alt.Chart(clock).mark_arc(color=GRID, innerRadius=r_in, outerRadius=r_out, opacity=0.35).encode(
            theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q")
        arc = alt.Chart(clock[clock.n > 0]).mark_arc(color=UP, innerRadius=r_in, cornerRadius=3).encode(
            theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q", radius=alt.Radius("r1:Q", scale=None),
            tooltip=[alt.Tooltip("label:N", title="Fell asleep"), alt.Tooltip("n:Q", title="Nights")])
        txt = alt.Chart(clock[clock.hour % 3 == 0]).mark_text(radius=r_out + 16, color=MUTED, fontSize=11).encode(
            theta=alt.Theta("tm:Q", scale=None), text="label:N")
        arc = ring + arc
        show((arc + txt).properties(height=400, width=400,
                                    title="When you fall asleep (24-hour clock, midnight at top)"))
        med = cyc["onset_hours"].median() % 24
        st.markdown(f'<p class="small">Median bedtime {int(med):02d}:{int((med % 1) * 60):02d}. Typical '
                    f'night-to-night spread over a week: {cyc["onset_sd7_hours"].median():.1f} h.</p>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="panel"><p class="kicker">Pre-registered test, decided before looking</p>
<div class="big">{irr_head}</div>
<div class="verdict" style="margin-top:.5rem">Once sleep length and quality were counted:
{prim.coef_points_per_unit:+.1f} points per extra hour of spread (95% CI
{prim.ci95_lo:.1f} to {prim.ci95_hi:+.1f}). Pre-registered verdict: {prim.primary_verdict}.</div></div>""",
                    unsafe_allow_html=True)
        st.write("")
        r = T["irr"].copy()
        r["term"] = r["term"].map({"onset_sd7_hours": "Bedtime spread (week)",
                                   "onset_shift_abs_hours": "Shift from last night",
                                   "onset_hours": "Later bedtime"})
        r["model"] = r["model"].map({"total": "Sleep held fixed", "direct": "Sleep, HRV and RHR held fixed"})
        base = alt.Chart(r).encode(y=alt.Y("term:N", title=None, axis=alt.Axis(labelLimit=220), sort=["Bedtime spread (week)",
                                                                       "Shift from last night", "Later bedtime"]),
                                   yOffset="model:N",
                                   color=alt.Color("model:N", scale=alt.Scale(range=[UP, DOWN]), title=None))
        ch = alt.layer(
            alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_2).encode(x="x:Q"),
            alt.Chart(pd.DataFrame({"x": [-2]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q"),
            base.mark_rule(strokeWidth=3).encode(x=alt.X("ci95_lo:Q", title="points per hour"), x2="ci95_hi:Q"),
            base.mark_circle(size=110, opacity=1).encode(
                x="coef_points_per_unit:Q",
                tooltip=[alt.Tooltip("term:N", title="Timing"), alt.Tooltip("model:N", title="Model"),
                         alt.Tooltip("coef_points_per_unit:Q", title="Points per hour", format="+.2f"),
                         alt.Tooltip("p_holm:Q", title="Holm p", format=".2f")]))
        show(ch.properties(height=260, title="Every timing effect sits near zero (dashed line: pass mark)"))

# ---- 6. what if
with tabs[5]:
    st.markdown(f'<p class="kicker">Start from {row["sleep_end_local"]:%A %d %B} and change one thing at a time. '
                'The formula recomputes the score.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        base_sleep, base_need = row["total_sleep_hours"], row["sleep_need_total_hours"]
        base_strain = row["prior_strain"] if pd.notna(row["prior_strain"]) else 10.0
        WI = {"wi_sleep": 0.0, "wi_need": 0.0, "wi_hrv": 0, "wi_rhr": 0, "wi_strain": 0.0}

        def preset(**kw):
            for k, v in WI.items():
                st.session_state[k] = kw.get(k, v)
        p1, p2 = st.columns(2)
        p3, p4 = st.columns(2)
        p1.button("One more hour of sleep", on_click=preset, kwargs={"wi_sleep": 1.0}, width="stretch")
        p2.button("About a 4-hour night", on_click=preset, kwargs={"wi_sleep": float(np.clip(round((4 - base_sleep) * 4) / 4, -6, 4))},
                  width="stretch")
        p3.button("Lighter yesterday", on_click=preset, kwargs={"wi_strain": -5.0}, width="stretch")
        p4.button("Reset", on_click=preset, width="stretch")
        ds = st.slider(f"Sleep: change from {base_sleep:.1f} h", -6.0, 4.0, 0.0, 0.25, format="%+.2f h",
                       key="wi_sleep")
        dn = st.slider(f"Sleep need: change from {base_need:.1f} h", -3.0, 3.0, 0.0, 0.25, format="%+.2f h",
                       key="wi_need")
        hrv_d = st.slider(f"HRV: change from {row['hrv_rmssd_milli']:.0f} ms", -40, 40, 0, 1, format="%+d ms",
                          key="wi_hrv")
        rhr_d = st.slider(f"Resting HR: change from {row['resting_heart_rate']:.0f} bpm", -8, 8, 0, 1,
                          format="%+d bpm", key="wi_rhr")
        dstr = st.slider(f"Yesterday's strain: change from {base_strain:.1f}", -10.0, 10.0, 0.0, 0.5,
                         format="%+.1f", key="wi_strain")
    new = dd.what_if(row, max(0.0, base_sleep + ds), max(0.0, base_need + dn), hrv_d, rhr_d,
                     float(np.clip(base_strain + dstr, 0, 21)))
    new_pred = float(b.model.predict(pd.DataFrame([new[dd.COLS]]))[0])
    with c2:
        d1, d2, d3 = st.columns(3)
        d1.metric("Formula, as it was", f"{pred:.0f}")
        d2.metric("Formula, with your changes", f"{new_pred:.0f}",
                  f"{new_pred - pred:+.1f} pts" if abs(new_pred - pred) >= 0.05 else None)
        d3.metric("WHOOP actually said", f"{actual:.0f}")
        new_phi = dd.explain_rows(b, pd.DataFrame([new[dd.COLS]])).iloc[0]
        diff = (new_phi - phi)
        diff = diff[diff.abs() > 0.05].sort_values()
        if len(diff):
            dfd = diff.rename("points").reset_index().rename(columns={"index": "group"})
            dfd["kind"] = np.where(dfd["points"] >= 0, "up", "down")
            ch = alt.Chart(dfd).mark_bar(cornerRadiusEnd=4, height=18).encode(
                y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                x=alt.X("points:Q", title="change in points", scale=padded(dfd["points"])),
                color=alt.Color("kind:N", scale=alt.Scale(domain=["up", "down"], range=[UP, DOWN]), legend=None))
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_2).encode(x="x:Q")
            show((zero + ch + signed_labels(ch, "points")).properties(height=alt.Step(30),
                                                                      title="Where the change comes from"))
            st.markdown('<p class="small">An input you didn\'t touch can still shift a little: the score curve '
                        'flattens near 0 and 100, so each input\'s share depends on where the others put you.</p>',
                        unsafe_allow_html=True)
        else:
            st.info("Move a slider to see where the change comes from.")
        warns = dd.guardrail_warnings(new)
        for w in warns:
            st.warning(w, icon="⚠️")
        st.markdown('<p class="small">This is the model\'s arithmetic, not a promise about your body: it shows what '
                    'the formula would say, which tracks WHOOP within ~6.6 points on unseen mornings.</p>',
                    unsafe_allow_html=True)

# ---- 7. how this works
with tabs[6]:
    st.markdown(f"""<div class="note">
<p><b>What this is.</b> An approximation of WHOOP's recovery score built from your own data through the WHOOP
API. A readable formula (ridge regression on the logit of the score) is fitted on your earlier mornings and
tested on mornings it never saw: {test.loc['logit_ridge__set_b_full', 'mae']:.1f} points average error,
R² {test.loc['logit_ridge__set_b_full', 'r2']:.2f}. A flexible model (XGBoost) gets
{test.loc['xgboost__set_b_full', 'mae']:.1f}, and the two agree on what matters.</p>
<p><b>How a morning is split into points.</b> Exact Shapley values over 11 groups of inputs, measured from one fixed
reference morning (B = {b.B:.1f}). The pieces always add up to the estimate. Day-to-day changes are differences of
these pieces, so the reference cancels out.</p>
<p><b>What it is not.</b> Not WHOOP's real formula, and not physiology. "HRV accounts for 48%" describes the model,
not a lever you can pull. One person's data, about {len(cyc)} mornings.</p>
<p><b>Privacy.</b> Everything runs on your machine from files in <code>data/</code>, which git ignores. Nothing is
uploaded.</p></div>""", unsafe_allow_html=True)
