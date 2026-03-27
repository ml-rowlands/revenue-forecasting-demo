"""
Page 3: Walk to Target
Interactive scenario builder with optimization and feasibility scoring.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forecast.walk import walk_to_target, monte_carlo_feasibility
from forecast.waterfall import compute_waterfall
from utils.charts import apply_theme, waterfall_chart, NAVY, TEAL, CORAL, GOLD, SILVER
from utils.formatting import fmt_millions, fmt_pct, fmt_number, percentile_color, percentile_label

st.set_page_config(page_title="Walk to Target — NCLH", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0B2545 !important; }
[data-testid="stSidebar"] * { color: #F0F4F8 !important; }
.main .block-container { padding-top: 1.2rem; max-width: 1400px; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0B2545 0%, #1a3a6b 100%);
    border-radius:10px; padding:16px; border:1px solid rgba(19,168,158,0.2);
}
div[data-testid="metric-container"] label { color:#13A89E !important; font-size:0.75rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: white !important; }
.driver-card {
    background: #F8FAFB; border:1px solid #E0E6EE;
    border-radius:10px; padding:14px 16px; margin-bottom:10px;
}
.pct-pill {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:600; color:white;
}
.feasibility-box {
    border-radius:12px; padding:20px 24px; margin:16px 0;
    background: linear-gradient(135deg, #0B2545 0%, #1a3a6b 100%);
    color:white;
}
.disclaimer { font-size:0.72rem; color:#8E9BAB; text-align:center;
    border-top:1px solid #E8EDF2; padding-top:8px; margin-top:32px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
if "hist_df" not in st.session_state:
    st.warning("Please start from the main app page to load data.")
    st.stop()

hist_df      = st.session_state["hist_df"]
fcst_df      = st.session_state["fcst_df"]
driver_stats = st.session_state["driver_stats"]

# Aggregate forecast parameters
total_passengers     = float(fcst_df["passengers_booked"].sum())
avg_itinerary_length = float(fcst_df["itinerary_length"].mean())
baseline_ntr         = float(fcst_df["net_ticket_revenue"].sum())

# ── Derive baseline drivers from forecast ──────────────────────────────────────
baseline_drivers = {
    "load_factor":        float(fcst_df["load_factor"].mean()),
    "gross_fare_per_diem":float(fcst_df["gross_fare_per_diem"].mean()),
    "discount_rate":      float(fcst_df["discount_rate"].mean()),
    "commission_rate":    float(fcst_df["commission_rate"].mean()),
    "air_inclusive_pct":  float(fcst_df["air_inclusive_pct"].mean()),
    "promo_cost_per_pax": float(fcst_df["promo_cost_per_pax"].mean()),
    "direct_booking_pct": float(fcst_df["direct_booking_pct"].mean()),
    "ta_commission_rate": float(fcst_df["ta_commission_rate"].mean()),
    "override_pct":       float(fcst_df["override_pct"].mean()),
    "kicker_per_cabin":   float(fcst_df["kicker_per_cabin"].mean()),
    "air_cost_per_pax":   float(fcst_df["air_cost_per_pax"].mean()),
    "taxes_fees_per_pax": float(fcst_df["taxes_fees_per_pax"].mean()),
}

# ── Driver display config ──────────────────────────────────────────────────────
DRIVER_CONFIG = [
    {
        "key":    "load_factor",
        "label":  "Load Factor",
        "fmt":    lambda x: f"{x*100:.2f}%",
        "step":   0.005,
        "format": "%.3f",
        "desc":   "Average passengers booked as % of berth capacity",
        "is_pct": True,
        "multiplier": 1.0,
    },
    {
        "key":    "gross_fare_per_diem",
        "label":  "Gross Fare Per Diem",
        "fmt":    lambda x: f"${x:,.2f}",
        "step":   5.0,
        "format": "%.1f",
        "desc":   "Average daily gross fare per passenger",
        "is_pct": False,
        "multiplier": 1.0,
    },
    {
        "key":    "discount_rate",
        "label":  "Discount Rate",
        "fmt":    lambda x: f"{x*100:.2f}%",
        "step":   0.005,
        "format": "%.3f",
        "desc":   "Revenue reduction from promotions and close-in discounting",
        "is_pct": True,
        "multiplier": -1.0,
    },
    {
        "key":    "commission_rate",
        "label":  "Blended Commission Rate",
        "fmt":    lambda x: f"{x*100:.2f}%",
        "step":   0.005,
        "format": "%.3f",
        "desc":   "Blended TA commission rate (after direct booking mix)",
        "is_pct": True,
        "multiplier": -1.0,
    },
    {
        "key":    "air_inclusive_pct",
        "label":  "Air-Inclusive Mix",
        "fmt":    lambda x: f"{x*100:.2f}%",
        "step":   0.005,
        "format": "%.3f",
        "desc":   "% of passengers booking air-inclusive packages",
        "is_pct": True,
        "multiplier": -1.0,
    },
    {
        "key":    "promo_cost_per_pax",
        "label":  "Promo Cost / Pax (Free at Sea)",
        "fmt":    lambda x: f"${x:,.2f}",
        "step":   1.0,
        "format": "%.1f",
        "desc":   "Average perk/promotion cost per passenger (Free at Sea)",
        "is_pct": False,
        "multiplier": -1.0,
    },
]


def compute_scenario_ntr(drivers: dict) -> float:
    base_lf     = baseline_drivers["load_factor"]
    adj_pax     = total_passengers * (drivers["load_factor"] / base_lf)
    direct_pct  = baseline_drivers["direct_booking_pct"]
    ta_rate     = drivers["commission_rate"] / (1 - direct_pct) if (1 - direct_pct) > 0 else drivers["commission_rate"]
    wf = compute_waterfall(
        passengers          = adj_pax,
        itinerary_length    = avg_itinerary_length,
        gross_fare_per_diem = drivers["gross_fare_per_diem"],
        discount_rate       = drivers["discount_rate"],
        promo_cost_per_pax  = drivers["promo_cost_per_pax"],
        direct_booking_pct  = direct_pct,
        ta_commission_rate  = ta_rate,
        override_pct        = baseline_drivers["override_pct"],
        kicker_per_cabin    = baseline_drivers["kicker_per_cabin"],
        air_inclusive_pct   = drivers["air_inclusive_pct"],
        air_cost_per_pax    = baseline_drivers["air_cost_per_pax"],
        taxes_fees_per_pax  = baseline_drivers["taxes_fees_per_pax"],
    )
    return wf["net_ticket_revenue"]


def get_percentile(driver: str, value: float) -> float:
    if driver not in driver_stats.index:
        return 50.0
    pcts = [10, 25, 50, 75, 90]
    vals = [driver_stats.loc[driver, f"p{p}"] for p in pcts]
    return float(np.interp(value, vals, pcts))


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎯 Walk to Target")
st.markdown("Set a revenue target, adjust driver assumptions, and assess scenario feasibility.")

# ── Target Setting ─────────────────────────────────────────────────────────────
st.markdown("---")
target_col, _ = st.columns([2, 3])
with target_col:
    st.markdown("### Annual Net Ticket Revenue Target")
    target_ntr_m = st.number_input(
        "Target NTR ($M)",
        min_value=float(baseline_ntr * 0.5 / 1e6),
        max_value=float(baseline_ntr * 1.5 / 1e6),
        value=float(round(baseline_ntr / 1e6, 0)),
        step=10.0,
        format="%.0f",
        help="Set your annual net ticket revenue target in millions",
    )
    target_ntr = target_ntr_m * 1e6

    gap = target_ntr - baseline_ntr
    gap_pct = gap / baseline_ntr * 100
    col_a, col_b = st.columns(2)
    col_a.metric("Baseline Forecast", fmt_millions(baseline_ntr))
    col_b.metric("Gap to Target",
                 fmt_millions(abs(gap)),
                 delta=f"{'▲' if gap >= 0 else '▼'} {abs(gap_pct):.1f}%",
                 delta_color="normal" if gap >= 0 else "inverse")

    run_optimization = st.button("🚀 Walk to Target", type="primary", use_container_width=True)

st.markdown("---")

# ── Driver Adjustment Panel ────────────────────────────────────────────────────
st.markdown("### Driver Adjustment Panel")
st.caption("Adjust assumptions manually using the sliders below. The NTR estimate updates in real time.")

# Initialize current drivers in session state
if "current_drivers" not in st.session_state:
    st.session_state["current_drivers"] = {k: baseline_drivers[k] for k in baseline_drivers}
if "walk_result" not in st.session_state:
    st.session_state["walk_result"] = None

# Run optimization if button pressed
if run_optimization:
    with st.spinner("Running walk-to-target optimization…"):
        result = walk_to_target(
            target_ntr           = target_ntr,
            baseline_drivers     = baseline_drivers,
            driver_stats         = driver_stats,
            total_passengers     = total_passengers,
            avg_itinerary_length = avg_itinerary_length,
        )
        st.session_state["walk_result"] = result
        # Update sliders to optimized values
        for k, v in result["optimized_drivers"].items():
            if k in st.session_state["current_drivers"]:
                st.session_state["current_drivers"][k] = v
    st.success(f"Optimization {'succeeded' if result['optimizer_success'] else 'converged (check assumptions)'}.")

# Render driver sliders
current = st.session_state["current_drivers"]

left_col, right_col = st.columns(2)
driver_vals = {}

for i, cfg in enumerate(DRIVER_CONFIG):
    key   = cfg["key"]
    label = cfg["label"]
    stats = driver_stats.loc[key] if key in driver_stats.index else None

    col = left_col if i % 2 == 0 else right_col

    with col:
        with st.container():
            p10 = float(stats["p10"]) if stats is not None else 0.0
            p90 = float(stats["p90"]) if stats is not None else 1.0
            p25 = float(stats["p25"]) if stats is not None else p10
            p50 = float(stats["p50"]) if stats is not None else (p10 + p90) / 2
            p75 = float(stats["p75"]) if stats is not None else p90

            slider_min = max(0.0001, p10 * 0.85)
            slider_max = p90 * 1.15

            baseline_val = baseline_drivers[key]
            current_val  = current.get(key, baseline_val)
            current_val  = float(np.clip(current_val, slider_min, slider_max))

            pct = get_percentile(key, current_val)
            color   = percentile_color(pct)
            plabel  = percentile_label(pct)

            st.markdown(
                f"**{label}** — "
                f"<span style='background:{color}; color:white; padding:2px 8px; "
                f"border-radius:12px; font-size:0.72rem;'>{plabel} · P{pct:.0f}</span>",
                unsafe_allow_html=True,
            )
            st.caption(cfg["desc"])

            new_val = st.slider(
                label,
                min_value=slider_min,
                max_value=slider_max,
                value=current_val,
                step=cfg["step"],
                format=cfg["format"],
                label_visibility="collapsed",
                key=f"slider_{key}",
            )
            driver_vals[key] = new_val

            # Sensitivity: $ impact of 1-step change
            base_ntr_val = compute_scenario_ntr({**current, key: baseline_val})
            nudge_val    = new_val + cfg["step"] * cfg["multiplier"]
            nudge_val    = float(np.clip(nudge_val, slider_min, slider_max))
            nudge_ntr    = compute_scenario_ntr({**current, key: nudge_val})
            sensitivity  = abs(nudge_ntr - base_ntr_val)

            dcol1, dcol2, dcol3 = st.columns([2, 2, 2])
            dcol1.markdown(f"<small>Current: **{cfg['fmt'](new_val)}**</small>", unsafe_allow_html=True)
            dcol2.markdown(f"<small>Baseline: **{cfg['fmt'](baseline_val)}**</small>", unsafe_allow_html=True)
            dcol3.markdown(
                f"<small>Sensitivity: **{fmt_millions(sensitivity)}/unit**</small>",
                unsafe_allow_html=True,
            )

            # Percentile sparkline
            if stats is not None:
                rng = p90 - p10
                pct_pos = max(0, min(100, (new_val - p10) / rng * 100)) if rng > 0 else 50
                st.markdown(
                    f"""<div style="background:#E8EDF2;border-radius:4px;height:6px;width:100%;position:relative;margin:4px 0 2px;">
                    <div style="position:absolute;left:{(p25-p10)/rng*100 if rng>0 else 25:.1f}%;
                    width:{(p75-p25)/rng*100 if rng>0 else 50:.1f}%;height:100%;
                    background:rgba(46,204,113,0.3);border-radius:2px;"></div>
                    <div style="position:absolute;left:{pct_pos:.1f}%;width:3px;height:10px;top:-2px;
                    background:{color};border-radius:2px;transform:translateX(-50%);"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.62rem;color:#8E9BAB;margin-bottom:8px;">
                    <span>P10: {cfg['fmt'](p10)}</span><span>P50: {cfg['fmt'](p50)}</span><span>P90: {cfg['fmt'](p90)}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("---")

# Update session state
st.session_state["current_drivers"].update(driver_vals)
full_current = {**baseline_drivers, **driver_vals}

# ── Live NTR estimate ──────────────────────────────────────────────────────────
live_ntr = compute_scenario_ntr(full_current)
live_gap = live_ntr - target_ntr
live_gap_pct = live_gap / target_ntr * 100 if target_ntr else 0

st.markdown("---")
st.markdown("### Live Scenario NTR Estimate")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Scenario NTR",  fmt_millions(live_ntr))
m2.metric("Target NTR",    fmt_millions(target_ntr))
m3.metric("Gap",           fmt_millions(abs(live_gap)),
          delta=f"{'▲' if live_gap >= 0 else '▼'} {abs(live_gap_pct):.1f}%",
          delta_color="normal" if live_gap >= 0 else "inverse")
m4.metric("vs Baseline",   fmt_millions(abs(live_ntr - baseline_ntr)),
          delta=f"{'▲' if live_ntr >= baseline_ntr else '▼'}{abs((live_ntr - baseline_ntr)/baseline_ntr*100):.1f}%",
          delta_color="normal" if live_ntr >= baseline_ntr else "inverse")

# ── Feasibility Assessment ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Feasibility Assessment")

with st.spinner("Running Monte Carlo simulation (5,000 draws)…"):
    feasibility = monte_carlo_feasibility(
        target_ntr           = target_ntr,
        driver_stats         = driver_stats,
        total_passengers     = total_passengers,
        avg_itinerary_length = avg_itinerary_length,
        scenario_drivers     = full_current,
        n_simulations        = 5000,
    )

# Color-code feasibility
feas_pct = feasibility * 100
if feas_pct >= 70:
    feas_color = "#2ecc71"; feas_icon = "✅"
elif feas_pct >= 40:
    feas_color = "#f39c12"; feas_icon = "⚠️"
else:
    feas_color = "#e74c3c"; feas_icon = "🚨"

st.markdown(
    f"""<div class="feasibility-box">
    <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div style="font-size:2.5rem;">{feas_icon}</div>
        <div>
            <div style="font-size:0.75rem; color:#13A89E; text-transform:uppercase; letter-spacing:1px;">
                Joint Feasibility Score</div>
            <div style="font-size:2.2rem; font-weight:800; color:{feas_color};">
                {feas_pct:.1f}%</div>
            <div style="font-size:0.85rem; color:#A8C4D8;">
                Estimated probability of achieving this scenario based on 5,000 Monte Carlo draws
                from historical driver distributions.</div>
        </div>
    </div>
    </div>""",
    unsafe_allow_html=True,
)

# Driver percentile table
st.markdown("#### Required Driver Values vs Historical Distribution")
rows = []
for cfg in DRIVER_CONFIG:
    key = cfg["key"]
    val = driver_vals.get(key, baseline_drivers[key])
    pct = get_percentile(key, val)
    base_val = baseline_drivers[key]
    stats_row = driver_stats.loc[key] if key in driver_stats.index else None
    rows.append({
        "Driver":          cfg["label"],
        "Scenario Value":  cfg["fmt"](val),
        "Baseline Value":  cfg["fmt"](base_val),
        "Change":          f"{((val - base_val) / base_val * 100):+.1f}%" if base_val else "—",
        "Hist Percentile": f"P{pct:.0f}",
        "Assessment":      percentile_label(pct),
        "_color":          percentile_color(pct),
        "_pct":            pct,
    })

driver_tbl = pd.DataFrame(rows)

def color_row(row):
    c = row["_color"]
    return [f"background-color: {c}22; color: #1a1a2e" if col == "Assessment"
            else "" for col in driver_tbl.columns]

styled = (driver_tbl.drop(columns=["_color", "_pct"])
                    .style.apply(lambda r: [
                        f"background-color: {rows[r.name]['_color']}22"
                        if col == "Assessment" else ""
                        for col in driver_tbl.drop(columns=["_color","_pct"]).columns
                    ], axis=1))
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Adjustment Waterfall ──────────────────────────────────────────────────────
st.markdown("#### Getting from Baseline to Scenario")

wf_labels  = ["Baseline NTR"]
wf_values  = [baseline_ntr]
wf_meas    = ["absolute"]

driver_impacts = {}
running = baseline_ntr
for cfg in DRIVER_CONFIG:
    key = cfg["key"]
    val = driver_vals.get(key, baseline_drivers[key])
    base_val = baseline_drivers[key]
    if abs(val - base_val) < 1e-8:
        continue
    # NTR with just this change applied
    test_drivers = {**baseline_drivers, key: val}
    ntr_with_change = compute_scenario_ntr({**baseline_drivers, key: val})
    impact = ntr_with_change - baseline_ntr
    if abs(impact) > 1000:
        driver_impacts[cfg["label"]] = impact

for label, impact in driver_impacts.items():
    wf_labels.append(label)
    wf_values.append(impact)
    wf_meas.append("relative")

wf_labels.append("Scenario NTR")
wf_values.append(live_ntr)
wf_meas.append("total")

fig_adj_wf = waterfall_chart(wf_labels, wf_values, wf_meas,
                              "Driver Adjustments: Baseline → Scenario NTR")
fig_adj_wf.update_layout(height=400)
st.plotly_chart(fig_adj_wf, use_container_width=True)

# ── Target proximity gauge ─────────────────────────────────────────────────────
st.markdown("#### Scenario vs Target")
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=live_ntr / 1e6,
    delta={"reference": target_ntr / 1e6, "valueformat": ".1f",
           "prefix": "$", "suffix": "M"},
    title={"text": "Scenario NTR ($M) vs Target"},
    gauge={
        "axis":  {"range": [baseline_ntr * 0.8 / 1e6, baseline_ntr * 1.3 / 1e6]},
        "bar":   {"color": TEAL},
        "steps": [
            {"range": [baseline_ntr * 0.8 / 1e6, target_ntr * 0.95 / 1e6], "color": "#FEECEC"},
            {"range": [target_ntr * 0.95 / 1e6, target_ntr * 1.05 / 1e6], "color": "#E5F7F5"},
            {"range": [target_ntr * 1.05 / 1e6, baseline_ntr * 1.3 / 1e6], "color": "#E9F7EF"},
        ],
        "threshold": {
            "line": {"color": CORAL, "width": 3},
            "thickness": 0.8,
            "value": target_ntr / 1e6,
        },
    },
    number={"prefix": "$", "suffix": "M", "valueformat": ",.1f"},
))
fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
apply_theme(fig_gauge)
st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown('<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data</div>',
            unsafe_allow_html=True)
