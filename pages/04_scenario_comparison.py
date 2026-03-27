"""
Page 4: Scenario Comparison
Save, name, and compare multiple scenarios side by side.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forecast.waterfall import compute_waterfall
from forecast.walk import monte_carlo_feasibility
from utils.charts import apply_theme, waterfall_chart, NAVY, TEAL, CORAL, GOLD, SILVER
from utils.formatting import fmt_millions, fmt_pct, percentile_color, percentile_label

st.set_page_config(page_title="Scenario Comparison — NCLH", layout="wide")

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
.scenario-card {
    border-radius:10px; padding:16px 18px; margin-bottom:10px;
    border: 1px solid #E0E6EE;
}
.disclaimer { font-size:0.72rem; color:#8E9BAB; text-align:center;
    border-top:1px solid #E8EDF2; padding-top:8px; margin-top:32px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
if "hist_df" not in st.session_state:
    st.warning("Please start from the main app page to load data.")
    st.stop()

hist_df       = st.session_state["hist_df"]
fcst_df       = st.session_state["fcst_df"]
driver_stats  = st.session_state["driver_stats"]
saved_scenarios = st.session_state.get("saved_scenarios", [])

total_passengers     = float(fcst_df["passengers_booked"].sum())
avg_itinerary_length = float(fcst_df["itinerary_length"].mean())
baseline_ntr         = float(fcst_df["net_ticket_revenue"].sum())

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


SCENARIO_COLORS = [TEAL, NAVY, CORAL, GOLD, SILVER, "#8E44AD", "#2E86C1"]

DRIVER_LABELS = {
    "load_factor":        "Load Factor",
    "gross_fare_per_diem":"Gross Per Diem",
    "discount_rate":      "Discount Rate",
    "commission_rate":    "Commission Rate",
    "air_inclusive_pct":  "Air-Inclusive %",
    "promo_cost_per_pax": "Promo Cost/Pax",
}

DRIVER_FMT = {
    "load_factor":        lambda x: f"{x*100:.2f}%",
    "gross_fare_per_diem":lambda x: f"${x:,.1f}",
    "discount_rate":      lambda x: f"{x*100:.2f}%",
    "commission_rate":    lambda x: f"{x*100:.2f}%",
    "air_inclusive_pct":  lambda x: f"{x*100:.2f}%",
    "promo_cost_per_pax": lambda x: f"${x:,.1f}",
}


def compute_scenario_ntr(drivers: dict) -> float:
    base_lf    = baseline_drivers["load_factor"]
    adj_pax    = total_passengers * (drivers["load_factor"] / base_lf)
    direct_pct = baseline_drivers["direct_booking_pct"]
    ta_rate    = drivers["commission_rate"] / (1 - direct_pct) if (1 - direct_pct) > 0 else drivers["commission_rate"]
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
st.markdown("# ⚖️ Scenario Comparison")
st.markdown("Save and compare multiple revenue scenarios side by side.")

# ── Save Current Scenario ─────────────────────────────────────────────────────
st.markdown("### Save Current Scenario")
with st.expander("Save a new scenario from current Walk to Target settings", expanded=False):
    current_drivers = st.session_state.get("current_drivers", baseline_drivers)
    scenario_name   = st.text_input("Scenario name", value="My Scenario", key="new_scenario_name")
    col_sv1, col_sv2 = st.columns([1, 4])
    with col_sv1:
        if st.button("💾 Save Scenario", type="primary"):
            ntr_val = compute_scenario_ntr(current_drivers)
            new_scen = {
                "name":    scenario_name,
                "drivers": {k: current_drivers.get(k, baseline_drivers.get(k, 0))
                            for k in DRIVER_LABELS},
                "ntr":     ntr_val,
            }
            # Avoid duplicates
            existing_names = [s["name"] for s in st.session_state["saved_scenarios"]]
            if scenario_name not in existing_names:
                st.session_state["saved_scenarios"].append(new_scen)
                st.success(f"Saved '{scenario_name}'")
            else:
                # Update existing
                for s in st.session_state["saved_scenarios"]:
                    if s["name"] == scenario_name:
                        s.update(new_scen)
                st.info(f"Updated '{scenario_name}'")
    with col_sv2:
        driver_preview = " · ".join(
            f"{DRIVER_LABELS[k]}: {DRIVER_FMT[k](current_drivers.get(k, baseline_drivers[k]))}"
            for k in DRIVER_LABELS
        )
        st.caption(f"Current assumptions: {driver_preview}")

# Reload scenarios
scenarios = st.session_state["saved_scenarios"]

if not scenarios:
    st.info("No scenarios saved yet. Save one from the Walk to Target page.")
    st.stop()

# ── Manage saved scenarios ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Saved Scenarios")

cols_top = st.columns(min(len(scenarios), 4))
for i, scen in enumerate(scenarios):
    ntr = scen.get("ntr") or compute_scenario_ntr(scen["drivers"])
    scen["ntr"] = ntr
    delta_pct = (ntr - baseline_ntr) / baseline_ntr * 100
    color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]

    with cols_top[i % min(len(scenarios), 4)]:
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,{color}ee,{color}88);
            border-radius:10px; padding:14px 16px; color:white; margin-bottom:8px;">
            <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;
                        opacity:0.8; margin-bottom:4px;">{scen['name']}</div>
            <div style="font-size:1.7rem; font-weight:800;">{fmt_millions(ntr)}</div>
            <div style="font-size:0.8rem; opacity:0.85;">
                {'▲' if delta_pct >= 0 else '▼'}{abs(delta_pct):.1f}% vs baseline</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Comparison Table ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Driver Assumptions Comparison")

tbl_rows = {}
for drv_key, drv_label in DRIVER_LABELS.items():
    row = {"Driver": drv_label}
    for scen in scenarios:
        val = scen["drivers"].get(drv_key, baseline_drivers.get(drv_key, 0))
        pct = get_percentile(drv_key, val)
        fmt = DRIVER_FMT[drv_key]
        row[scen["name"]] = f"{fmt(val)} (P{pct:.0f})"
    tbl_rows[drv_key] = row

tbl_df = pd.DataFrame(list(tbl_rows.values()))
st.dataframe(tbl_df, use_container_width=True, hide_index=True)

# ── NTR Comparison Bar Chart ──────────────────────────────────────────────────
st.markdown("### NTR Comparison")

fig_bar = go.Figure()
for i, scen in enumerate(scenarios):
    fig_bar.add_trace(go.Bar(
        name=scen["name"],
        x=[scen["name"]],
        y=[scen["ntr"] / 1e6],
        marker_color=SCENARIO_COLORS[i % len(SCENARIO_COLORS)],
        text=[f"${scen['ntr']/1e6:,.1f}M"],
        textposition="outside",
        hovertemplate=f"<b>{scen['name']}</b><br>NTR: $%{{y:,.1f}}M<extra></extra>",
    ))

# Add baseline reference line
fig_bar.add_hline(
    y=baseline_ntr / 1e6,
    line_dash="dash", line_color=SILVER, line_width=2,
    annotation_text=f"Baseline: {fmt_millions(baseline_ntr)}",
    annotation_position="right",
)
fig_bar.update_layout(
    title="Net Ticket Revenue by Scenario",
    yaxis_title="NTR ($M)", yaxis_tickformat="$,.0f",
    showlegend=False, height=380, barmode="group",
    margin=dict(t=60, b=40),
)
apply_theme(fig_bar)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Waterfall diff between scenarios ──────────────────────────────────────────
st.markdown("### Scenario Delta Analysis")
if len(scenarios) >= 2:
    sel_a = st.selectbox("Compare:", [s["name"] for s in scenarios], index=0)
    sel_b = st.selectbox("vs:", [s["name"] for s in scenarios], index=min(1, len(scenarios)-1))

    scen_a = next(s for s in scenarios if s["name"] == sel_a)
    scen_b = next(s for s in scenarios if s["name"] == sel_b)

    delta_labels = ["Start: " + sel_a]
    delta_values = [scen_a["ntr"]]
    delta_meas   = ["absolute"]

    for drv_key, drv_label in DRIVER_LABELS.items():
        val_a = scen_a["drivers"].get(drv_key, baseline_drivers.get(drv_key, 0))
        val_b = scen_b["drivers"].get(drv_key, baseline_drivers.get(drv_key, 0))
        if abs(val_a - val_b) < 1e-8:
            continue
        # Impact of changing this driver from A to B
        ntr_a_only = compute_scenario_ntr({**scen_a["drivers"], drv_key: val_b})
        impact = ntr_a_only - scen_a["ntr"]
        if abs(impact) > 1000:
            delta_labels.append(drv_label)
            delta_values.append(impact)
            delta_meas.append("relative")

    delta_labels.append("End: " + sel_b)
    delta_values.append(scen_b["ntr"])
    delta_meas.append("total")

    fig_delta = waterfall_chart(
        delta_labels, delta_values, delta_meas,
        f"Driver Delta: {sel_a} → {sel_b}",
    )
    fig_delta.update_layout(height=400)
    st.plotly_chart(fig_delta, use_container_width=True)
else:
    st.info("Save at least 2 scenarios to see the delta waterfall.")

# ── Driver radar chart ─────────────────────────────────────────────────────────
st.markdown("### Driver Profile (Percentile Radar)")
RADAR_DRIVERS = list(DRIVER_LABELS.keys())
RADAR_LABELS  = list(DRIVER_LABELS.values())

fig_radar = go.Figure()
for i, scen in enumerate(scenarios):
    r_vals = []
    for d in RADAR_DRIVERS:
        val = scen["drivers"].get(d, baseline_drivers.get(d, 0))
        pct = get_percentile(d, val)
        r_vals.append(pct)
    r_vals.append(r_vals[0])  # close the polygon

    fig_radar.add_trace(go.Scatterpolar(
        r=r_vals,
        theta=RADAR_LABELS + [RADAR_LABELS[0]],
        name=scen["name"],
        line=dict(color=SCENARIO_COLORS[i % len(SCENARIO_COLORS)], width=2),
        fill="toself",
        fillcolor=SCENARIO_COLORS[i % len(SCENARIO_COLORS)],
        opacity=0.18,
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75],
                        ticktext=["P25", "P50", "P75"]),
    ),
    title="Driver Percentile Profiles by Scenario",
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    height=450,
    margin=dict(l=40, r=40, t=60, b=80),
)
apply_theme(fig_radar)
fig_radar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_radar, use_container_width=True)

# ── Feasibility grid ──────────────────────────────────────────────────────────
st.markdown("### Feasibility Scores")
st.caption("Monte Carlo feasibility: probability of each scenario achieving its own NTR target, given historical driver variance.")

feas_cols = st.columns(min(len(scenarios), 4))
for i, scen in enumerate(scenarios):
    feas = monte_carlo_feasibility(
        target_ntr           = scen["ntr"],
        driver_stats         = driver_stats,
        total_passengers     = total_passengers,
        avg_itinerary_length = avg_itinerary_length,
        scenario_drivers     = {**baseline_drivers, **scen["drivers"]},
        n_simulations        = 2000,
    )
    feas_pct = feas * 100
    if feas_pct >= 70: feas_color = "#2ecc71"; icon = "✅"
    elif feas_pct >= 40: feas_color = "#f39c12"; icon = "⚠️"
    else: feas_color = "#e74c3c"; icon = "🚨"

    with feas_cols[i % min(len(scenarios), 4)]:
        st.markdown(
            f"""<div style="text-align:center; background:#F8FAFB;
            border:2px solid {feas_color}; border-radius:10px; padding:14px 10px; margin-bottom:8px;">
            <div style="font-size:0.72rem; color:#8E9BAB; text-transform:uppercase; letter-spacing:1px;">
                {scen['name']}</div>
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="font-size:1.5rem; font-weight:800; color:{feas_color};">
                {feas_pct:.1f}%</div>
            <div style="font-size:0.72rem; color:#8E9BAB;">feasibility</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Manage / delete scenarios ──────────────────────────────────────────────────
st.markdown("---")
with st.expander("Manage saved scenarios"):
    to_delete = st.multiselect(
        "Select scenarios to delete",
        [s["name"] for s in scenarios]
    )
    if st.button("🗑️ Delete selected"):
        st.session_state["saved_scenarios"] = [
            s for s in scenarios if s["name"] not in to_delete
        ]
        st.rerun()

st.markdown('<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data</div>',
            unsafe_allow_html=True)
