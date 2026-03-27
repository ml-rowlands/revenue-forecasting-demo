"""
Page 2: Baseline Forecast
Automated 12-month driver and revenue forecast with confidence bands.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.charts import (
    apply_theme, waterfall_chart,
    NAVY, TEAL, CORAL, GOLD, SILVER, BRAND_COLORS, TRADE_COLORS,
)
from utils.formatting import fmt_dollars, fmt_pct, fmt_number, fmt_millions

st.set_page_config(page_title="Baseline Forecast — NCLH", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0B2545 !important; }
[data-testid="stSidebar"] * { color: #F0F4F8 !important; }
.main .block-container { padding-top: 1.2rem; max-width: 1400px; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0B2545 0%, #1a3a6b 100%);
    border-radius: 10px; padding: 16px; border: 1px solid rgba(19,168,158,0.2);
}
div[data-testid="metric-container"] label { color: #13A89E !important; font-size:0.75rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: white !important; }
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
driver_fcst  = st.session_state["driver_fcst"]
monthly_ts   = st.session_state["monthly_ts"]

st.markdown("# 📈 Baseline Forecast")
st.markdown("Nixtla-powered driver-based forecast for the next 12 months with hierarchical reconciliation.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## View Controls")
    agg_by    = st.radio("Aggregate by", ["Month", "Trade", "Brand", "Ship"])
    model_sel = st.radio("Forecast model", ["Statistical (StatsForecast)", "ML (LightGBM)", "Ensemble"])
    show_ci   = st.checkbox("Show confidence bands", value=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Forecast NTR",  fmt_millions(fcst_df["net_ticket_revenue"].sum()))
c2.metric("Total Gross Revenue", fmt_millions(fcst_df["gross_ticket_revenue"].sum()))
c3.metric("Forecast Sailings",   fmt_number(len(fcst_df)))
c4.metric("Avg Load Factor",     fmt_pct(fcst_df["load_factor"].mean()))
c5.metric("Avg Per Diem",        f"${fcst_df['gross_fare_per_diem'].mean():,.0f}")

# YoY comparison
hist_2024 = hist_df[hist_df["departure_date"].dt.year == 2024]
if not hist_2024.empty:
    prior_ntr  = hist_2024["net_ticket_revenue"].sum()
    fcst_ntr   = fcst_df["net_ticket_revenue"].sum()
    yoy_delta  = (fcst_ntr - prior_ntr) / prior_ntr * 100
    st.markdown(
        f"<div style='background:#EAF6F6; border-left:4px solid #13A89E; padding:10px 16px; "
        f"border-radius:6px; margin-bottom:16px;'>"
        f"Forecast NTR of <strong>{fmt_millions(fcst_ntr)}</strong> represents a "
        f"<strong>{'▲' if yoy_delta > 0 else '▼'}{abs(yoy_delta):.1f}%</strong> "
        f"vs prior year ({fmt_millions(prior_ntr)}).</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Driver Forecast Charts ────────────────────────────────────────────────────
st.markdown("### Forecasted Driver Time Series")

# Build base forecasts from driver_fcst — use statistical by default
lf_key  = "lf_stat"
gpd_key = "gpd_stat"
if "ML" in model_sel:
    lf_key, gpd_key = "lf_ml", "gpd_ml"

lf_fcst_df  = driver_fcst[lf_key]["forecasts_df"]
gpd_fcst_df = driver_fcst[gpd_key]["forecasts_df"]

# Get column names
def _main_col(df):
    return [c for c in df.columns if c not in ("unique_id", "ds", "lo_80", "hi_80")][0]

lf_col  = _main_col(lf_fcst_df)
gpd_col = _main_col(gpd_fcst_df)

# Aggregate to total across all unique_ids
lf_agg = (lf_fcst_df.groupby("ds")[[lf_col]]
                     .mean().reset_index().rename(columns={lf_col: "load_factor"}))
gpd_agg = (gpd_fcst_df.groupby("ds")[[gpd_col]]
                        .mean().reset_index().rename(columns={gpd_col: "gross_fare_per_diem"}))

# Historical monthly averages for comparison
hist_monthly_lf  = (hist_df.assign(ds=lambda x: x["departure_date"].dt.to_period("M").dt.to_timestamp())
                           .groupby("ds")[["load_factor", "gross_fare_per_diem"]].mean().reset_index())

tab1, tab2 = st.tabs(["Load Factor Forecast", "Per Diem Forecast"])

with tab1:
    fig_lf = go.Figure()
    fig_lf.add_trace(go.Scatter(
        x=hist_monthly_lf["ds"], y=hist_monthly_lf["load_factor"],
        mode="lines", name="Historical",
        line=dict(color=SILVER, width=1.5), opacity=0.7,
        hovertemplate="<b>Historical</b><br>%{x|%b %Y}: %{y:.1%}<extra></extra>",
    ))
    fig_lf.add_trace(go.Scatter(
        x=lf_agg["ds"], y=lf_agg["load_factor"],
        mode="lines+markers", name="Forecast",
        line=dict(color=TEAL, width=2.5),
        marker=dict(size=6, color=TEAL),
        hovertemplate="<b>Forecast</b><br>%{x|%b %Y}: %{y:.1%}<extra></extra>",
    ))
    if show_ci and "lo_80" in lf_fcst_df.columns:
        lo80 = lf_fcst_df.groupby("ds")["lo_80"].mean().reset_index()
        hi80 = lf_fcst_df.groupby("ds")["hi_80"].mean().reset_index()
        ci_x = pd.concat([hi80["ds"], lo80["ds"].iloc[::-1]])
        ci_y = pd.concat([hi80["lo_80"], lo80["lo_80"].iloc[::-1]])
        fig_lf.add_trace(go.Scatter(
            x=ci_x, y=ci_y, fill="toself",
            fillcolor="rgba(19,168,158,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI", hoverinfo="skip",
        ))
    fig_lf.update_layout(
        title="Load Factor Forecast vs Historical",
        yaxis_title="Load Factor", yaxis_tickformat=".1%",
        height=350, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_theme(fig_lf)
    st.plotly_chart(fig_lf, use_container_width=True)

with tab2:
    fig_gpd = go.Figure()
    fig_gpd.add_trace(go.Scatter(
        x=hist_monthly_lf["ds"], y=hist_monthly_lf["gross_fare_per_diem"],
        mode="lines", name="Historical",
        line=dict(color=SILVER, width=1.5), opacity=0.7,
        hovertemplate="<b>Historical</b><br>%{x|%b %Y}: $%{y:,.0f}<extra></extra>",
    ))
    fig_gpd.add_trace(go.Scatter(
        x=gpd_agg["ds"], y=gpd_agg["gross_fare_per_diem"],
        mode="lines+markers", name="Forecast",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=6, color=GOLD),
        hovertemplate="<b>Forecast</b><br>%{x|%b %Y}: $%{y:,.0f}<extra></extra>",
    ))
    fig_gpd.update_layout(
        title="Gross Fare Per Diem Forecast vs Historical",
        yaxis_title="Gross Per Diem ($)", yaxis_tickformat="$,.0f",
        height=350, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_theme(fig_gpd)
    st.plotly_chart(fig_gpd, use_container_width=True)

st.markdown("---")

# ── Revenue Waterfall ─────────────────────────────────────────────────────────
st.markdown("### Forecast Revenue Waterfall")

wf = {
    "Gross Revenue":   fcst_df["gross_ticket_revenue"].sum(),
    "Discounts":      -fcst_df["discount_amount"].sum(),
    "Promo Costs":    -fcst_df["promo_cost_total"].sum(),
    "TA Commissions": -fcst_df["ta_commission"].sum(),
    "Overrides":      -fcst_df["override_amount"].sum(),
    "Kickers":        -fcst_df["kicker_total"].sum(),
    "Air Costs":      -fcst_df["air_cost_total"].sum(),
    "Taxes & Fees":   -fcst_df["taxes_total"].sum(),
    "Net Revenue":     fcst_df["net_ticket_revenue"].sum(),
}
labels   = list(wf.keys())
values   = list(wf.values())
measures = ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]

fig_wf = waterfall_chart(labels, values, measures, "Forecast: Gross → Net Ticket Revenue")
fig_wf.update_layout(height=400)
st.plotly_chart(fig_wf, use_container_width=True)

# ── Monthly NTR forecast + prior year comparison ──────────────────────────────
st.markdown("### Forecast vs Prior Year Comparison")

fcst_monthly = (fcst_df.assign(month=lambda x: x["departure_date"].dt.to_period("M").dt.to_timestamp())
                        .groupby("month")["net_ticket_revenue"].sum().reset_index())
hist_2024_monthly = (hist_df[hist_df["departure_date"].dt.year == 2024]
                     .assign(month=lambda x: x["departure_date"].dt.to_period("M").dt.to_timestamp())
                     .groupby("month")["net_ticket_revenue"].sum().reset_index())
# Align months for YoY
hist_2024_monthly["month_label"] = hist_2024_monthly["month"].dt.strftime("%b")
fcst_monthly["month_label"]      = fcst_monthly["month"].dt.strftime("%b")

fig_yoy = go.Figure()
fig_yoy.add_trace(go.Bar(
    x=hist_2024_monthly["month_label"], y=hist_2024_monthly["net_ticket_revenue"],
    name="Prior Year (2024)", marker_color=SILVER, opacity=0.7,
    hovertemplate="<b>Prior Year</b><br>%{x}: $%{y:,.0f}<extra></extra>",
))
fig_yoy.add_trace(go.Bar(
    x=fcst_monthly["month_label"], y=fcst_monthly["net_ticket_revenue"],
    name="Forecast (2025)", marker_color=TEAL,
    hovertemplate="<b>Forecast</b><br>%{x}: $%{y:,.0f}<extra></extra>",
))
fig_yoy.update_layout(
    title="Monthly NTR: Forecast (2025) vs Prior Year (2024)",
    barmode="group", yaxis_title="Net Ticket Revenue ($)",
    yaxis_tickformat="$,.0s", height=370,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
apply_theme(fig_yoy)
st.plotly_chart(fig_yoy, use_container_width=True)

# ── Aggregated sailing table ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Forecasted Sailings Detail")

agg_options = {
    "Month":  ["departure_month"],
    "Trade":  ["trade"],
    "Brand":  ["brand"],
    "Ship":   ["ship_name"],
}
grp_cols = agg_options[agg_by]

tbl_df = (fcst_df.groupby(grp_cols).agg(
    sailings              = ("sailing_id",           "count"),
    passengers            = ("passengers_booked",    "sum"),
    gross_revenue         = ("gross_ticket_revenue", "sum"),
    net_revenue           = ("net_ticket_revenue",   "sum"),
    avg_load_factor       = ("load_factor",           "mean"),
    avg_per_diem          = ("gross_fare_per_diem",   "mean"),
    avg_discount          = ("discount_rate",         "mean"),
).reset_index())

tbl_df["net_margin_pct"] = tbl_df["net_revenue"] / tbl_df["gross_revenue"]

# Format for display
tbl_disp = tbl_df.copy()
tbl_disp["gross_revenue"]   = tbl_disp["gross_revenue"].map(lambda x: f"${x/1e6:,.1f}M")
tbl_disp["net_revenue"]     = tbl_disp["net_revenue"].map(lambda x: f"${x/1e6:,.1f}M")
tbl_disp["passengers"]      = tbl_disp["passengers"].map(lambda x: f"{x:,}")
tbl_disp["avg_load_factor"] = tbl_disp["avg_load_factor"].map(lambda x: f"{x*100:.1f}%")
tbl_disp["avg_per_diem"]    = tbl_disp["avg_per_diem"].map(lambda x: f"${x:,.0f}")
tbl_disp["avg_discount"]    = tbl_disp["avg_discount"].map(lambda x: f"{x*100:.1f}%")
tbl_disp["net_margin_pct"]  = tbl_disp["net_margin_pct"].map(lambda x: f"{x*100:.1f}%")

col_rename = {
    "departure_month": "Month", "trade": "Trade", "brand": "Brand", "ship_name": "Ship",
    "sailings": "Sailings", "passengers": "Total Pax",
    "gross_revenue": "Gross Revenue", "net_revenue": "Net Revenue",
    "avg_load_factor": "Avg LF", "avg_per_diem": "Avg Per Diem",
    "avg_discount": "Avg Discount", "net_margin_pct": "Net Margin",
}
tbl_disp = tbl_disp.rename(columns=col_rename)
st.dataframe(tbl_disp, use_container_width=True, hide_index=True)

st.markdown('<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data</div>',
            unsafe_allow_html=True)
