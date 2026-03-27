"""
Page 1: Revenue Overview
Historical revenue waterfall, trends, KPIs, and driver distributions.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.charts import (
    apply_theme, waterfall_chart, time_series_chart,
    NAVY, TEAL, CORAL, GOLD, BRAND_COLORS, TRADE_COLORS,
)
from utils.formatting import fmt_dollars, fmt_pct, fmt_number

st.set_page_config(page_title="Revenue Overview — NCLH", layout="wide")

# ── Shared CSS ────────────────────────────────────────────────────────────────
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
driver_stats = st.session_state["driver_stats"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Revenue Overview")
st.markdown("Historical revenue waterfall, trends, and driver distributions across the fleet.")

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    brands  = ["All"] + sorted(hist_df["brand"].unique().tolist())
    trades  = ["All"] + sorted(hist_df["trade"].unique().tolist())
    years   = ["All"] + sorted(hist_df["departure_date"].dt.year.unique().tolist())

    sel_brand = st.selectbox("Brand",  brands)
    sel_trade = st.selectbox("Trade",  trades)
    sel_year  = st.selectbox("Year",   years)
    color_by  = st.radio("Color by", ["Brand", "Trade"])

df = hist_df.copy()
if sel_brand != "All": df = df[df["brand"] == sel_brand]
if sel_trade != "All": df = df[df["trade"] == sel_trade]
if sel_year  != "All": df = df[df["departure_date"].dt.year == int(sel_year)]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
last12_cutoff = hist_df["departure_date"].max() - pd.DateOffset(months=12)
df_l12 = df[df["departure_date"] >= last12_cutoff]
if df_l12.empty:
    df_l12 = df

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net Ticket Revenue (L12M)",  fmt_dollars(df_l12["net_ticket_revenue"].sum()))
c2.metric("Gross Revenue (L12M)",       fmt_dollars(df_l12["gross_ticket_revenue"].sum()))
c3.metric("Avg Load Factor",            fmt_pct(df["load_factor"].mean()))
c4.metric("Avg Gross Per Diem",         f"${df['gross_fare_per_diem'].mean():,.0f}")
c5.metric("Avg Blended Commission",     fmt_pct(df["commission_rate"].mean()))

st.markdown("---")

# ── Revenue Waterfall ─────────────────────────────────────────────────────────
st.markdown("### Revenue Waterfall")

wf_totals = {
    "Gross Revenue":    df["gross_ticket_revenue"].sum(),
    "Discounts":       -df["discount_amount"].sum(),
    "Promo Costs":     -df["promo_cost_total"].sum(),
    "TA Commissions":  -df["ta_commission"].sum(),
    "Override Comm.":  -df["override_amount"].sum(),
    "Kicker Costs":    -df["kicker_total"].sum(),
    "Air Costs":       -df["air_cost_total"].sum(),
    "Taxes & Fees":    -df["taxes_total"].sum(),
    "Net Revenue":      df["net_ticket_revenue"].sum(),
}

labels   = list(wf_totals.keys())
values   = list(wf_totals.values())
measures = ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]

fig_wf = waterfall_chart(labels, values, measures, title="Gross → Net Ticket Revenue Waterfall")
fig_wf.update_layout(height=420)
st.plotly_chart(fig_wf, use_container_width=True)

# ── Time Series ───────────────────────────────────────────────────────────────
st.markdown("### Monthly Net Ticket Revenue")

df_ts = df.copy()
df_ts["month"] = df_ts["departure_date"].dt.to_period("M").dt.to_timestamp()

if color_by == "Brand":
    grp_col = "brand"; cmap = BRAND_COLORS
else:
    grp_col = "trade"; cmap = TRADE_COLORS

monthly = (df_ts.groupby(["month", grp_col])["net_ticket_revenue"]
               .sum().reset_index())

fig_ts = time_series_chart(
    monthly, "month", "net_ticket_revenue",
    color_col=grp_col, title="Monthly Net Ticket Revenue",
    y_label="Net Ticket Revenue ($)", color_map=cmap,
)
fig_ts.update_layout(height=380)
# Format y-axis in millions
fig_ts.update_yaxes(tickformat="$,.0s")
st.plotly_chart(fig_ts, use_container_width=True)

# ── Monthly with trend (total) ────────────────────────────────────────────────
st.markdown("### Revenue Trend (Total) with Trend Line")
monthly_total = df_ts.groupby("month")["net_ticket_revenue"].sum().reset_index()
monthly_total = monthly_total.sort_values("month")

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=monthly_total["month"], y=monthly_total["net_ticket_revenue"],
    mode="lines+markers", name="NTR",
    line=dict(color=TEAL, width=2), marker=dict(size=5),
    hovertemplate="<b>%{x|%b %Y}</b><br>NTR: $%{y:,.0f}<extra></extra>",
))
# Trend line
x_num = np.arange(len(monthly_total))
z     = np.polyfit(x_num, monthly_total["net_ticket_revenue"].values, 1)
trend = np.poly1d(z)(x_num)
fig_trend.add_trace(go.Scatter(
    x=monthly_total["month"], y=trend,
    mode="lines", name="Trend",
    line=dict(color=CORAL, width=2, dash="dash"),
    hoverinfo="skip",
))
fig_trend.update_layout(
    title="Monthly NTR — Total Fleet (Trend)",
    yaxis_title="Net Ticket Revenue ($)",
    yaxis_tickformat="$,.0s",
    height=300,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
apply_theme(fig_trend)
st.plotly_chart(fig_trend, use_container_width=True)

# ── Driver Distributions ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Driver Distributions")
st.markdown("Historical distribution of key revenue drivers. These distributions power feasibility scoring on the Walk to Target page.")

driver_display = {
    "load_factor":        ("Load Factor",       lambda x: f"{x*100:.1f}%"),
    "gross_fare_per_diem":("Gross Per Diem",    lambda x: f"${x:,.0f}"),
    "discount_rate":      ("Discount Rate",     lambda x: f"{x*100:.1f}%"),
    "commission_rate":    ("Commission Rate",   lambda x: f"{x*100:.1f}%"),
    "air_inclusive_pct":  ("Air Inclusive %",   lambda x: f"{x*100:.1f}%"),
    "promo_cost_per_pax": ("Promo Cost/Pax",    lambda x: f"${x:,.0f}"),
}

cols = st.columns(3)
for i, (col_name, (label, fmt_fn)) in enumerate(driver_display.items()):
    if col_name not in df.columns:
        continue
    series = df[col_name].dropna()
    stats  = driver_stats.loc[col_name] if col_name in driver_stats.index else None

    with cols[i % 3]:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=series, nbinsx=30,
            marker_color=TEAL, opacity=0.75, name=label,
        ))
        if stats is not None:
            for pct, col in [("p10", "#8E9BAB"), ("p25", GOLD), ("p50", CORAL), ("p75", GOLD), ("p90", "#8E9BAB")]:
                fig_hist.add_vline(
                    x=stats[pct],
                    line_dash="dash", line_color=col, line_width=1.5,
                    annotation_text=pct.upper(), annotation_font_size=9,
                )
        fig_hist.update_layout(
            title=label, height=220, showlegend=False,
            margin=dict(l=20, r=10, t=35, b=20),
            xaxis_title=None, yaxis_title="Count",
        )
        apply_theme(fig_hist)
        # Clean up x axis ticks
        if "pct" in col_name or col_name in ("load_factor", "discount_rate", "commission_rate", "air_inclusive_pct"):
            fig_hist.update_xaxes(tickformat=".1%")
        else:
            fig_hist.update_xaxes(tickformat="$,.0f")
        st.plotly_chart(fig_hist, use_container_width=True)

        if stats is not None:
            st.caption(
                f"P10: {fmt_fn(stats['p10'])} · P50: {fmt_fn(stats['p50'])} · P90: {fmt_fn(stats['p90'])}"
            )

# ── Brand mix ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Revenue Mix by Brand & Trade")

col_a, col_b = st.columns(2)

with col_a:
    brand_ntr = df.groupby("brand")["net_ticket_revenue"].sum().reset_index()
    fig_pie_b = go.Figure(go.Pie(
        labels=brand_ntr["brand"], values=brand_ntr["net_ticket_revenue"],
        marker_colors=[BRAND_COLORS.get(b, TEAL) for b in brand_ntr["brand"]],
        hole=0.4, textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>NTR: $%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig_pie_b.update_layout(title="NTR by Brand", height=320, showlegend=False)
    apply_theme(fig_pie_b)
    st.plotly_chart(fig_pie_b, use_container_width=True)

with col_b:
    trade_ntr = df.groupby("trade")["net_ticket_revenue"].sum().reset_index()
    fig_pie_t = go.Figure(go.Pie(
        labels=trade_ntr["trade"], values=trade_ntr["net_ticket_revenue"],
        marker_colors=[TRADE_COLORS.get(t, TEAL) for t in trade_ntr["trade"]],
        hole=0.4, textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>NTR: $%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig_pie_t.update_layout(title="NTR by Trade", height=320, showlegend=False)
    apply_theme(fig_pie_t)
    st.plotly_chart(fig_pie_t, use_container_width=True)

# ── Year-over-Year Table ───────────────────────────────────────────────────────
st.markdown("### Year-over-Year Summary")
df["year"] = df["departure_date"].dt.year
yoy = df.groupby("year").agg(
    gross_revenue    = ("gross_ticket_revenue", "sum"),
    net_revenue      = ("net_ticket_revenue",   "sum"),
    sailings         = ("sailing_id",           "count"),
    avg_load_factor  = ("load_factor",           "mean"),
    avg_per_diem     = ("gross_fare_per_diem",   "mean"),
    avg_discount     = ("discount_rate",         "mean"),
).reset_index()
yoy["net_margin_pct"] = yoy["net_revenue"] / yoy["gross_revenue"]
yoy_display = yoy.rename(columns={
    "year": "Year", "gross_revenue": "Gross Revenue",
    "net_revenue": "Net Revenue", "sailings": "Sailings",
    "avg_load_factor": "Avg Load Factor", "avg_per_diem": "Avg Per Diem",
    "avg_discount": "Avg Discount", "net_margin_pct": "Net Margin %",
})
yoy_display["Gross Revenue"]   = yoy_display["Gross Revenue"].map(lambda x: f"${x/1e6:,.0f}M")
yoy_display["Net Revenue"]     = yoy_display["Net Revenue"].map(lambda x: f"${x/1e6:,.0f}M")
yoy_display["Avg Load Factor"] = yoy_display["Avg Load Factor"].map(lambda x: f"{x*100:.1f}%")
yoy_display["Avg Per Diem"]    = yoy_display["Avg Per Diem"].map(lambda x: f"${x:,.0f}")
yoy_display["Avg Discount"]    = yoy_display["Avg Discount"].map(lambda x: f"{x*100:.1f}%")
yoy_display["Net Margin %"]    = yoy_display["Net Margin %"].map(lambda x: f"{x*100:.1f}%")
st.dataframe(yoy_display, use_container_width=True, hide_index=True)

st.markdown('<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data</div>',
            unsafe_allow_html=True)
