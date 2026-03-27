"""
NCLH Revenue Forecasting Engine — Demo App
Main entry point. Initialises data and forecast engine, then routes to pages.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
st.set_page_config(
    page_title="NCLH Revenue Forecasting Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from data.generate import (
    generate_historical_sailings,
    generate_future_sailings,
    build_monthly_timeseries,
    get_overall_driver_stats,
)
from forecast.engine import forecast_drivers, apply_driver_forecasts
from forecast.waterfall import apply_waterfall_to_sailings


# ──────────────────────────────────────────────────────────────────────────────
# CSS overrides
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar header */
[data-testid="stSidebar"] { background: #0B2545 !important; }
[data-testid="stSidebar"] * { color: #F0F4F8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #A8C4D8 !important; }

/* Main background */
.main .block-container { padding-top: 1.5rem; max-width: 1400px; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0B2545 0%, #1a3a6b 100%);
    border-radius: 10px; padding: 16px;
    border: 1px solid rgba(19,168,158,0.2);
}
div[data-testid="metric-container"] label { color: #13A89E !important; font-size:0.75rem; text-transform:uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: white !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { fill: currentColor; }

/* Disclaimer footer */
.disclaimer {
    font-size: 0.72rem; color: #8E9BAB; text-align: center;
    border-top: 1px solid #E8EDF2; padding-top: 8px; margin-top: 32px;
}

/* Tabs */
[data-baseweb="tab-list"] { gap: 8px; }
[data-baseweb="tab"] {
    background: #F0F4F8; border-radius: 6px 6px 0 0;
    padding: 8px 18px; font-weight: 500;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #0B2545 !important; color: white !important;
}

/* Slider label color */
.stSlider label { color: #2C3E50; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached data / model loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Generating synthetic sailing data…")
def load_historical():
    return generate_historical_sailings()


@st.cache_data(show_spinner="Preparing future sailings schedule…")
def load_future(_hist_df):
    return generate_future_sailings(_hist_df)


@st.cache_data(show_spinner="Building monthly time series…")
def load_monthly_ts(_hist_df):
    return build_monthly_timeseries(_hist_df)


@st.cache_resource(show_spinner="Fitting forecasting models (this takes ~30 s)…")
def load_driver_forecasts(_monthly_ts):
    return forecast_drivers(_monthly_ts, h=12)


@st.cache_data(show_spinner="Applying forecasts to future sailings…")
def load_forecast_sailings(_future_df, _driver_fcsts, _hist_df, model="statistical"):
    fs = apply_driver_forecasts(_future_df, _driver_fcsts, _hist_df, model_choice=model)
    fs = apply_waterfall_to_sailings(fs)
    return fs


@st.cache_data
def load_driver_stats(_hist_df):
    return get_overall_driver_stats(_hist_df)


# ── Boot up ──────────────────────────────────────────────────────────────────
hist_df     = load_historical()
future_df   = load_future(hist_df)
monthly_ts  = load_monthly_ts(hist_df)
driver_fcst = load_driver_forecasts(monthly_ts)
driver_stats = load_driver_stats(hist_df)
fcst_df     = load_forecast_sailings(future_df, driver_fcst, hist_df)

# Store in session state so pages can access
st.session_state["hist_df"]      = hist_df
st.session_state["future_df"]    = future_df
st.session_state["monthly_ts"]   = monthly_ts
st.session_state["driver_fcst"]  = driver_fcst
st.session_state["driver_stats"] = driver_stats
st.session_state["fcst_df"]      = fcst_df

# Pre-populate saved scenarios if not already present
if "saved_scenarios" not in st.session_state:
    base_drivers = {
        "load_factor":        driver_stats.loc["load_factor", "p50"],
        "gross_fare_per_diem":driver_stats.loc["gross_fare_per_diem", "p50"],
        "discount_rate":      driver_stats.loc["discount_rate", "p50"],
        "commission_rate":    driver_stats.loc["commission_rate", "p50"],
        "air_inclusive_pct":  driver_stats.loc["air_inclusive_pct", "p50"],
        "promo_cost_per_pax": driver_stats.loc["promo_cost_per_pax", "p50"],
    }
    cfo_drivers = {
        k: float(np.interp(70, [10,25,50,75,90],
                           [driver_stats.loc[k, f"p{p}"] for p in [10,25,50,75,90]]))
        for k in base_drivers
    }
    cons_drivers = {
        k: float(np.interp(40, [10,25,50,75,90],
                           [driver_stats.loc[k, f"p{p}"] for p in [10,25,50,75,90]]))
        for k in base_drivers
    }
    base_ntr  = fcst_df["net_ticket_revenue"].sum()
    cfo_ntr   = base_ntr * 1.08
    cons_ntr  = base_ntr * 0.93

    st.session_state["saved_scenarios"] = [
        {"name": "Base Case",    "drivers": base_drivers, "ntr": base_ntr},
        {"name": "CFO Target",   "drivers": cfo_drivers,  "ntr": cfo_ntr},
        {"name": "Conservative", "drivers": cons_drivers, "ntr": cons_ntr},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 20px 0;">
        <div style="font-size:1.6rem;">🚢</div>
        <div style="font-size:0.95rem; font-weight:700; color:#13A89E; letter-spacing:0.5px;">
            NCLH Revenue<br>Forecasting Engine
        </div>
        <div style="font-size:0.65rem; color:#8E9BAB; margin-top:4px;">Demo — Synthetic Data</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("pages/01_revenue_overview.py",   label="📊 Revenue Overview",    icon=None)
    st.page_link("pages/02_baseline_forecast.py",  label="📈 Baseline Forecast",   icon=None)
    st.page_link("pages/03_walk_to_target.py",     label="🎯 Walk to Target",      icon=None)
    st.page_link("pages/04_scenario_comparison.py",label="⚖️ Scenario Comparison", icon=None)

    st.markdown("---")
    st.markdown("### Data Summary")
    st.metric("Historical Sailings", f"{len(hist_df):,}")
    st.metric("Future Sailings",     f"{len(future_df):,}")
    st.metric("Forecast Horizon",    "12 months")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem; color:#8E9BAB; text-align:center;">'
        'Powered by Nixtla StatsForecast<br>+ HierarchicalForecast + MLForecast'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Default landing content
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## NCLH Revenue Forecasting Engine")
st.markdown(
    "Select a page from the sidebar to explore the forecasting tool. "
    "Use the navigation links above or click below to jump to a section."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("**📊 Revenue Overview**\nHistorical waterfall & trends")
with col2:
    st.info("**📈 Baseline Forecast**\nAutomated 12-month forecast")
with col3:
    st.success("**🎯 Walk to Target**\nInteractive scenario builder")
with col4:
    st.warning("**⚖️ Scenario Comparison**\nSave & compare scenarios")

# Quick stats
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
last12 = hist_df[hist_df["departure_date"] >= hist_df["departure_date"].max() - pd.DateOffset(months=12)]
c1.metric("NTR Last 12M",         f"${last12['net_ticket_revenue'].sum()/1e6:,.0f}M")
c2.metric("Avg Load Factor",      f"{hist_df['load_factor'].mean()*100:.1f}%")
c3.metric("Avg Gross Per Diem",   f"${hist_df['gross_fare_per_diem'].mean():,.0f}")
c4.metric("Forecast NTR (12M)",   f"${fcst_df['net_ticket_revenue'].sum()/1e6:,.0f}M")

st.markdown(
    '<div class="disclaimer">Demo with synthetic data — not based on actual NCLH data</div>',
    unsafe_allow_html=True,
)
