"""
Forecasting engine.
Fits statistical models (statsforecast + hierarchicalforecast) and
an ML model (mlforecast + LightGBM) on historical monthly data,
then generates forecasts for the next 12 months.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date

# ── statsforecast ──────────────────────────────────────────────────────────────
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoETS, AutoARIMA, AutoCES, SeasonalNaive
)

# ── hierarchicalforecast ───────────────────────────────────────────────────────
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace, BottomUp
from hierarchicalforecast.utils import aggregate

# ── mlforecast ─────────────────────────────────────────────────────────────────
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from lightgbm import LGBMRegressor

from forecast.waterfall import compute_waterfall


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build Nixtla-format time series
# ──────────────────────────────────────────────────────────────────────────────

def _build_nixtla_df(monthly_ts: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert monthly_ts to Nixtla format: unique_id, ds, y.
    Also returns the hierarchy spec.
    """
    df = monthly_ts[["unique_id", "ds", target_col]].copy()
    df = df.rename(columns={target_col: "y"})
    df = df.dropna(subset=["y"])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


def _build_hierarchy(bottom_level_ids: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Given bottom-level unique_ids of form 'brand/trade/ship_class',
    build the aggregation DataFrame and tags dict for hierarchicalforecast.
    """
    rows = []
    for uid in bottom_level_ids:
        parts = uid.split("/")
        if len(parts) == 3:
            brand, trade, ship_class = parts
        else:
            continue
        rows.append({
            "brand":        brand,
            "trade":        trade,
            "ship_class":   ship_class,
            "unique_id":    uid,
        })
    df_hier = pd.DataFrame(rows)

    # Use hierarchicalforecast's aggregate utility to build Y_df and tags
    spec = [
        ["brand"],
        ["brand", "trade"],
        ["brand", "trade", "ship_class"],
    ]
    return df_hier, spec


def _aggregate_series(
    Y_df: pd.DataFrame, hier_df: pd.DataFrame, spec: list
) -> tuple[pd.DataFrame, dict]:
    """
    Use hierarchicalforecast.utils.aggregate to build all aggregation levels.
    Returns (Y_df_all_levels, tags).
    """
    # Merge Y_df with hierarchy info
    bottom_ids = Y_df["unique_id"].unique().tolist()
    df_merge = hier_df.set_index("unique_id")

    # Build a wide dataframe: rows = ds, cols = unique_id
    Y_wide = Y_df.pivot(index="ds", columns="unique_id", values="y").reset_index()
    Y_wide.columns.name = None

    # Manually build aggregated series
    groups_df = hier_df[["brand", "trade", "ship_class", "unique_id"]].copy()

    # Build summed series at each level
    dfs = []
    for _, row in Y_df.iterrows():
        dfs.append(row.to_dict())

    # Use hierarchicalforecast aggregate
    try:
        S_df, tags = aggregate(Y_df, spec)
        return S_df, tags
    except Exception:
        # Fallback: return bottom-level only
        return Y_df.copy(), {"bottom": bottom_ids}


# ──────────────────────────────────────────────────────────────────────────────
# Statistical forecasting (statsforecast + hierarchicalforecast)
# ──────────────────────────────────────────────────────────────────────────────

def fit_statistical_models(
    monthly_ts: pd.DataFrame,
    target_col: str = "load_factor",
    h: int = 12,
    season_length: int = 12,
) -> dict:
    """
    Fit AutoETS, AutoARIMA, AutoCES, SeasonalNaive at bottom level,
    then reconcile using MinTrace(mint_shrink) via hierarchicalforecast.

    Returns a dict with:
        - forecasts_df    : reconciled forecasts at all hierarchy levels
        - base_forecasts  : raw statsforecast output (for diagnostics)
        - model_name      : str label
    """
    Y_df = _build_nixtla_df(monthly_ts, target_col)

    # Ensure we have enough history per series (need at least 2 × season_length)
    min_obs = 2 * season_length
    counts  = Y_df.groupby("unique_id")["ds"].count()
    valid   = counts[counts >= min_obs].index.tolist()
    if not valid:
        valid = counts.index.tolist()
    Y_df = Y_df[Y_df["unique_id"].isin(valid)].copy()

    # Build hierarchy
    hier_df, spec = _build_hierarchy(valid)
    hier_df = hier_df[hier_df["unique_id"].isin(valid)]

    # Aggregate series to all levels
    try:
        S_df, tags = aggregate(Y_df[["unique_id", "ds", "y"]], spec)
    except Exception:
        S_df = Y_df.copy()
        tags = {"bottom": valid}

    # Fit base models with statsforecast
    models = [
        AutoETS(season_length=season_length, model="ZZZ"),
        AutoARIMA(season_length=season_length),
        AutoCES(season_length=season_length),
        SeasonalNaive(season_length=season_length),
    ]

    sf = StatsForecast(
        models=models,
        freq="MS",
        n_jobs=1,
        fallback_model=SeasonalNaive(season_length=season_length),
    )

    # Use top-level S_df that has all aggregation levels
    sf.fit(S_df)

    # Predict with prediction intervals
    try:
        base_fcst = sf.predict(h=h, level=[80, 95])
    except Exception:
        base_fcst = sf.predict(h=h)

    base_fcst = base_fcst.reset_index()

    # Pick the best model per series (use AutoETS as primary, fall back to others)
    # hierarchicalforecast expects columns: unique_id, ds, <model>
    model_cols = [c for c in base_fcst.columns if c not in ("unique_id", "ds")]
    # Prefer AutoETS
    primary_col = next((c for c in model_cols if "AutoETS" in c and "-lo" not in c and "-hi" not in c), None)
    if primary_col is None:
        primary_col = next((c for c in model_cols if "-lo" not in c and "-hi" not in c), model_cols[0])

    # Build Y_hat_df for reconciliation: unique_id, ds, model_col
    Y_hat_df = base_fcst[["unique_id", "ds", primary_col]].copy()
    Y_hat_df = Y_hat_df.rename(columns={primary_col: "model"})

    # Also grab interval columns if available
    lo_80_col = next((c for c in model_cols if "lo-80" in c and "AutoETS" in c), None)
    hi_80_col = next((c for c in model_cols if "hi-80" in c and "AutoETS" in c), None)

    # Reconcile
    try:
        hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method="mint_shrink")])
        reconciled = hrec.reconcile(
            Y_hat_df=Y_hat_df,
            Y_df=S_df,
            tags=tags,
        )
    except Exception:
        # Fallback to BottomUp if MinTrace fails
        try:
            hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])
            reconciled = hrec.reconcile(
                Y_hat_df=Y_hat_df,
                Y_df=S_df,
                tags=tags,
            )
        except Exception:
            reconciled = Y_hat_df.copy()
            reconciled = reconciled.rename(columns={"model": "reconciled"})

    # Add interval columns from base forecast (not reconciled, but indicative)
    if lo_80_col and hi_80_col:
        intervals = base_fcst[["unique_id", "ds", lo_80_col, hi_80_col]].copy()
        intervals.columns = ["unique_id", "ds", "lo_80", "hi_80"]
        reconciled = reconciled.merge(intervals, on=["unique_id", "ds"], how="left")

    return {
        "forecasts_df":   reconciled,
        "base_forecasts": base_fcst,
        "model_name":     f"StatsForecast/{primary_col}+MinTrace",
        "s_df":           S_df,
        "tags":           tags,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ML forecasting (mlforecast + LightGBM)
# ──────────────────────────────────────────────────────────────────────────────

def fit_ml_model(
    monthly_ts: pd.DataFrame,
    target_col: str = "load_factor",
    h: int = 12,
) -> dict:
    """
    Fit a LightGBM-based model using mlforecast.
    Incorporates capacity, n_sailings, and lagged features.

    Returns a dict with forecasts_df (unique_id, ds, LGBMRegressor).
    """
    Y_df = _build_nixtla_df(monthly_ts, target_col)

    # Add extra feature columns if available in monthly_ts
    extra_cols = [c for c in monthly_ts.columns
                  if c not in ("unique_id", "ds", target_col, "load_factor", "gross_fare_per_diem")]

    feat_df = monthly_ts[["unique_id", "ds"] + extra_cols].copy()
    feat_df["ds"] = pd.to_datetime(feat_df["ds"])

    Y_df = Y_df.merge(feat_df, on=["unique_id", "ds"], how="left")

    # Fill missing with forward-fill
    Y_df = Y_df.sort_values(["unique_id", "ds"])
    Y_df[extra_cols] = Y_df.groupby("unique_id")[extra_cols].ffill().bfill()

    # Fill gaps in the time series (missing months -> forward fill)
    all_ids = Y_df["unique_id"].unique()
    all_ds  = pd.date_range(Y_df["ds"].min(), Y_df["ds"].max(), freq="MS")
    idx = pd.MultiIndex.from_product([all_ids, all_ds], names=["unique_id", "ds"])
    Y_df = (Y_df.set_index(["unique_id", "ds"])
                .reindex(idx)
                .reset_index())
    Y_df["y"] = Y_df.groupby("unique_id")["y"].ffill().bfill()
    Y_df = Y_df.dropna(subset=["y"])

    mlf = MLForecast(
        models=[LGBMRegressor(n_estimators=80, learning_rate=0.05, random_state=42,
                              verbose=-1, n_jobs=1)],
        freq="MS",
        lags=[1, 2, 3, 12],
        num_threads=1,
    )

    try:
        mlf.fit(Y_df[["unique_id", "ds", "y"]])
        ml_fcst = mlf.predict(h=h)
        # Rename model column
        non_id_cols = [c for c in ml_fcst.columns if c not in ("unique_id", "ds")]
        if non_id_cols:
            ml_fcst = ml_fcst.rename(columns={non_id_cols[0]: "LGBMRegressor"})
    except Exception:
        # Fallback: repeat last 12 months shifted by 1 year
        fallback_rows = []
        for uid, grp in Y_df.groupby("unique_id"):
            last = grp.tail(12).copy()
            last["ds"] = last["ds"] + pd.DateOffset(years=1)
            last["LGBMRegressor"] = last["y"]
            fallback_rows.append(last[["unique_id", "ds", "LGBMRegressor"]])
        ml_fcst = pd.concat(fallback_rows, ignore_index=True)

    return {
        "forecasts_df": ml_fcst,
        "model_name":   "MLForecast/LightGBM",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Driver forecasting: load factor + per diem
# ──────────────────────────────────────────────────────────────────────────────

def forecast_drivers(
    monthly_ts: pd.DataFrame,
    h: int = 12,
) -> dict:
    """
    Forecast load_factor and gross_fare_per_diem for the next h months.

    Returns a dict:
        - lf_stat:   statistical model result for load_factor
        - lf_ml:     ML model result for load_factor
        - gpd_stat:  statistical model result for gross_fare_per_diem
        - gpd_ml:    ML model result for gross_fare_per_diem
    """
    lf_stat  = fit_statistical_models(monthly_ts, "load_factor",         h=h)
    gpd_stat = fit_statistical_models(monthly_ts, "gross_fare_per_diem", h=h)
    lf_ml    = fit_ml_model(monthly_ts, "load_factor",         h=h)
    gpd_ml   = fit_ml_model(monthly_ts, "gross_fare_per_diem", h=h)

    return {
        "lf_stat":  lf_stat,
        "lf_ml":    lf_ml,
        "gpd_stat": gpd_stat,
        "gpd_ml":   gpd_ml,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Apply forecasted drivers to future sailings
# ──────────────────────────────────────────────────────────────────────────────

def apply_driver_forecasts(
    future_sailings: pd.DataFrame,
    driver_forecasts: dict,
    historical_df: pd.DataFrame,
    model_choice: str = "statistical",
    pricing_overrides: dict | None = None,
) -> pd.DataFrame:
    """
    Join forecasted load_factor and gross_fare_per_diem onto future_sailings,
    then fill remaining drivers from historical averages by trade × season.

    model_choice: 'statistical' | 'ml' | 'ensemble'
    pricing_overrides: dict of driver -> scalar override value
    """
    fs = future_sailings.copy()
    fs["ds"] = fs["departure_date"].dt.to_period("M").dt.to_timestamp()
    fs["unique_id"] = fs["brand"] + "/" + fs["trade"] + "/" + fs["ship_class"]

    if pricing_overrides is None:
        pricing_overrides = {}

    # Choose forecast source
    lf_key  = "lf_stat"  if model_choice == "statistical" else "lf_ml"
    gpd_key = "gpd_stat" if model_choice == "statistical" else "gpd_ml"
    if model_choice == "ensemble":
        lf_key  = "lf_stat"
        gpd_key = "gpd_stat"

    # Extract forecast dataframes
    lf_fcst  = driver_forecasts[lf_key]["forecasts_df"]
    gpd_fcst = driver_forecasts[gpd_key]["forecasts_df"]

    # Rename forecast column
    def _get_forecast_col(df: pd.DataFrame) -> str:
        candidates = [c for c in df.columns if c not in ("unique_id", "ds", "lo_80", "hi_80")]
        return candidates[0] if candidates else df.columns[-1]

    lf_col  = _get_forecast_col(lf_fcst)
    gpd_col = _get_forecast_col(gpd_fcst)

    lf_fcst  = lf_fcst[["unique_id", "ds", lf_col]].rename(columns={lf_col: "lf_forecast"})
    gpd_fcst = gpd_fcst[["unique_id", "ds", gpd_col]].rename(columns={gpd_col: "gpd_forecast"})

    # Merge onto future sailings
    fs = fs.merge(lf_fcst,  on=["unique_id", "ds"], how="left")
    fs = fs.merge(gpd_fcst, on=["unique_id", "ds"], how="left")

    # For ML ensemble: average with statistical if requested
    if model_choice == "ensemble":
        lf_ml_fcst  = driver_forecasts["lf_ml"]["forecasts_df"]
        gpd_ml_fcst = driver_forecasts["gpd_ml"]["forecasts_df"]
        lf_ml_col   = _get_forecast_col(lf_ml_fcst)
        gpd_ml_col  = _get_forecast_col(gpd_ml_fcst)

        lf_ml_fcst  = lf_ml_fcst[["unique_id", "ds", lf_ml_col]].rename(columns={lf_ml_col: "lf_ml"})
        gpd_ml_fcst = gpd_ml_fcst[["unique_id", "ds", gpd_ml_col]].rename(columns={gpd_ml_col: "gpd_ml"})

        fs = fs.merge(lf_ml_fcst,  on=["unique_id", "ds"], how="left")
        fs = fs.merge(gpd_ml_fcst, on=["unique_id", "ds"], how="left")
        fs["lf_forecast"]  = fs[["lf_forecast", "lf_ml"]].mean(axis=1)
        fs["gpd_forecast"] = fs[["gpd_forecast", "gpd_ml"]].mean(axis=1)

    # Fall back to historical medians by brand/trade/ship_class/month where forecast is missing
    hist_medians = (historical_df
                    .groupby(["brand", "trade", "ship_class", "departure_month"])
                    [["load_factor", "gross_fare_per_diem"]]
                    .median()
                    .reset_index())

    # Build fallback dict
    def _fallback_val(row, col: str, hist_col: str) -> float:
        if not pd.isna(row.get(col)):
            return row[col]
        match = hist_medians[
            (hist_medians["brand"]       == row["brand"]) &
            (hist_medians["trade"]       == row["trade"]) &
            (hist_medians["ship_class"]  == row["ship_class"]) &
            (hist_medians["departure_month"] == row["departure_month"])
        ]
        if not match.empty:
            return match[hist_col].values[0]
        # Broader fallback: trade + month
        match2 = hist_medians[
            (hist_medians["trade"]           == row["trade"]) &
            (hist_medians["departure_month"] == row["departure_month"])
        ]
        if not match2.empty:
            return match2[hist_col].mean()
        return historical_df[hist_col].median()

    fs["load_factor"]         = fs.apply(lambda r: _fallback_val(r, "lf_forecast",  "load_factor"),        axis=1)
    fs["gross_fare_per_diem"] = fs.apply(lambda r: _fallback_val(r, "gpd_forecast", "gross_fare_per_diem"), axis=1)

    # Apply pricing overrides
    if "load_factor" in pricing_overrides:
        fs["load_factor"] = fs["load_factor"] * pricing_overrides["load_factor"]
    if "gross_fare_per_diem" in pricing_overrides:
        fs["gross_fare_per_diem"] = fs["gross_fare_per_diem"] * pricing_overrides["gross_fare_per_diem"]

    fs["load_factor"]         = fs["load_factor"].clip(0.70, 1.15)
    fs["gross_fare_per_diem"] = fs["gross_fare_per_diem"].clip(150, 600)
    fs["passengers_booked"]   = (fs["lower_berth_capacity"] * fs["load_factor"]).round().astype(int)

    # Fill remaining drivers from historical averages by trade × season
    driver_cols = {
        "discount_rate":        ("discount_rate",       "mean"),
        "promo_cost_per_pax":   ("promo_cost_per_pax",  "mean"),
        "direct_booking_pct":   ("direct_booking_pct",  "mean"),
        "ta_commission_rate":   ("ta_commission_rate",  "mean"),
        "commission_rate":      ("commission_rate",     "mean"),
        "override_pct":         ("override_pct",        "mean"),
        "kicker_per_cabin":     ("kicker_per_cabin",    "mean"),
        "air_inclusive_pct":    ("air_inclusive_pct",   "mean"),
        "air_cost_per_pax":     ("air_cost_per_pax",    "mean"),
        "taxes_fees_per_pax":   ("taxes_fees_per_pax",  "mean"),
    }

    for col, (hist_col, agg_fn) in driver_cols.items():
        if hist_col not in historical_df.columns:
            continue
        override_val = pricing_overrides.get(col)
        if override_val is not None:
            fs[col] = override_val
            continue
        medians = (historical_df
                   .groupby(["trade", "season"])[hist_col]
                   .agg(agg_fn)
                   .reset_index()
                   .rename(columns={hist_col: col}))
        fs = fs.merge(medians, on=["trade", "season"], how="left", suffixes=("", "_fill"))
        fill_col = col + "_fill"
        if fill_col in fs.columns:
            fs[col] = fs[col].fillna(fs[fill_col])
            fs.drop(columns=[fill_col], inplace=True)
        if col not in fs.columns:
            fs[col] = historical_df[hist_col].median()
        else:
            fs[col] = fs[col].fillna(historical_df[hist_col].median())

    return fs
