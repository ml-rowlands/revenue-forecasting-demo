"""
Forecasting engine.
Fits statistical models (statsforecast + hierarchicalforecast) and
an ML model (mlforecast + LightGBM) on historical monthly data,
then generates forecasts for the next 12 months.

All heavy dependencies (statsforecast, hierarchicalforecast, mlforecast,
lightgbm, numba) are imported lazily so the app can run in degraded mode
with pure-pandas seasonal fallbacks if those packages are unavailable.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date

# ── Optional heavy imports ──────────────────────────────────────────────────
_STATS_AVAILABLE = False
_ML_AVAILABLE = False

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoARIMA, AutoCES, SeasonalNaive
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import MinTrace, BottomUp
    from hierarchicalforecast.utils import aggregate as hf_aggregate
    _STATS_AVAILABLE = True
except Exception:
    pass

try:
    from mlforecast import MLForecast
    from lightgbm import LGBMRegressor
    _ML_AVAILABLE = True
except Exception:
    pass

from forecast.waterfall import compute_waterfall


# ──────────────────────────────────────────────────────────────────────────────
# Pure-pandas fallback forecaster
# ──────────────────────────────────────────────────────────────────────────────

def _seasonal_trend_forecast(Y_df: pd.DataFrame, h: int = 12) -> pd.DataFrame:
    """
    Simple seasonal-trend decomposition forecast using pure pandas/numpy.
    Used as fallback when statsforecast/numba are unavailable.

    Algorithm:
    1. Compute the mean value for each (unique_id, month-of-year) — the seasonal component.
    2. Estimate a linear trend per series over the last 24 months.
    3. Project h months ahead: seasonal_mean × (1 + monthly_trend × step).
    Returns a DataFrame with columns: unique_id, ds, y_hat.
    """
    rows = []
    Y_df = Y_df.copy()
    Y_df["month"] = Y_df["ds"].dt.month

    for uid, grp in Y_df.groupby("unique_id"):
        grp = grp.sort_values("ds")
        if grp.empty:
            continue

        # Seasonal means (month 1–12)
        seasonal = grp.groupby("month")["y"].mean()

        # Linear trend on recent data
        recent = grp.tail(24)
        if len(recent) >= 4:
            x = np.arange(len(recent), dtype=float)
            y_vals = recent["y"].values
            try:
                slope, intercept = np.polyfit(x, y_vals, 1)
            except Exception:
                slope = 0.0
                intercept = recent["y"].mean()
        else:
            slope = 0.0
            intercept = grp["y"].mean()

        # Monthly growth rate: spread slope over mean level
        base_level = max(abs(grp["y"].mean()), 1e-9)
        monthly_rate = slope / base_level

        last_ds = grp["ds"].max()
        for step in range(1, h + 1):
            future_ds = last_ds + pd.DateOffset(months=step)
            m = future_ds.month
            base = seasonal.get(m, grp["y"].mean())
            trend_factor = 1.0 + monthly_rate * step
            # Dampen trend so it doesn't explode
            trend_factor = np.clip(trend_factor, 0.85, 1.20)
            rows.append({
                "unique_id": uid,
                "ds":        future_ds,
                "y_hat":     max(base * trend_factor, 0),
            })

    return pd.DataFrame(rows, columns=["unique_id", "ds", "y_hat"])


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build Nixtla-format time series
# ──────────────────────────────────────────────────────────────────────────────

def _build_nixtla_df(monthly_ts: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = monthly_ts[["unique_id", "ds", target_col]].copy()
    df = df.rename(columns={target_col: "y"})
    df = df.dropna(subset=["y"])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


def _build_hierarchy(bottom_level_ids: list) -> tuple:
    rows = []
    for uid in bottom_level_ids:
        parts = uid.split("/")
        if len(parts) == 3:
            brand, trade, ship_class = parts
        else:
            continue
        rows.append({
            "brand":      brand,
            "trade":      trade,
            "ship_class": ship_class,
            "unique_id":  uid,
        })
    df_hier = pd.DataFrame(rows)
    spec = [
        ["brand"],
        ["brand", "trade"],
        ["brand", "trade", "ship_class"],
    ]
    return df_hier, spec


# ──────────────────────────────────────────────────────────────────────────────
# Statistical forecasting (statsforecast + hierarchicalforecast, with fallback)
# ──────────────────────────────────────────────────────────────────────────────

def fit_statistical_models(
    monthly_ts: pd.DataFrame,
    target_col: str = "load_factor",
    h: int = 12,
    season_length: int = 12,
) -> dict:
    """
    Fit AutoETS/AutoARIMA/AutoCES at bottom level, reconcile with MinTrace.
    Falls back to a pure-pandas seasonal-trend model if statsforecast/numba
    are unavailable or raise an error.
    """
    Y_df = _build_nixtla_df(monthly_ts, target_col)

    min_obs = 2 * season_length
    counts  = Y_df.groupby("unique_id")["ds"].count()
    valid   = counts[counts >= min_obs].index.tolist()
    if not valid:
        valid = counts.index.tolist()
    Y_df = Y_df[Y_df["unique_id"].isin(valid)].copy()

    # ── Try full statsforecast pipeline ──────────────────────────────────────
    if _STATS_AVAILABLE:
        try:
            hier_df, spec = _build_hierarchy(valid)
            hier_df = hier_df[hier_df["unique_id"].isin(valid)]

            try:
                S_df, tags = hf_aggregate(Y_df[["unique_id", "ds", "y"]], spec)
            except Exception:
                S_df = Y_df.copy()
                tags = {"bottom": valid}

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
            sf.fit(S_df)

            try:
                base_fcst = sf.predict(h=h, level=[80, 95])
            except Exception:
                base_fcst = sf.predict(h=h)

            base_fcst = base_fcst.reset_index()

            model_cols   = [c for c in base_fcst.columns if c not in ("unique_id", "ds")]
            primary_col  = next(
                (c for c in model_cols if "AutoETS" in c and "-lo" not in c and "-hi" not in c),
                None,
            )
            if primary_col is None:
                primary_col = next(
                    (c for c in model_cols if "-lo" not in c and "-hi" not in c),
                    model_cols[0],
                )

            Y_hat_df = base_fcst[["unique_id", "ds", primary_col]].copy()
            Y_hat_df = Y_hat_df.rename(columns={primary_col: "model"})

            lo_80_col = next((c for c in model_cols if "lo-80" in c and "AutoETS" in c), None)
            hi_80_col = next((c for c in model_cols if "hi-80" in c and "AutoETS" in c), None)

            try:
                hrec = HierarchicalReconciliation(
                    reconcilers=[MinTrace(method="mint_shrink")]
                )
                reconciled = hrec.reconcile(
                    Y_hat_df=Y_hat_df,
                    Y_df=S_df,
                    tags=tags,
                )
            except Exception:
                try:
                    hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])
                    reconciled = hrec.reconcile(
                        Y_hat_df=Y_hat_df,
                        Y_df=S_df,
                        tags=tags,
                    )
                except Exception:
                    reconciled = Y_hat_df.rename(columns={"model": "reconciled"})

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

        except Exception:
            pass  # Fall through to pandas fallback

    # ── Pure-pandas fallback ─────────────────────────────────────────────────
    fcst = _seasonal_trend_forecast(Y_df, h=h)
    fcst = fcst.rename(columns={"y_hat": "SeasonalTrend"})

    # Add approximate confidence bands (±10% and ±18%)
    fcst["lo_80"] = fcst["SeasonalTrend"] * 0.90
    fcst["hi_80"] = fcst["SeasonalTrend"] * 1.10

    return {
        "forecasts_df":   fcst,
        "base_forecasts": fcst,
        "model_name":     "SeasonalTrend (fallback)",
        "s_df":           Y_df,
        "tags":           {"bottom": valid},
    }


# ──────────────────────────────────────────────────────────────────────────────
# ML forecasting (mlforecast + LightGBM, with fallback)
# ──────────────────────────────────────────────────────────────────────────────

def fit_ml_model(
    monthly_ts: pd.DataFrame,
    target_col: str = "load_factor",
    h: int = 12,
) -> dict:
    """
    Fit LightGBM via mlforecast with lagged features.
    Falls back to seasonal-trend if mlforecast/lightgbm are unavailable.
    """
    Y_df = _build_nixtla_df(monthly_ts, target_col)

    if _ML_AVAILABLE:
        try:
            extra_cols = [
                c for c in monthly_ts.columns
                if c not in ("unique_id", "ds", target_col, "load_factor", "gross_fare_per_diem")
            ]
            feat_df = monthly_ts[["unique_id", "ds"] + extra_cols].copy()
            feat_df["ds"] = pd.to_datetime(feat_df["ds"])
            Y_df = Y_df.merge(feat_df, on=["unique_id", "ds"], how="left")

            Y_df = Y_df.sort_values(["unique_id", "ds"])
            if extra_cols:
                Y_df[extra_cols] = (
                    Y_df.groupby("unique_id")[extra_cols].ffill().bfill()
                )

            all_ids = Y_df["unique_id"].unique()
            all_ds  = pd.date_range(Y_df["ds"].min(), Y_df["ds"].max(), freq="MS")
            idx = pd.MultiIndex.from_product(
                [all_ids, all_ds], names=["unique_id", "ds"]
            )
            Y_df = (
                Y_df.set_index(["unique_id", "ds"])
                    .reindex(idx)
                    .reset_index()
            )
            Y_df["y"] = Y_df.groupby("unique_id")["y"].ffill().bfill()
            Y_df = Y_df.dropna(subset=["y"])

            mlf = MLForecast(
                models=[
                    LGBMRegressor(
                        n_estimators=80,
                        learning_rate=0.05,
                        random_state=42,
                        verbose=-1,
                        n_jobs=1,
                    )
                ],
                freq="MS",
                lags=[1, 2, 3, 12],
                num_threads=1,
            )
            mlf.fit(Y_df[["unique_id", "ds", "y"]])
            ml_fcst = mlf.predict(h=h)

            non_id_cols = [
                c for c in ml_fcst.columns if c not in ("unique_id", "ds")
            ]
            if non_id_cols:
                ml_fcst = ml_fcst.rename(columns={non_id_cols[0]: "LGBMRegressor"})

            return {
                "forecasts_df": ml_fcst,
                "model_name":   "MLForecast/LightGBM",
            }

        except Exception:
            pass  # Fall through to pandas fallback

    # ── Pure-pandas fallback ─────────────────────────────────────────────────
    fcst = _seasonal_trend_forecast(Y_df, h=h)
    fcst = fcst.rename(columns={"y_hat": "LGBMRegressor"})

    return {
        "forecasts_df": fcst,
        "model_name":   "SeasonalTrend (fallback)",
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
    Returns a dict with lf_stat, lf_ml, gpd_stat, gpd_ml.
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
    pricing_overrides: "dict | None" = None,
) -> pd.DataFrame:
    """
    Join forecasted load_factor and gross_fare_per_diem onto future_sailings,
    then fill remaining drivers from historical averages by trade × season.
    """
    fs = future_sailings.copy()
    fs["ds"] = fs["departure_date"].dt.to_period("M").dt.to_timestamp()
    fs["unique_id"] = fs["brand"] + "/" + fs["trade"] + "/" + fs["ship_class"]

    if pricing_overrides is None:
        pricing_overrides = {}

    lf_key  = "lf_stat"  if model_choice == "statistical" else "lf_ml"
    gpd_key = "gpd_stat" if model_choice == "statistical" else "gpd_ml"
    if model_choice == "ensemble":
        lf_key  = "lf_stat"
        gpd_key = "gpd_stat"

    lf_fcst  = driver_forecasts[lf_key]["forecasts_df"]
    gpd_fcst = driver_forecasts[gpd_key]["forecasts_df"]

    def _get_forecast_col(df: pd.DataFrame) -> str:
        candidates = [
            c for c in df.columns
            if c not in ("unique_id", "ds", "lo_80", "hi_80")
        ]
        return candidates[0] if candidates else df.columns[-1]

    lf_col  = _get_forecast_col(lf_fcst)
    gpd_col = _get_forecast_col(gpd_fcst)

    lf_fcst  = lf_fcst[["unique_id", "ds", lf_col]].rename(
        columns={lf_col: "lf_forecast"}
    )
    gpd_fcst = gpd_fcst[["unique_id", "ds", gpd_col]].rename(
        columns={gpd_col: "gpd_forecast"}
    )

    fs = fs.merge(lf_fcst,  on=["unique_id", "ds"], how="left")
    fs = fs.merge(gpd_fcst, on=["unique_id", "ds"], how="left")

    if model_choice == "ensemble":
        lf_ml_fcst  = driver_forecasts["lf_ml"]["forecasts_df"]
        gpd_ml_fcst = driver_forecasts["gpd_ml"]["forecasts_df"]
        lf_ml_col   = _get_forecast_col(lf_ml_fcst)
        gpd_ml_col  = _get_forecast_col(gpd_ml_fcst)

        lf_ml_fcst  = lf_ml_fcst[["unique_id", "ds", lf_ml_col]].rename(
            columns={lf_ml_col: "lf_ml"}
        )
        gpd_ml_fcst = gpd_ml_fcst[["unique_id", "ds", gpd_ml_col]].rename(
            columns={gpd_ml_col: "gpd_ml"}
        )
        fs = fs.merge(lf_ml_fcst,  on=["unique_id", "ds"], how="left")
        fs = fs.merge(gpd_ml_fcst, on=["unique_id", "ds"], how="left")
        fs["lf_forecast"]  = fs[["lf_forecast",  "lf_ml"]].mean(axis=1)
        fs["gpd_forecast"] = fs[["gpd_forecast", "gpd_ml"]].mean(axis=1)

    # Fall back to historical medians where forecast is missing
    hist_medians = (
        historical_df
        .groupby(["brand", "trade", "ship_class", "departure_month"])[
            ["load_factor", "gross_fare_per_diem"]
        ]
        .median()
        .reset_index()
    )

    def _fallback_val(row, col: str, hist_col: str) -> float:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return val
        match = hist_medians[
            (hist_medians["brand"]           == row["brand"]) &
            (hist_medians["trade"]           == row["trade"]) &
            (hist_medians["ship_class"]      == row["ship_class"]) &
            (hist_medians["departure_month"] == row["departure_month"])
        ]
        if not match.empty:
            return match[hist_col].values[0]
        match2 = hist_medians[
            (hist_medians["trade"]           == row["trade"]) &
            (hist_medians["departure_month"] == row["departure_month"])
        ]
        if not match2.empty:
            return match2[hist_col].mean()
        return historical_df[hist_col].median()

    fs["load_factor"] = fs.apply(
        lambda r: _fallback_val(r, "lf_forecast", "load_factor"), axis=1
    )
    fs["gross_fare_per_diem"] = fs.apply(
        lambda r: _fallback_val(r, "gpd_forecast", "gross_fare_per_diem"), axis=1
    )

    if "load_factor" in pricing_overrides:
        fs["load_factor"] = fs["load_factor"] * pricing_overrides["load_factor"]
    if "gross_fare_per_diem" in pricing_overrides:
        fs["gross_fare_per_diem"] = (
            fs["gross_fare_per_diem"] * pricing_overrides["gross_fare_per_diem"]
        )

    fs["load_factor"]         = fs["load_factor"].clip(0.70, 1.15)
    fs["gross_fare_per_diem"] = fs["gross_fare_per_diem"].clip(150, 600)
    fs["passengers_booked"]   = (
        fs["lower_berth_capacity"] * fs["load_factor"]
    ).round().astype(int)

    driver_cols = {
        "discount_rate":      ("discount_rate",      "mean"),
        "promo_cost_per_pax": ("promo_cost_per_pax", "mean"),
        "direct_booking_pct": ("direct_booking_pct", "mean"),
        "ta_commission_rate": ("ta_commission_rate", "mean"),
        "commission_rate":    ("commission_rate",    "mean"),
        "override_pct":       ("override_pct",       "mean"),
        "kicker_per_cabin":   ("kicker_per_cabin",   "mean"),
        "air_inclusive_pct":  ("air_inclusive_pct",  "mean"),
        "air_cost_per_pax":   ("air_cost_per_pax",   "mean"),
        "taxes_fees_per_pax": ("taxes_fees_per_pax", "mean"),
    }

    for col, (hist_col, agg_fn) in driver_cols.items():
        if hist_col not in historical_df.columns:
            continue
        override_val = pricing_overrides.get(col)
        if override_val is not None:
            fs[col] = override_val
            continue
        medians = (
            historical_df
            .groupby(["trade", "season"])[hist_col]
            .agg(agg_fn)
            .reset_index()
            .rename(columns={hist_col: col})
        )
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
