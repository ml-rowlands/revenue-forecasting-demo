"""
Synthetic cruise line booking and revenue data generation.
Produces realistic historical sailings and future scheduled sailings.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

# ──────────────────────────────────────────────
# Constants / lookup tables
# ──────────────────────────────────────────────

SHIPS = {
    # (brand, ship_class, lower_berth_capacity)
    "Norwegian Prima":     ("NCL",     "Prima",     3215),
    "Norwegian Viva":      ("NCL",     "Prima",     3215),
    "Norwegian Bliss":     ("NCL",     "Breakaway", 3898),
    "Norwegian Encore":    ("NCL",     "Breakaway", 3998),
    "Norwegian Escape":    ("NCL",     "Breakaway", 4266),
    "Norwegian Breakaway": ("NCL",     "Breakaway", 3963),
    "Norwegian Getaway":   ("NCL",     "Breakaway", 3963),
    "Norwegian Jewel":     ("NCL",     "Jewel",     2376),
    "Norwegian Pearl":     ("NCL",     "Jewel",     2394),
    "Norwegian Gem":       ("NCL",     "Jewel",     2394),
    "Marina":              ("Oceania", "Regatta",   1250),
    "Riviera":             ("Oceania", "Regatta",   1250),
    "Vista":               ("Oceania", "Allura",    1200),
    "Seven Seas Explorer": ("Regent",  "Explorer",   750),
    "Seven Seas Splendor": ("Regent",  "Explorer",   750),
    "Seven Seas Grandeur": ("Regent",  "Explorer",   700),
}

# Trade routes and their seasonal weights by month (1-12)
TRADE_CONFIG = {
    "Caribbean": {
        "brands":       ["NCL", "Oceania", "Regent"],
        "itineraries":  [7, 7, 7, 10, 5, 3],     # weighted itinerary lengths
        "seasonal_idx": [1.15, 1.10, 1.05, 0.95, 0.85, 0.80,
                         0.80, 0.85, 0.90, 0.95, 1.05, 1.15],  # by month
        "per_diem_base": {"NCL": 285, "Oceania": 370, "Regent": 440},
        "load_base":    1.03,
    },
    "Mediterranean": {
        "brands":       ["NCL", "Oceania", "Regent"],
        "itineraries":  [7, 7, 10, 10, 14, 12],
        "seasonal_idx": [0.60, 0.65, 0.80, 0.95, 1.10, 1.20,
                         1.25, 1.20, 1.10, 0.90, 0.65, 0.55],
        "per_diem_base": {"NCL": 310, "Oceania": 390, "Regent": 450},
        "load_base":    0.99,
    },
    "Alaska": {
        "brands":       ["NCL"],
        "itineraries":  [7, 7, 7, 10],
        "seasonal_idx": [0.0, 0.0, 0.0, 0.0, 0.80, 1.05,
                         1.20, 1.20, 1.05, 0.0, 0.0, 0.0],
        "per_diem_base": {"NCL": 295},
        "load_base":    1.02,
    },
    "Northern Europe": {
        "brands":       ["NCL", "Oceania", "Regent"],
        "itineraries":  [7, 10, 12, 14],
        "seasonal_idx": [0.0, 0.0, 0.0, 0.55, 0.80, 1.10,
                         1.20, 1.15, 0.90, 0.55, 0.0, 0.0],
        "per_diem_base": {"NCL": 320, "Oceania": 400, "Regent": 460},
        "load_base":    0.97,
    },
    "Bermuda": {
        "brands":       ["NCL"],
        "itineraries":  [5, 7],
        "seasonal_idx": [0.0, 0.0, 0.0, 0.70, 0.95, 1.05,
                         1.10, 1.10, 1.00, 0.75, 0.0, 0.0],
        "per_diem_base": {"NCL": 265},
        "load_base":    1.01,
    },
}

SEASON_MAP = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
              5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall",   10: "Fall",  11: "Winter", 12: "Winter"}

rng = np.random.default_rng(42)


def _season(month: int) -> str:
    return SEASON_MAP[month]


def _growth_factor(dep_date: date, base_date: date, annual_rate: float = 0.065) -> float:
    years = (dep_date - base_date).days / 365.25
    return (1 + annual_rate) ** years


def _compute_waterfall(row: pd.Series, rng_local) -> dict:
    """Compute all waterfall components and net ticket revenue for one sailing."""
    passengers = row["passengers_booked"]
    nights     = row["itinerary_length"]
    gpd        = row["gross_fare_per_diem"]
    trade      = row["trade"]
    season     = row["season"]
    brand      = row["brand"]

    gross_revenue = passengers * gpd * nights

    # Discount: higher in shoulder/close-in, lower in peak
    seasonal_discount_adj = {"Winter": 0.12, "Spring": 0.10, "Summer": 0.07, "Fall": 0.11}
    base_disc = seasonal_discount_adj.get(season, 0.10)
    discount_rate = float(np.clip(rng_local.normal(base_disc, 0.025), 0.03, 0.22))

    # Promo cost (Free at Sea perks): Oceania/Regent slightly higher
    promo_base = {"NCL": 52, "Oceania": 60, "Regent": 65}.get(brand, 52)
    promo_cost_per_pax = float(np.clip(rng_local.normal(promo_base, 8), 25, 95))

    # Commission: mix of direct vs TA
    direct_pct_mean = {"NCL": 0.22, "Oceania": 0.18, "Regent": 0.15}.get(brand, 0.20)
    direct_booking_pct = float(np.clip(rng_local.normal(direct_pct_mean, 0.04), 0.05, 0.45))
    ta_commission_rate = float(np.clip(rng_local.normal(0.13, 0.015), 0.08, 0.18))
    commission_rate    = ta_commission_rate * (1 - direct_booking_pct)  # blended

    # Override commission (~40% of TA bookings)
    override_flag   = rng_local.random() < 0.40
    override_pct    = float(rng_local.uniform(0.01, 0.03)) if override_flag else 0.0

    # Kicker (~30% of sailings)
    kicker_flag     = rng_local.random() < 0.30
    kicker_per_cabin = float(rng_local.uniform(0, 75)) if kicker_flag else 0.0

    # Air inclusive
    air_pct_mean = {"Caribbean": 0.15, "Mediterranean": 0.22, "Alaska": 0.18,
                    "Northern Europe": 0.25, "Bermuda": 0.12}.get(trade, 0.18)
    air_inclusive_pct = float(np.clip(rng_local.normal(air_pct_mean, 0.04), 0.05, 0.45))
    air_cost_base = {"Caribbean": 320, "Mediterranean": 680, "Alaska": 420,
                     "Northern Europe": 730, "Bermuda": 280}.get(trade, 450)
    air_cost_per_pax = float(np.clip(rng_local.normal(air_cost_base, 60), 150, 950))

    # Taxes & fees
    taxes_fees_base = {"Caribbean": 115, "Mediterranean": 155, "Alaska": 130,
                       "Northern Europe": 165, "Bermuda": 95}.get(trade, 130)
    taxes_fees_per_pax = float(np.clip(rng_local.normal(taxes_fees_base, 15), 60, 220))

    # Waterfall computations
    discount_amount  = gross_revenue * discount_rate
    promo_cost_total = passengers * promo_cost_per_pax
    net_after_disc   = gross_revenue - discount_amount
    ta_commission    = net_after_disc * ta_commission_rate * (1 - direct_booking_pct)
    override_amount  = ta_commission * override_pct
    kicker_total     = (passengers / 2) * kicker_per_cabin
    air_cost_total   = passengers * air_inclusive_pct * air_cost_per_pax
    taxes_total      = passengers * taxes_fees_per_pax

    net_ticket_revenue = (gross_revenue - discount_amount - promo_cost_total
                          - ta_commission - override_amount - kicker_total
                          - air_cost_total - taxes_total)

    return {
        "gross_ticket_revenue":  round(gross_revenue, 0),
        "discount_rate":         round(discount_rate, 4),
        "discount_amount":       round(discount_amount, 0),
        "promo_cost_per_pax":    round(promo_cost_per_pax, 2),
        "promo_cost_total":      round(promo_cost_total, 0),
        "direct_booking_pct":    round(direct_booking_pct, 4),
        "ta_commission_rate":    round(ta_commission_rate, 4),
        "commission_rate":       round(commission_rate, 4),
        "override_pct":          round(override_pct, 4),
        "override_amount":       round(override_amount, 0),
        "kicker_per_cabin":      round(kicker_per_cabin, 2),
        "kicker_total":          round(kicker_total, 0),
        "air_inclusive_pct":     round(air_inclusive_pct, 4),
        "air_cost_per_pax":      round(air_cost_per_pax, 2),
        "air_cost_total":        round(air_cost_total, 0),
        "taxes_fees_per_pax":    round(taxes_fees_per_pax, 2),
        "taxes_total":           round(taxes_total, 0),
        "net_ticket_revenue":    round(net_ticket_revenue, 0),
    }


# ──────────────────────────────────────────────
# Main generation functions
# ──────────────────────────────────────────────

def generate_historical_sailings() -> pd.DataFrame:
    """
    Generate ~2,000 historical sailings across 3 years.
    Returns a DataFrame with all driver and revenue columns.
    """
    base_date = date(2022, 1, 1)
    end_date  = date(2024, 12, 31)

    records = []
    sailing_id = 1

    # Build a pool of ship×trade assignments that rotate realistically
    ship_trade_pool = [
        ("Norwegian Bliss",     "Caribbean"),
        ("Norwegian Encore",    "Caribbean"),
        ("Norwegian Escape",    "Caribbean"),
        ("Norwegian Breakaway", "Caribbean"),
        ("Norwegian Getaway",   "Caribbean"),
        ("Norwegian Prima",     "Caribbean"),
        ("Norwegian Viva",      "Caribbean"),
        ("Norwegian Jewel",     "Caribbean"),
        ("Norwegian Pearl",     "Caribbean"),
        ("Norwegian Gem",       "Caribbean"),
        ("Norwegian Bliss",     "Alaska"),
        ("Norwegian Encore",    "Alaska"),
        ("Norwegian Escape",    "Mediterranean"),
        ("Norwegian Prima",     "Mediterranean"),
        ("Norwegian Breakaway", "Northern Europe"),
        ("Norwegian Jewel",     "Bermuda"),
        ("Marina",              "Mediterranean"),
        ("Riviera",             "Mediterranean"),
        ("Vista",               "Caribbean"),
        ("Seven Seas Explorer", "Mediterranean"),
        ("Seven Seas Splendor", "Northern Europe"),
        ("Seven Seas Grandeur", "Caribbean"),
    ]

    current_date = base_date

    while current_date <= end_date:
        for ship_name, trade in ship_trade_pool:
            brand, ship_class, lbc = SHIPS[ship_name]
            cfg = TRADE_CONFIG[trade]

            if brand not in cfg["brands"]:
                continue

            month = current_date.month
            seas_idx = cfg["seasonal_idx"][month - 1]
            if seas_idx < 0.05:
                continue  # trade not operating this month

            # Pick itinerary length
            itin_len = int(rng.choice(cfg["itineraries"]))

            dep_date = current_date + timedelta(days=int(rng.integers(0, 28)))
            if dep_date > end_date:
                dep_date = end_date

            # Capacity grows over time (fleet expansion) — apply to load factor
            growth = _growth_factor(dep_date, base_date, annual_rate=0.065)

            # Load factor: base + seasonal + noise
            lf_base  = cfg["load_base"]
            lf_seas  = lf_base * seas_idx / max(cfg["seasonal_idx"])
            lf       = float(np.clip(rng.normal(lf_seas, 0.035), 0.78, 1.15))
            # Apply slight upward trend over time
            lf       = min(1.15, lf * (1 + (growth - 1) * 0.3))

            passengers_booked = int(lbc * lf)

            # Per diem: base + seasonal premium + noise + growth trend
            gpd_base  = cfg["per_diem_base"][brand]
            gpd_seas  = gpd_base * (1 + 0.12 * (seas_idx / max(cfg["seasonal_idx"]) - 0.5))
            gpd       = float(np.clip(rng.normal(gpd_seas, gpd_base * 0.06), 180, 500))
            gpd       = gpd * growth  # pricing power grows with time

            row_base = {
                "sailing_id":            sailing_id,
                "ship_name":             ship_name,
                "ship_class":            ship_class,
                "brand":                 brand,
                "trade":                 trade,
                "itinerary_length":      itin_len,
                "departure_date":        pd.Timestamp(dep_date),
                "departure_month":       dep_date.month,
                "season":                _season(dep_date.month),
                "lower_berth_capacity":  lbc,
                "passengers_booked":     passengers_booked,
                "load_factor":           round(lf, 4),
                "gross_fare_per_diem":   round(gpd, 2),
            }

            waterfall = _compute_waterfall(pd.Series(row_base), rng)
            records.append({**row_base, **waterfall})
            sailing_id += 1

        # Advance to next "wave" of sailings (approximately weekly blocks)
        current_date += timedelta(days=int(rng.integers(9, 13)))

    df = pd.DataFrame(records)
    df = df.sort_values("departure_date").reset_index(drop=True)
    return df


def generate_future_sailings(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the next 12 months of scheduled sailings (ship assignments + capacity,
    no revenue data yet — that's what we forecast).
    """
    start_date = date(2025, 1, 1)
    end_date   = date(2025, 12, 31)

    # 2025 ship-trade schedule (slightly expanded fleet)
    ship_trade_pool = [
        ("Norwegian Bliss",     "Caribbean"),
        ("Norwegian Encore",    "Caribbean"),
        ("Norwegian Escape",    "Caribbean"),
        ("Norwegian Breakaway", "Caribbean"),
        ("Norwegian Getaway",   "Caribbean"),
        ("Norwegian Prima",     "Caribbean"),
        ("Norwegian Viva",      "Caribbean"),
        ("Norwegian Jewel",     "Caribbean"),
        ("Norwegian Pearl",     "Caribbean"),
        ("Norwegian Gem",       "Caribbean"),
        ("Norwegian Bliss",     "Alaska"),
        ("Norwegian Encore",    "Alaska"),
        ("Norwegian Escape",    "Mediterranean"),
        ("Norwegian Prima",     "Mediterranean"),
        ("Norwegian Breakaway", "Northern Europe"),
        ("Norwegian Jewel",     "Bermuda"),
        ("Marina",              "Mediterranean"),
        ("Riviera",             "Mediterranean"),
        ("Vista",               "Caribbean"),
        ("Seven Seas Explorer", "Mediterranean"),
        ("Seven Seas Splendor", "Northern Europe"),
        ("Seven Seas Grandeur", "Caribbean"),
    ]

    records = []
    sailing_id = 9000

    current_date = start_date
    while current_date <= end_date:
        for ship_name, trade in ship_trade_pool:
            brand, ship_class, lbc = SHIPS[ship_name]
            cfg = TRADE_CONFIG[trade]

            if brand not in cfg["brands"]:
                continue

            month = current_date.month
            seas_idx = cfg["seasonal_idx"][month - 1]
            if seas_idx < 0.05:
                continue

            itin_len = int(rng.choice(cfg["itineraries"]))
            dep_date = current_date + timedelta(days=int(rng.integers(0, 28)))
            if dep_date > end_date:
                dep_date = end_date

            records.append({
                "sailing_id":           sailing_id,
                "ship_name":            ship_name,
                "ship_class":           ship_class,
                "brand":                brand,
                "trade":                trade,
                "itinerary_length":     itin_len,
                "departure_date":       pd.Timestamp(dep_date),
                "departure_month":      dep_date.month,
                "season":               _season(dep_date.month),
                "lower_berth_capacity": lbc,
            })
            sailing_id += 1

        current_date += timedelta(days=int(rng.integers(9, 13)))

    df = pd.DataFrame(records)
    df = df.sort_values("departure_date").reset_index(drop=True)
    return df


def build_monthly_timeseries(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monthly time series at brand × trade × ship_class level for
    use with statsforecast / hierarchicalforecast.
    Returns a DataFrame with columns: unique_id, ds, y (load_factor),
    plus gross_fare_per_diem as a separate series.
    """
    df = historical_df.copy()
    df["ds"] = df["departure_date"].dt.to_period("M").dt.to_timestamp()
    df["unique_id"] = (df["brand"] + "/" + df["trade"] + "/" + df["ship_class"])

    agg = (df.groupby(["unique_id", "ds"])
             .agg(
                 load_factor          = ("load_factor",       "mean"),
                 gross_fare_per_diem  = ("gross_fare_per_diem", "mean"),
                 n_sailings           = ("sailing_id",        "count"),
             )
             .reset_index())

    return agg


def get_driver_distributions(historical_df: pd.DataFrame) -> dict:
    """
    Return P10/P25/P50/P75/P90 for each key driver, grouped by trade × season.
    """
    drivers = ["load_factor", "gross_fare_per_diem", "discount_rate",
               "commission_rate", "air_inclusive_pct", "promo_cost_per_pax",
               "taxes_fees_per_pax", "kicker_per_cabin"]
    result = {}
    for drv in drivers:
        if drv not in historical_df.columns:
            continue
        grp = historical_df.groupby(["trade", "season"])[drv].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        result[drv] = grp
    return result


def get_overall_driver_stats(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return overall P10/P25/P50/P75/P90 + mean + std for each key driver.
    """
    drivers = ["load_factor", "gross_fare_per_diem", "discount_rate",
               "commission_rate", "air_inclusive_pct", "promo_cost_per_pax",
               "taxes_fees_per_pax", "kicker_per_cabin"]
    rows = []
    for drv in drivers:
        if drv not in historical_df.columns:
            continue
        s = historical_df[drv]
        rows.append({
            "driver": drv,
            "mean":   s.mean(),
            "std":    s.std(),
            "p10":    s.quantile(0.10),
            "p25":    s.quantile(0.25),
            "p50":    s.quantile(0.50),
            "p75":    s.quantile(0.75),
            "p90":    s.quantile(0.90),
            "min":    s.min(),
            "max":    s.max(),
        })
    return pd.DataFrame(rows).set_index("driver")
