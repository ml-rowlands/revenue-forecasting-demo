"""
Walk-to-target optimization.
Given a net revenue target, find the minimal driver adjustments to reach it.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from forecast.waterfall import compute_waterfall


def walk_to_target(
    target_ntr: float,
    baseline_drivers: dict,
    driver_stats: pd.DataFrame,
    total_passengers: float,
    avg_itinerary_length: float,
    fixed_drivers: list[str] | None = None,
) -> dict:
    """
    Optimize driver values to hit `target_ntr` with minimal deviation from
    historical medians (weighted by inverse variance).

    Parameters
    ----------
    target_ntr            : target net ticket revenue (total across all sailings)
    baseline_drivers      : dict of driver_name -> current baseline value
    driver_stats          : DataFrame with index=driver, columns incl. mean/std/p10-p90/min/max
    total_passengers      : total pax across all forecast sailings
    avg_itinerary_length  : weighted-average itinerary length
    fixed_drivers         : drivers to hold at baseline (not optimized)

    Returns
    -------
    dict with:
        - optimized_drivers : dict driver -> value
        - percentiles       : dict driver -> historical percentile (0-100)
        - delta_from_base   : dict driver -> absolute change
        - feasibility       : estimated probability (from simulation)
        - status            : optimizer status string
    """
    if fixed_drivers is None:
        fixed_drivers = []

    # The drivers we actually optimize
    ADJUSTABLE = [
        "load_factor",
        "gross_fare_per_diem",
        "discount_rate",
        "commission_rate",
        "air_inclusive_pct",
        "promo_cost_per_pax",
    ]
    active = [d for d in ADJUSTABLE if d not in fixed_drivers]

    # Build index mapping active driver -> position in x vector
    idx_map = {d: i for i, d in enumerate(active)}

    # Fixed driver baseline values
    fixed_vals = {d: baseline_drivers[d] for d in fixed_drivers}

    def get_driver(x: np.ndarray, name: str) -> float:
        if name in idx_map:
            return x[idx_map[name]]
        return fixed_vals.get(name, baseline_drivers.get(name, 0.0))

    # Compute NTR from a driver vector
    # We scale total_passengers by load_factor deviation
    base_lf        = baseline_drivers.get("load_factor", 1.0)
    base_passengers = total_passengers  # already at baseline load factor

    def compute_ntr(x: np.ndarray) -> float:
        lf      = get_driver(x, "load_factor")
        gpd     = get_driver(x, "gross_fare_per_diem")
        disc    = get_driver(x, "discount_rate")
        comm    = get_driver(x, "commission_rate")
        air_pct = get_driver(x, "air_inclusive_pct")
        promo   = get_driver(x, "promo_cost_per_pax")

        # Adjust passengers proportionally to load factor change
        adj_passengers = base_passengers * (lf / base_lf)

        # Derive TA commission rate and direct booking pct from blended commission_rate
        # Assume direct_booking_pct fixed at baseline
        direct_pct = baseline_drivers.get("direct_booking_pct", 0.20)
        ta_rate    = comm / (1 - direct_pct) if (1 - direct_pct) > 0 else comm

        wf = compute_waterfall(
            passengers          = adj_passengers,
            itinerary_length    = avg_itinerary_length,
            gross_fare_per_diem = gpd,
            discount_rate       = disc,
            promo_cost_per_pax  = promo,
            direct_booking_pct  = direct_pct,
            ta_commission_rate  = ta_rate,
            override_pct        = baseline_drivers.get("override_pct", 0.01),
            kicker_per_cabin    = baseline_drivers.get("kicker_per_cabin", 20.0),
            air_inclusive_pct   = air_pct,
            air_cost_per_pax    = baseline_drivers.get("air_cost_per_pax", 450),
            taxes_fees_per_pax  = baseline_drivers.get("taxes_fees_per_pax", 130),
        )
        return wf["net_ticket_revenue"]

    # Objective: minimize sum of weighted squared z-scores
    def objective(x: np.ndarray) -> float:
        total = 0.0
        for d in active:
            if d in driver_stats.index:
                mu  = driver_stats.loc[d, "mean"]
                sig = driver_stats.loc[d, "std"] + 1e-10
                z   = (x[idx_map[d]] - mu) / sig
                total += z ** 2
        return total

    # Constraint: NTR must equal target
    constraints = [
        {
            "type": "eq",
            "fun":  lambda x: compute_ntr(x) - target_ntr,
        }
    ]

    # Bounds: clamp to historical min/max
    bounds = []
    x0     = []
    for d in active:
        x0.append(baseline_drivers.get(d, driver_stats.loc[d, "mean"] if d in driver_stats.index else 0))
        lo = driver_stats.loc[d, "min"] if d in driver_stats.index else 0.0
        hi = driver_stats.loc[d, "max"] if d in driver_stats.index else 1.0
        # Give a tiny buffer so optimizer isn't constrained exactly on data boundary
        lo = max(lo * 0.90, lo - 0.01)
        hi = min(hi * 1.10, hi + 0.01)
        bounds.append((lo, hi))

    x0 = np.array(x0)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    # Build output
    optimized = {**baseline_drivers}
    for d in active:
        optimized[d] = float(result.x[idx_map[d]])

    # Compute percentiles for each active driver
    percentiles = {}
    deltas      = {}
    for d in active:
        val = optimized[d]
        deltas[d] = val - baseline_drivers.get(d, 0)
        if d in driver_stats.index:
            # Interpolate percentile from P10/P25/P50/P75/P90
            pcts = [10, 25, 50, 75, 90]
            vals = [driver_stats.loc[d, f"p{p}"] for p in pcts]
            percentiles[d] = float(np.interp(val, vals, pcts))
        else:
            percentiles[d] = 50.0

    # Revenue impact breakdown per driver
    ntr_impacts = {}
    base_ntr = compute_ntr(x0)
    for d in active:
        x_copy = x0.copy()
        x_copy[idx_map[d]] = result.x[idx_map[d]]
        ntr_at_change = compute_ntr(x_copy)
        ntr_impacts[d] = ntr_at_change - base_ntr

    return {
        "optimized_drivers": optimized,
        "percentiles":        percentiles,
        "delta_from_base":    deltas,
        "ntr_impacts":        ntr_impacts,
        "achieved_ntr":       compute_ntr(result.x),
        "optimizer_success":  result.success,
        "status":             result.message,
    }


def monte_carlo_feasibility(
    target_ntr: float,
    driver_stats: pd.DataFrame,
    total_passengers: float,
    avg_itinerary_length: float,
    scenario_drivers: dict,
    n_simulations: int = 5000,
    seed: int = 99,
) -> float:
    """
    Estimate the probability that a scenario's NTR meets the target,
    given historical driver variability.

    Uses empirical percentile-based sampling centered on scenario_drivers.
    Returns a float in [0, 1].
    """
    rng = np.random.default_rng(seed)

    drivers_to_sample = [
        "load_factor", "gross_fare_per_diem", "discount_rate",
        "commission_rate", "air_inclusive_pct", "promo_cost_per_pax",
    ]

    # Build covariance-like sampling using truncated normal around scenario mean
    samples = {}
    for d in drivers_to_sample:
        if d not in driver_stats.index:
            samples[d] = np.full(n_simulations, scenario_drivers.get(d, 0))
            continue
        mu  = scenario_drivers.get(d, driver_stats.loc[d, "mean"])
        sig = driver_stats.loc[d, "std"]
        lo  = driver_stats.loc[d, "min"]
        hi  = driver_stats.loc[d, "max"]
        raw = rng.normal(mu, sig, n_simulations)
        samples[d] = np.clip(raw, lo, hi)

    # Fixed drivers
    direct_pct     = scenario_drivers.get("direct_booking_pct", 0.20)
    override_pct   = scenario_drivers.get("override_pct", 0.01)
    kicker_cabin   = scenario_drivers.get("kicker_per_cabin", 20.0)
    air_cost_pax   = scenario_drivers.get("air_cost_per_pax", 450)
    taxes_pax      = scenario_drivers.get("taxes_fees_per_pax", 130)
    base_lf        = scenario_drivers.get("load_factor", 1.0)

    ntrs = np.zeros(n_simulations)
    for i in range(n_simulations):
        lf      = samples["load_factor"][i]
        gpd     = samples["gross_fare_per_diem"][i]
        disc    = samples["discount_rate"][i]
        comm    = samples["commission_rate"][i]
        air_pct = samples["air_inclusive_pct"][i]
        promo   = samples["promo_cost_per_pax"][i]

        adj_pax = total_passengers * (lf / base_lf)
        ta_rate = comm / (1 - direct_pct) if (1 - direct_pct) > 0 else comm

        wf = compute_waterfall(
            passengers          = adj_pax,
            itinerary_length    = avg_itinerary_length,
            gross_fare_per_diem = gpd,
            discount_rate       = disc,
            promo_cost_per_pax  = promo,
            direct_booking_pct  = direct_pct,
            ta_commission_rate  = ta_rate,
            override_pct        = override_pct,
            kicker_per_cabin    = kicker_cabin,
            air_inclusive_pct   = air_pct,
            air_cost_per_pax    = air_cost_pax,
            taxes_fees_per_pax  = taxes_pax,
        )
        ntrs[i] = wf["net_ticket_revenue"]

    return float((ntrs >= target_ntr).mean())
