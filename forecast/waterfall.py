"""
Revenue waterfall computation.
Given a sailing's driver values, compute gross → net ticket revenue.
"""

import numpy as np
import pandas as pd


def compute_waterfall(
    passengers: float,
    itinerary_length: float,
    gross_fare_per_diem: float,
    discount_rate: float,
    promo_cost_per_pax: float,
    direct_booking_pct: float,
    ta_commission_rate: float,
    override_pct: float,
    kicker_per_cabin: float,
    air_inclusive_pct: float,
    air_cost_per_pax: float,
    taxes_fees_per_pax: float,
) -> dict:
    """
    Compute the full revenue waterfall for a single sailing.
    Returns a dict with all intermediate values and net_ticket_revenue.
    """
    gross_revenue    = passengers * gross_fare_per_diem * itinerary_length
    discount_amount  = gross_revenue * discount_rate
    promo_cost_total = passengers * promo_cost_per_pax
    net_after_disc   = gross_revenue - discount_amount
    ta_commission    = net_after_disc * ta_commission_rate * (1 - direct_booking_pct)
    override_amount  = ta_commission * override_pct
    kicker_total     = (passengers / 2) * kicker_per_cabin
    air_cost_total   = passengers * air_inclusive_pct * air_cost_per_pax
    taxes_total      = passengers * taxes_fees_per_pax

    net_ticket_revenue = (
        gross_revenue
        - discount_amount
        - promo_cost_total
        - ta_commission
        - override_amount
        - kicker_total
        - air_cost_total
        - taxes_total
    )

    return {
        "gross_ticket_revenue": gross_revenue,
        "discount_amount":      discount_amount,
        "promo_cost_total":     promo_cost_total,
        "ta_commission":        ta_commission,
        "override_amount":      override_amount,
        "kicker_total":         kicker_total,
        "air_cost_total":       air_cost_total,
        "taxes_total":          taxes_total,
        "net_ticket_revenue":   net_ticket_revenue,
    }


def apply_waterfall_to_sailings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply waterfall to a DataFrame of sailings that already has driver columns.
    Returns df with computed revenue columns appended.
    """
    results = []
    for _, row in df.iterrows():
        wf = compute_waterfall(
            passengers          = row["passengers_booked"],
            itinerary_length    = row["itinerary_length"],
            gross_fare_per_diem = row["gross_fare_per_diem"],
            discount_rate       = row.get("discount_rate", 0.10),
            promo_cost_per_pax  = row.get("promo_cost_per_pax", 52),
            direct_booking_pct  = row.get("direct_booking_pct", 0.20),
            ta_commission_rate  = row.get("ta_commission_rate", 0.13),
            override_pct        = row.get("override_pct", 0.0),
            kicker_per_cabin    = row.get("kicker_per_cabin", 0.0),
            air_inclusive_pct   = row.get("air_inclusive_pct", 0.18),
            air_cost_per_pax    = row.get("air_cost_per_pax", 450),
            taxes_fees_per_pax  = row.get("taxes_fees_per_pax", 130),
        )
        results.append(wf)
    wf_df = pd.DataFrame(results, index=df.index)
    # Merge back, overwriting revenue columns if present
    for col in wf_df.columns:
        df[col] = wf_df[col]
    return df


def aggregate_waterfall(df: pd.DataFrame) -> dict:
    """
    Aggregate all waterfall components across a set of sailings.
    Returns totals and derived rates.
    """
    total_gross   = df["gross_ticket_revenue"].sum()
    total_disc    = df["discount_amount"].sum()
    total_promo   = df["promo_cost_total"].sum()
    total_ta      = df["ta_commission"].sum()
    total_over    = df["override_amount"].sum()
    total_kicker  = df["kicker_total"].sum()
    total_air     = df["air_cost_total"].sum()
    total_taxes   = df["taxes_total"].sum()
    total_net     = df["net_ticket_revenue"].sum()

    return {
        "gross_revenue":    total_gross,
        "discount":         total_disc,
        "promo_cost":       total_promo,
        "ta_commission":    total_ta,
        "override":         total_over,
        "kicker":           total_kicker,
        "air_cost":         total_air,
        "taxes_fees":       total_taxes,
        "net_revenue":      total_net,
        "net_margin_pct":   total_net / total_gross if total_gross else 0,
    }
