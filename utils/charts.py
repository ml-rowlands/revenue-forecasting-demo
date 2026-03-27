"""
Plotly chart helper functions.
NCLH brand color scheme: navy #0B2545, teal #13A89E, coral #E8593C.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Brand colors ────────────────────────────────────────────────────────────
NAVY   = "#0B2545"
TEAL   = "#13A89E"
CORAL  = "#E8593C"
GOLD   = "#D4A843"
SILVER = "#8E9BAB"
WHITE  = "#FFFFFF"
LIGHT_GRAY = "#F0F4F8"

BRAND_COLORS = {
    "NCL":     TEAL,
    "Oceania": NAVY,
    "Regent":  GOLD,
}

TRADE_COLORS = {
    "Caribbean":      TEAL,
    "Mediterranean":  NAVY,
    "Alaska":         "#1E8449",
    "Northern Europe":"#7D3C98",
    "Bermuda":        CORAL,
}

PLOTLY_TEMPLATE = {
    "layout": {
        "font":       {"family": "Inter, Arial, sans-serif", "color": "#2C3E50"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "xaxis":      {"gridcolor": "#E8EDF2", "linecolor": "#D0D7DE"},
        "yaxis":      {"gridcolor": "#E8EDF2", "linecolor": "#D0D7DE"},
        "colorway":   [TEAL, NAVY, CORAL, GOLD, SILVER, "#2E86C1", "#117864"],
    }
}


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, Arial, sans-serif", color="#2C3E50"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(gridcolor="#E8EDF2", linecolor="#D0D7DE", showgrid=True)
    fig.update_yaxes(gridcolor="#E8EDF2", linecolor="#D0D7DE", showgrid=True)
    return fig


def waterfall_chart(
    labels: list[str],
    values: list[float],
    measures: list[str],
    title: str = "Revenue Waterfall",
) -> go.Figure:
    """
    Create a Plotly waterfall chart.
    measures: list of 'absolute' | 'relative' | 'total'
    """
    colors = []
    for m, v in zip(measures, values):
        if m == "absolute":
            colors.append(TEAL)
        elif m == "total":
            colors.append(NAVY)
        else:
            colors.append(CORAL if v < 0 else "#2ecc71")

    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        textposition="outside",
        text=[f"${abs(v)/1e6:.1f}M" for v in values],
        connector={"line": {"color": "#8E9BAB", "width": 1}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": CORAL}},
        totals={"marker":    {"color": NAVY}},
    ))
    fig.update_layout(title=title, showlegend=False)
    return apply_theme(fig)


def time_series_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str = "",
    y_label: str = "",
    color_map: dict | None = None,
    add_trend: bool = False,
    ci_lo_col: str | None = None,
    ci_hi_col: str | None = None,
) -> go.Figure:
    """Generic time series line chart."""
    fig = go.Figure()

    if color_col:
        groups = df[color_col].unique()
        for grp in groups:
            sub  = df[df[color_col] == grp].sort_values(x_col)
            col  = (color_map or {}).get(grp, TEAL)
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode="lines+markers",
                name=str(grp),
                line=dict(color=col, width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>{grp}</b><br>%{{x}}<br>{y_label}: %{{y:,.0f}}<extra></extra>",
            ))
    else:
        sub = df.sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col],
            mode="lines+markers",
            name=y_label or y_col,
            line=dict(color=TEAL, width=2),
            marker=dict(size=4),
            hovertemplate=f"%{{x}}<br>{y_label}: %{{y:,.0f}}<extra></extra>",
        ))
        if ci_lo_col and ci_hi_col and ci_lo_col in sub.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([sub[x_col], sub[x_col].iloc[::-1]]),
                y=pd.concat([sub[ci_hi_col], sub[ci_lo_col].iloc[::-1]]),
                fill="toself",
                fillcolor=f"rgba(19,168,158,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                name="80% CI",
                hoverinfo="skip",
            ))
        if add_trend:
            x_num = np.arange(len(sub))
            z     = np.polyfit(x_num, sub[y_col].fillna(method="ffill").values, 1)
            trend = np.poly1d(z)(x_num)
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=trend,
                mode="lines",
                name="Trend",
                line=dict(color=CORAL, width=2, dash="dash"),
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return apply_theme(fig)


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str = "",
    y_label: str = "",
    color_map: dict | None = None,
    barmode: str = "group",
) -> go.Figure:
    if color_col:
        fig = px.bar(
            df, x=x_col, y=y_col, color=color_col,
            barmode=barmode,
            color_discrete_map=color_map or {},
            title=title,
            labels={y_col: y_label},
        )
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title, labels={y_col: y_label},
                     color_discrete_sequence=[TEAL])
    return apply_theme(fig)


def histogram_chart(
    series: pd.Series,
    title: str = "",
    x_label: str = "",
    percentiles: dict | None = None,
    color: str = TEAL,
) -> go.Figure:
    """Histogram with optional percentile vlines."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=series.dropna(),
        nbinsx=30,
        marker_color=color,
        opacity=0.75,
        name=x_label,
    ))
    if percentiles:
        pct_colors = {"p10": SILVER, "p25": GOLD, "p50": CORAL, "p75": GOLD, "p90": SILVER}
        for label, val in percentiles.items():
            fig.add_vline(
                x=val,
                line_dash="dash",
                line_color=pct_colors.get(label, CORAL),
                annotation_text=label.upper(),
                annotation_position="top",
            )
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Count", showlegend=False)
    return apply_theme(fig)


def kpi_card_html(label: str, value: str, delta: str = "", delta_positive: bool = True) -> str:
    """Return HTML for a styled KPI card."""
    delta_color = "#2ecc71" if delta_positive else CORAL
    delta_html = (
        f'<div style="font-size:0.85rem; color:{delta_color}; margin-top:2px;">{delta}</div>'
        if delta else ""
    )
    return f"""
    <div style="
        background: linear-gradient(135deg, {NAVY} 0%, #1a3a6b 100%);
        border-radius: 10px; padding: 18px 22px;
        color: white; min-width: 160px;
        box-shadow: 0 2px 8px rgba(11,37,69,0.15);
    ">
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;
                    color:{TEAL}; margin-bottom:6px;">{label}</div>
        <div style="font-size:1.6rem; font-weight:700; line-height:1.1;">{value}</div>
        {delta_html}
    </div>
    """


def sparkline_percentile_html(
    value: float,
    p10: float, p25: float, p50: float, p75: float, p90: float,
    label: str = "",
    fmt: str = "{:.2f}",
) -> str:
    """
    Return an inline HTML bar showing where `value` falls on the P10–P90 range.
    Color-coded: green P25–P75, amber P10–P25 or P75–P90, red outside P10/P90.
    """
    rng = p90 - p10
    if rng < 1e-9:
        pct = 50.0
    else:
        pct = (value - p10) / rng * 100
    pct = max(0, min(100, pct))

    if p25 <= value <= p75:
        color = "#2ecc71"
        status = "Normal"
    elif p10 <= value <= p90:
        color = "#f39c12"
        status = "Stretch"
    else:
        color = "#e74c3c"
        status = "Heroic"

    bar_html = f"""
    <div style="background:#E8EDF2; border-radius:4px; height:8px; width:100%; position:relative; margin:4px 0;">
        <!-- P25-P75 band -->
        <div style="
            position:absolute;
            left:{((p25-p10)/rng*100 if rng>0 else 25):.1f}%;
            width:{((p75-p25)/rng*100 if rng>0 else 50):.1f}%;
            height:100%; background:rgba(46,204,113,0.25); border-radius:2px;
        "></div>
        <!-- Current value marker -->
        <div style="
            position:absolute; left:{pct:.1f}%;
            width:3px; height:12px; top:-2px;
            background:{color}; border-radius:2px;
            transform:translateX(-50%);
        "></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#8E9BAB;">
        <span>P10</span><span>P25</span><span>P50</span><span>P75</span><span>P90</span>
    </div>
    """
    return bar_html
