"""Number and string formatting utilities."""


def fmt_dollars(value: float, decimals: int = 0) -> str:
    """Format a number as a dollar amount with commas."""
    if value is None:
        return "—"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    return f"${value:,.{decimals}f}"


def fmt_millions(value: float, decimals: int = 1) -> str:
    """Format a value in millions with $ sign."""
    if value is None:
        return "—"
    return f"${value / 1_000_000:,.{decimals}f}M"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a fraction (0–1) as a percentage string."""
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f}%"


def fmt_number(value: float, decimals: int = 0) -> str:
    """Format a number with commas."""
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


def percentile_color(percentile: float) -> str:
    """
    Return a CSS color string based on which percentile band the value falls in.
    Green = P25–P75 (normal)
    Amber = P10–P25 or P75–P90 (stretch)
    Red   = <P10 or >P90 (heroic)
    """
    if 25 <= percentile <= 75:
        return "#2ecc71"   # green
    if 10 <= percentile <= 90:
        return "#f39c12"   # amber
    return "#e74c3c"       # red


def percentile_label(percentile: float) -> str:
    if 25 <= percentile <= 75:
        return "Normal range"
    if 10 <= percentile <= 90:
        return "Stretch"
    return "Heroic assumption"
