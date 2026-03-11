"""
Pulse Dashboard - Week-over-Week Utilities
==========================================
Extracts the duplicated WoW comparison logic into a shared module.
Used by: 1_Executive_Summary.py, 3_Trends.py, 8_Comparison.py.
"""
import pandas as pd
import numpy as np
from typing import Optional


def get_previous_week(week: int, year: int) -> tuple[int, int]:
    """Return (prev_week, prev_year) handling year rollover."""
    prev_wk = week - 1
    prev_yr = year
    if prev_wk < 1:
        prev_wk = 52
        prev_yr -= 1
    return prev_wk, prev_yr


def compute_wow_delta(df: pd.DataFrame, year: int, week: int) -> Optional[float]:
    """Compute portfolio-level WoW delta. Returns None if no previous data."""
    prev_wk, prev_yr = get_previous_week(week, year)
    current = df[(df['Year'] == year) & (df['Wk'] == week)]
    previous = df[(df['Year'] == prev_yr) & (df['Wk'] == prev_wk)]
    if current.empty or previous.empty:
        return None
    return current['Total Score'].mean() - previous['Total Score'].mean()


def generate_anomaly_narrative(
    df: pd.DataFrame, year: int, week: int,
    score_dimensions: list[str],
    threshold: float = 0.8,
) -> Optional[str]:
    """Generate a rule-based narrative explaining sharp score changes.

    Compares current week vs previous week. If the portfolio delta exceeds
    ``threshold`` (absolute), identifies the top contributing projects,
    dimensions, and regions.

    Returns
    -------
    str or None
        HTML-formatted narrative using the ``insight-callout`` CSS class,
        or None if no anomaly detected.
    """
    prev_wk, prev_yr = get_previous_week(week, year)
    current = df[(df['Year'] == year) & (df['Wk'] == week)]
    previous = df[(df['Year'] == prev_yr) & (df['Wk'] == prev_wk)]

    if current.empty or previous.empty:
        return None

    portfolio_delta = current['Total Score'].mean() - previous['Total Score'].mean()
    if abs(portfolio_delta) < threshold:
        return None

    direction = "improved" if portfolio_delta > 0 else "declined"
    color = "#22c55e" if portfolio_delta > 0 else "#ef4444"

    # Top project movers
    cur_proj = current.groupby('Project')['Total Score'].mean()
    prev_proj = previous.groupby('Project')['Total Score'].mean()
    proj_delta = (cur_proj - prev_proj).dropna().sort_values()

    if portfolio_delta < 0:
        top_movers = proj_delta.head(3)
    else:
        top_movers = proj_delta.tail(3).sort_values(ascending=False)

    # Dimension drivers
    cur_dims = current[score_dimensions].mean()
    prev_dims = previous[score_dimensions].mean()
    dim_delta = (cur_dims - prev_dims).sort_values()
    if portfolio_delta < 0:
        driver_dim = dim_delta.index[0]
        driver_delta = dim_delta.iloc[0]
    else:
        driver_dim = dim_delta.index[-1]
        driver_delta = dim_delta.iloc[-1]

    # Region drivers
    cur_reg = current.groupby('Region')['Total Score'].mean()
    prev_reg = previous.groupby('Region')['Total Score'].mean()
    reg_delta = (cur_reg - prev_reg).dropna().sort_values()
    if portfolio_delta < 0 and len(reg_delta) > 0:
        worst_region = reg_delta.index[0]
        worst_reg_delta = reg_delta.iloc[0]
    elif len(reg_delta) > 0:
        worst_region = reg_delta.index[-1]
        worst_reg_delta = reg_delta.iloc[-1]
    else:
        worst_region = "N/A"
        worst_reg_delta = 0

    mover_bullets = "".join(
        f"<li><b>{proj}</b>: <span style='color:{color};'>{delta:+.1f}</span> pts</li>"
        for proj, delta in top_movers.items()
    )

    return f"""
    <div class="insight-callout">
        <b style="font-size: 1.05rem;">Week {week} Anomaly:</b> Portfolio pulse
        <b style="color:{color};">{direction}</b> by
        <b style="color:{color};">{abs(portfolio_delta):.1f}</b> points WoW.
        <br><br>
        <b>Primary dimension driver:</b> {driver_dim}
        shifted <span style="color:{color};">{driver_delta:+.2f}</span> across the portfolio.
        <br>
        <b>Most affected region:</b> {worst_region}
        (<span style="color:{color};">{worst_reg_delta:+.1f}</span> pts).
        <br><br>
        <b>Top project movers:</b>
        <ul style="margin: 4px 0; padding-left: 20px;">{mover_bullets}</ul>
    </div>
    """
