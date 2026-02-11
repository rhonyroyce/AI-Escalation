"""
Pulse Dashboard - Trends & Forecasting

This page provides time-series analysis of the portfolio's Pulse Score,
helping leadership understand trajectory and momentum rather than just
a single-point-in-time snapshot.

PURPOSE
-------
- Visualize how the portfolio pulse has evolved week-over-week
- Forecast where the pulse is heading using linear extrapolation
- Show per-region sparkline trends for quick visual comparison
- Present detailed Week-over-Week (WoW) change tables at both the
  region and dimension levels

SCORING SYSTEM CONTEXT
----------------------
Each project is rated across 8 dimensions (SCORE_DIMENSIONS):
    Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential

Each dimension is scored 0-3 (integer):
    0 = Critical / Escalation
    1 = Issues / Complaints
    2 = Acceptable / On-track
    3 = Exceptional / Positive

The Total Score ("Pulse Score") is the sum of all 8 dimensions (0-24).
Status thresholds:
    Red        :  1-13  (Critical)
    Yellow     : 14-15  (At Risk)
    Green      : 16-19  (On Track)
    Dark Green : 20-24  (Exceptional)

PAGE LAYOUT (top to bottom)
---------------------------
1. Trend + Forecast chart   - Line chart of weekly avg pulse with linear
                              forecast and target reference line
2. Sparklines by Region     - Small multiples showing each region's trend
3. WoW Changes table        - Region-level: this week vs last week deltas
4. Dimension-Level WoW      - Per-dimension: this week vs last week deltas

DATA NOTES
----------
- The trend chart uses the FULL unfiltered dataset (df) so the complete
  time series is always visible regardless of sidebar filters.
- The WoW tables use the sidebar-selected week/year to determine
  "this week" and "last week".
- Week rollover is handled: if current week is 1, previous is week 52
  of the prior year.
"""

# ---------------------------------------------------------------------------
# PATH SETUP: Allow imports from the pulse_dashboard package root.
# This insert ensures `from utils.sidebar import ...` resolves correctly
# even when Streamlit runs this file directly as a "page".
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# UTILITY IMPORTS
# render_sidebar       - shared sidebar; returns the sidebar-filtered DataFrame
# inject_css           - injects the global dark-theme CSS
# get_pulse_color      - score (float) -> hex color string (for conditional formatting)
# SCORE_DIMENSIONS     - list of the 8 scoring dimension column names
# chart_trend_forecast - Plotly line chart: weekly avg pulse + linear forecast
# chart_sparklines     - Plotly small-multiples: one sparkline per region
# ---------------------------------------------------------------------------
from utils.sidebar import render_sidebar
from utils.styles import inject_css, get_pulse_color, SCORE_DIMENSIONS
from utils.mckinsey_charts import chart_trend_forecast, chart_sparklines

# Inject the global dark-theme CSS into the Streamlit page
inject_css()

# ---------------------------------------------------------------------------
# SIDEBAR & DATA LOADING
# render_sidebar() draws the sidebar widgets and returns the filtered DataFrame.
# If no data is loaded or nothing matches the filters, bail out early.
# ---------------------------------------------------------------------------
filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

# `df` is the FULL unfiltered dataset -- used for the trend chart and
# sparklines so the complete time series is always visible.
df = st.session_state.df
# Target pulse score (horizontal reference line on the trend chart)
target = st.session_state.get('pulse_target', 17.0)
# Currently selected week/year from the sidebar (for WoW comparison)
selected_week = st.session_state.get('selected_week')
selected_year = st.session_state.get('selected_year')

# Page title rendered as styled HTML
st.markdown('<p class="main-header">Trends & Forecasting</p>', unsafe_allow_html=True)

# ============================================================================
# TREND + FORECAST (full dataset for time series)
#
# This chart plots the weekly average Total Score across all projects as a
# solid line, with a dashed linear-regression forecast extending several
# weeks into the future.  A horizontal reference line marks the target.
# Uses the full (unfiltered) dataset so the trend is always complete.
# ============================================================================
fig = chart_trend_forecast(df, target)
st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SPARKLINES BY REGION
#
# A grid of small sparkline charts, one per region, showing that region's
# weekly average pulse over time.  This gives a quick visual comparison
# of regional trajectories without needing to drill down.
# Also uses the full dataset for completeness.
# ============================================================================
fig_spark = chart_sparklines(df)
st.plotly_chart(fig_spark, use_container_width=True)

# ============================================================================
# WEEK-OVER-WEEK CHANGES TABLE
#
# Compares the current sidebar-selected week to the immediately preceding
# week, showing the delta for each region.  Trend indicators:
#   up-arrow   = improved by more than 0.1 points
#   down-arrow = declined by more than 0.1 points
#   right-arrow = essentially flat (within +/- 0.1)
#   dash       = data missing for comparison
# ============================================================================
st.markdown("### Week-over-Week Changes")

if selected_week and selected_year:
    # Calculate previous week, handling year boundary rollover
    prev_wk = selected_week - 1
    prev_yr = selected_year
    if prev_wk < 1:
        prev_wk = 52
        prev_yr -= 1

    # Filter the full dataset for the current and previous weeks
    current = df[(df['Year'] == selected_year) & (df['Wk'] == selected_week)]
    previous = df[(df['Year'] == prev_yr) & (df['Wk'] == prev_wk)]

    if not current.empty and not previous.empty:
        # Compute average pulse per region for both weeks
        cur_region = current.groupby('Region')['Total Score'].mean()
        prev_region = previous.groupby('Region')['Total Score'].mean()

        # Build a comparison row for each region present in either week
        wow_data = []
        for region in sorted(set(cur_region.index) | set(prev_region.index)):
            cur_val = cur_region.get(region, np.nan)
            prev_val = prev_region.get(region, np.nan)
            # Calculate delta; handle cases where one week is missing
            delta = cur_val - prev_val if pd.notna(cur_val) and pd.notna(prev_val) else np.nan
            # Trend indicator with 0.1-point dead zone for "flat"
            trend = "▲" if delta > 0.1 else ("▼" if delta < -0.1 else "→") if pd.notna(delta) else "—"
            wow_data.append({
                'Region': region,
                'This Week': f"{cur_val:.1f}" if pd.notna(cur_val) else "—",
                'Last Week': f"{prev_val:.1f}" if pd.notna(prev_val) else "—",
                'Delta': f"{delta:+.2f}" if pd.notna(delta) else "—",
                'Trend': trend,
            })

        wow_df = pd.DataFrame(wow_data)
        st.dataframe(wow_df, use_container_width=True, hide_index=True)

        # ====================================================================
        # DIMENSION-LEVEL WoW
        #
        # Same comparison as above, but for each of the 8 scoring dimensions
        # rather than by region.  This reveals which specific dimensions
        # improved or declined, guiding targeted interventions.
        # Trend dead zone is tighter (0.05) since dimension scores are 0-3.
        # ====================================================================
        st.markdown("### Dimension-Level WoW")
        # Average each dimension across all projects for both weeks
        cur_dims = current[SCORE_DIMENSIONS].mean()
        prev_dims = previous[SCORE_DIMENSIONS].mean()
        dim_wow = []
        for dim in SCORE_DIMENSIONS:
            cv = cur_dims[dim]
            pv = prev_dims[dim]
            d = cv - pv
            # Tighter dead zone (0.05) because dimension range is only 0-3
            trend = "▲" if d > 0.05 else ("▼" if d < -0.05 else "→")
            dim_wow.append({
                'Dimension': dim,
                'This Week': f"{cv:.2f}",
                'Last Week': f"{pv:.2f}",
                'Delta': f"{d:+.2f}",
                'Trend': trend,
            })
        dim_df = pd.DataFrame(dim_wow)
        st.dataframe(dim_df, use_container_width=True, hide_index=True)
    else:
        st.info("No previous week data available for comparison.")
else:
    st.info("Select a week in the sidebar to see WoW changes.")
