"""
Pulse Dashboard - Trends & Forecasting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.sidebar import render_sidebar
from utils.styles import inject_css, get_pulse_color, SCORE_DIMENSIONS
from utils.mckinsey_charts import chart_trend_forecast, chart_sparklines

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

df = st.session_state.df
target = st.session_state.get('pulse_target', 17.0)
selected_week = st.session_state.get('selected_week')
selected_year = st.session_state.get('selected_year')

st.markdown('<p class="main-header">Trends & Forecasting</p>', unsafe_allow_html=True)

# ============================================================================
# TREND + FORECAST (full dataset for time series)
# ============================================================================
fig = chart_trend_forecast(df, target)
st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SPARKLINES BY REGION
# ============================================================================
fig_spark = chart_sparklines(df)
st.plotly_chart(fig_spark, use_container_width=True)

# ============================================================================
# WEEK-OVER-WEEK CHANGES TABLE
# ============================================================================
st.markdown("### Week-over-Week Changes")

if selected_week and selected_year:
    prev_wk = selected_week - 1
    prev_yr = selected_year
    if prev_wk < 1:
        prev_wk = 52
        prev_yr -= 1

    current = df[(df['Year'] == selected_year) & (df['Wk'] == selected_week)]
    previous = df[(df['Year'] == prev_yr) & (df['Wk'] == prev_wk)]

    if not current.empty and not previous.empty:
        cur_region = current.groupby('Region')['Total Score'].mean()
        prev_region = previous.groupby('Region')['Total Score'].mean()

        wow_data = []
        for region in sorted(set(cur_region.index) | set(prev_region.index)):
            cur_val = cur_region.get(region, np.nan)
            prev_val = prev_region.get(region, np.nan)
            delta = cur_val - prev_val if pd.notna(cur_val) and pd.notna(prev_val) else np.nan
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

        # Dimension-level WoW
        st.markdown("### Dimension-Level WoW")
        cur_dims = current[SCORE_DIMENSIONS].mean()
        prev_dims = previous[SCORE_DIMENSIONS].mean()
        dim_wow = []
        for dim in SCORE_DIMENSIONS:
            cv = cur_dims[dim]
            pv = prev_dims[dim]
            d = cv - pv
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
