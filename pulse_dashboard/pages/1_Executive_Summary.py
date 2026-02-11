"""
Pulse Dashboard - Executive Summary (McKinsey SCR Format)

This is the primary landing page for senior leadership. It provides a
high-level, one-screen overview of the entire project portfolio's health
using the "Pulse Score" system.

SCORING SYSTEM CONTEXT
----------------------
Each project is rated across 8 dimensions (defined in SCORE_DIMENSIONS):
    Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential

Each dimension is scored 0-3 (integer):
    0 = Critical / Escalation
    1 = Issues / Complaints
    2 = Acceptable / On-track
    3 = Exceptional / Positive

The Total Score (aka "Pulse Score") is the sum of all 8 dimensions,
ranging from 0 to 24.  Status thresholds:
    Red        :  1-13  (Critical)
    Yellow     : 14-15  (At Risk)
    Green      : 16-19  (On Track)
    Dark Green : 20-24  (Exceptional)

PAGE LAYOUT (top to bottom)
---------------------------
1. Header bar       - Overall pulse score with status badge and WoW delta
2. SCR boxes        - McKinsey Situation/Complication/Resolution narrative
3. KPI cards        - Five headline KPIs (pulse, green, yellow, red, total)
4. Charts Row 1     - Variance bullet gauge + trend/forecast line chart
5. Insight callout  - Auto-generated key insight about weakest dimension
6. Charts Row 2     - Waterfall decomposition + Pareto analysis
7. Recommendations  - Auto-generated priority action table

All data is filtered through the shared sidebar (region, week, year).
The sidebar populates `st.session_state` which this page reads.
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
# render_sidebar  - shared sidebar that returns the filtered DataFrame
# inject_css      - injects the dark-theme CSS into the page
# SCORE_DIMENSIONS - list of the 8 dimension column names
# STATUS_CONFIG   - dict mapping status names to color / threshold metadata
# get_pulse_status - maps a numeric score to its status tier string
# chart_*         - McKinsey-style Plotly chart builders
# ---------------------------------------------------------------------------
from utils.sidebar import render_sidebar
from utils.styles import (
    inject_css, SCORE_DIMENSIONS, STATUS_CONFIG, get_pulse_status,
)
from utils.mckinsey_charts import (
    chart_variance_bullet, chart_trend_forecast,
    chart_waterfall_decomposition, chart_pareto,
)

# Inject the global dark-theme CSS into the Streamlit page
inject_css()

# ---------------------------------------------------------------------------
# SIDEBAR & DATA LOADING
# render_sidebar() draws the sidebar widgets (region, year, week selectors)
# and returns the filtered DataFrame.  If nothing matches, bail out early.
# ---------------------------------------------------------------------------
filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

# `df` is the FULL unfiltered dataset (used for WoW comparisons and trends).
# `filtered_df` is the subset matching the sidebar selections.
df = st.session_state.df

# Retrieve configurable targets from session state (set via sidebar or defaults):
#   pulse_target   - the "par" score the portfolio should hit (default 17)
#   pulse_stretch  - aspirational stretch goal (default 19)
#   green_pct_target - % of projects that should be Green/Dark Green (default 80)
#   max_red_target - maximum acceptable count of Red projects (default 3)
target = st.session_state.get('pulse_target', 17.0)
stretch = st.session_state.get('pulse_stretch', 19.0)
green_pct_target = st.session_state.get('green_pct_target', 80)
max_red_target = st.session_state.get('max_red_target', 3)
selected_week = st.session_state.get('selected_week')
selected_year = st.session_state.get('selected_year')

# ============================================================================
# METRICS - Compute all headline numbers from the filtered dataset
# ============================================================================
# Average pulse score across all filtered project-week rows
avg_pulse = filtered_df['Total Score'].mean()

# Variance = how far the portfolio average sits above/below the target
variance = avg_pulse - target

# Count projects by their status tier (Red / Yellow / Green / Dark Green).
# Green and Dark Green are combined for the "green project" KPI.
status_counts = filtered_df['Pulse_Status'].value_counts()
red_count = status_counts.get('Red', 0)
yellow_count = status_counts.get('Yellow', 0)
green_count = status_counts.get('Green', 0) + status_counts.get('Dark Green', 0)
total_entries = len(filtered_df)

# Percentage of entries that are Green or Dark Green
green_pct = (green_count / total_entries * 100) if total_entries > 0 else 0

# ---------------------------------------------------------------------------
# WEEK-OVER-WEEK (WoW) COMPARISON
# Compare the current week's average pulse to the immediately preceding week.
# Handles year rollover (if current week is 1, previous is week 52 of prior year).
# ---------------------------------------------------------------------------
if selected_week and selected_year:
    prev_wk = selected_week - 1
    prev_yr = selected_year
    # Handle year boundary: week 0 wraps to week 52 of the prior year
    if prev_wk < 1:
        prev_wk = 52
        prev_yr -= 1
    # Filter the full dataset for the previous week
    prev_df = df[(df['Year'] == prev_yr) & (df['Wk'] == prev_wk)]
    prev_avg = prev_df['Total Score'].mean() if not prev_df.empty else None
    # WoW delta: positive = improvement, negative = decline
    wow_delta = avg_pulse - prev_avg if prev_avg is not None else None
else:
    prev_avg = None
    wow_delta = None

# ---------------------------------------------------------------------------
# DIMENSION AVERAGES
# Compute the mean score for each of the 8 dimensions, sorted ascending
# so that the worst (lowest-scoring) dimensions are first.
# ---------------------------------------------------------------------------
dim_means = filtered_df[SCORE_DIMENSIONS].mean().sort_values()
worst_dim = dim_means.index[0]            # Name of the weakest dimension
worst_dim_score = dim_means.iloc[0]       # Its average score (0.00 - 3.00)
second_worst_dim = dim_means.index[1] if len(dim_means) > 1 else None
second_worst_score = dim_means.iloc[1] if len(dim_means) > 1 else None

# ---------------------------------------------------------------------------
# WORST REGION
# Identify the region with the lowest average pulse for the Resolution box.
# ---------------------------------------------------------------------------
region_means = filtered_df.groupby('Region')['Total Score'].mean()
worst_region = region_means.idxmin() if not region_means.empty else 'N/A'
worst_region_score = region_means.min() if not region_means.empty else 0

# ---------------------------------------------------------------------------
# REGION-LEVEL WoW DELTAS
# Find regions that have declined by more than 0.5 points week-over-week.
# These surface as "Complications" in the SCR narrative.
# ---------------------------------------------------------------------------
if prev_avg is not None and not prev_df.empty:
    prev_region_means = prev_df.groupby('Region')['Total Score'].mean()
    # subtract aligns on region index; fill_value=0 handles new/missing regions
    region_deltas = region_means.subtract(prev_region_means, fill_value=0)
    declining_regions = region_deltas[region_deltas < -0.5]
else:
    declining_regions = pd.Series(dtype=float)

# List of project names currently in Red status (for escalation callouts)
red_projects = filtered_df[filtered_df['Pulse_Status'] == 'Red']['Project'].unique()

# ============================================================================
# HEADER - Status badge + pulse score vs target + WoW delta
# ============================================================================
# Determine the overall portfolio status badge based on average pulse.
# These thresholds are slightly different from per-project thresholds
# because they characterize the *portfolio* health, not individual projects.
if avg_pulse >= 20:
    status_badge = '<span class="badge badge-exceptional">EXCEPTIONAL</span>'
elif avg_pulse >= target:
    status_badge = '<span class="badge badge-success">ON TRACK</span>'
elif avg_pulse >= 14:
    status_badge = '<span class="badge badge-warning">AT RISK</span>'
else:
    status_badge = '<span class="badge badge-critical">OFF TRACK</span>'

# WoW text snippet: shows the week-over-week delta with green/red coloring.
# Only rendered when previous-week data exists.
wow_text = f' | WoW: <span class="{"delta-positive" if wow_delta >= 0 else "delta-negative"}">{wow_delta:+.2f}</span>' if wow_delta is not None else ''

# Render the header card with flexbox layout: title on left, badge + score on right
st.markdown(f"""
<div class="exec-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <p class="main-header" style="margin: 0;">Executive Summary</p>
            <p style="color: #94a3b8;">Week {selected_week}, {selected_year}</p>
        </div>
        <div style="text-align: right;">
            {status_badge}
            <p style="color: #94a3b8; margin-top: 8px;">
                Pulse: <b style="color: white;">{avg_pulse:.1f}</b> vs {target:.0f}
                (<span class="{'delta-positive' if variance >= 0 else 'delta-negative'}">{variance:+.1f}</span>)
                {wow_text}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SCR BOXES (Situation / Complications / Resolution)
# This is the McKinsey "SCR" narrative framework:
#   Situation    - neutral summary of the current state
#   Complications - the problems / red flags that need attention
#   Resolution   - recommended actions to address the complications
# Each box is auto-generated from the computed metrics above.
# ============================================================================
situation_text = f"Week {selected_week}: Portfolio pulse at {avg_pulse:.1f} ({variance:+.1f} vs target of {target:.0f}). {total_entries} project entries across {filtered_df['Region'].nunique()} regions."

# Build the complications list dynamically based on threshold breaches
complications = []
if red_count > max_red_target:
    complications.append(f"{red_count} projects in red status (>{max_red_target} threshold)")
if green_pct < green_pct_target:
    complications.append(f"Only {green_pct:.0f}% green vs {green_pct_target}% target")
if wow_delta is not None and wow_delta < 0:
    complications.append(f"Pulse declined {abs(wow_delta):.2f} pts WoW")
# Call out any region that dropped more than 0.5 points WoW
for region, delta in declining_regions.items():
    complications.append(f"{region} region down {abs(delta):.1f} pts WoW")
if not complications:
    complications.append("No major complications identified")

# Build the resolutions list: actionable next steps derived from the data
resolutions = []
resolutions.append(f"Focus on {worst_dim} (avg {worst_dim_score:.2f})")
if second_worst_dim:
    resolutions.append(f"and {second_worst_dim} (avg {second_worst_score:.2f})")
if len(red_projects) > 0:
    # Show at most 3 red project names to keep the box concise
    proj_list = ', '.join(red_projects[:3])
    resolutions.append(f"Escalate: {proj_list}")
resolutions.append(f"Deploy support to {worst_region} region (avg {worst_region_score:.1f})")

# Render the three SCR columns side by side with color-coded borders
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="scr-situation">
        <p class="scr-title" style="color: #3b82f6;">Situation</p>
        <p style="color: #e2e8f0; font-size: 0.9rem;">{situation_text}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Join complications into an HTML unordered list
    comp_html = ''.join(f'<li style="margin: 4px 0;">{c}</li>' for c in complications)
    st.markdown(f"""
    <div class="scr-complication">
        <p class="scr-title" style="color: #ef4444;">Complications</p>
        <ul style="color: #e2e8f0; font-size: 0.9rem; padding-left: 16px; margin: 0;">{comp_html}</ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Join resolutions into an HTML unordered list
    res_html = ''.join(f'<li style="margin: 4px 0;">{r}</li>' for r in resolutions)
    st.markdown(f"""
    <div class="scr-resolution">
        <p class="scr-title" style="color: #22c55e;">Resolution</p>
        <ul style="color: #e2e8f0; font-size: 0.9rem; padding-left: 16px; margin: 0;">{res_html}</ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# KPI CARDS - Five headline metrics displayed as styled cards in a row
#
# Each card has: value (big number), label (description), optional subtitle,
# and an optional CSS class for conditional coloring (e.g. "critical" for red).
# ============================================================================
cols = st.columns(5)
kpi_items = [
    ("Pulse Score", f"{avg_pulse:.1f}", f"{variance:+.1f} vs target", ""),
    ("Green Projects", f"{green_count}", f"{green_pct:.0f}% of total", "success"),
    ("Yellow Projects", f"{yellow_count}", "", "warning"),
    ("Red Projects", f"{red_count}", f"{'OK' if red_count <= max_red_target else 'OVER LIMIT'}", "critical" if red_count > max_red_target else ""),
    ("Total Entries", f"{total_entries}", f"{filtered_df['Project'].nunique()} projects", ""),
]

# Iterate over columns and KPI tuples, rendering each as an HTML card
for col, (label, value, sub, css) in zip(cols, kpi_items):
    with col:
        sub_html = f'<p style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">{sub}</p>' if sub else ''
        st.markdown(f"""
        <div class="kpi-container {css}">
            <p class="kpi-value">{value}</p>
            <p class="kpi-label">{label}</p>
            {sub_html}
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# CHARTS ROW 1: Variance Bullet | Trend + Forecast
#
# Left (1/3 width): Bullet gauge showing current pulse vs target and stretch.
# Right (2/3 width): Time-series line chart with historical pulse and a
#   linear forecast extrapolation.
# ============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    # Bullet chart: a horizontal gauge showing avg_pulse relative to
    # target (good) and stretch (exceptional) thresholds
    fig_bullet = chart_variance_bullet(avg_pulse, target, stretch)
    st.plotly_chart(fig_bullet, use_container_width=True)

with col2:
    # Trend + forecast: plots weekly average pulse over time with a
    # dashed linear forecast extending into future weeks.
    # Uses the FULL dataset (df) so the time series is complete.
    fig_trend = chart_trend_forecast(df, target)
    st.plotly_chart(fig_trend, use_container_width=True)

# ============================================================================
# INSIGHT CALLOUT
# Auto-generated narrative insight highlighting the weakest dimension and
# its drag on the portfolio.  If the dimension averages below 2.0 (out of 3),
# an additional "what-if" sentence quantifies the improvement opportunity.
# ============================================================================
st.markdown(f"""
<div class="insight-callout">
    <b>Key Insight:</b> The weakest dimension is <b>{worst_dim}</b> (avg {worst_dim_score:.2f}/3.00),
    pulling the portfolio pulse {3.0 - worst_dim_score:.2f} points below potential.
    {'Improving ' + worst_dim + ' by just 0.5 points across all projects would lift the portfolio pulse by ~0.5 points.' if worst_dim_score < 2.0 else ''}
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CHARTS ROW 2: Waterfall | Pareto
#
# Left: Waterfall decomposition showing how each dimension's average
#   contributes to (or drags down) the total pulse score.
# Right: Pareto chart identifying which projects/groups account for
#   the majority of the gap to target (80/20 rule).
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    fig_waterfall = chart_waterfall_decomposition(filtered_df)
    st.plotly_chart(fig_waterfall, use_container_width=True)

with col2:
    fig_pareto = chart_pareto(filtered_df, target)
    st.plotly_chart(fig_pareto, use_container_width=True)

# ============================================================================
# RECOMMENDATIONS TABLE
# Auto-generated action items ranked by priority (P1 = immediate, P2 = soon).
# These are data-driven recommendations, not manually authored:
#   - P1: Worst dimension improvement program
#   - P1: Red project escalation (if any exist)
#   - P2: Worst region resource deployment
#   - P2: Green % improvement (if below target)
# ============================================================================
st.markdown("### Recommendations")

recommendations = []
# 1. Worst dimension - always present as the primary action item
recommendations.append({
    'Priority': 'P1',
    'Category': 'Dimension Gap',
    'Action': f'Launch targeted improvement program for {worst_dim}',
    'Rationale': f'Lowest scoring dimension at {worst_dim_score:.2f}/3.00',
    'Owner': '',
    'Timeline': 'Next 2 weeks',
})

# 2. Red projects - only included if any projects are in Red status
if len(red_projects) > 0:
    recommendations.append({
        'Priority': 'P1',
        'Category': 'Critical Projects',
        'Action': f'Escalate and assign dedicated support to {len(red_projects)} red projects',
        'Rationale': f'{", ".join(red_projects[:3])} require immediate attention',
        'Owner': '',
        'Timeline': 'Immediate',
    })

# 3. Worst region - always present to focus regional support
recommendations.append({
    'Priority': 'P2',
    'Category': 'Regional Focus',
    'Action': f'Deploy additional resources to {worst_region} region',
    'Rationale': f'Lowest regional average at {worst_region_score:.1f}',
    'Owner': '',
    'Timeline': 'This week',
})

# 4. Green % gap - only included when green percentage is below target
if green_pct < green_pct_target:
    gap = green_pct_target - green_pct
    recommendations.append({
        'Priority': 'P2',
        'Category': 'Portfolio Health',
        'Action': f'Move {int(gap * total_entries / 100) + 1} yellow projects to green',
        'Rationale': f'Green % at {green_pct:.0f}% vs {green_pct_target}% target',
        'Owner': '',
        'Timeline': 'Next 4 weeks',
    })

# Render the recommendations as a Streamlit dataframe (sortable, full-width)
rec_df = pd.DataFrame(recommendations)
st.dataframe(rec_df, use_container_width=True, hide_index=True)
