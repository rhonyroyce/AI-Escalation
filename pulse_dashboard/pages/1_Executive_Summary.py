"""
Pulse Dashboard - Executive Summary (McKinsey SCR Format)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.sidebar import render_sidebar
from utils.styles import (
    inject_css, SCORE_DIMENSIONS, STATUS_CONFIG, get_pulse_status,
)
from utils.mckinsey_charts import (
    chart_variance_bullet, chart_trend_forecast,
    chart_waterfall_decomposition, chart_pareto,
)

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

df = st.session_state.df
target = st.session_state.get('pulse_target', 17.0)
stretch = st.session_state.get('pulse_stretch', 19.0)
green_pct_target = st.session_state.get('green_pct_target', 80)
max_red_target = st.session_state.get('max_red_target', 3)
selected_week = st.session_state.get('selected_week')
selected_year = st.session_state.get('selected_year')

# ============================================================================
# METRICS
# ============================================================================
avg_pulse = filtered_df['Total Score'].mean()
variance = avg_pulse - target
status_counts = filtered_df['Pulse_Status'].value_counts()
red_count = status_counts.get('Red', 0)
yellow_count = status_counts.get('Yellow', 0)
green_count = status_counts.get('Green', 0) + status_counts.get('Dark Green', 0)
total_entries = len(filtered_df)
green_pct = (green_count / total_entries * 100) if total_entries > 0 else 0

# WoW comparison
if selected_week and selected_year:
    prev_wk = selected_week - 1
    prev_yr = selected_year
    if prev_wk < 1:
        prev_wk = 52
        prev_yr -= 1
    prev_df = df[(df['Year'] == prev_yr) & (df['Wk'] == prev_wk)]
    prev_avg = prev_df['Total Score'].mean() if not prev_df.empty else None
    wow_delta = avg_pulse - prev_avg if prev_avg is not None else None
else:
    prev_avg = None
    wow_delta = None

# Dimension averages
dim_means = filtered_df[SCORE_DIMENSIONS].mean().sort_values()
worst_dim = dim_means.index[0]
worst_dim_score = dim_means.iloc[0]
second_worst_dim = dim_means.index[1] if len(dim_means) > 1 else None
second_worst_score = dim_means.iloc[1] if len(dim_means) > 1 else None

# Worst region
region_means = filtered_df.groupby('Region')['Total Score'].mean()
worst_region = region_means.idxmin() if not region_means.empty else 'N/A'
worst_region_score = region_means.min() if not region_means.empty else 0

# Region WoW
if prev_avg is not None and not prev_df.empty:
    prev_region_means = prev_df.groupby('Region')['Total Score'].mean()
    region_deltas = region_means.subtract(prev_region_means, fill_value=0)
    declining_regions = region_deltas[region_deltas < -0.5]
else:
    declining_regions = pd.Series(dtype=float)

# Red projects
red_projects = filtered_df[filtered_df['Pulse_Status'] == 'Red']['Project'].unique()

# ============================================================================
# HEADER
# ============================================================================
if avg_pulse >= 20:
    status_badge = '<span class="badge badge-exceptional">EXCEPTIONAL</span>'
elif avg_pulse >= target:
    status_badge = '<span class="badge badge-success">ON TRACK</span>'
elif avg_pulse >= 14:
    status_badge = '<span class="badge badge-warning">AT RISK</span>'
else:
    status_badge = '<span class="badge badge-critical">OFF TRACK</span>'

wow_text = f' | WoW: <span class="{"delta-positive" if wow_delta >= 0 else "delta-negative"}">{wow_delta:+.2f}</span>' if wow_delta is not None else ''

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
# ============================================================================
situation_text = f"Week {selected_week}: Portfolio pulse at {avg_pulse:.1f} ({variance:+.1f} vs target of {target:.0f}). {total_entries} project entries across {filtered_df['Region'].nunique()} regions."

complications = []
if red_count > max_red_target:
    complications.append(f"{red_count} projects in red status (>{max_red_target} threshold)")
if green_pct < green_pct_target:
    complications.append(f"Only {green_pct:.0f}% green vs {green_pct_target}% target")
if wow_delta is not None and wow_delta < 0:
    complications.append(f"Pulse declined {abs(wow_delta):.2f} pts WoW")
for region, delta in declining_regions.items():
    complications.append(f"{region} region down {abs(delta):.1f} pts WoW")
if not complications:
    complications.append("No major complications identified")

resolutions = []
resolutions.append(f"Focus on {worst_dim} (avg {worst_dim_score:.2f})")
if second_worst_dim:
    resolutions.append(f"and {second_worst_dim} (avg {second_worst_score:.2f})")
if len(red_projects) > 0:
    proj_list = ', '.join(red_projects[:3])
    resolutions.append(f"Escalate: {proj_list}")
resolutions.append(f"Deploy support to {worst_region} region (avg {worst_region_score:.1f})")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="scr-situation">
        <p class="scr-title" style="color: #3b82f6;">Situation</p>
        <p style="color: #e2e8f0; font-size: 0.9rem;">{situation_text}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    comp_html = ''.join(f'<li style="margin: 4px 0;">{c}</li>' for c in complications)
    st.markdown(f"""
    <div class="scr-complication">
        <p class="scr-title" style="color: #ef4444;">Complications</p>
        <ul style="color: #e2e8f0; font-size: 0.9rem; padding-left: 16px; margin: 0;">{comp_html}</ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    res_html = ''.join(f'<li style="margin: 4px 0;">{r}</li>' for r in resolutions)
    st.markdown(f"""
    <div class="scr-resolution">
        <p class="scr-title" style="color: #22c55e;">Resolution</p>
        <ul style="color: #e2e8f0; font-size: 0.9rem; padding-left: 16px; margin: 0;">{res_html}</ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# KPI CARDS
# ============================================================================
cols = st.columns(5)
kpi_items = [
    ("Pulse Score", f"{avg_pulse:.1f}", f"{variance:+.1f} vs target", ""),
    ("Green Projects", f"{green_count}", f"{green_pct:.0f}% of total", "success"),
    ("Yellow Projects", f"{yellow_count}", "", "warning"),
    ("Red Projects", f"{red_count}", f"{'OK' if red_count <= max_red_target else 'OVER LIMIT'}", "critical" if red_count > max_red_target else ""),
    ("Total Entries", f"{total_entries}", f"{filtered_df['Project'].nunique()} projects", ""),
]

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
# ============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    fig_bullet = chart_variance_bullet(avg_pulse, target, stretch)
    st.plotly_chart(fig_bullet, use_container_width=True)

with col2:
    fig_trend = chart_trend_forecast(df, target)
    st.plotly_chart(fig_trend, use_container_width=True)

# ============================================================================
# INSIGHT CALLOUT
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
# ============================================================================
st.markdown("### Recommendations")

recommendations = []
# 1. Worst dimension
recommendations.append({
    'Priority': 'P1',
    'Category': 'Dimension Gap',
    'Action': f'Launch targeted improvement program for {worst_dim}',
    'Rationale': f'Lowest scoring dimension at {worst_dim_score:.2f}/3.00',
    'Owner': '',
    'Timeline': 'Next 2 weeks',
})

# 2. Red projects
if len(red_projects) > 0:
    recommendations.append({
        'Priority': 'P1',
        'Category': 'Critical Projects',
        'Action': f'Escalate and assign dedicated support to {len(red_projects)} red projects',
        'Rationale': f'{", ".join(red_projects[:3])} require immediate attention',
        'Owner': '',
        'Timeline': 'Immediate',
    })

# 3. Worst region
recommendations.append({
    'Priority': 'P2',
    'Category': 'Regional Focus',
    'Action': f'Deploy additional resources to {worst_region} region',
    'Rationale': f'Lowest regional average at {worst_region_score:.1f}',
    'Owner': '',
    'Timeline': 'This week',
})

# 4. Green % gap
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

rec_df = pd.DataFrame(recommendations)
st.dataframe(rec_df, use_container_width=True, hide_index=True)
