"""
Pulse Dashboard - Rankings & Leaderboard

Top/Bottom performing Regions, Areas, or Projects ranked by any metric.
Includes: horizontal bar chart, dimension heatmap, trend sparklines,
most improved/declined movers, and percentile badges.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.sidebar import render_sidebar
from utils.styles import (
    inject_css, SCORE_DIMENSIONS, STATUS_CONFIG, STATUS_ORDER,
    get_pulse_status, get_pulse_color, pulse_css_class, score_css_class,
    get_plotly_theme, AXIS_STYLE, DIMENSION_COLORS, REGION_LINE_COLORS,
)
from utils.mckinsey_charts import _apply_theme, mini_sparkline_svg

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

df = st.session_state.df
selected_regions = st.session_state.get('selected_regions', [])
target = st.session_state.get('pulse_target', 17.0)

st.markdown('<p class="main-header">Rankings & Leaderboard</p>', unsafe_allow_html=True)

# ============================================================================
# WEEK SCOPE — same pattern as Drill Down
# ============================================================================
mw_df = df.copy()
if selected_regions:
    mw_df = mw_df[mw_df['Region'].isin(selected_regions)]

yw_pairs = (
    mw_df[['Year', 'Wk']].drop_duplicates()
    .sort_values(['Year', 'Wk'])
    .values.tolist()
)
week_labels = [f"{int(y)}-W{int(w):02d}" for y, w in yw_pairs]
week_options = ['All Weeks (Average)'] + week_labels

sidebar_year = st.session_state.get('selected_year')
sidebar_week = st.session_state.get('selected_week')
sidebar_label = f"{int(sidebar_year)}-W{int(sidebar_week):02d}" if sidebar_year and sidebar_week else None
default_idx = (week_labels.index(sidebar_label) + 1) if sidebar_label in week_labels else 0

# ============================================================================
# CONTROLS ROW
# ============================================================================
c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 0.8])

with c1:
    week_scope = st.selectbox('Week Scope', week_options, index=default_idx, key='rank_week_scope')

with c2:
    direction = st.radio('Direction', ['Bottom', 'Top'], index=0, key='rank_direction', horizontal=True)

with c3:
    level = st.selectbox('Level', ['Project', 'Area', 'Region'], key='rank_level')

with c4:
    metric_options = ['Total Score'] + list(SCORE_DIMENSIONS)
    metric = st.selectbox('Metric', metric_options, key='rank_metric')

with c5:
    count = st.slider('Count', 5, 30, 10, step=5, key='rank_count')

# Optional region/area filter
filter_cols = st.columns([1, 1, 2])
with filter_cols[0]:
    all_regions = sorted(mw_df['Region'].dropna().unique())
    filter_region = st.selectbox('Filter Region', ['All'] + all_regions, key='rank_filter_region')

with filter_cols[1]:
    if filter_region != 'All':
        all_areas = sorted(mw_df[mw_df['Region'] == filter_region]['Area'].dropna().unique())
    else:
        all_areas = sorted(mw_df['Area'].dropna().unique())
    filter_area = st.selectbox('Filter Area', ['All'] + all_areas, key='rank_filter_area')

# ============================================================================
# BUILD ANALYSIS DATAFRAME
# ============================================================================
if week_scope == 'All Weeks (Average)':
    agg_cols = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg_cols['Total Score'] = 'mean'
    latest_idx = mw_df.sort_values(['Year', 'Wk']).groupby('Project').tail(1).index
    meta_df = mw_df.loc[latest_idx, ['Project', 'Region', 'Area', 'PM Name']].drop_duplicates('Project')
    avg_scores = mw_df.groupby('Project').agg(agg_cols).reset_index()
    for dim in SCORE_DIMENSIONS:
        avg_scores[dim] = avg_scores[dim].round(2)
    avg_scores['Total Score'] = avg_scores['Total Score'].round(2)
    analysis_df = meta_df.merge(avg_scores, on='Project', how='inner')
    scope_label = f"Average across {len(yw_pairs)} weeks"
else:
    yr_str, wk_str = week_scope.split('-W')
    sel_yr, sel_wk = int(yr_str), int(wk_str)
    analysis_df = mw_df[(mw_df['Year'] == sel_yr) & (mw_df['Wk'] == sel_wk)].copy()
    scope_label = f"Week {week_scope}"

if analysis_df.empty:
    st.warning("No data for the selected scope.")
    st.stop()

# Apply region/area filter
if filter_region != 'All':
    analysis_df = analysis_df[analysis_df['Region'] == filter_region]
if filter_area != 'All':
    analysis_df = analysis_df[analysis_df['Area'] == filter_area]

if analysis_df.empty:
    st.warning("No data after applying filters.")
    st.stop()

st.caption(f"{scope_label} | {len(analysis_df)} projects | Ranked by **{metric}** | {direction} {count}")

# ============================================================================
# AGGREGATE BY LEVEL
# ============================================================================
if level == 'Project':
    rank_df = analysis_df.copy()
    rank_df['_entity'] = rank_df['Project']
    rank_df['_count'] = 1
    entity_col = 'Project'
elif level == 'Area':
    agg = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg['Total Score'] = 'mean'
    agg['Project'] = 'nunique'
    rank_df = analysis_df.groupby(['Region', 'Area']).agg(agg).reset_index()
    rank_df.rename(columns={'Project': '_count'}, inplace=True)
    rank_df['_entity'] = rank_df['Area']
    entity_col = 'Area'
else:  # Region
    agg = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg['Total Score'] = 'mean'
    agg['Project'] = 'nunique'
    rank_df = analysis_df.groupby('Region').agg(agg).reset_index()
    rank_df.rename(columns={'Project': '_count'}, inplace=True)
    rank_df['_entity'] = rank_df['Region']
    entity_col = 'Region'

# Round all scores
for col in SCORE_DIMENSIONS + ['Total Score']:
    if col in rank_df.columns:
        rank_df[col] = rank_df[col].round(2)

# Sort and slice
ascending = direction == 'Bottom'
rank_df = rank_df.sort_values(metric, ascending=ascending).head(count).reset_index(drop=True)
rank_df['_rank'] = range(1, len(rank_df) + 1)

# Compute percentiles from full dataset
if level == 'Project':
    full_values = analysis_df[metric].dropna()
elif level == 'Area':
    full_agg = analysis_df.groupby('Area')[metric].mean()
    full_values = full_agg
else:
    full_agg = analysis_df.groupby('Region')[metric].mean()
    full_values = full_agg

rank_df['_percentile'] = rank_df[metric].apply(
    lambda x: (full_values < x).sum() / len(full_values) * 100 if len(full_values) > 0 else 50
)

# ============================================================================
# HORIZONTAL BAR CHART + SCORECARD HEATMAP (side by side)
# ============================================================================
col_chart, col_heat = st.columns([1.2, 1.8])

with col_chart:
    st.markdown("**Ranked Bar Chart**")

    entities = rank_df['_entity'].tolist()
    values = rank_df[metric].tolist()
    counts = rank_df.get('_count', pd.Series([1] * len(rank_df))).tolist()
    percentiles = rank_df['_percentile'].tolist()

    if metric == 'Total Score':
        colors = [get_pulse_color(v) for v in values]
        x_range = [0, 24]
    else:
        from utils.mckinsey_charts import _dim_color
        colors = [_dim_color(v) for v in values]
        x_range = [0, 3.2]

    # Build text labels with percentile badge
    texts = []
    for v, c, p in zip(values, counts, percentiles):
        if level == 'Project':
            texts.append(f"{v:.1f}  (P{p:.0f})")
        else:
            texts.append(f"{v:.1f}  ({c} proj, P{p:.0f})")

    fig = go.Figure(go.Bar(
        y=entities[::-1],
        x=values[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=texts[::-1],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=10),
    ))
    _apply_theme(fig)
    fig.update_layout(
        height=max(250, len(entities) * 32 + 80),
        margin=dict(l=140, r=100, t=10, b=30),
    )
    fig.update_xaxes(range=x_range, title_text=metric)

    # Target/average line
    portfolio_avg = analysis_df[metric].mean()
    fig.add_vline(
        x=portfolio_avg, line_dash='dash', line_color='#60a5fa', line_width=1,
        annotation_text=f'Avg ({portfolio_avg:.1f})',
        annotation_font_color='#94a3b8',
        annotation_position='top',
    )
    if metric == 'Total Score':
        fig.add_vline(
            x=target, line_dash='dot', line_color='white', line_width=1,
            annotation_text=f'Target ({target:.0f})',
            annotation_font_color='#94a3b8',
        )

    st.plotly_chart(fig, use_container_width=True)

with col_heat:
    st.markdown("**Dimension Scorecard**")

    # Build heatmap HTML table
    html = (
        '<div class="matrix-container" style="max-height:500px; overflow-y:auto;">'
        '<table class="matrix-table">'
        '<thead><tr>'
        '<th style="text-align:left; min-width:30px;">#</th>'
        f'<th style="text-align:left; min-width:120px;">{level}</th>'
        '<th>Pulse</th>'
    )
    for dim in SCORE_DIMENSIONS:
        short = dim.replace('PM Performance', 'PM Perf')
        html += f'<th>{short}</th>'
    html += '<th>Pctl</th></tr></thead><tbody>'

    for _, row in rank_df.iterrows():
        rank_num = int(row['_rank'])
        entity = row['_entity']
        total = row['Total Score']
        pctl = row['_percentile']
        p_cls = pulse_css_class(total)

        # Percentile badge color
        if pctl >= 75:
            pctl_style = 'background:#059669; color:white;'
        elif pctl >= 50:
            pctl_style = 'background:#22c55e; color:white;'
        elif pctl >= 25:
            pctl_style = 'background:#f59e0b; color:#212529;'
        else:
            pctl_style = 'background:#ef4444; color:white;'

        html += '<tr>'
        html += f'<td style="text-align:left; color:#64748b;">{rank_num}</td>'
        html += f'<td style="text-align:left; font-weight:600; color:#e2e8f0;">{entity}</td>'
        html += f'<td><span class="score-cell {p_cls}">{total:.1f}</span></td>'

        for dim in SCORE_DIMENSIONS:
            val = row[dim]
            cls = score_css_class(val)
            html += f'<td><span class="score-cell {cls}">{val:.2f}</span></td>'

        html += (
            f'<td><span style="display:inline-block; padding:2px 6px; border-radius:10px;'
            f' font-size:0.65rem; font-weight:600; {pctl_style}">P{pctl:.0f}</span></td>'
        )
        html += '</tr>'

    html += '</tbody></table></div>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# TREND SPARKLINES
# ============================================================================
st.markdown("---")
st.markdown("**Trend Sparklines** (score trajectory over all weeks)")

# Build weekly history for each ranked entity
sparkline_data = []
for _, row in rank_df.iterrows():
    entity = row['_entity']
    if level == 'Project':
        hist = mw_df[mw_df['Project'] == entity].sort_values(['Year', 'Wk'])
        weekly_vals = hist.groupby(['Year', 'Wk'])[metric].mean().values
    elif level == 'Area':
        hist = mw_df[mw_df['Area'] == entity].sort_values(['Year', 'Wk'])
        weekly_vals = hist.groupby(['Year', 'Wk'])[metric].mean().values
    else:
        hist = mw_df[mw_df['Region'] == entity].sort_values(['Year', 'Wk'])
        weekly_vals = hist.groupby(['Year', 'Wk'])[metric].mean().values

    if len(weekly_vals) >= 2:
        delta = weekly_vals[-1] - weekly_vals[0]
        recent_delta = weekly_vals[-1] - weekly_vals[-2] if len(weekly_vals) >= 2 else 0
    else:
        delta = 0
        recent_delta = 0

    sparkline_data.append({
        'entity': entity,
        'values': weekly_vals,
        'current': weekly_vals[-1] if len(weekly_vals) > 0 else 0,
        'delta': delta,
        'recent_delta': recent_delta,
        'n_weeks': len(weekly_vals),
    })

# Render as HTML table with inline SVG sparklines
if metric == 'Total Score':
    spark_color_fn = get_pulse_color
else:
    spark_color_fn = lambda v: _dim_color(v)

spark_html = (
    '<div class="matrix-container">'
    '<table class="matrix-table">'
    '<thead><tr>'
    f'<th style="text-align:left;">{level}</th>'
    '<th>Current</th>'
    '<th>Trend</th>'
    '<th>Overall Delta</th>'
    '<th>Last Week Delta</th>'
    '<th>Weeks</th>'
    '</tr></thead><tbody>'
)

for s in sparkline_data:
    color = spark_color_fn(s['current'])
    svg = mini_sparkline_svg(s['values'], width=100, height=24, color=color)

    delta_cls = 'delta-positive' if s['delta'] >= 0 else 'delta-negative'
    recent_cls = 'delta-positive' if s['recent_delta'] >= 0 else 'delta-negative'

    spark_html += '<tr>'
    spark_html += f'<td style="text-align:left; font-weight:600; color:#e2e8f0;">{s["entity"]}</td>'
    spark_html += f'<td style="color:{color}; font-weight:600;">{s["current"]:.1f}</td>'
    spark_html += f'<td>{svg}</td>'
    spark_html += f'<td><span class="{delta_cls}">{s["delta"]:+.1f}</span></td>'
    spark_html += f'<td><span class="{recent_cls}">{s["recent_delta"]:+.1f}</span></td>'
    spark_html += f'<td style="color:#94a3b8;">{s["n_weeks"]}</td>'
    spark_html += '</tr>'

spark_html += '</tbody></table></div>'
st.markdown(spark_html, unsafe_allow_html=True)

# ============================================================================
# MOVERS — Most Improved & Most Declined
# ============================================================================
st.markdown("---")
st.markdown("**Movers** (biggest score changes over all weeks)")

# Compute deltas for ALL entities at the selected level, then show top 5 improved / declined
if level == 'Project':
    group_col = 'Project'
elif level == 'Area':
    group_col = 'Area'
else:
    group_col = 'Region'

# Get first and last week scores for each entity
first_week = mw_df.sort_values(['Year', 'Wk']).groupby(group_col).first()[metric]
last_week = mw_df.sort_values(['Year', 'Wk']).groupby(group_col).last()[metric]

movers_df = pd.DataFrame({
    'first': first_week,
    'last': last_week,
}).dropna()
movers_df['delta'] = movers_df['last'] - movers_df['first']
movers_df = movers_df[movers_df['delta'] != 0]  # exclude no-change

# Also compute recent delta (last 2 weeks)
sorted_yw = mw_df[['Year', 'Wk']].drop_duplicates().sort_values(['Year', 'Wk'])
if len(sorted_yw) >= 2:
    prev_yw = sorted_yw.iloc[-2]
    last_yw = sorted_yw.iloc[-1]

    prev_scores = mw_df[(mw_df['Year'] == prev_yw['Year']) & (mw_df['Wk'] == prev_yw['Wk'])].groupby(group_col)[metric].mean()
    last_scores = mw_df[(mw_df['Year'] == last_yw['Year']) & (mw_df['Wk'] == last_yw['Wk'])].groupby(group_col)[metric].mean()

    recent_df = pd.DataFrame({'prev': prev_scores, 'last': last_scores}).dropna()
    recent_df['recent_delta'] = recent_df['last'] - recent_df['prev']
    movers_df = movers_df.join(recent_df[['recent_delta']], how='left')
else:
    movers_df['recent_delta'] = 0

movers_df['recent_delta'] = movers_df['recent_delta'].fillna(0)

mover_count = 5

m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown('<span style="color:#22c55e; font-weight:700;">Most Improved</span>', unsafe_allow_html=True)
    improved = movers_df.nlargest(mover_count, 'delta')

    if improved.empty:
        st.info("No improvements detected.")
    else:
        imp_html = (
            '<div class="matrix-container"><table class="matrix-table">'
            '<thead><tr>'
            f'<th style="text-align:left;">{level}</th>'
            '<th>First</th><th>Last</th><th>Change</th><th>Recent</th>'
            '</tr></thead><tbody>'
        )
        for entity, row in improved.iterrows():
            delta_cls = 'delta-positive' if row['delta'] >= 0 else 'delta-negative'
            recent_cls = 'delta-positive' if row['recent_delta'] >= 0 else 'delta-negative'
            imp_html += (
                f'<tr>'
                f'<td style="text-align:left; font-weight:600; color:#e2e8f0;">{entity}</td>'
                f'<td style="color:#94a3b8;">{row["first"]:.1f}</td>'
                f'<td style="color:#94a3b8;">{row["last"]:.1f}</td>'
                f'<td><span class="{delta_cls}" style="font-weight:700;">{row["delta"]:+.1f}</span></td>'
                f'<td><span class="{recent_cls}">{row["recent_delta"]:+.1f}</span></td>'
                f'</tr>'
            )
        imp_html += '</tbody></table></div>'
        st.markdown(imp_html, unsafe_allow_html=True)

with m_col2:
    st.markdown('<span style="color:#ef4444; font-weight:700;">Most Declined</span>', unsafe_allow_html=True)
    declined = movers_df.nsmallest(mover_count, 'delta')
    declined = declined[declined['delta'] < 0]

    if declined.empty:
        st.success("No declines detected.")
    else:
        dec_html = (
            '<div class="matrix-container"><table class="matrix-table">'
            '<thead><tr>'
            f'<th style="text-align:left;">{level}</th>'
            '<th>First</th><th>Last</th><th>Change</th><th>Recent</th>'
            '</tr></thead><tbody>'
        )
        for entity, row in declined.iterrows():
            delta_cls = 'delta-positive' if row['delta'] >= 0 else 'delta-negative'
            recent_cls = 'delta-positive' if row['recent_delta'] >= 0 else 'delta-negative'
            dec_html += (
                f'<tr>'
                f'<td style="text-align:left; font-weight:600; color:#e2e8f0;">{entity}</td>'
                f'<td style="color:#94a3b8;">{row["first"]:.1f}</td>'
                f'<td style="color:#94a3b8;">{row["last"]:.1f}</td>'
                f'<td><span class="{delta_cls}" style="font-weight:700;">{row["delta"]:+.1f}</span></td>'
                f'<td><span class="{recent_cls}">{row["recent_delta"]:+.1f}</span></td>'
                f'</tr>'
            )
        dec_html += '</tbody></table></div>'
        st.markdown(dec_html, unsafe_allow_html=True)
