"""
Pulse Dashboard - Page 7: Rankings & Leaderboard

This page ranks Projects, Areas, or Regions by any scoring metric and
visualises the results through four interconnected sections:

1. **Ranked Horizontal Bar Chart** -- A horizontal bar chart showing the top
   or bottom N entities (Projects, Areas, or Regions) ranked by the selected
   metric (Total Score or any of the 8 dimensions).  Each bar is color-coded
   by the entity's Pulse status tier (for Total Score) or dimension severity
   (for individual dimensions).  A dashed vertical line marks the portfolio
   average; for Total Score, an additional dotted line marks the target.
   Each bar's text label includes the numeric score and a percentile badge
   (e.g. "P25") showing where the entity falls relative to the full dataset.

2. **Dimension Scorecard Heatmap** -- An HTML table beside the bar chart that
   shows each ranked entity's scores across all 8 dimensions, color-coded
   from red (0) through yellow (1-2) to green (3).  The first numeric column
   is the Pulse Total Score, followed by the 8 dimension scores, and finally
   a percentile badge (P0-P100) with quartile-based coloring.

3. **Trend Sparklines** -- For each ranked entity, a mini inline SVG sparkline
   shows the score trajectory over all available weeks.  Alongside the
   sparkline, the table shows the current value, overall delta (first-to-last
   week change), last-week delta, and number of weeks with data.

4. **Movers (Most Improved / Most Declined)** -- Two side-by-side tables
   showing the 5 entities with the largest positive delta (improved) and the
   5 with the largest negative delta (declined) between their first and last
   recorded weeks.  A "Recent" column also shows the change from the
   penultimate to the latest week.

**Control Panel**
A row of controls at the top allows the user to configure:
- **Week Scope**: "All Weeks (Average)" aggregates scores across all weeks;
  selecting a specific week shows only that week's data.
- **Direction**: "Bottom" (ascending, worst first) or "Top" (descending, best
  first).
- **Level**: Project, Area, or Region -- determines the grouping entity.
- **Metric**: Total Score or any of the 8 individual dimensions.
- **Count**: slider for how many entities to show (5-30).
- Optional **Region** and **Area** filters to narrow the ranking scope.

**Scoring Context**
The Pulse scoring system evaluates each project across 8 dimensions
(Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential), each
scored 0-3.  The Total Score (sum) ranges 0-24.  The 4-tier Pulse Status is:
- Red (Critical):            1-13
- Yellow (At Risk):          14-15
- Green (On Track):          16-19
- Dark Green (Exceptional):  20-24

When the Level is "Area" or "Region", scores are averaged across all projects
within that group.

**AI Cache / Session State Context**
This page does not directly use the AI cache keys (``ai_exec_summary``,
``ai_issue_categories``, ``embeddings_index``).  It reads from:
- ``st.session_state.df`` -- the full unfiltered DataFrame (all weeks)
- ``st.session_state.selected_regions`` -- list of regions selected in sidebar
- ``st.session_state.pulse_target`` -- target score (default 17.0)
- ``st.session_state.selected_year`` / ``selected_week`` -- sidebar week selection
"""

# ---------------------------------------------------------------------------
# Path setup -- allow imports from the parent ``pulse_dashboard`` package.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# render_sidebar: applies global filters, returns filtered DataFrame.
# inject_css: pushes the shared dark-theme CSS into the page.
# SCORE_DIMENSIONS: list of 8 dimension column names.
# STATUS_CONFIG / STATUS_ORDER: 4-tier status definitions and ordering.
# get_pulse_status / get_pulse_color: map a Total Score to its status/color.
# pulse_css_class / score_css_class: return CSS class names for HTML table cells.
# get_plotly_theme / AXIS_STYLE: Plotly theme dict and axis styling.
# DIMENSION_COLORS / REGION_LINE_COLORS: palettes for charts.
from utils.sidebar import render_sidebar
from utils.styles import (
    inject_css, SCORE_DIMENSIONS, STATUS_CONFIG, STATUS_ORDER,
    get_pulse_status, get_pulse_color, pulse_css_class, score_css_class,
    get_plotly_theme, AXIS_STYLE, DIMENSION_COLORS, REGION_LINE_COLORS,
)

# _apply_theme: applies the McKinsey dark-theme styling to a Plotly figure.
# mini_sparkline_svg: generates an inline SVG polyline for a sequence of values.
from utils.mckinsey_charts import _apply_theme, mini_sparkline_svg

# ---------------------------------------------------------------------------
# Page initialisation
# ---------------------------------------------------------------------------

# Inject the shared dark-theme CSS.
inject_css()

# Apply sidebar filters and get the filtered DataFrame.
filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

# Full unfiltered DataFrame -- used for multi-week aggregations, trend
# sparklines, and mover calculations that need the complete history.
df = st.session_state.df

# Regions selected in the sidebar (may be empty = all regions).
selected_regions = st.session_state.get('selected_regions', [])

# Configurable target score drawn as a reference line on the bar chart.
target = st.session_state.get('pulse_target', 17.0)

# Page title.
st.markdown('<p class="main-header">Rankings & Leaderboard</p>', unsafe_allow_html=True)

# ============================================================================
# WEEK SCOPE -- build the list of available Year-Week pairs
# ============================================================================
# ``mw_df`` ("multi-week DataFrame") is a copy of the full DF, optionally
# pre-filtered by the sidebar's region selection.  This is the base dataset
# for all ranking computations -- it includes ALL weeks so we can compute
# averages and trends.
mw_df = df.copy()
if selected_regions:
    mw_df = mw_df[mw_df['Region'].isin(selected_regions)]

# Extract the unique (Year, Week) pairs present in the data, sorted
# chronologically.  These become the options for the Week Scope dropdown.
yw_pairs = (
    mw_df[['Year', 'Wk']].drop_duplicates()
    .sort_values(['Year', 'Wk'])
    .values.tolist()
)
# Format as human-readable labels: "2025-W01", "2025-W02", etc.
week_labels = [f"{int(y)}-W{int(w):02d}" for y, w in yw_pairs]
# Prepend the "All Weeks" option for cross-week averaging.
week_options = ['All Weeks (Average)'] + week_labels

# Try to default the Week Scope dropdown to the week selected in the sidebar
# (if the user has chosen a specific Year + Week).
sidebar_year = st.session_state.get('selected_year')
sidebar_week = st.session_state.get('selected_week')
sidebar_label = f"{int(sidebar_year)}-W{int(sidebar_week):02d}" if sidebar_year and sidebar_week else None
# If the sidebar week exists in our list, select it; otherwise default to
# index 0 ("All Weeks (Average)").
default_idx = (week_labels.index(sidebar_label) + 1) if sidebar_label in week_labels else 0

# ============================================================================
# CONTROLS ROW -- user-facing ranking parameters
# ============================================================================
# Five columns in a single row for compact layout.
c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 0.8])

with c1:
    # Week Scope: "All Weeks (Average)" or a specific week.
    week_scope = st.selectbox('Week Scope', week_options, index=default_idx, key='rank_week_scope')

with c2:
    # Direction: "Bottom" shows lowest scores first (worst performers);
    # "Top" shows highest scores first (best performers).
    direction = st.radio('Direction', ['Bottom', 'Top'], index=0, key='rank_direction', horizontal=True)

with c3:
    # Level: the entity type to rank (individual Project, Area, or Region).
    level = st.selectbox('Level', ['Project', 'Area', 'Region'], key='rank_level')

with c4:
    # Metric: which score column to rank by.  Total Score (0-24) or any
    # of the 8 individual dimensions (0-3 each).
    metric_options = ['Total Score'] + list(SCORE_DIMENSIONS)
    metric = st.selectbox('Metric', metric_options, key='rank_metric')

with c5:
    # Count: how many entities to display in the ranking (5 to 30).
    count = st.slider('Count', 5, 30, 10, step=5, key='rank_count')

# ── Optional secondary filters (Region / Area) ──
# These allow further narrowing beyond the sidebar's global filters.
filter_cols = st.columns([1, 1, 2])
with filter_cols[0]:
    all_regions = sorted(mw_df['Region'].dropna().unique())
    filter_region = st.selectbox('Filter Region', ['All'] + all_regions, key='rank_filter_region')

with filter_cols[1]:
    # If a specific region is selected, only show areas within that region.
    if filter_region != 'All':
        all_areas = sorted(mw_df[mw_df['Region'] == filter_region]['Area'].dropna().unique())
    else:
        all_areas = sorted(mw_df['Area'].dropna().unique())
    filter_area = st.selectbox('Filter Area', ['All'] + all_areas, key='rank_filter_area')

# ============================================================================
# BUILD ANALYSIS DATAFRAME
# ============================================================================
# Depending on the Week Scope, we either average across all weeks or filter
# to a single week.
if week_scope == 'All Weeks (Average)':
    # Compute per-project mean scores across all weeks.
    agg_cols = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg_cols['Total Score'] = 'mean'

    # To attach metadata (Region, Area, PM Name) we take each project's
    # *latest* row (by Year/Wk) as the canonical metadata source.
    latest_idx = mw_df.sort_values(['Year', 'Wk']).groupby('Project').tail(1).index
    meta_df = mw_df.loc[latest_idx, ['Project', 'Region', 'Area', 'PM Name']].drop_duplicates('Project')

    # Aggregate scores per project.
    avg_scores = mw_df.groupby('Project').agg(agg_cols).reset_index()
    for dim in SCORE_DIMENSIONS:
        avg_scores[dim] = avg_scores[dim].round(2)
    avg_scores['Total Score'] = avg_scores['Total Score'].round(2)

    # Merge metadata with aggregated scores.
    analysis_df = meta_df.merge(avg_scores, on='Project', how='inner')
    scope_label = f"Average across {len(yw_pairs)} weeks"
else:
    # Parse the selected "YYYY-WNN" label into year and week integers.
    yr_str, wk_str = week_scope.split('-W')
    sel_yr, sel_wk = int(yr_str), int(wk_str)
    # Filter to exactly that week.
    analysis_df = mw_df[(mw_df['Year'] == sel_yr) & (mw_df['Wk'] == sel_wk)].copy()
    scope_label = f"Week {week_scope}"

if analysis_df.empty:
    st.warning("No data for the selected scope.")
    st.stop()

# Apply the optional Region / Area filters from the controls row.
if filter_region != 'All':
    analysis_df = analysis_df[analysis_df['Region'] == filter_region]
if filter_area != 'All':
    analysis_df = analysis_df[analysis_df['Area'] == filter_area]

if analysis_df.empty:
    st.warning("No data after applying filters.")
    st.stop()

# Display a summary caption showing the current scope and parameters.
st.caption(f"{scope_label} | {len(analysis_df)} projects | Ranked by **{metric}** | {direction} {count}")

# ============================================================================
# AGGREGATE BY LEVEL (Project / Area / Region)
# ============================================================================
# When the level is Area or Region, we group projects and compute mean scores
# per group.  The ``_count`` column tracks how many projects are in each group.
# ``_entity`` is a normalised column holding the entity name regardless of level.
if level == 'Project':
    # No aggregation needed at the Project level.
    rank_df = analysis_df.copy()
    rank_df['_entity'] = rank_df['Project']
    rank_df['_count'] = 1
    entity_col = 'Project'
elif level == 'Area':
    # Group by (Region, Area) and compute mean scores.  Also count distinct
    # projects per area via 'nunique'.
    agg = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg['Total Score'] = 'mean'
    agg['Project'] = 'nunique'
    rank_df = analysis_df.groupby(['Region', 'Area']).agg(agg).reset_index()
    rank_df.rename(columns={'Project': '_count'}, inplace=True)
    rank_df['_entity'] = rank_df['Area']
    entity_col = 'Area'
else:  # Region
    # Group by Region and compute mean scores.
    agg = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg['Total Score'] = 'mean'
    agg['Project'] = 'nunique'
    rank_df = analysis_df.groupby('Region').agg(agg).reset_index()
    rank_df.rename(columns={'Project': '_count'}, inplace=True)
    rank_df['_entity'] = rank_df['Region']
    entity_col = 'Region'

# Round all score columns to 2 decimal places for display.
for col in SCORE_DIMENSIONS + ['Total Score']:
    if col in rank_df.columns:
        rank_df[col] = rank_df[col].round(2)

# Sort by the selected metric and take the top N entities.
# ``ascending=True`` for "Bottom" direction (worst first);
# ``ascending=False`` for "Top" direction (best first).
ascending = direction == 'Bottom'
rank_df = rank_df.sort_values(metric, ascending=ascending).head(count).reset_index(drop=True)
# Assign 1-based rank numbers for display.
rank_df['_rank'] = range(1, len(rank_df) + 1)

# ── Percentile computation ──
# Compute the percentile of each ranked entity relative to ALL entities at
# the same level (not just the displayed top-N).  This gives context:
# e.g. "this project is at the 15th percentile of all projects".
if level == 'Project':
    full_values = analysis_df[metric].dropna()
elif level == 'Area':
    full_agg = analysis_df.groupby('Area')[metric].mean()
    full_values = full_agg
else:
    full_agg = analysis_df.groupby('Region')[metric].mean()
    full_values = full_agg

# Percentile = fraction of all entities with a lower score, times 100.
rank_df['_percentile'] = rank_df[metric].apply(
    lambda x: (full_values < x).sum() / len(full_values) * 100 if len(full_values) > 0 else 50
)

# ============================================================================
# HORIZONTAL BAR CHART + SCORECARD HEATMAP (side by side)
# ============================================================================
# Left column (1.2 fraction): bar chart.  Right column (1.8 fraction): heatmap.
col_chart, col_heat = st.columns([1.2, 1.8])

with col_chart:
    st.markdown("**Ranked Bar Chart**")

    # Extract parallel lists for entities, values, project counts, percentiles.
    entities = rank_df['_entity'].tolist()
    values = rank_df[metric].tolist()
    counts = rank_df.get('_count', pd.Series([1] * len(rank_df))).tolist()
    percentiles = rank_df['_percentile'].tolist()

    # Choose bar colors based on the metric type.
    if metric == 'Total Score':
        # Use the 4-tier Pulse status color mapping (red/yellow/green/dark-green).
        colors = [get_pulse_color(v) for v in values]
        x_range = [0, 24]  # Total Score axis runs 0-24
    else:
        # For individual dimensions (0-3), use the dimension color function
        # which maps severity similarly but on a 0-3 scale.
        from utils.mckinsey_charts import _dim_color
        colors = [_dim_color(v) for v in values]
        x_range = [0, 3.2]  # Slight padding beyond max of 3

    # Build text labels shown outside each bar, including score and percentile.
    texts = []
    for v, c, p in zip(values, counts, percentiles):
        if level == 'Project':
            # Projects don't need a project count (it's always 1).
            texts.append(f"{v:.1f}  (P{p:.0f})")
        else:
            # Areas/Regions show how many projects contribute to the average.
            texts.append(f"{v:.1f}  ({c} proj, P{p:.0f})")

    # Build the Plotly horizontal bar chart.
    # Note: lists are reversed ([::-1]) because Plotly renders horizontal bars
    # bottom-to-top, and we want the #1 ranked entity at the top.
    fig = go.Figure(go.Bar(
        y=entities[::-1],
        x=values[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=texts[::-1],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=10),
    ))
    # Apply the shared McKinsey dark-theme styling to the figure.
    _apply_theme(fig)
    # Dynamic height: at least 250px, growing by 32px per entity.
    fig.update_layout(
        height=max(250, len(entities) * 32 + 80),
        margin=dict(l=140, r=100, t=10, b=30),
    )
    fig.update_xaxes(range=x_range, title_text=metric)

    # Add a dashed vertical line at the portfolio average for reference.
    portfolio_avg = analysis_df[metric].mean()
    fig.add_vline(
        x=portfolio_avg, line_dash='dash', line_color='#60a5fa', line_width=1,
        annotation_text=f'Avg ({portfolio_avg:.1f})',
        annotation_font_color='#94a3b8',
        annotation_position='top',
    )
    # For Total Score, also add a dotted target line.
    if metric == 'Total Score':
        fig.add_vline(
            x=target, line_dash='dot', line_color='white', line_width=1,
            annotation_text=f'Target ({target:.0f})',
            annotation_font_color='#94a3b8',
        )

    st.plotly_chart(fig, use_container_width=True)

with col_heat:
    st.markdown("**Dimension Scorecard**")

    # ── Build a custom HTML table for the dimension heatmap ──
    # This uses raw HTML rather than st.dataframe because we need fine-grained
    # control over per-cell background colors (CSS classes from pulse_css_class
    # and score_css_class) and the percentile badge styling.
    html = (
        '<div class="matrix-container" style="max-height:500px; overflow-y:auto;">'
        '<table class="matrix-table">'
        '<thead><tr>'
        '<th style="text-align:left; min-width:30px;">#</th>'
        f'<th style="text-align:left; min-width:120px;">{level}</th>'
        '<th>Pulse</th>'  # Total Score column header
    )
    # Add a header for each of the 8 scoring dimensions.
    for dim in SCORE_DIMENSIONS:
        # Shorten "PM Performance" to "PM Perf" so headers fit better.
        short = dim.replace('PM Performance', 'PM Perf')
        html += f'<th>{short}</th>'
    html += '<th>Pctl</th></tr></thead><tbody>'  # Percentile column header

    # One row per ranked entity.
    for _, row in rank_df.iterrows():
        rank_num = int(row['_rank'])
        entity = row['_entity']
        total = row['Total Score']
        pctl = row['_percentile']
        # ``pulse_css_class`` returns a CSS class name (e.g. "score-red")
        # that sets the cell background to the appropriate Pulse tier color.
        p_cls = pulse_css_class(total)

        # Percentile badge color: quartile-based green/yellow/red scheme.
        if pctl >= 75:
            pctl_style = 'background:#059669; color:white;'    # Top quartile (dark green)
        elif pctl >= 50:
            pctl_style = 'background:#22c55e; color:white;'    # Above median (green)
        elif pctl >= 25:
            pctl_style = 'background:#f59e0b; color:#212529;'  # Below median (amber)
        else:
            pctl_style = 'background:#ef4444; color:white;'    # Bottom quartile (red)

        html += '<tr>'
        html += f'<td style="text-align:left; color:#64748b;">{rank_num}</td>'
        html += f'<td style="text-align:left; font-weight:600; color:#e2e8f0;">{entity}</td>'
        # Pulse Total Score cell with tier-based background color.
        html += f'<td><span class="score-cell {p_cls}">{total:.1f}</span></td>'

        # Individual dimension score cells, each with its own color class.
        for dim in SCORE_DIMENSIONS:
            val = row[dim]
            # ``score_css_class`` maps dimension scores (0-3) to CSS classes
            # that produce red (0), amber (1), green (2), or dark-green (3).
            cls = score_css_class(val)
            html += f'<td><span class="score-cell {cls}">{val:.2f}</span></td>'

        # Percentile badge pill.
        html += (
            f'<td><span style="display:inline-block; padding:2px 6px; border-radius:10px;'
            f' font-size:0.65rem; font-weight:600; {pctl_style}">P{pctl:.0f}</span></td>'
        )
        html += '</tr>'

    html += '</tbody></table></div>'
    # Render the raw HTML table in Streamlit.
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# TREND SPARKLINES
# ============================================================================
st.markdown("---")
st.markdown("**Trend Sparklines** (score trajectory over all weeks)")

# For each ranked entity, gather its weekly score history from the multi-week
# DataFrame and compute deltas.
sparkline_data = []
for _, row in rank_df.iterrows():
    entity = row['_entity']
    # Filter the multi-week DF to this entity's rows and sort chronologically.
    if level == 'Project':
        hist = mw_df[mw_df['Project'] == entity].sort_values(['Year', 'Wk'])
        # Compute mean per week (in case of duplicates).
        weekly_vals = hist.groupby(['Year', 'Wk'])[metric].mean().values
    elif level == 'Area':
        hist = mw_df[mw_df['Area'] == entity].sort_values(['Year', 'Wk'])
        weekly_vals = hist.groupby(['Year', 'Wk'])[metric].mean().values
    else:
        hist = mw_df[mw_df['Region'] == entity].sort_values(['Year', 'Wk'])
        weekly_vals = hist.groupby(['Year', 'Wk'])[metric].mean().values

    # Calculate overall delta (first week to last week) and recent delta
    # (penultimate week to last week).
    if len(weekly_vals) >= 2:
        delta = weekly_vals[-1] - weekly_vals[0]
        recent_delta = weekly_vals[-1] - weekly_vals[-2] if len(weekly_vals) >= 2 else 0
    else:
        delta = 0
        recent_delta = 0

    sparkline_data.append({
        'entity': entity,
        'values': weekly_vals,           # Array of per-week scores for the sparkline
        'current': weekly_vals[-1] if len(weekly_vals) > 0 else 0,
        'delta': delta,                  # Overall change (first-to-last)
        'recent_delta': recent_delta,    # Week-over-week change
        'n_weeks': len(weekly_vals),     # Number of weeks with data
    })

# Choose the color function for sparklines based on the metric type.
if metric == 'Total Score':
    # Use the Pulse status color (maps 0-24 to red/yellow/green/dark-green).
    spark_color_fn = get_pulse_color
else:
    # Use the dimension color function (maps 0-3).
    spark_color_fn = lambda v: _dim_color(v)

# ── Build an HTML table with inline SVG sparklines ──
spark_html = (
    '<div class="matrix-container">'
    '<table class="matrix-table">'
    '<thead><tr>'
    f'<th style="text-align:left;">{level}</th>'
    '<th>Current</th>'
    '<th>Trend</th>'          # Sparkline SVG column
    '<th>Overall Delta</th>'  # First-to-last change
    '<th>Last Week Delta</th>'  # Most recent week-over-week change
    '<th>Weeks</th>'          # Number of data points
    '</tr></thead><tbody>'
)

for s in sparkline_data:
    # Color the current value and the sparkline stroke based on the score.
    color = spark_color_fn(s['current'])
    # mini_sparkline_svg generates a small inline SVG polyline.
    svg = mini_sparkline_svg(s['values'], width=100, height=24, color=color)

    # CSS classes for positive (green arrow-up) and negative (red arrow-down)
    # delta styling, defined in the shared stylesheet.
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
# MOVERS -- Most Improved & Most Declined
# ============================================================================
st.markdown("---")
st.markdown("**Movers** (biggest score changes over all weeks)")

# Determine the grouping column based on the selected Level.
# Movers are computed across ALL entities at this level (not just the
# displayed top-N), then sliced to the top 5 improved and top 5 declined.
if level == 'Project':
    group_col = 'Project'
elif level == 'Area':
    group_col = 'Area'
else:
    group_col = 'Region'

# ── Compute overall delta for every entity ──
# Get the first and last week's score for each entity by sorting
# chronologically and taking the first/last row per group.
first_week = mw_df.sort_values(['Year', 'Wk']).groupby(group_col).first()[metric]
last_week = mw_df.sort_values(['Year', 'Wk']).groupby(group_col).last()[metric]

movers_df = pd.DataFrame({
    'first': first_week,
    'last': last_week,
}).dropna()
movers_df['delta'] = movers_df['last'] - movers_df['first']
# Exclude entities with zero change (they are neither improved nor declined).
movers_df = movers_df[movers_df['delta'] != 0]

# ── Compute recent delta (last 2 weeks) ──
# Find the two most recent (Year, Week) pairs in the data.
sorted_yw = mw_df[['Year', 'Wk']].drop_duplicates().sort_values(['Year', 'Wk'])
if len(sorted_yw) >= 2:
    prev_yw = sorted_yw.iloc[-2]  # Penultimate week
    last_yw = sorted_yw.iloc[-1]  # Most recent week

    # Compute mean scores per entity for each of these two weeks.
    prev_scores = mw_df[(mw_df['Year'] == prev_yw['Year']) & (mw_df['Wk'] == prev_yw['Wk'])].groupby(group_col)[metric].mean()
    last_scores = mw_df[(mw_df['Year'] == last_yw['Year']) & (mw_df['Wk'] == last_yw['Wk'])].groupby(group_col)[metric].mean()

    recent_df = pd.DataFrame({'prev': prev_scores, 'last': last_scores}).dropna()
    recent_df['recent_delta'] = recent_df['last'] - recent_df['prev']
    # Join the recent delta onto the movers DataFrame.
    movers_df = movers_df.join(recent_df[['recent_delta']], how='left')
else:
    # If fewer than 2 weeks exist, recent delta is meaningless.
    movers_df['recent_delta'] = 0

# Fill NaN recent deltas (entities missing from one of the two weeks).
movers_df['recent_delta'] = movers_df['recent_delta'].fillna(0)

# Show top 5 improved and top 5 declined.
mover_count = 5

# ── Two-column layout: improved on the left, declined on the right ──
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown('<span style="color:#22c55e; font-weight:700;">Most Improved</span>', unsafe_allow_html=True)
    # nlargest(5, 'delta') gives the entities with the biggest positive changes.
    improved = movers_df.nlargest(mover_count, 'delta')

    if improved.empty:
        st.info("No improvements detected.")
    else:
        # Render as an HTML table with the shared matrix-table styling.
        imp_html = (
            '<div class="matrix-container"><table class="matrix-table">'
            '<thead><tr>'
            f'<th style="text-align:left;">{level}</th>'
            '<th>First</th><th>Last</th><th>Change</th><th>Recent</th>'
            '</tr></thead><tbody>'
        )
        for entity, row in improved.iterrows():
            # Apply green/red CSS class based on the sign of the delta.
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
    # nsmallest(5, 'delta') gives the entities with the most negative changes.
    declined = movers_df.nsmallest(mover_count, 'delta')
    # Only show entities that actually declined (delta < 0).
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
