"""
Pulse Dashboard - Drill-Down Visualizations

Multi-level drill-down: Portfolio -> Region -> Area -> Project
With interactive charts and detailed analysis panels at every level.

PURPOSE
-------
While the Executive Summary provides a bird's-eye view, this page lets
users interactively navigate the hierarchy:

    Portfolio (all regions)
      -> Region  (e.g. EMEA, APAC)
        -> Area  (sub-region or business unit)
          -> Project (individual engagement)

At each level the page shows contextually appropriate visualizations,
KPI cards, comparison charts, dimension heatmaps, and text summaries.

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
1. Week scope selector     - Choose a specific week or "All Weeks" average
2. Cascading nav dropdowns - Region -> Area -> Project (with breadcrumb)
3. Interactive chart tabs   - Sunburst, Treemap, Sankey, Icicle (click-to-drill)
4. Deep detail panel       - Context-sensitive panel that adapts to the
                             drill level (portfolio / region / area / project)

INTERACTIVITY
-------------
- Sunburst/Treemap/Icicle charts support on_select: clicking a segment
  opens an inline detail panel with radar charts, score comparisons, etc.
- The Sankey tab includes a dimension deep-dive selector.
- The detail panel at the bottom adapts to the selected drill level:
    Portfolio -> region comparison bar chart + Region x Dimension heatmap
    Region    -> area comparison + Area x Dimension heatmap + bottom projects
    Area      -> project comparison + Project x Dimension heatmap + weak dims
    Project   -> full project detail with radar, trend, dimension comparison
"""

# ---------------------------------------------------------------------------
# PATH SETUP: Allow imports from the pulse_dashboard package root.
# sys.path.insert ensures `from utils.* import ...` resolves correctly
# when Streamlit runs this file directly as a multi-page "page".
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# UTILITY IMPORTS
#
# render_sidebar       - shared sidebar; returns the sidebar-filtered DataFrame
# inject_css           - injects the global dark-theme CSS
# SCORE_DIMENSIONS     - list of the 8 scoring dimension column names
# STATUS_CONFIG        - dict: status name -> {color, label, range, min, max}
# STATUS_ORDER         - canonical ordering: ['Red', 'Yellow', 'Green', 'Dark Green']
# get_pulse_status     - score (float) -> status name string
# get_pulse_color      - score (float) -> hex color string
# pulse_css_class      - total score -> CSS class for HTML table cells
# score_css_class      - dimension score (0-3) -> CSS class for HTML badges
# DIMENSION_COLORS     - dict mapping each dimension name to a brand hex color
# REGION_LINE_COLORS   - dict mapping region names to line chart hex colors
# get_plotly_theme     - returns the dark-theme Plotly layout dict
# AXIS_STYLE           - common axis styling dict for Plotly charts
# chart_*              - McKinsey-style Plotly chart builders
# mini_sparkline_svg   - generates a tiny inline SVG sparkline
# _apply_theme         - applies the dark theme to a Plotly figure in-place
# ---------------------------------------------------------------------------
from utils.sidebar import render_sidebar
from utils.styles import (
    inject_css, SCORE_DIMENSIONS, STATUS_CONFIG, STATUS_ORDER,
    get_pulse_status, get_pulse_color,
    pulse_css_class, score_css_class, DIMENSION_COLORS, REGION_LINE_COLORS,
    get_plotly_theme, AXIS_STYLE,
)
from utils.mckinsey_charts import (
    chart_sunburst, chart_treemap, chart_sankey, chart_icicle,
    chart_radar, chart_project_trend, _apply_theme,
    chart_dimension_by_region, chart_dimension_distribution,
    mini_sparkline_svg,
)

# Inject the global dark-theme CSS into the page
inject_css()

# ---------------------------------------------------------------------------
# SIDEBAR & DATA LOADING
# render_sidebar() draws the sidebar widgets and returns the filtered DataFrame.
# If no data is loaded or no rows match the current filters, bail out early.
# ---------------------------------------------------------------------------
filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

# `df` is the FULL unfiltered dataset (used for project trend history).
# `target` is the portfolio pulse target score from sidebar config.
# `selected_regions` is the list of region filters chosen in the sidebar.
df = st.session_state.df
target = st.session_state.get('pulse_target', 17.0)
selected_regions = st.session_state.get('selected_regions', [])

# Page title rendered as styled HTML
st.markdown('<p class="main-header">Drill-Down Analysis</p>', unsafe_allow_html=True)

# ============================================================================
# WEEK SCOPE SELECTOR -- choose any week or overall average
#
# Unlike the sidebar week filter (which pre-filters the global DataFrame),
# this selector lets the user view a specific week OR the cross-week average
# for the drill-down analysis.  It uses the region-filtered (but NOT
# week-filtered) dataset so all weeks remain available.
# ============================================================================
# Build multi-week dataset: apply region filter only, keep all weeks
mw_df = df.copy()
if selected_regions:
    mw_df = mw_df[mw_df['Region'].isin(selected_regions)]

# Extract and sort all available (Year, Week) pairs chronologically
yw_pairs = (
    mw_df[['Year', 'Wk']].drop_duplicates()
    .sort_values(['Year', 'Wk'])
    .values.tolist()
)
# Format as "YYYY-W##" labels for the selectbox
week_labels = [f"{int(y)}-W{int(w):02d}" for y, w in yw_pairs]
week_options = ['All Weeks (Average)'] + week_labels

# Default the selectbox to whatever week the sidebar has selected
sidebar_year = st.session_state.get('selected_year')
sidebar_week = st.session_state.get('selected_week')
sidebar_label = f"{int(sidebar_year)}-W{int(sidebar_week):02d}" if sidebar_year and sidebar_week else None
default_idx = (week_labels.index(sidebar_label) + 1) if sidebar_label in week_labels else 0

# Render the scope selector in a narrow left column
scope_col1, scope_col2 = st.columns([1, 3])
with scope_col1:
    week_scope = st.selectbox(
        'Analysis Scope',
        week_options,
        index=default_idx,
        key='dd_week_scope',
    )

# ---------------------------------------------------------------------------
# BUILD THE ANALYSIS DATAFRAME based on the selected scope.
#
# "All Weeks (Average)" mode:
#   - Averages each project's dimension scores across all weeks
#   - Rounds to 1 decimal place for readability
#   - Keeps the latest week's metadata (Region, Area, PM Name) per project
#   - Recomputes Pulse_Status and Pulse_Color from the averaged Total Score
#
# Specific week mode:
#   - Simply filters to the selected (Year, Week) pair
# ---------------------------------------------------------------------------
if week_scope == 'All Weeks (Average)':
    # Average all dimension scores and Total Score per project (latest region/area/PM)
    agg_cols = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg_cols['Total Score'] = 'mean'
    # Keep latest metadata per project (last row when sorted chronologically)
    latest_idx = mw_df.sort_values(['Year', 'Wk']).groupby('Project').tail(1).index
    meta_df = mw_df.loc[latest_idx, ['Project', 'Region', 'Area', 'PM Name']].drop_duplicates('Project')

    avg_scores = mw_df.groupby('Project').agg(agg_cols).reset_index()
    # Round scores to 1 decimal for display; dimension scores stay as floats
    for dim in SCORE_DIMENSIONS:
        avg_scores[dim] = avg_scores[dim].round(1)
    avg_scores['Total Score'] = avg_scores['Total Score'].round(1)

    # Merge metadata with averaged scores
    analysis_df = meta_df.merge(avg_scores, on='Project', how='inner')
    # Recompute status columns from the averaged Total Score
    analysis_df['Pulse_Status'] = analysis_df['Total Score'].apply(get_pulse_status)
    analysis_df['Pulse_Color'] = analysis_df['Total Score'].apply(get_pulse_color)

    with scope_col2:
        st.caption(f"Showing average across {len(yw_pairs)} weeks | {len(analysis_df)} projects")
else:
    # Specific week selected: parse "YYYY-W##" format
    yr_str, wk_str = week_scope.split('-W')
    sel_yr, sel_wk = int(yr_str), int(wk_str)
    analysis_df = mw_df[(mw_df['Year'] == sel_yr) & (mw_df['Wk'] == sel_wk)].copy()

    with scope_col2:
        st.caption(f"Week {week_scope} | {len(analysis_df)} projects")

# Guard: if the chosen scope yields no data, stop rendering
if analysis_df.empty:
    st.warning("No data for the selected scope.")
    st.stop()

# Replace filtered_df with the scope-specific analysis DataFrame.
# All downstream code on this page uses `filtered_df` as the working dataset.
filtered_df = analysis_df


# ============================================================================
# HELPER FUNCTIONS
# These are page-local utility functions used by the drill-down panels below.
# ============================================================================

def _dimension_heatmap_html(data_df, entity_col, entity_label='Entity'):
    """
    Generate an HTML heatmap table: rows = entities, columns = dimensions.

    Renders a styled HTML table where each cell is color-coded by its
    score (using CSS classes from utils.styles).  The table has two
    fixed columns on the left (Pulse total score and project count N),
    followed by one column per scoring dimension.

    Parameters
    ----------
    data_df : pd.DataFrame
        The data to aggregate (must contain entity_col, 'Total Score',
        'Project', and all SCORE_DIMENSIONS columns).
    entity_col : str
        Column name to group by for rows (e.g. 'Region', 'Area', 'Project').
    entity_label : str
        Human-readable label for the first column header.

    Returns
    -------
    str
        Complete HTML string for the heatmap table, wrapped in a
        scrollable container div.
    """
    entities = sorted(data_df[entity_col].dropna().unique())

    # Build table header: entity name, Pulse score, count N, then each dimension
    html = (
        '<div class="matrix-container"><table class="matrix-table">'
        '<thead><tr>'
        f'<th style="text-align:left; min-width:120px;">{entity_label}</th>'
        '<th>Pulse</th>'
        '<th>N</th>'
    )
    for dim in SCORE_DIMENSIONS:
        # Shorten "PM Performance" to "PM Perf" to save horizontal space
        short = dim.replace('PM Performance', 'PM Perf')
        html += f'<th>{short}</th>'
    html += '</tr></thead><tbody>'

    # Build one row per entity with color-coded score cells
    for entity in entities:
        edf = data_df[data_df[entity_col] == entity]
        avg_pulse = edf['Total Score'].mean()
        n_projects = edf['Project'].nunique()
        # pulse_css_class returns e.g. "pulse-green" for Total Score coloring
        p_cls = pulse_css_class(avg_pulse)

        html += '<tr>'
        html += f'<td style="text-align:left; font-weight:600; color:#e2e8f0;">{entity}</td>'
        html += f'<td><span class="score-cell {p_cls}">{avg_pulse:.1f}</span></td>'
        html += f'<td style="color:#94a3b8;">{n_projects}</td>'

        # Each dimension cell: mean score for this entity, color-coded 0-3
        for dim in SCORE_DIMENSIONS:
            val = edf[dim].mean()
            # score_css_class returns e.g. "score-red" for dimension-level coloring
            cls = score_css_class(val)
            html += f'<td><span class="score-cell {cls}">{val:.2f}</span></td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html


def _text_summary_panel(data_df, max_items=8):
    """
    Display qualitative text fields (pain points, comments, resolutions, etc.)
    as collapsible Streamlit expanders.

    Scans for known text columns in the DataFrame and renders each non-empty
    column as an expander with a bullet list of entries.  "Pain Points" is
    expanded by default since it is typically the most actionable.

    Parameters
    ----------
    data_df : pd.DataFrame
        Rows to extract text from (typically drill-filtered).
    max_items : int
        Maximum number of text entries to show per column (prevents
        overwhelming the UI on large datasets).
    """
    # Map of column name -> emoji icon for the expander header
    text_cols = {
        'Pain Points': '\U0001f534',
        'Comments': '\U0001f4ac',
        'Resolution Plan': '\u2705',
        'Issue': '\u26a0\ufe0f',
        'Escalations (External)': '\U0001f4e2',
        'Escalations (Internal)': '\U0001f4e2',
        'Potential Escalations': '\u26a1',
    }

    found_any = False
    for col, icon in text_cols.items():
        if col in data_df.columns:
            # Drop nulls and empty/whitespace-only strings
            texts = data_df[col].dropna()
            texts = texts[texts.astype(str).str.strip() != '']
            if not texts.empty:
                found_any = True
                # Pain Points expanded by default; others collapsed
                with st.expander(f"{icon} {col} ({len(texts)})", expanded=(col == 'Pain Points')):
                    for text in texts.head(max_items):
                        # Truncate individual entries to 300 chars for readability
                        st.markdown(f"- {str(text)[:300]}")

    if not found_any:
        st.info("No text data available for this selection.")


def _comparison_bar_chart(data_df, group_col, title='Comparison'):
    """
    Create a horizontal bar chart comparing average pulse scores by group.

    Each bar is color-coded by the group's average pulse status (red/yellow/
    green/dark-green).  A vertical dashed line marks the target score.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data to aggregate.
    group_col : str
        Column to group by (e.g. 'Region', 'Area', 'Project').
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Configured horizontal bar chart.
    """
    # Aggregate: mean pulse score and unique project count per group
    agg = (
        data_df.groupby(group_col)
        .agg(AvgPulse=('Total Score', 'mean'), Count=('Project', 'nunique'))
        .reset_index()
        .sort_values('AvgPulse', ascending=True)  # worst at top for horizontal bars
    )

    # Color each bar by its average pulse status
    colors = [get_pulse_color(v) for v in agg['AvgPulse']]

    fig = go.Figure(go.Bar(
        y=agg[group_col],
        x=agg['AvgPulse'],
        orientation='h',
        marker_color=colors,
        # Text labels show score and project count outside each bar
        text=[f"{v:.1f} ({c} proj)" for v, c in zip(agg['AvgPulse'], agg['Count'])],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=10),
    ))
    # Apply the shared dark theme
    _apply_theme(fig)
    # Dynamic height: scales with number of bars, minimum 200px
    fig.update_layout(
        title=title,
        height=max(200, len(agg) * 35 + 80),
        margin=dict(l=120, r=80, t=40, b=30),
    )
    # X-axis fixed at 0-24 (full pulse score range)
    fig.update_xaxes(range=[0, 24], title_text='Avg Pulse Score')
    # Add a dashed vertical target line for reference
    fig.add_vline(x=target, line_dash='dash', line_color='white', line_width=1,
                  annotation_text=f'Target ({target:.0f})',
                  annotation_font_color='#94a3b8')
    return fig


def _project_table_html(data_df, max_rows=30):
    """
    Generate a styled HTML table showing individual project scores.

    Columns: Project name, PM Name, each of the 8 dimension scores (0-3),
    and Total Score.  Sorted ascending by Total Score (worst first).
    Dimension scores use rating-badge CSS classes; total scores use
    pulse-status CSS classes.

    Parameters
    ----------
    data_df : pd.DataFrame
        Project-level data.
    max_rows : int
        Maximum number of rows to display (prevents huge tables).

    Returns
    -------
    str
        Complete HTML string wrapped in a scrollable container.
    """
    cols = ['Project', 'PM Name'] + SCORE_DIMENSIONS + ['Total Score']
    available = [c for c in cols if c in data_df.columns]
    # Sort worst-first and limit rows
    sorted_df = data_df[available].sort_values('Total Score', ascending=True).head(max_rows)

    # Build table header
    html = (
        '<div class="matrix-container" style="max-height:400px; overflow-y:auto;">'
        '<table class="matrix-table"><thead><tr>'
    )
    for c in available:
        # Left-align text columns; center numeric columns
        align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
        label = c.replace('PM Performance', 'PM Perf')
        html += f'<th style="{align}">{label}</th>'
    html += '</tr></thead><tbody>'

    # Build table rows with color-coded cells
    for _, r in sorted_df.iterrows():
        html += '<tr>'
        for c in available:
            val = r[c]
            if c == 'Total Score':
                # Total Score uses pulse-level coloring (red/yellow/green/dark-green)
                cls = pulse_css_class(val)
                display = f'<span class="score-cell {cls}">{int(val)}</span>' if pd.notna(val) else '\u2014'
            elif c in SCORE_DIMENSIONS:
                # Dimension scores use rating-badge classes: rating-0, rating-1, etc.
                cls = f"rating-{int(val)}" if pd.notna(val) else ""
                display = f'<span class="rating-badge {cls}">{int(val)}</span>' if pd.notna(val) else '\u2014'
            elif c in ('Project', 'PM Name'):
                # Text columns: truncate long names at 50 chars
                display = str(val)[:50] if pd.notna(val) else '\u2014'
            else:
                display = str(val) if pd.notna(val) else '\u2014'
            align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
            html += f'<td style="{align}">{display}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html


def _kpi_row(data_df):
    """
    Render a row of 4 KPI cards for the given data subset.

    Cards shown:
        1. Avg Pulse    - mean Total Score
        2. Projects     - count of unique projects
        3. Green Rate   - % of entries with Total Score >= 16
        4. Red Projects - count of entries with Total Score < 14

    Uses inline HTML with CSS classes from the global theme.

    Parameters
    ----------
    data_df : pd.DataFrame
        The subset of data to compute KPIs from.
    """
    avg_pulse = data_df['Total Score'].mean()
    n_projects = data_df['Project'].nunique()
    # Green: score >= 16 (covers both Green and Dark Green tiers)
    green_count = len(data_df[data_df['Total Score'] >= 16])
    # Red: score < 14 (the Red tier threshold)
    red_count = len(data_df[data_df['Total Score'] < 14])
    green_pct = (green_count / len(data_df) * 100) if len(data_df) > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{avg_pulse:.1f}</p>
            <p class="kpi-label">Avg Pulse</p>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{n_projects}</p>
            <p class="kpi-label">Projects</p>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-container success">
            <p class="kpi-value green">{green_pct:.0f}%</p>
            <p class="kpi-label">Green Rate</p>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        # Red projects card gets "critical" styling when any reds exist
        st.markdown(f"""
        <div class="kpi-container {'critical' if red_count > 0 else ''}">
            <p class="kpi-value {'red' if red_count > 0 else ''}">{red_count}</p>
            <p class="kpi-label">Red Projects</p>
        </div>
        """, unsafe_allow_html=True)


def _click_project_detail(click_df, label, drill_df, full_df, target_score):
    """
    Full project detail panel shown when a single project is clicked.

    Displays:
    1. Header with project name, status badge, PM, region/area
    2. Dimension score cards (8 small cards in a flex row)
    3. Radar chart (left) + multi-week trend chart (right)
    4. Dimension comparison table: project score vs region avg vs portfolio avg
    5. Text fields (pain points, comments, resolution plans, etc.)

    Parameters
    ----------
    click_df : pd.DataFrame
        DataFrame containing the clicked project's row(s).
    label : str
        The clicked label (project name from the chart).
    drill_df : pd.DataFrame
        The currently drill-filtered dataset (for portfolio averages).
    full_df : pd.DataFrame
        The full unfiltered dataset (for historical trend lookup).
    target_score : float
        The pulse target score (for the trend chart reference line).
    """
    row = click_df.iloc[0]
    project_name = row.get('Project', label)
    status = get_pulse_status(row['Total Score'])
    color = STATUS_CONFIG[status]['color']

    # Header panel with project name, status, PM, and region/area
    st.markdown(f"""
    <div class="drilldown-panel">
        <div class="drilldown-header">
            <span class="drilldown-badge">{project_name}</span>
            <span class="drilldown-context" style="color:{color};">
                Score: {int(row['Total Score'])} &bull; {status} &bull;
                PM: {row.get('PM Name', '\u2014')} &bull;
                {row.get('Region', '')} / {row.get('Area', '')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dimension score cards: a horizontal flex row of 8 small cards,
    # each showing the dimension's integer score with color-coding
    dim_html = '<div style="display:flex; gap:6px; flex-wrap:wrap; margin:8px 0;">'
    for dim in SCORE_DIMENSIONS:
        val = int(row.get(dim, 0))
        cls = score_css_class(val)
        short = dim.replace('PM Performance', 'PM Perf')
        dim_html += (
            f'<div style="background:rgba(255,255,255,0.04); border-radius:8px;'
            f' padding:8px 12px; text-align:center; min-width:70px;">'
            f'<div><span class="score-cell {cls}" style="font-size:1.3rem;">{val}</span></div>'
            f'<div style="font-size:0.65rem; color:#94a3b8; margin-top:2px;">{short}</div>'
            f'</div>'
        )
    dim_html += '</div>'
    st.markdown(dim_html, unsafe_allow_html=True)

    # Side-by-side: Radar chart (dimension profile) + Trend chart (week history)
    c1, c2 = st.columns(2)
    with c1:
        # Radar: plots this project's 8 dimension scores as a filled polygon
        fig_radar = chart_radar(row)
        st.plotly_chart(fig_radar, use_container_width=True)
    with c2:
        # Trend: plots the project's Total Score across all available weeks.
        # Only shown if more than one week of history exists.
        proj_history = full_df[full_df['Project'] == project_name]
        if len(proj_history) > 1:
            fig_trend = chart_project_trend(full_df, project_name, target_score)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Only one week of data \u2014 no trend available.")

    # Dimension comparison table: project score vs region average vs portfolio average
    st.markdown("**Dimension Scores vs Averages**")
    avg_region = full_df[full_df['Region'] == row['Region']][SCORE_DIMENSIONS].mean()
    avg_portfolio = drill_df[SCORE_DIMENSIONS].mean()

    comp_html = (
        '<div class="matrix-container"><table class="matrix-table">'
        '<thead><tr>'
        '<th style="text-align:left;">Dimension</th>'
        '<th>Score</th><th>Region Avg</th><th>Portfolio Avg</th><th>vs Region</th>'
        '</tr></thead><tbody>'
    )
    for dim in SCORE_DIMENSIONS:
        val = row.get(dim, 0)
        r_avg = avg_region[dim]
        p_avg = avg_portfolio[dim]
        # Delta vs region average: positive = above average, negative = below
        diff = val - r_avg
        diff_cls = 'delta-positive' if diff >= 0 else 'delta-negative'
        cls = f"rating-{int(val)}" if pd.notna(val) else ""
        short = dim.replace('PM Performance', 'PM Perf')
        comp_html += (
            f'<tr><td style="text-align:left; color:#e2e8f0;">{short}</td>'
            f'<td><span class="rating-badge {cls}">{int(val)}</span></td>'
            f'<td style="color:#94a3b8;">{r_avg:.2f}</td>'
            f'<td style="color:#94a3b8;">{p_avg:.2f}</td>'
            f'<td><span class="{diff_cls}">{diff:+.2f}</span></td></tr>'
        )
    comp_html += '</tbody></table></div>'
    st.markdown(comp_html, unsafe_allow_html=True)

    # Show all qualitative text fields for this project
    _text_summary_panel(click_df)


def _click_detail_panel(click_df, label):
    """
    Show an inline detail panel when a chart element is clicked.

    Adapts its display based on the number of projects in the clicked
    subset:
    - Single project: shows a radar chart + dimension comparison table
    - Multiple projects: shows a styled project table + text summaries

    Parameters
    ----------
    click_df : pd.DataFrame
        Rows matching the clicked chart element.
    label : str
        The label text from the clicked element (region/area/project name).
    """
    # Header with label, average pulse, and project count
    st.markdown(f"""
    <div class="drilldown-panel">
        <div class="drilldown-header">
            <span class="drilldown-badge">{label}</span>
            <span class="drilldown-context">
                Avg Pulse: {click_df['Total Score'].mean():.1f} &bull;
                {click_df['Project'].nunique()} projects
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if click_df['Project'].nunique() == 1:
        # Single project -> show radar chart + score comparison table
        row = click_df.iloc[0]
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_radar = chart_radar(row)
            st.plotly_chart(fig_radar, use_container_width=True)
        with c2:
            # Build a comparison table: dimension score vs portfolio average
            comp_html = '<div class="matrix-container"><table class="matrix-table"><thead><tr>'
            comp_html += '<th style="text-align:left;">Dim</th><th>Score</th><th>Avg</th><th>Gap</th>'
            comp_html += '</tr></thead><tbody>'
            # Portfolio-level averages for comparison
            avg_all = filtered_df[SCORE_DIMENSIONS].mean()
            for dim in SCORE_DIMENSIONS:
                val = row.get(dim, 0)
                avg_v = avg_all[dim]
                diff = val - avg_v
                diff_cls = 'delta-positive' if diff >= 0 else 'delta-negative'
                cls = f"rating-{int(val)}" if pd.notna(val) else ""
                short = dim.replace('PM Performance', 'PM Perf')
                comp_html += (
                    f'<tr><td style="text-align:left; color:#e2e8f0;">{short}</td>'
                    f'<td><span class="rating-badge {cls}">{int(val)}</span></td>'
                    f'<td style="color:#94a3b8;">{avg_v:.2f}</td>'
                    f'<td><span class="{diff_cls}">{diff:+.2f}</span></td></tr>'
                )
            comp_html += '</tbody></table></div>'
            st.markdown(comp_html, unsafe_allow_html=True)

        _text_summary_panel(click_df, max_items=3)
    else:
        # Multiple projects -> show a ranked project table + text summaries
        st.markdown(_project_table_html(click_df, max_rows=10), unsafe_allow_html=True)
        _text_summary_panel(click_df, max_items=5)


# ============================================================================
# DRILL-DOWN NAVIGATION (cascading: Region -> Area -> Project)
#
# Three linked selectboxes that cascade left to right:
#   Region selection filters the Area list
#   Area selection filters the Project list
# "All Regions" / "All Areas" / "All Projects" are always the first option,
# meaning no filter is applied at that level.
# ============================================================================

nav_cols = st.columns([1, 1, 1.5])

with nav_cols[0]:
    # Region selector: "All Regions" + sorted unique region names
    regions = ['All Regions'] + sorted(filtered_df['Region'].dropna().unique().tolist())
    selected_region = st.selectbox('Region', regions, key='dd_region')

with nav_cols[1]:
    # Area selector: filtered by selected region (if any)
    if selected_region != 'All Regions':
        areas_list = ['All Areas'] + sorted(
            filtered_df[filtered_df['Region'] == selected_region]['Area'].dropna().unique().tolist()
        )
    else:
        areas_list = ['All Areas'] + sorted(filtered_df['Area'].dropna().unique().tolist())
    selected_area = st.selectbox('Area', areas_list, key='dd_area')

with nav_cols[2]:
    # Project selector: filtered by selected area and/or region
    if selected_area != 'All Areas':
        proj_mask = filtered_df['Area'] == selected_area
        if selected_region != 'All Regions':
            proj_mask &= filtered_df['Region'] == selected_region
        projects_list = ['All Projects'] + sorted(
            filtered_df[proj_mask]['Project'].dropna().unique().tolist()
        )
    elif selected_region != 'All Regions':
        projects_list = ['All Projects'] + sorted(
            filtered_df[filtered_df['Region'] == selected_region]['Project'].dropna().unique().tolist()
        )
    else:
        projects_list = ['All Projects'] + sorted(filtered_df['Project'].dropna().unique().tolist())
    selected_project = st.selectbox('Project', projects_list, key='dd_project')

# ---------------------------------------------------------------------------
# BUILD DRILL-FILTERED DATA
# Apply the cascading filter selections to produce the working dataset
# for the chart tabs and detail panel below.
# ---------------------------------------------------------------------------
drill_df = filtered_df.copy()
if selected_region != 'All Regions':
    drill_df = drill_df[drill_df['Region'] == selected_region]
if selected_area != 'All Areas':
    drill_df = drill_df[drill_df['Area'] == selected_area]
if selected_project != 'All Projects':
    drill_df = drill_df[drill_df['Project'] == selected_project]

# Breadcrumb trail: shows the current navigation path (e.g. "Portfolio -> EMEA -> UK -> ProjectX")
crumbs = ['Portfolio']
if selected_region != 'All Regions':
    crumbs.append(selected_region)
if selected_area != 'All Areas':
    crumbs.append(selected_area)
if selected_project != 'All Projects':
    crumbs.append(selected_project)

st.markdown(
    f'<div style="font-size:0.8rem; color:#64748b; margin-bottom:0.5rem;">'
    f'{" \u2192 ".join(crumbs)} &bull; {len(drill_df)} records</div>',
    unsafe_allow_html=True,
)

# ============================================================================
# CHART TABS (with click-to-drill on all tabs)
#
# Four hierarchical chart types, each showing the same data from a different
# visual perspective:
#   Sunburst - concentric rings: center=portfolio, rings=region/area/project
#   Treemap  - nested rectangles sized by project count
#   Sankey   - flow diagram: dimensions -> score levels -> pulse status
#   Icicle   - top-down partition chart (similar to sunburst, linear layout)
#
# Sunburst, Treemap, and Icicle support click-to-drill: clicking a segment
# triggers st.plotly_chart's on_select callback, which renders an inline
# detail panel below the chart.
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Sunburst", "Treemap", "Sankey", "Icicle"])

with tab1:
    fig = chart_sunburst(drill_df)
    # on_select="rerun" makes Streamlit re-run the script when a segment is clicked,
    # passing the click event data back through the `event` return value
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="sunburst")

    # Process click event: extract the label of the clicked segment
    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        label = pt.get('label', '') if isinstance(pt, dict) else getattr(pt, 'label', '')
        if label:
            # Check if clicked label is a dimension name (sunburst may include dimensions)
            if label in SCORE_DIMENSIONS:
                # Dimension leaf clicked -- show dimension-specific analysis
                st.markdown(f"""
                <div class="drilldown-panel">
                    <div class="drilldown-header">
                        <span class="drilldown-badge">{label}</span>
                        <span class="drilldown-context">Dimension Analysis</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                dim_color = DIMENSION_COLORS.get(label, '#2563eb')
                dc1, dc2 = st.columns(2)
                with dc1:
                    # Distribution histogram: how scores are distributed for this dimension
                    fig_dist = chart_dimension_distribution(drill_df, label)
                    st.plotly_chart(fig_dist, use_container_width=True)
                with dc2:
                    # Bar chart: average score for this dimension across regions
                    fig_bar = chart_dimension_by_region(drill_df, label, dim_color)
                    st.plotly_chart(fig_bar, use_container_width=True)
                # Show projects that scored 0 or 1 on this dimension (need attention)
                low = drill_df[drill_df[label] <= 1]
                if not low.empty:
                    st.markdown(f"**Projects scoring 0-1 on {label}** ({len(low)})")
                    show_cols = ['Project', 'Region', 'Area', 'PM Name', label, 'Total Score']
                    avail = [c for c in show_cols if c in low.columns]
                    st.dataframe(low[avail].sort_values(label), use_container_width=True, hide_index=True)
            else:
                # Region / Area / Project clicked: filter to matching rows
                click_df = drill_df[
                    (drill_df['Region'] == label) |
                    (drill_df['Area'] == label) |
                    (drill_df['Project'] == label)
                ]
                if not click_df.empty:
                    if click_df['Project'].nunique() == 1:
                        # Single project -- show full project detail with radar, trend, etc.
                        _click_project_detail(click_df, label, drill_df, df, target)
                    else:
                        # Multiple projects -- show summary panel
                        _click_detail_panel(click_df, label)

with tab2:
    fig = chart_treemap(drill_df)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="treemap")

    # Treemap click handler: only matches Region or Area labels
    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        label = pt.get('label', '') if isinstance(pt, dict) else getattr(pt, 'label', '')
        if label:
            click_df = drill_df[
                (drill_df['Region'] == label) |
                (drill_df['Area'] == label)
            ]
            if not click_df.empty:
                _click_detail_panel(click_df, label)

with tab3:
    # Sankey diagram: shows the flow from scoring dimensions to score levels
    # to pulse status.  Not click-interactive, but includes a dimension
    # deep-dive selector below.
    fig = chart_sankey(drill_df)
    st.plotly_chart(fig, use_container_width=True, key="sankey")
    st.caption("Flow: Scoring Dimensions \u2192 Score Level \u2192 Pulse Status")

    # Dimension breakdown below Sankey: user selects a dimension to analyze
    st.markdown("**Dimension Deep Dive**")
    dim_selected = st.selectbox(
        'Select dimension to analyze',
        SCORE_DIMENSIONS,
        key='sankey_dim',
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        # Distribution: histogram of scores for the selected dimension
        fig_dist = chart_dimension_distribution(drill_df, dim_selected)
        st.plotly_chart(fig_dist, use_container_width=True)
    with sc2:
        # Regional breakdown: bar chart of average score per region
        dim_color = DIMENSION_COLORS.get(dim_selected, '#2563eb')
        fig_bar = chart_dimension_by_region(drill_df, dim_selected, dim_color)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Projects scoring 0-1 on this dimension (critical attention needed)
    low_scorers = drill_df[drill_df[dim_selected] <= 1]
    if not low_scorers.empty:
        st.markdown(f"**Projects scoring 0\u20131 on {dim_selected}** ({len(low_scorers)} total)")
        show_cols = ['Project', 'Region', 'Area', 'PM Name', dim_selected, 'Total Score']
        available_cols = [c for c in show_cols if c in low_scorers.columns]
        st.dataframe(
            low_scorers[available_cols].sort_values(dim_selected),
            use_container_width=True, hide_index=True,
        )
    else:
        st.success(f"No projects scoring 0\u20131 on {dim_selected}")

with tab4:
    fig = chart_icicle(drill_df)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="icicle")

    # Icicle click handler: matches Region, Area, or Pulse_Status labels
    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        label = pt.get('label', '') if isinstance(pt, dict) else getattr(pt, 'label', '')
        if label:
            click_df = drill_df[
                (drill_df['Region'] == label) |
                (drill_df['Area'] == label) |
                (drill_df['Pulse_Status'] == label)
            ]
            if not click_df.empty:
                _click_detail_panel(click_df, label)

# ============================================================================
# DEEP DRILL-DOWN DETAIL PANEL
#
# This is the bottom section of the page.  It adapts its content based on
# the deepest level of the cascading navigation:
#
#   Portfolio level (no filters) -> Region comparison + Region x Dimension
#       heatmap + status distribution + bottom 10 projects
#
#   Region level -> Area comparison + Area x Dimension heatmap +
#       bottom projects table + text summaries
#
#   Area level -> Project comparison + Project x Dimension heatmap +
#       weakest dimensions list + text summaries
#
#   Project level (deepest) -> Full project detail: KPI cards, radar,
#       trend, dimension comparison table, text fields
# ============================================================================
st.markdown("---")
st.markdown('<div class="section-title">Detail Panel</div>', unsafe_allow_html=True)

if selected_project != 'All Projects':
    # ---- PROJECT LEVEL (deepest drill) ----
    # Show the full detail view for a single selected project
    proj_row = drill_df.iloc[0] if len(drill_df) > 0 else None
    if proj_row is not None:
        # KPI row: 4 cards showing Total Score, Status, PM Name, Region/Area
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            status = get_pulse_status(proj_row['Total Score'])
            color = STATUS_CONFIG[status]['color']
            # Total Score card with gradient text matching the status color
            st.markdown(f"""
            <div class="kpi-container" style="border-left-color:{color};">
                <p class="kpi-value" style="font-size:2.4rem; background:{color};
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    {int(proj_row['Total Score'])}
                </p>
                <p class="kpi-label">Total Score</p>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-container">
                <p class="kpi-value" style="font-size:1.2rem;">{status}</p>
                <p class="kpi-label">Pulse Status</p>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            pm_name = proj_row.get('PM Name', '\u2014')
            st.markdown(f"""
            <div class="kpi-container">
                <p class="kpi-value" style="font-size:1rem;">{pm_name}</p>
                <p class="kpi-label">PM Name</p>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            area_name = proj_row.get('Area', '\u2014')
            region_name = proj_row.get('Region', '\u2014')
            st.markdown(f"""
            <div class="kpi-container">
                <p class="kpi-value" style="font-size:1rem;">{region_name} / {area_name}</p>
                <p class="kpi-label">Region / Area</p>
            </div>
            """, unsafe_allow_html=True)

        # Radar chart + Trend chart side by side
        c1, c2 = st.columns(2)
        with c1:
            # Radar: 8-dimension polygon for this project
            fig_radar = chart_radar(proj_row)
            st.plotly_chart(fig_radar, use_container_width=True)
        with c2:
            # Trend: multi-week Total Score line for this project.
            # Uses the full unfiltered dataset to get complete history.
            proj_history = df[df['Project'] == selected_project]
            if len(proj_history) > 1:
                fig_trend = chart_project_trend(df, selected_project, target)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Only one week of data \u2014 no trend available.")

        # Dimension comparison table: project score vs region avg vs portfolio avg
        st.markdown("**Dimension Scores vs Averages**")
        avg_region = df[df['Region'] == proj_row['Region']][SCORE_DIMENSIONS].mean()
        avg_portfolio = filtered_df[SCORE_DIMENSIONS].mean()

        comp_html = (
            '<div class="matrix-container"><table class="matrix-table">'
            '<thead><tr>'
            '<th style="text-align:left;">Dimension</th>'
            '<th>Score</th>'
            '<th>Region Avg</th>'
            '<th>Portfolio Avg</th>'
            '<th>vs Region</th>'
            '</tr></thead><tbody>'
        )
        for dim in SCORE_DIMENSIONS:
            val = proj_row.get(dim, 0)
            r_avg = avg_region[dim]
            p_avg = avg_portfolio[dim]
            diff = val - r_avg
            diff_cls = 'delta-positive' if diff >= 0 else 'delta-negative'
            cls = f"rating-{int(val)}" if pd.notna(val) else ""
            short = dim.replace('PM Performance', 'PM Perf')

            comp_html += (
                f'<tr>'
                f'<td style="text-align:left; color:#e2e8f0;">{short}</td>'
                f'<td><span class="rating-badge {cls}">{int(val)}</span></td>'
                f'<td style="color:#94a3b8;">{r_avg:.2f}</td>'
                f'<td style="color:#94a3b8;">{p_avg:.2f}</td>'
                f'<td><span class="{diff_cls}">{diff:+.2f}</span></td>'
                f'</tr>'
            )
        comp_html += '</tbody></table></div>'
        st.markdown(comp_html, unsafe_allow_html=True)

        # All qualitative text fields for this project
        st.markdown("**Notes & Issues**")
        _text_summary_panel(drill_df)

elif selected_area != 'All Areas':
    # ---- AREA LEVEL ----
    # Show area-level analysis with project comparison
    _kpi_row(drill_df)

    # Horizontal bar chart: compare projects within this area by avg pulse
    fig_comp = _comparison_bar_chart(drill_df, 'Project', f'Projects in {selected_area}')
    st.plotly_chart(fig_comp, use_container_width=True)

    # Dimension heatmap: rows = projects, columns = dimensions
    st.markdown("**Project \u00d7 Dimension Scores**")
    st.markdown(_dimension_heatmap_html(drill_df, 'Project', 'Project'),
                unsafe_allow_html=True)

    # Highlight dimensions averaging below 2.0 (indicating systemic weakness)
    st.markdown("**Weakest Dimensions (avg < 2.0)**")
    dim_avgs = drill_df[SCORE_DIMENSIONS].mean().sort_values()
    weak_dims = dim_avgs[dim_avgs < 2.0]
    if not weak_dims.empty:
        for dim_name, dim_avg in weak_dims.items():
            dim_color = DIMENSION_COLORS.get(dim_name, '#2563eb')
            st.markdown(
                f'<span style="color:{dim_color}; font-weight:600;">{dim_name}</span>: '
                f'{dim_avg:.2f} avg \u2014 '
                f'{len(drill_df[drill_df[dim_name] <= 1])} projects at 0\u20131',
                unsafe_allow_html=True,
            )
    else:
        st.success("All dimensions averaging 2.0+")

    # Qualitative text summaries for all projects in this area
    st.markdown("**Notes & Issues**")
    _text_summary_panel(drill_df)

elif selected_region != 'All Regions':
    # ---- REGION LEVEL ----
    # Show region-level analysis with area comparison
    _kpi_row(drill_df)

    # Horizontal bar chart: compare areas within this region
    fig_comp = _comparison_bar_chart(drill_df, 'Area', f'Areas in {selected_region}')
    st.plotly_chart(fig_comp, use_container_width=True)

    # Dimension heatmap: rows = areas, columns = dimensions
    st.markdown("**Area \u00d7 Dimension Scores**")
    st.markdown(_dimension_heatmap_html(drill_df, 'Area', 'Area'),
                unsafe_allow_html=True)

    # Table of bottom projects in this region (sorted worst-first)
    st.markdown("**Bottom Projects (by Total Score)**")
    st.markdown(_project_table_html(drill_df, max_rows=15), unsafe_allow_html=True)

    # Qualitative text summaries
    st.markdown("**Notes & Issues**")
    _text_summary_panel(drill_df)

else:
    # ---- PORTFOLIO LEVEL (no drill filter active) ----
    # Show portfolio-wide analysis with region comparison
    _kpi_row(drill_df)

    # Horizontal bar chart: compare regions by average pulse
    fig_comp = _comparison_bar_chart(drill_df, 'Region', 'Average Pulse by Region')
    st.plotly_chart(fig_comp, use_container_width=True)

    # Dimension heatmap: rows = regions, columns = dimensions
    st.markdown("**Region \u00d7 Dimension Scores**")
    st.markdown(_dimension_heatmap_html(drill_df, 'Region', 'Region'),
                unsafe_allow_html=True)

    # Status distribution: colored cards showing count and % per status tier
    st.markdown("**Status Distribution**")
    status_counts = drill_df['Pulse_Status'].value_counts()
    status_html = '<div style="display:flex; gap:0.5rem; flex-wrap:wrap;">'
    for status_name in STATUS_ORDER:
        count = status_counts.get(status_name, 0)
        color = STATUS_CONFIG[status_name]['color']
        pct = (count / len(drill_df) * 100) if len(drill_df) > 0 else 0
        status_html += (
            f'<div style="background:rgba(255,255,255,0.05); border-left:4px solid {color};'
            f' padding:0.5rem 1rem; border-radius:0 6px 6px 0; min-width:100px;">'
            f'<div style="font-size:1.2rem; font-weight:700; color:{color};">{count}</div>'
            f'<div style="font-size:0.7rem; color:#94a3b8;">{status_name} ({pct:.0f}%)</div>'
            f'</div>'
        )
    status_html += '</div>'
    st.markdown(status_html, unsafe_allow_html=True)

    # Bottom 10 projects across the entire portfolio
    st.markdown("**Bottom 10 Projects**")
    st.markdown(_project_table_html(drill_df, max_rows=10), unsafe_allow_html=True)
