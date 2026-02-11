"""
Pulse Dashboard - PULSE HOME Page (Main Landing Page)
=====================================================

This is the main entry point and landing page for the **Project Pulse** dashboard,
a McKinsey-grade executive view of telecom project portfolio health.  It is built
entirely with Streamlit and renders a combination of native Streamlit widgets and
raw HTML tables (injected via `st.markdown(..., unsafe_allow_html=True)`).

Layout overview (top to bottom):
    1. **Week Scope selector** - "All Weeks (Average)" or a specific Year-Week.
    2. **Header bar** - CSE Unit branding + live Pulse vs Target indicator.
    3. **Pulse Ranking chart** - Horizontal stacked-bar chart (Plotly) showing
       each dimension's contribution to the total Pulse score per week.
    4. **Two-column body**:
       - *Left*  : Interactive Region -> Area -> Project drill-down HTML table
                    with expand/collapse checkboxes and inline sparklines.
       - *Right* : Weekly Trend Heatmap showing per-region Pulse scores over time.
    5. **Ratings Legend** - Reference table explaining the 0-3 scoring scheme per
       dimension and the Pulse color bands (red / yellow / green / dark-green).
    6. **Drill-Down Panel** - Appears when the user clicks a dimension band in the
       Pulse Ranking chart; shows per-region bar, distribution, project detail
       table, and weekly trend line chart.
    7. **Footer** - Simple branded footer.

Key DataFrames
--------------
* **df** (`st.session_state.df`):
    The *full* dataset loaded from ``ProjectPulse.xlsx``.  Never filtered.

* **filtered_df**:
    The sidebar-filtered view of ``df`` (region + week filters applied).
    Used only to decide whether there is any data to show; the actual rendering
    is driven by ``scope_df`` and ``mw_df``.

* **mw_df** (multi-week DataFrame):
    A copy of ``df`` filtered by selected *regions* only -- week filters are
    intentionally **not** applied so that trend sparklines and the weekly
    heatmap always show all available weeks for context.

* **scope_df**:
    Derived from ``mw_df`` based on the **Week Scope** selector at the top of
    the page:
    - "All Weeks (Average)": scores are averaged across every week per project,
      then merged back with metadata (Region, Area, PM Name) taken from the
      most-recent week entry.
    - Specific week (e.g. "2025-W12"): ``mw_df`` filtered to that single week.
    This is the DataFrame that drives the *score values* shown in the Region /
    Area table and the project sub-rows.

* **region_info** (dict):
    A nested dictionary built from ``scope_df`` + ``mw_df``:
        { "Central": { "pulse": 18.5, "trend": [...], <dim scores>,
                        "areas": { "DFW": { "pulse": ..., "trend": [...], ... } }
                      },
          ...
        }
    Used to generate the Region -> Area HTML table rows.

Key helper function
-------------------
* ``_score_row_html(name, data, level)``:
    Renders one ``<tr>`` of the matrix table.  ``data`` is a dict with keys
    ``pulse``, ``trend`` (array for sparkline), and each dimension name.
    ``level`` controls CSS class and indentation ("region", "area", or "total").

Expand/collapse mechanism
-------------------------
Streamlit cannot embed interactive widgets (checkboxes) inside raw HTML tables.
The workaround is:
    1. Render Streamlit ``st.checkbox`` widgets *above* the HTML table in a row
       of ``st.columns`` -- one checkbox per region.
    2. Track which regions (and areas) are expanded via the ``expanded_regions``
       and ``expanded_areas`` Python sets.
    3. When building the HTML table, conditionally include area sub-rows (and
       project sub-sub-rows) only if the parent is in the expanded set.
This produces a visually seamless drill-down experience even though the toggles
live outside the HTML table DOM.

Usage:
    streamlit run pulse_dashboard/app.py --server.port 8502
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
# Ensure the pulse_dashboard package directory is on sys.path so that
# relative imports like ``from utils.data_loader import ...`` resolve
# correctly regardless of where Streamlit is launched from.
# This keeps the dashboard self-contained -- it does NOT import from the
# sibling ``escalation_ai`` package.
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Local utility imports
# ---------------------------------------------------------------------------
from utils.data_loader import load_pulse_data, get_default_file_path
from utils.styles import (
    inject_css, STATUS_CONFIG, STATUS_ORDER, SCORE_DIMENSIONS,
    get_pulse_status, get_pulse_color, MCKINSEY_COLORS, DIMENSION_COLORS,
    REGION_LINE_COLORS, pulse_css_class, heat_css_class, score_css_class,
)

# ============================================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================================
# inject_css() internally calls st.set_page_config() and then injects the
# custom CSS stylesheet (dark theme, matrix table styles, heatmap cells, etc.)
# into the page via st.markdown.  Streamlit requires set_page_config to be
# the very first st.* call, so this must come before any other rendering.
inject_css()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Streamlit re-runs the entire script on every user interaction.  We use
# st.session_state to persist data across reruns.  The DEFAULTS dict lists
# every key used throughout the app together with its initial value.
DEFAULTS = {
    'df': None,                    # Full raw DataFrame (all regions, all weeks)
    'filtered_df': None,           # Sidebar-filtered DataFrame (set by render_sidebar)
    'selected_year': None,         # Currently selected year filter (sidebar)
    'selected_week': None,         # Currently selected week filter (sidebar)
    'selected_regions': [],        # List of region strings currently selected
    'pulse_target': 17.0,          # Target Pulse score (shown in header delta)
    'pulse_stretch': 19.0,         # Stretch goal (used elsewhere for KPI coloring)
    'green_pct_target': 80,        # % of projects expected to be green
    'max_red_target': 3,           # Max acceptable number of red projects
    'embeddings_index': None,      # Reserved for semantic search / Ollama embeddings
    'selected_project': None,      # Single selected project for detail view
    'ollama_available': None,      # Flag: is the Ollama LLM backend reachable?
    'selected_drill': None,        # Dict describing the active drill-down panel
}
# Only set defaults for keys that do not already exist so that user
# selections survive across Streamlit reruns.
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================================
# DATA LOADING
# ============================================================================
# On the very first run (or if the data was cleared), attempt to locate and
# load the ProjectPulse.xlsx file from the project root.  Once loaded, a
# rerun is triggered so the rest of the script sees the data immediately.
if st.session_state.df is None:
    default_path = get_default_file_path()
    if default_path:
        with st.spinner("Loading ProjectPulse.xlsx..."):
            st.session_state.df = load_pulse_data(str(default_path))
        st.rerun()  # Force a full re-execution now that df is populated
    else:
        st.error("ProjectPulse.xlsx not found in project root.")
        st.stop()  # Halt rendering -- nothing more we can do without data

# ============================================================================
# LANDING PAGE (data loaded -- all code below assumes df is available)
# ============================================================================
# These imports are deferred to here (after data load) because they pull in
# heavier charting code that is unnecessary if the app is still waiting for data.
from utils.sidebar import render_sidebar
from utils.mckinsey_charts import (
    chart_pulse_ranking, mini_sparkline_svg,
    chart_dimension_by_region, chart_dimension_distribution,
    chart_dimension_trend_by_region,
)

# Render the sidebar (region / week / year filters) and obtain the filtered
# DataFrame.  The sidebar also mutates session state keys like
# selected_regions, selected_year, selected_week.
filtered_df = render_sidebar()

# Keep a convenience reference to the full (unfiltered) DataFrame.
df = st.session_state.df

# Guard: if sidebar filters eliminate all data, show a warning and stop.
if filtered_df is None or filtered_df.empty:
    st.warning("No data matches the current filters. Adjust sidebar filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Build the multi-week dataset (mw_df)
# ---------------------------------------------------------------------------
# mw_df is region-filtered but NOT week-filtered.  This is intentional:
# sparklines and the weekly heatmap need access to *all* weeks within the
# selected regions to show a meaningful trend.  Week scoping is applied
# separately via scope_df below.
selected_regions = st.session_state.get('selected_regions', [])
mw_df = df.copy()
if selected_regions:
    mw_df = mw_df[mw_df['Region'].isin(selected_regions)]

# Retrieve the user-configured Pulse target for delta calculations.
target = st.session_state.get('pulse_target', 17.0)

# ============================================================================
# WEEK SCOPE SELECTOR -- default to "All Weeks (Average)"
# ============================================================================
# Collect every distinct (Year, Wk) pair present in the multi-week data and
# sort chronologically.  These become the options in the Week Scope dropdown.
yw_pairs_all = (
    mw_df[['Year', 'Wk']].drop_duplicates()
    .sort_values(['Year', 'Wk'])
    .values.tolist()
)
# Format each pair as "YYYY-WNN" (e.g. "2025-W04") for display.
week_labels_all = [f"{int(y)}-W{int(w):02d}" for y, w in yw_pairs_all]
# Prepend the aggregate option so the user can view a cross-week average.
week_options_all = ['All Weeks (Average)'] + week_labels_all

# Layout: narrow column for the selector, wider column for the context label.
scope_c1, scope_c2 = st.columns([1, 3])
with scope_c1:
    home_week_scope = st.selectbox(
        'Week Scope',
        week_options_all,
        index=0,  # default to "All Weeks (Average)"
        key='home_week_scope',
    )

# ---------------------------------------------------------------------------
# Build scope_df based on Week Scope selection
# ---------------------------------------------------------------------------
# scope_df is the DataFrame whose scores are displayed in the Region/Area
# matrix table and used for the current-scope header metrics.
if home_week_scope == 'All Weeks (Average)':
    # --- Aggregate mode: average every dimension score per project ----------
    # Build the aggregation dict: mean of each scoring dimension + Total Score.
    agg_cols = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg_cols['Total Score'] = 'mean'

    # For metadata (Region, Area, PM Name) use the *most recent* week entry
    # per project.  This avoids issues where a project changed region mid-period.
    latest_idx = mw_df.sort_values(['Year', 'Wk']).groupby('Project').tail(1).index
    meta_df = mw_df.loc[latest_idx, ['Project', 'Region', 'Area', 'PM Name']].drop_duplicates('Project')

    # Compute per-project average scores across all weeks.
    avg_scores = mw_df.groupby('Project').agg(agg_cols).reset_index()
    # Round to 2 decimals for clean display.
    for dim in SCORE_DIMENSIONS:
        avg_scores[dim] = avg_scores[dim].round(2)
    avg_scores['Total Score'] = avg_scores['Total Score'].round(2)

    # Merge metadata with averaged scores.
    scope_df = meta_df.merge(avg_scores, on='Project', how='inner')
    # Derive a categorical Pulse status (Red/Yellow/Green/DarkGreen) from Total Score.
    scope_df['Pulse_Status'] = scope_df['Total Score'].apply(get_pulse_status)
    scope_label = f"Average across {len(yw_pairs_all)} weeks | {scope_df['Project'].nunique()} projects"
else:
    # --- Single-week mode: filter mw_df to the chosen year/week pair --------
    yr_str, wk_str = home_week_scope.split('-W')
    sel_yr, sel_wk = int(yr_str), int(wk_str)
    scope_df = mw_df[(mw_df['Year'] == sel_yr) & (mw_df['Wk'] == sel_wk)].copy()
    scope_label = f"Week {home_week_scope} | {len(scope_df)} projects"

# Show the scope context label to the right of the dropdown.
with scope_c2:
    st.caption(scope_label)

# Compute the portfolio-wide average Pulse and its variance from target.
avg_pulse = scope_df['Total Score'].mean() if not scope_df.empty else 0
variance = avg_pulse - target

# ---------------------------------------------------------------------------
# Header bar: branding + live Pulse vs Target indicator
# ---------------------------------------------------------------------------
# Condense scope text for inline display.
scope_text = "All Weeks Avg" if home_week_scope == 'All Weeks (Average)' else home_week_scope

# The header is a flexbox row:
#   - Left:  construction emoji + "CSE UNIT" title + subtitle
#   - Right: scope label | Pulse value vs target with +/- delta styled green/red
st.markdown(f"""
<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
    <div style="font-size:2rem;">\U0001f3d7\ufe0f</div>
    <div>
        <div style="font-size:1.3rem; font-weight:700; color:#60a5fa;">CSE UNIT</div>
        <div style="font-size:0.65rem; color:#64748b; letter-spacing:1px;">PROJECT PULSE TRACKER</div>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <div style="font-size:0.8rem; color:#94a3b8;">
            {scope_text} |
            Pulse: <b style="color:white;">{avg_pulse:.1f}</b> vs {target:.0f}
            (<span class="{'delta-positive' if variance >= 0 else 'delta-negative'}">{variance:+.1f}</span>)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PULSE RANKING CHART
# ============================================================================
# Section title rendered as styled HTML for visual consistency.
st.markdown('<div class="section-title">Pulse Ranking</div>', unsafe_allow_html=True)

# chart_pulse_ranking returns a Plotly Figure: a horizontal stacked bar chart
# where each bar segment represents one scoring dimension's contribution.
# It uses the full mw_df (all weeks) so the user can compare weeks.
fig_ranking = chart_pulse_ranking(mw_df)

# Render with on_select="rerun" so clicking a bar segment triggers a Streamlit
# rerun and populates `event` with the click coordinates.
event = st.plotly_chart(fig_ranking, use_container_width=True, on_select="rerun",
                        key="pulse_ranking_chart")

# ---------------------------------------------------------------------------
# Handle chart click -> open the drill-down panel
# ---------------------------------------------------------------------------
# When the user clicks a dimension band in the ranking chart, Plotly returns
# the clicked point's customdata.  We store it in session state so the
# drill-down panel (rendered further below) knows what to show.
if event and event.selection and event.selection.points:
    pt = event.selection.points[0]
    if 'customdata' in pt and pt['customdata']:
        st.session_state.selected_drill = {
            'type': 'dimension',           # Drill-down type (currently only "dimension")
            'value': pt['customdata'][0],  # Dimension name (e.g. "CSAT")
            'week': pt['customdata'][1],   # Year-Week string (e.g. "2025-W04")
        }

# ============================================================================
# MAIN TABLES: Region/Area Matrix (left) + Weekly Heatmap (right)
# ============================================================================

# Sorted list of all distinct regions present in the multi-week data.
# Used as the iteration order for both the matrix table and the heatmap.
all_regions = sorted(mw_df['Region'].dropna().unique())


# ---------------------------------------------------------------------------
# Helper: _score_row_html
# ---------------------------------------------------------------------------
def _score_row_html(name, data, level='region'):
    """
    Build one HTML ``<tr>`` for the Region/Area matrix table.

    Parameters
    ----------
    name : str
        Display label for the row (region name, area name, or "Total").
    data : dict
        Must contain:
        - ``'pulse'``  : float -- the Total Score value for this row.
        - ``'trend'``  : array-like -- weekly Total Score values used to draw
          a mini SVG sparkline.
        - One key per dimension in ``SCORE_DIMENSIONS`` (e.g. "CSAT", "LOB",
          "PM Performance", "Potential") with float values.
    level : str, optional
        One of ``'region'``, ``'area'``, or ``'total'``.  Controls:
        - The CSS class applied to the ``<tr>`` (``region-row`` / ``area-row``
          / ``total-row``), which in turn controls background color and font.
        - Whether the name cell uses ``region-name`` or ``area-name`` styling
          (indentation, weight).

    Returns
    -------
    str
        A complete ``<tr>...</tr>`` HTML string ready to be concatenated into
        the ``<tbody>`` of the matrix table.

    Cell layout (left to right):
        Name | Sparkline SVG | Pulse score (color-coded) | dim1 | dim2 | ...
    """
    # Choose sparkline color: pink if below 16 (at-risk), blue otherwise.
    spark_color = "#f472b6" if data['pulse'] < 16 else "#60a5fa"
    # mini_sparkline_svg returns an inline <svg> element rendered as a tiny
    # line chart of the trend array.
    sparkline = mini_sparkline_svg(data['trend'], color=spark_color)

    # Determine the CSS class for the Pulse score cell (pulse-red, pulse-yellow,
    # pulse-green, or pulse-darkgreen) based on value thresholds.
    p_cls = pulse_css_class(data['pulse'])
    p_val = f"{data['pulse']:.2f}" if pd.notna(data['pulse']) else "\u2014"

    # Build one <td> per scoring dimension, each with a color-coded badge.
    score_cells = ""
    for dim in SCORE_DIMENSIONS:
        val = data.get(dim)
        cls = score_css_class(val)  # e.g. "rating-0", "rating-1", ... "rating-3"
        display = f"{val:.2f}" if pd.notna(val) else "\u2014"
        score_cells += f'<td><span class="score-cell {cls}">{display}</span></td>'

    # Name cell and row class depend on the hierarchy level.
    if level == 'total':
        name_cell = '<td class="region-name" style="color:#60a5fa;">Total</td>'
        row_class = 'total-row'
    elif level == 'area':
        name_cell = f'<td class="area-name">{name}</td>'
        row_class = 'area-row'
    else:
        name_cell = f'<td class="region-name">{name}</td>'
        row_class = 'region-row'

    # Assemble the full row: Name | Sparkline | Pulse | dimension scores...
    return (
        f'<tr class="{row_class}">'
        f'{name_cell}'
        f'<td>{sparkline}</td>'
        f'<td><span class="score-cell {p_cls}">{p_val}</span></td>'
        f'{score_cells}'
        f'</tr>'
    )


# ---------------------------------------------------------------------------
# Reusable HTML table header row
# ---------------------------------------------------------------------------
# The matrix table always has the same columns:
#   Region | Trend (sparkline) | Pulse | <one column per scoring dimension>
# We build the <thead> once and reuse it.
_table_header = (
    '<thead><tr>'
    '<th style="text-align:left; min-width:100px;">Region</th>'
    '<th style="min-width:75px;">Trend</th>'
    '<th>Pulse</th>'
)
for _dim in SCORE_DIMENSIONS:
    # Shorten "PM Performance" to "PM Perf" to save horizontal space.
    _short = _dim.replace('PM Performance', 'PM Perf')
    _table_header += f'<th>{_short}</th>'
_table_header += '</tr></thead>'

# ---------------------------------------------------------------------------
# Build region_info: nested hierarchy of Region -> Area -> scores/trends
# ---------------------------------------------------------------------------
# region_info is the single data structure that drives the entire left-column
# matrix table.  Each region entry contains:
#   - 'trend'  : array of weekly average Total Score (from mw_df, all weeks)
#   - 'pulse'  : average Total Score (from scope_df, respects week scope)
#   - <dim>    : average score per dimension (from scope_df)
#   - 'areas'  : dict of area_name -> { trend, pulse, <dims> }
#
# The trend sparkline intentionally uses mw_df (all weeks) for richer context,
# while the numeric scores use scope_df (averaged or week-specific) to match
# whatever the user selected in the Week Scope dropdown.
region_info = {}
for region in all_regions:
    # Scope-filtered data for this region (scores shown in table).
    rdf_scope = scope_df[scope_df['Region'] == region]
    # Full multi-week data for this region (used only for trend sparklines).
    rdf_full = mw_df[mw_df['Region'] == region]

    if len(rdf_scope) == 0:
        continue  # Skip regions that have no data in the current scope

    # Trend sparkline: average Total Score per week across all projects in region.
    region_trend = rdf_full.groupby('Wk')['Total Score'].mean().sort_index().values
    # Scores from scope_df (averaged across projects within the region).
    region_avg = rdf_scope['Total Score'].mean()
    region_scores = {c: rdf_scope[c].mean() for c in SCORE_DIMENSIONS}

    region_info[region] = {
        'trend': region_trend, 'pulse': region_avg, **region_scores,
        'areas': {}
    }

    # Iterate over every area within this region.
    for area in sorted(rdf_scope['Area'].dropna().unique()):
        adf_scope = rdf_scope[rdf_scope['Area'] == area]
        adf_full = rdf_full[rdf_full['Area'] == area]
        if len(adf_scope) == 0:
            continue
        # Same pattern: trend from all weeks, scores from scope.
        area_trend = adf_full.groupby('Wk')['Total Score'].mean().sort_index().values
        area_avg = adf_scope['Total Score'].mean()
        area_scores = {c: adf_scope[c].mean() for c in SCORE_DIMENSIONS}
        region_info[region]['areas'][area] = {
            'trend': area_trend, 'pulse': area_avg, **area_scores,
        }

# Portfolio-wide totals (for the "Total" row at the bottom of the table).
total_trend = mw_df.groupby('Wk')['Total Score'].mean().sort_index().values
total_avg = scope_df['Total Score'].mean()
total_scores = {c: scope_df[c].mean() for c in SCORE_DIMENSIONS}

# ---------------------------------------------------------------------------
# Two-column layout: left = Region/Area matrix, right = Weekly heatmap
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([2.2, 1])

# ── LEFT COLUMN: Region | Area Interactive Matrix ──
with col_left:
    st.markdown('<div class="section-title">Project Pulse \u2013 Region | Area</div>',
                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Region expand toggles (row of compact checkboxes)
    # ------------------------------------------------------------------
    # Because Streamlit cannot embed widgets inside raw HTML, we render
    # one st.checkbox per region in a horizontal row of st.columns ABOVE
    # the HTML table.  Checking a box adds the region to expanded_regions.
    avail_regions = [r for r in all_regions if r in region_info]
    r_cols = st.columns(len(avail_regions))
    expanded_regions = set()
    for i, region in enumerate(avail_regions):
        with r_cols[i]:
            # Triangle-right icon hints that clicking expands the region.
            if st.checkbox(f'\u25B6 {region}', key=f'exp_r_{region}'):
                expanded_regions.add(region)

    # ------------------------------------------------------------------
    # Area expand toggles (only shown for regions that are expanded)
    # ------------------------------------------------------------------
    # Gather all (region, area) pairs where the parent region is expanded.
    expandable_areas = []
    for r in expanded_regions:
        for a in region_info[r]['areas']:
            expandable_areas.append((r, a))

    expanded_areas = set()
    if expandable_areas:
        # Lay out area checkboxes in up to 6 columns, wrapping as needed.
        a_cols = st.columns(min(len(expandable_areas), 6))
        for i, (r, a) in enumerate(expandable_areas):
            col_idx = i % min(len(expandable_areas), 6)
            with a_cols[col_idx]:
                if st.checkbox(f'\u25B6 {a}', key=f'exp_a_{r}_{a}'):
                    # Track expanded areas as "Region/Area" strings for lookup.
                    expanded_areas.add(f'{r}/{a}')

    # ------------------------------------------------------------------
    # Build ONE unified HTML table for the Region/Area/Project hierarchy
    # ------------------------------------------------------------------
    # The table has a fixed column layout (11 columns):
    #   Name | Trend (sparkline) | Pulse | dim1 | dim2 | ... | dim8
    # When a region is expanded, area sub-rows appear indented below it.
    # When an area is expanded, individual project rows appear below the
    # area, reusing the same 11 columns but with a different mapping:
    #   Project | PM Name | dim1 | ... | dim8 | Total Score
    unified_html = f'<div class="matrix-container"><table class="matrix-table">{_table_header}<tbody>'

    for region in avail_regions:
        info = region_info[region]
        is_expanded = region in expanded_regions

        # Region summary row -- triangle indicates expanded/collapsed state.
        indicator = '\u25BC' if is_expanded else '\u25B6'
        unified_html += _score_row_html(f'{indicator} {region}', info, 'region')

        if is_expanded:
            # Render one area row per area within this region.
            for area_name, area_data in info['areas'].items():
                area_key = f'{region}/{area_name}'
                area_expanded = area_key in expanded_areas
                a_indicator = '\u25BC' if area_expanded else '\u25B6'

                unified_html += _score_row_html(f'{a_indicator} {area_name}', area_data, 'area')

                if area_expanded:
                    # --------------------------------------------------
                    # Project sub-rows (individual project-level detail)
                    # --------------------------------------------------
                    # Define the column order for project rows.
                    proj_cols_ordered = ['Project', 'PM Name'] + SCORE_DIMENSIONS + ['Total Score']
                    # Only include columns that actually exist in scope_df.
                    proj_avail = [c for c in proj_cols_ordered if c in scope_df.columns]

                    # Sub-header row: column labels for the project detail section.
                    # Styled with a darker background to visually separate it from
                    # the Region/Area rows above.
                    unified_html += '<tr style="background:#0c1a30; border-top:1px solid #1e3a5f;">'
                    for c in proj_avail:
                        align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
                        label = c.replace('PM Performance', 'PM Perf')
                        unified_html += (
                            f'<th style="{align} font-size:0.6rem; padding:0.25rem;'
                            f' color:#64748b; background:#0c1a30;">{label}</th>'
                        )
                    unified_html += '</tr>'

                    # Fetch projects for this Region + Area from scope_df,
                    # sorted ascending by Total Score (worst first for quick
                    # identification of underperformers).
                    proj_df = scope_df[
                        (scope_df['Region'] == region) &
                        (scope_df['Area'] == area_name)
                    ].sort_values('Total Score', ascending=True)

                    # Render one row per project.
                    for _, r in proj_df.iterrows():
                        unified_html += '<tr style="background:rgba(255,255,255,0.02);">'
                        for c in proj_avail:
                            val = r[c]
                            # Color-code cells based on their column type:
                            if c == 'Total Score':
                                # Pulse-style coloring (red/yellow/green/darkgreen)
                                cls = pulse_css_class(val)
                                display = f'<span class="score-cell {cls}">{val:.1f}</span>' if pd.notna(val) else '\u2014'
                            elif c in SCORE_DIMENSIONS:
                                # Dimension scores use rating-badge coloring (0-3 scale)
                                cls = f"rating-{int(round(val))}" if pd.notna(val) else ""
                                display = f'<span class="rating-badge {cls}">{val:.1f}</span>' if pd.notna(val) else '\u2014'
                            elif c in ('Project', 'PM Name'):
                                # Text columns: truncate to 40 chars.
                                display = str(val)[:40] if pd.notna(val) else '\u2014'
                            else:
                                # Fallback for any other column.
                                display = str(val) if pd.notna(val) else '\u2014'
                            align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
                            # Indent the first column (Project name) for visual nesting.
                            unified_html += f'<td style="{align} padding-left:2.5rem;">{display}</td>' if c == proj_avail[0] else f'<td style="{align}">{display}</td>'
                        unified_html += '</tr>'

    # Portfolio-wide "Total" row at the bottom of the table.
    total_data = {'trend': total_trend, 'pulse': total_avg, **total_scores}
    unified_html += _score_row_html('Total', total_data, 'total')

    # Close the table and its scrollable container div.
    unified_html += '</tbody></table></div>'

    # Inject the completed HTML table into the Streamlit page.
    st.markdown(unified_html, unsafe_allow_html=True)

# ── RIGHT COLUMN: Weekly Trend Heatmap (scrollable, all weeks) ──
with col_right:
    st.markdown('<div class="section-title">Project Pulse \u2013 Weekly Trend</div>',
                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Gather all (Year, Wk) pairs sorted chronologically for the heatmap
    # columns.  Uses mw_df (all weeks) so the heatmap always shows full
    # history regardless of the Week Scope selector.
    # ------------------------------------------------------------------
    yw_pairs = (mw_df[['Year', 'Wk']].drop_duplicates()
                .sort_values(['Year', 'Wk'])
                .values.tolist())

    # ------------------------------------------------------------------
    # Build heatmap data: one dict per region, keyed by "YY-WNN" strings.
    # Each value is the average Total Score for that region in that week.
    # ------------------------------------------------------------------
    heat_data = []
    for region in all_regions:
        rdf = mw_df[mw_df['Region'] == region]
        row = {'Region': region}
        for yr, wk in yw_pairs:
            key = f"{int(yr) % 100}-W{int(wk):02d}"  # e.g. "25-W04"
            wdf = rdf[(rdf['Year'] == yr) & (rdf['Wk'] == wk)]
            row[key] = wdf['Total Score'].mean() if len(wdf) > 0 else None
        heat_data.append(row)

    # Append a portfolio-wide "Total" row.
    total_heat = {'Region': 'Total'}
    for yr, wk in yw_pairs:
        key = f"{int(yr) % 100}-W{int(wk):02d}"
        wdf = mw_df[(mw_df['Year'] == yr) & (mw_df['Wk'] == wk)]
        total_heat[key] = wdf['Total Score'].mean() if len(wdf) > 0 else None
    heat_data.append(total_heat)

    # Visual icons per region for the heatmap's Region column.
    region_icons = {
        'Central': '\U0001f7e6', 'NE': '\U0001f7e9',
        'South': '\U0001f7e8', 'West': '\U0001f7ea', 'Total': '\u2b1c',
    }

    # ------------------------------------------------------------------
    # Build the heatmap HTML table
    # ------------------------------------------------------------------
    # The table sits inside a horizontally scrollable container so that
    # many weeks can be displayed without breaking the layout.
    heat_html = (
        '<div class="heatmap-scroll-container"><table class="heatmap-table">'
        '<thead><tr><th style="text-align:left;">Region</th>'
    )
    # One column header per week.
    for yr, wk in yw_pairs:
        label = f"{int(yr) % 100}-W{int(wk):02d}"
        heat_html += f'<th style="white-space:nowrap;">{label}</th>'
    heat_html += '</tr></thead><tbody>'

    # One row per region (plus the Total row appended earlier).
    for row in heat_data:
        is_total = row['Region'] == 'Total'
        row_class = "total-row" if is_total else ""
        icon = region_icons.get(row['Region'], '')

        heat_html += f'<tr class="{row_class}">'
        heat_html += f'<td class="region-col">{icon} {row["Region"]}</td>'

        for yr, wk in yw_pairs:
            key = f"{int(yr) % 100}-W{int(wk):02d}"
            val = row.get(key)
            if val is not None and pd.notna(val):
                # Color-code the cell using the heatmap CSS classes
                # (heat-red, heat-yellow, heat-green, heat-darkgreen).
                cls = heat_css_class(val)
                heat_html += f'<td><span class="heatmap-cell {cls}">{val:.1f}</span></td>'
            else:
                heat_html += '<td>\u2014</td>'  # Em-dash for missing data
        heat_html += '</tr>'

    heat_html += '</tbody></table></div>'
    st.markdown(heat_html, unsafe_allow_html=True)

# ============================================================================
# RATINGS LEGEND
# ============================================================================
# A six-column reference section explaining what each score (0-3) means for
# every dimension, plus the Pulse color bands and a notes panel.
st.markdown('<div class="section-title" style="margin-top:0.8rem;">Ratings Reference</div>',
            unsafe_allow_html=True)

leg1, leg2, leg3, leg4, leg5, leg6 = st.columns([1.2, 1, 1, 1.2, 1, 1.5])

# --- Column 1: LOB (Line of Business) rating definitions ---
with leg1:
    st.markdown('''
    <table class="legend-table">
        <tr><th colspan="2">Ratings \u2013 LOB</th></tr>
        <tr><td><span class="rating-badge rating-0">0</span>Escalation</td></tr>
        <tr><td><span class="rating-badge rating-1">1</span>Complaint/Concern</td></tr>
        <tr><td><span class="rating-badge rating-2">2</span>BAU / NA</td></tr>
        <tr><td><span class="rating-badge rating-3">3</span>Appreciation</td></tr>
    </table>
    ''', unsafe_allow_html=True)

# --- Column 2: CSAT rating definitions ---
with leg2:
    st.markdown('''
    <table class="legend-table">
        <tr><th>CSAT</th></tr>
        <tr><td>Escalation</td></tr>
        <tr><td>Complaint</td></tr>
        <tr><td>Mixed</td></tr>
        <tr><td>Positive</td></tr>
    </table>
    ''', unsafe_allow_html=True)

# --- Column 3: PM Performance rating definitions ---
with leg3:
    st.markdown('''
    <table class="legend-table">
        <tr><th>PM Perf</th></tr>
        <tr><td>Escalation</td></tr>
        <tr><td>Issues</td></tr>
        <tr><td>On-time / Good</td></tr>
        <tr><td>Exceptional</td></tr>
    </table>
    ''', unsafe_allow_html=True)

# --- Column 4: Potential / Growth rating definitions ---
with leg4:
    st.markdown('''
    <table class="legend-table">
        <tr><th>Potential</th></tr>
        <tr><td>Declining / At Risk</td></tr>
        <tr><td>Stagnant</td></tr>
        <tr><td>Moderate Opportunity</td></tr>
        <tr><td>Strong Future ROI</td></tr>
    </table>
    ''', unsafe_allow_html=True)

# --- Column 5: Project Pulse total-score color bands ---
with leg5:
    st.markdown('''
    <table class="legend-table">
        <tr><th colspan="2">Project Pulse</th></tr>
        <tr><td><span class="score-cell pulse-red" style="margin-right:0.4rem;">1-13</span></td></tr>
        <tr><td><span class="score-cell pulse-yellow" style="margin-right:0.4rem;">14-15</span></td></tr>
        <tr><td><span class="score-cell pulse-green" style="margin-right:0.4rem;">16-19</span></td></tr>
        <tr><td><span class="score-cell pulse-darkgreen" style="margin-right:0.4rem;">20-24</span></td></tr>
    </table>
    ''', unsafe_allow_html=True)

# --- Column 6: Usage notes ---
with leg6:
    st.markdown('''
    <div class="notes-box">
        <div class="notes-title">Notes</div>
        Pulse Ranking \u2013 Ranking of each contributor in Project Pulse per week.
        Click any band in the chart to drill down into that dimension.
    </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# DRILL-DOWN PANEL (conditionally shown when a dimension band is clicked)
# ============================================================================
# The drill-down is triggered by clicking a stacked-bar segment in the Pulse
# Ranking chart above.  The click handler stores the dimension name and week
# in st.session_state.selected_drill.  If that key is set, we render a full
# analytical panel for the selected dimension.
if st.session_state.get('selected_drill'):
    drill = st.session_state.selected_drill

    if drill['type'] == 'dimension':
        dim = drill['value']       # e.g. "CSAT", "LOB", "PM Performance", "Potential"
        wk = drill.get('week')     # e.g. "2025-W04" or None
        dim_color = DIMENSION_COLORS.get(dim, '#2563eb')  # Brand color for this dimension

        # Drill-down header badge: shows dimension name + context (week, regions).
        st.markdown(f"""
        <div class="drilldown-panel">
            <div class="drilldown-header">
                <span class="drilldown-badge" style="background:{dim_color};">{dim}</span>
                <span class="drilldown-context">
                    Drill-down &bull; Week {wk if wk else 'All'} &bull;
                    {len(selected_regions)} regions
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Filter the multi-week data to the specific week if one was clicked.
        # Uses 'Year_Week' column (format "YYYY-WNN") if available in the data.
        drill_df = mw_df.copy()
        if wk and 'Year_Week' in drill_df.columns:
            drill_df = drill_df[drill_df['Year_Week'] == wk]

        # ------------------------------------------------------------------
        # Drill-down charts: two side-by-side Plotly charts
        # ------------------------------------------------------------------
        dc1, dc2 = st.columns(2)
        with dc1:
            # Bar chart: average score for this dimension grouped by region.
            st.markdown(f"**{dim} \u2013 By Region**")
            fig_bar = chart_dimension_by_region(drill_df, dim, dim_color)
            st.plotly_chart(fig_bar, use_container_width=True)
        with dc2:
            # Histogram / distribution: how scores for this dimension are
            # distributed across all projects.
            st.markdown(f"**{dim} \u2013 Score Distribution**")
            fig_dist = chart_dimension_distribution(drill_df, dim)
            st.plotly_chart(fig_dist, use_container_width=True)

        # ------------------------------------------------------------------
        # Project detail table for the drilled dimension
        # ------------------------------------------------------------------
        # Shows individual project rows sorted by the drilled dimension score
        # (ascending = worst first) then by Total Score, capped at 50 rows.
        st.markdown(f"**Project Details \u2013 {dim}**")
        detail_cols = ['Wk', 'Region', 'Area', 'Project', 'PM Name', dim, 'Total Score']
        available = [c for c in detail_cols if c in drill_df.columns]
        detail = drill_df[available].sort_values(
            [dim, 'Total Score'], ascending=[True, True]
        ).head(50)

        # Build the detail table as raw HTML (same approach as the matrix table).
        detail_html = (
            '<div class="matrix-container" style="max-height:250px; overflow-y:auto;">'
            '<table class="matrix-table"><thead><tr>'
        )
        for c in available:
            align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
            label = c.replace('PM Performance', 'PM Perf')
            detail_html += f'<th style="{align}">{label}</th>'
        detail_html += '</tr></thead><tbody>'

        for _, r in detail.iterrows():
            detail_html += '<tr>'
            for c in available:
                val = r[c]
                if c == dim:
                    # Highlight the drilled dimension with a rating-badge.
                    cls = f"rating-{int(val)}" if pd.notna(val) else ""
                    display = f'<span class="rating-badge {cls}">{int(val)}</span>' if pd.notna(val) else '\u2014'
                elif c == 'Total Score':
                    # Pulse-style coloring for total score.
                    cls = pulse_css_class(val)
                    display = f'<span class="score-cell {cls}">{int(val)}</span>' if pd.notna(val) else '\u2014'
                elif c in ('Project', 'PM Name'):
                    display = str(val)[:40] if pd.notna(val) else '\u2014'
                else:
                    display = str(val) if pd.notna(val) else '\u2014'
                align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
                detail_html += f'<td style="{align}">{display}</td>'
            detail_html += '</tr>'

        detail_html += '</tbody></table></div>'
        st.markdown(detail_html, unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Weekly trend line chart for the drilled dimension by region
        # ------------------------------------------------------------------
        # Uses full mw_df (all weeks) so the trend is not limited by scope.
        st.markdown(f"**{dim} \u2013 Weekly Trend by Region**")
        fig_trend = chart_dimension_trend_by_region(mw_df, dim)
        st.plotly_chart(fig_trend, use_container_width=True)

        # "Close" button: clears the drill-down and triggers a rerun so the
        # panel disappears.
        if st.button("\u2715 Close Drill-Down", key="close_drill"):
            st.session_state.selected_drill = None
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
# Minimal branded footer with a top border separator.
st.markdown('''
<div style="text-align:center; padding:1rem; color:#334155; font-size:0.65rem;
            margin-top:1rem; border-top:1px solid #1e293b;">
    CSE Unit &bull; Pulse Tracker &bull; Powered by Streamlit + Plotly
</div>
''', unsafe_allow_html=True)
