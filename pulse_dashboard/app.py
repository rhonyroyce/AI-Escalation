"""
Pulse Dashboard - Main Entry Point

McKinsey-grade executive dashboard for telecom project portfolio management.
Analyzes weekly Pulse scores across projects, regions, and performance dimensions.

Usage:
    streamlit run pulse_dashboard/app.py --server.port 8502
"""

import sys
from pathlib import Path

# Self-contained path setup — no escalation_ai imports
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd

from utils.data_loader import load_pulse_data, get_default_file_path
from utils.styles import (
    inject_css, STATUS_CONFIG, STATUS_ORDER, SCORE_DIMENSIONS,
    get_pulse_status, get_pulse_color, MCKINSEY_COLORS, DIMENSION_COLORS,
    REGION_LINE_COLORS, pulse_css_class, heat_css_class, score_css_class,
)

# ============================================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================================
inject_css()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
DEFAULTS = {
    'df': None,
    'filtered_df': None,
    'selected_year': None,
    'selected_week': None,
    'selected_regions': [],
    'pulse_target': 17.0,
    'pulse_stretch': 19.0,
    'green_pct_target': 80,
    'max_red_target': 3,
    'embeddings_index': None,
    'selected_project': None,
    'ollama_available': None,
    'selected_drill': None,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================================
# DATA LOADING
# ============================================================================
if st.session_state.df is None:
    default_path = get_default_file_path()
    if default_path:
        with st.spinner("Loading ProjectPulse.xlsx..."):
            st.session_state.df = load_pulse_data(str(default_path))
        st.rerun()
    else:
        st.error("ProjectPulse.xlsx not found in project root.")
        st.stop()

# ============================================================================
# LANDING PAGE (data loaded)
# ============================================================================
from utils.sidebar import render_sidebar
from utils.mckinsey_charts import (
    chart_pulse_ranking, mini_sparkline_svg,
    chart_dimension_by_region, chart_dimension_distribution,
    chart_dimension_trend_by_region,
)

filtered_df = render_sidebar()
df = st.session_state.df

if filtered_df is None or filtered_df.empty:
    st.warning("No data matches the current filters. Adjust sidebar filters.")
    st.stop()

# Build multi-week dataset (region/area filtered, NOT week filtered)
selected_regions = st.session_state.get('selected_regions', [])
mw_df = df.copy()
if selected_regions:
    mw_df = mw_df[mw_df['Region'].isin(selected_regions)]

target = st.session_state.get('pulse_target', 17.0)

# ============================================================================
# WEEK SCOPE SELECTOR — default to All Weeks (Average)
# ============================================================================
yw_pairs_all = (
    mw_df[['Year', 'Wk']].drop_duplicates()
    .sort_values(['Year', 'Wk'])
    .values.tolist()
)
week_labels_all = [f"{int(y)}-W{int(w):02d}" for y, w in yw_pairs_all]
week_options_all = ['All Weeks (Average)'] + week_labels_all

scope_c1, scope_c2 = st.columns([1, 3])
with scope_c1:
    home_week_scope = st.selectbox(
        'Week Scope',
        week_options_all,
        index=0,  # default to All Weeks
        key='home_week_scope',
    )

# Build scope_df based on selection
if home_week_scope == 'All Weeks (Average)':
    # Average all dimension scores and Total Score per project
    agg_cols = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg_cols['Total Score'] = 'mean'
    latest_idx = mw_df.sort_values(['Year', 'Wk']).groupby('Project').tail(1).index
    meta_df = mw_df.loc[latest_idx, ['Project', 'Region', 'Area', 'PM Name']].drop_duplicates('Project')
    avg_scores = mw_df.groupby('Project').agg(agg_cols).reset_index()
    for dim in SCORE_DIMENSIONS:
        avg_scores[dim] = avg_scores[dim].round(2)
    avg_scores['Total Score'] = avg_scores['Total Score'].round(2)
    scope_df = meta_df.merge(avg_scores, on='Project', how='inner')
    scope_df['Pulse_Status'] = scope_df['Total Score'].apply(get_pulse_status)
    scope_label = f"Average across {len(yw_pairs_all)} weeks | {scope_df['Project'].nunique()} projects"
else:
    yr_str, wk_str = home_week_scope.split('-W')
    sel_yr, sel_wk = int(yr_str), int(wk_str)
    scope_df = mw_df[(mw_df['Year'] == sel_yr) & (mw_df['Wk'] == sel_wk)].copy()
    scope_label = f"Week {home_week_scope} | {len(scope_df)} projects"

with scope_c2:
    st.caption(scope_label)

avg_pulse = scope_df['Total Score'].mean() if not scope_df.empty else 0
variance = avg_pulse - target

# ── Header ──
scope_text = "All Weeks Avg" if home_week_scope == 'All Weeks (Average)' else home_week_scope
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
st.markdown('<div class="section-title">Pulse Ranking</div>', unsafe_allow_html=True)

fig_ranking = chart_pulse_ranking(mw_df)
event = st.plotly_chart(fig_ranking, use_container_width=True, on_select="rerun",
                        key="pulse_ranking_chart")

# Handle chart click → set drill-down target
if event and event.selection and event.selection.points:
    pt = event.selection.points[0]
    if 'customdata' in pt and pt['customdata']:
        st.session_state.selected_drill = {
            'type': 'dimension',
            'value': pt['customdata'][0],
            'week': pt['customdata'][1],
        }

# ============================================================================
# MAIN TABLES: Region/Area Matrix + Weekly Heatmap
# ============================================================================

# ── Shared data for both tables ──
all_regions = sorted(mw_df['Region'].dropna().unique())


def _score_row_html(name, data, level='region'):
    """Build one HTML <tr> for the matrix table."""
    spark_color = "#f472b6" if data['pulse'] < 16 else "#60a5fa"
    sparkline = mini_sparkline_svg(data['trend'], color=spark_color)
    p_cls = pulse_css_class(data['pulse'])
    p_val = f"{data['pulse']:.2f}" if pd.notna(data['pulse']) else "\u2014"

    score_cells = ""
    for dim in SCORE_DIMENSIONS:
        val = data.get(dim)
        cls = score_css_class(val)
        display = f"{val:.2f}" if pd.notna(val) else "\u2014"
        score_cells += f'<td><span class="score-cell {cls}">{display}</span></td>'

    if level == 'total':
        name_cell = '<td class="region-name" style="color:#60a5fa;">Total</td>'
        row_class = 'total-row'
    elif level == 'area':
        name_cell = f'<td class="area-name">{name}</td>'
        row_class = 'area-row'
    else:
        name_cell = f'<td class="region-name">{name}</td>'
        row_class = 'region-row'

    return (
        f'<tr class="{row_class}">'
        f'{name_cell}'
        f'<td>{sparkline}</td>'
        f'<td><span class="score-cell {p_cls}">{p_val}</span></td>'
        f'{score_cells}'
        f'</tr>'
    )


# Reusable table header
_table_header = (
    '<thead><tr>'
    '<th style="text-align:left; min-width:100px;">Region</th>'
    '<th style="min-width:75px;">Trend</th>'
    '<th>Pulse</th>'
)
for _dim in SCORE_DIMENSIONS:
    _short = _dim.replace('PM Performance', 'PM Perf')
    _table_header += f'<th>{_short}</th>'
_table_header += '</tr></thead>'

# Build region → area hierarchy from scope_df (respects week scope)
region_info = {}
for region in all_regions:
    rdf_scope = scope_df[scope_df['Region'] == region]
    rdf_full = mw_df[mw_df['Region'] == region]
    if len(rdf_scope) == 0:
        continue
    # Trend sparkline always uses full multi-week data
    region_trend = rdf_full.groupby('Wk')['Total Score'].mean().sort_index().values
    # Scores come from scope_df (averaged or week-specific)
    region_avg = rdf_scope['Total Score'].mean()
    region_scores = {c: rdf_scope[c].mean() for c in SCORE_DIMENSIONS}
    region_info[region] = {
        'trend': region_trend, 'pulse': region_avg, **region_scores,
        'areas': {}
    }
    for area in sorted(rdf_scope['Area'].dropna().unique()):
        adf_scope = rdf_scope[rdf_scope['Area'] == area]
        adf_full = rdf_full[rdf_full['Area'] == area]
        if len(adf_scope) == 0:
            continue
        area_trend = adf_full.groupby('Wk')['Total Score'].mean().sort_index().values
        area_avg = adf_scope['Total Score'].mean()
        area_scores = {c: adf_scope[c].mean() for c in SCORE_DIMENSIONS}
        region_info[region]['areas'][area] = {
            'trend': area_trend, 'pulse': area_avg, **area_scores,
        }

total_trend = mw_df.groupby('Wk')['Total Score'].mean().sort_index().values
total_avg = scope_df['Total Score'].mean()
total_scores = {c: scope_df[c].mean() for c in SCORE_DIMENSIONS}

col_left, col_right = st.columns([2.2, 1])

# ── Region | Area Interactive Matrix ──
with col_left:
    st.markdown('<div class="section-title">Project Pulse \u2013 Region | Area</div>',
                unsafe_allow_html=True)

    # ── Region expand toggles (compact checkboxes) ──
    avail_regions = [r for r in all_regions if r in region_info]
    r_cols = st.columns(len(avail_regions))
    expanded_regions = set()
    for i, region in enumerate(avail_regions):
        with r_cols[i]:
            if st.checkbox(f'\u25B6 {region}', key=f'exp_r_{region}'):
                expanded_regions.add(region)

    # ── Area expand toggles (only for expanded regions) ──
    expandable_areas = []
    for r in expanded_regions:
        for a in region_info[r]['areas']:
            expandable_areas.append((r, a))

    expanded_areas = set()
    if expandable_areas:
        a_cols = st.columns(min(len(expandable_areas), 6))
        for i, (r, a) in enumerate(expandable_areas):
            col_idx = i % min(len(expandable_areas), 6)
            with a_cols[col_idx]:
                if st.checkbox(f'\u25B6 {a}', key=f'exp_a_{r}_{a}'):
                    expanded_areas.add(f'{r}/{a}')

    # ── Build ONE unified HTML table ──
    # Main table has 11 cols: Name | Trend | Pulse | 8 dims
    # Project sub-rows reuse same 11 cols: Project | PM Name | 8 dims | Total Score (mapped)
    unified_html = f'<div class="matrix-container"><table class="matrix-table">{_table_header}<tbody>'

    for region in avail_regions:
        info = region_info[region]
        is_expanded = region in expanded_regions

        # Region row (with indicator)
        indicator = '\u25BC' if is_expanded else '\u25B6'
        unified_html += _score_row_html(f'{indicator} {region}', info, 'region')

        if is_expanded:
            for area_name, area_data in info['areas'].items():
                area_key = f'{region}/{area_name}'
                area_expanded = area_key in expanded_areas
                a_indicator = '\u25BC' if area_expanded else '\u25B6'

                unified_html += _score_row_html(f'{a_indicator} {area_name}', area_data, 'area')

                if area_expanded:
                    # Project sub-header row (reuses 11 cols)
                    proj_cols_ordered = ['Project', 'PM Name'] + SCORE_DIMENSIONS + ['Total Score']
                    proj_avail = [c for c in proj_cols_ordered if c in scope_df.columns]

                    unified_html += '<tr style="background:#0c1a30; border-top:1px solid #1e3a5f;">'
                    for c in proj_avail:
                        align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
                        label = c.replace('PM Performance', 'PM Perf')
                        unified_html += (
                            f'<th style="{align} font-size:0.6rem; padding:0.25rem;'
                            f' color:#64748b; background:#0c1a30;">{label}</th>'
                        )
                    unified_html += '</tr>'

                    # Project data rows
                    proj_df = scope_df[
                        (scope_df['Region'] == region) &
                        (scope_df['Area'] == area_name)
                    ].sort_values('Total Score', ascending=True)

                    for _, r in proj_df.iterrows():
                        unified_html += '<tr style="background:rgba(255,255,255,0.02);">'
                        for c in proj_avail:
                            val = r[c]
                            if c == 'Total Score':
                                cls = pulse_css_class(val)
                                display = f'<span class="score-cell {cls}">{val:.1f}</span>' if pd.notna(val) else '\u2014'
                            elif c in SCORE_DIMENSIONS:
                                cls = f"rating-{int(round(val))}" if pd.notna(val) else ""
                                display = f'<span class="rating-badge {cls}">{val:.1f}</span>' if pd.notna(val) else '\u2014'
                            elif c in ('Project', 'PM Name'):
                                display = str(val)[:40] if pd.notna(val) else '\u2014'
                            else:
                                display = str(val) if pd.notna(val) else '\u2014'
                            align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
                            unified_html += f'<td style="{align} padding-left:2.5rem;">{display}</td>' if c == proj_avail[0] else f'<td style="{align}">{display}</td>'
                        unified_html += '</tr>'

    # Total row
    total_data = {'trend': total_trend, 'pulse': total_avg, **total_scores}
    unified_html += _score_row_html('Total', total_data, 'total')
    unified_html += '</tbody></table></div>'
    st.markdown(unified_html, unsafe_allow_html=True)

# ── Weekly Trend Heatmap (scrollable, all weeks) ──
with col_right:
    st.markdown('<div class="section-title">Project Pulse \u2013 Weekly Trend</div>',
                unsafe_allow_html=True)

    # All Year+Wk pairs sorted chronologically
    yw_pairs = (mw_df[['Year', 'Wk']].drop_duplicates()
                .sort_values(['Year', 'Wk'])
                .values.tolist())

    heat_data = []
    for region in all_regions:
        rdf = mw_df[mw_df['Region'] == region]
        row = {'Region': region}
        for yr, wk in yw_pairs:
            key = f"{int(yr) % 100}-W{int(wk):02d}"
            wdf = rdf[(rdf['Year'] == yr) & (rdf['Wk'] == wk)]
            row[key] = wdf['Total Score'].mean() if len(wdf) > 0 else None
        heat_data.append(row)

    total_heat = {'Region': 'Total'}
    for yr, wk in yw_pairs:
        key = f"{int(yr) % 100}-W{int(wk):02d}"
        wdf = mw_df[(mw_df['Year'] == yr) & (mw_df['Wk'] == wk)]
        total_heat[key] = wdf['Total Score'].mean() if len(wdf) > 0 else None
    heat_data.append(total_heat)

    region_icons = {
        'Central': '\U0001f7e6', 'NE': '\U0001f7e9',
        'South': '\U0001f7e8', 'West': '\U0001f7ea', 'Total': '\u2b1c',
    }

    heat_html = (
        '<div class="heatmap-scroll-container"><table class="heatmap-table">'
        '<thead><tr><th style="text-align:left;">Region</th>'
    )
    for yr, wk in yw_pairs:
        label = f"{int(yr) % 100}-W{int(wk):02d}"
        heat_html += f'<th style="white-space:nowrap;">{label}</th>'
    heat_html += '</tr></thead><tbody>'

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
                cls = heat_css_class(val)
                heat_html += f'<td><span class="heatmap-cell {cls}">{val:.1f}</span></td>'
            else:
                heat_html += '<td>\u2014</td>'
        heat_html += '</tr>'

    heat_html += '</tbody></table></div>'
    st.markdown(heat_html, unsafe_allow_html=True)

# ============================================================================
# RATINGS LEGEND
# ============================================================================
st.markdown('<div class="section-title" style="margin-top:0.8rem;">Ratings Reference</div>',
            unsafe_allow_html=True)

leg1, leg2, leg3, leg4, leg5, leg6 = st.columns([1.2, 1, 1, 1.2, 1, 1.5])

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

with leg6:
    st.markdown('''
    <div class="notes-box">
        <div class="notes-title">Notes</div>
        Pulse Ranking \u2013 Ranking of each contributor in Project Pulse per week.
        Click any band in the chart to drill down into that dimension.
    </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# DRILL-DOWN PANEL (shown when a dimension band is clicked)
# ============================================================================
if st.session_state.get('selected_drill'):
    drill = st.session_state.selected_drill

    if drill['type'] == 'dimension':
        dim = drill['value']
        wk = drill.get('week')
        dim_color = DIMENSION_COLORS.get(dim, '#2563eb')

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

        # Filter data for drill-down (wk is now "YYYY-WNN" string)
        drill_df = mw_df.copy()
        if wk and 'Year_Week' in drill_df.columns:
            drill_df = drill_df[drill_df['Year_Week'] == wk]

        # Charts: bar by region + score distribution
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown(f"**{dim} \u2013 By Region**")
            fig_bar = chart_dimension_by_region(drill_df, dim, dim_color)
            st.plotly_chart(fig_bar, use_container_width=True)
        with dc2:
            st.markdown(f"**{dim} \u2013 Score Distribution**")
            fig_dist = chart_dimension_distribution(drill_df, dim)
            st.plotly_chart(fig_dist, use_container_width=True)

        # Project detail table
        st.markdown(f"**Project Details \u2013 {dim}**")
        detail_cols = ['Wk', 'Region', 'Area', 'Project', 'PM Name', dim, 'Total Score']
        available = [c for c in detail_cols if c in drill_df.columns]
        detail = drill_df[available].sort_values(
            [dim, 'Total Score'], ascending=[True, True]
        ).head(50)

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
                    cls = f"rating-{int(val)}" if pd.notna(val) else ""
                    display = f'<span class="rating-badge {cls}">{int(val)}</span>' if pd.notna(val) else '\u2014'
                elif c == 'Total Score':
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

        # Weekly trend by region
        st.markdown(f"**{dim} \u2013 Weekly Trend by Region**")
        fig_trend = chart_dimension_trend_by_region(mw_df, dim)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Close drill-down button
        if st.button("\u2715 Close Drill-Down", key="close_drill"):
            st.session_state.selected_drill = None
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown('''
<div style="text-align:center; padding:1rem; color:#334155; font-size:0.65rem;
            margin-top:1rem; border-top:1px solid #1e293b;">
    CSE Unit &bull; Pulse Tracker &bull; Powered by Streamlit + Plotly
</div>
''', unsafe_allow_html=True)
