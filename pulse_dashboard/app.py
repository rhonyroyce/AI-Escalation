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
    get_pulse_status, MCKINSEY_COLORS, DIMENSION_COLORS, REGION_LINE_COLORS,
    pulse_css_class, heat_css_class, score_css_class,
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
    st.markdown('<p class="main-header">Pulse Dashboard</p>', unsafe_allow_html=True)
    st.markdown("#### Project Portfolio Intelligence")
    st.markdown("---")

    # Try default file first
    default_path = get_default_file_path()

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Pulse Tracker (.xlsx)",
            type=['xlsx', 'xls'],
            help="Excel file must contain a 'Project Pulse' sheet",
        )

    with col2:
        if default_path:
            st.info(f"Default file found:\n`{default_path.name}`")
            use_default = st.button("Load Default File", type="primary")
        else:
            use_default = False
            st.warning("No ProjectPulse.xlsx found in project root")

    # Load data
    file_to_load = None
    if uploaded_file:
        file_to_load = uploaded_file
    elif use_default and default_path:
        file_to_load = str(default_path)

    if file_to_load:
        try:
            with st.spinner("Loading and cleaning data..."):
                df = load_pulse_data(file_to_load)
                st.session_state.df = df
                st.rerun()
        except ValueError as e:
            st.error(f"Data loading error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

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

latest_week = st.session_state.get('selected_week', df['Wk'].max())
latest_year = st.session_state.get('selected_year', df['Year'].max())
avg_pulse = filtered_df['Total Score'].mean()
target = st.session_state.get('pulse_target', 17.0)
variance = avg_pulse - target

# ── Header ──
st.markdown(f"""
<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
    <div style="font-size:2rem;">\U0001f3d7\ufe0f</div>
    <div>
        <div style="font-size:1.3rem; font-weight:700; color:#60a5fa;">CSE UNIT</div>
        <div style="font-size:0.65rem; color:#64748b; letter-spacing:1px;">PROJECT PULSE TRACKER</div>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <div style="font-size:0.8rem; color:#94a3b8;">
            Year {latest_year} | Week {latest_week} |
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

# Build region → area hierarchy
region_info = {}
for region in all_regions:
    rdf = mw_df[mw_df['Region'] == region]
    if len(rdf) == 0:
        continue
    region_trend = rdf.groupby('Wk')['Total Score'].mean().sort_index().values
    region_avg = rdf['Total Score'].mean()
    region_scores = {c: rdf[c].mean() for c in SCORE_DIMENSIONS}
    region_info[region] = {
        'trend': region_trend, 'pulse': region_avg, **region_scores,
        'areas': {}
    }
    for area in sorted(rdf['Area'].dropna().unique()):
        adf = rdf[rdf['Area'] == area]
        if len(adf) == 0:
            continue
        area_trend = adf.groupby('Wk')['Total Score'].mean().sort_index().values
        area_avg = adf['Total Score'].mean()
        area_scores = {c: adf[c].mean() for c in SCORE_DIMENSIONS}
        region_info[region]['areas'][area] = {
            'trend': area_trend, 'pulse': area_avg, **area_scores,
        }

total_trend = mw_df.groupby('Wk')['Total Score'].mean().sort_index().values
total_avg = mw_df['Total Score'].mean()
total_scores = {c: mw_df[c].mean() for c in SCORE_DIMENSIONS}

col_left, col_right = st.columns([2.2, 1])

# ── Region | Area Matrix (collapsible drill-down) ──
with col_left:
    st.markdown('<div class="section-title">Project Pulse \u2013 Region | Area</div>',
                unsafe_allow_html=True)

    # Summary: 4 regions + Total (always visible)
    summary_html = f'<div class="matrix-container"><table class="matrix-table">{_table_header}<tbody>'
    for region in all_regions:
        if region in region_info:
            summary_html += _score_row_html(region, region_info[region], 'region')
    summary_html += _score_row_html(
        'Total', {'trend': total_trend, 'pulse': total_avg, **total_scores}, 'total'
    )
    summary_html += '</tbody></table></div>'
    st.markdown(summary_html, unsafe_allow_html=True)

    # Expandable region drill-down
    for region in all_regions:
        if region not in region_info:
            continue
        info = region_info[region]
        pulse_val = info['pulse']
        status = get_pulse_status(pulse_val)
        color = STATUS_CONFIG[status]['color']

        with st.expander(f"\U0001f50d {region} \u2014 Avg Pulse: {pulse_val:.1f}"):
            # Area breakdown table
            area_html = f'<div class="matrix-container"><table class="matrix-table">{_table_header}<tbody>'
            for area_name, area_data in info['areas'].items():
                area_html += _score_row_html(area_name, area_data, 'area')
            area_html += '</tbody></table></div>'
            st.markdown(area_html, unsafe_allow_html=True)

            # Area-level project drill-down
            for area_name in info['areas']:
                with st.expander(f"\U0001f4cb {area_name} \u2014 Projects"):
                    proj_df = filtered_df[
                        (filtered_df['Region'] == region) &
                        (filtered_df['Area'] == area_name)
                    ]
                    if proj_df.empty:
                        st.info("No projects for the selected week.")
                    else:
                        proj_cols = ['Project', 'PM Name'] + SCORE_DIMENSIONS + ['Total Score']
                        available = [c for c in proj_cols if c in proj_df.columns]
                        proj_display = proj_df[available].sort_values('Total Score', ascending=True)

                        proj_html = (
                            '<div class="matrix-container" style="max-height:300px; overflow-y:auto;">'
                            '<table class="matrix-table"><thead><tr>'
                        )
                        for c in available:
                            align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
                            label = c.replace('PM Performance', 'PM Perf')
                            proj_html += f'<th style="{align}">{label}</th>'
                        proj_html += '</tr></thead><tbody>'

                        for _, r in proj_display.iterrows():
                            proj_html += '<tr>'
                            for c in available:
                                val = r[c]
                                if c == 'Total Score':
                                    cls = pulse_css_class(val)
                                    display = f'<span class="score-cell {cls}">{int(val)}</span>' if pd.notna(val) else '\u2014'
                                elif c in SCORE_DIMENSIONS:
                                    cls = f"rating-{int(val)}" if pd.notna(val) else ""
                                    display = f'<span class="rating-badge {cls}">{int(val)}</span>' if pd.notna(val) else '\u2014'
                                elif c in ('Project', 'PM Name'):
                                    display = str(val)[:50] if pd.notna(val) else '\u2014'
                                else:
                                    display = str(val) if pd.notna(val) else '\u2014'
                                align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
                                proj_html += f'<td style="{align}">{display}</td>'
                            proj_html += '</tr>'

                        proj_html += '</tbody></table></div>'
                        st.markdown(proj_html, unsafe_allow_html=True)

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
