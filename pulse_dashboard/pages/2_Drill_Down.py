"""
Pulse Dashboard - Drill-Down Visualizations

Multi-level drill-down: Portfolio → Region → Area → Project
With interactive charts and detailed analysis panels at every level.
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

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

df = st.session_state.df
target = st.session_state.get('pulse_target', 17.0)
selected_regions = st.session_state.get('selected_regions', [])

st.markdown('<p class="main-header">Drill-Down Analysis</p>', unsafe_allow_html=True)

# ============================================================================
# WEEK SCOPE SELECTOR — choose any week or overall average
# ============================================================================
# Build multi-week dataset (region-filtered, NOT week-filtered)
mw_df = df.copy()
if selected_regions:
    mw_df = mw_df[mw_df['Region'].isin(selected_regions)]

# Available weeks sorted chronologically
yw_pairs = (
    mw_df[['Year', 'Wk']].drop_duplicates()
    .sort_values(['Year', 'Wk'])
    .values.tolist()
)
week_labels = [f"{int(y)}-W{int(w):02d}" for y, w in yw_pairs]
week_options = ['All Weeks (Average)'] + week_labels

# Default to sidebar-selected week
sidebar_year = st.session_state.get('selected_year')
sidebar_week = st.session_state.get('selected_week')
sidebar_label = f"{int(sidebar_year)}-W{int(sidebar_week):02d}" if sidebar_year and sidebar_week else None
default_idx = (week_labels.index(sidebar_label) + 1) if sidebar_label in week_labels else 0

scope_col1, scope_col2 = st.columns([1, 3])
with scope_col1:
    week_scope = st.selectbox(
        'Analysis Scope',
        week_options,
        index=default_idx,
        key='dd_week_scope',
    )

# Build the analysis DataFrame based on scope
if week_scope == 'All Weeks (Average)':
    # Average all dimension scores and Total Score per project (latest region/area/PM)
    agg_cols = {dim: 'mean' for dim in SCORE_DIMENSIONS}
    agg_cols['Total Score'] = 'mean'
    # Keep latest metadata per project
    latest_idx = mw_df.sort_values(['Year', 'Wk']).groupby('Project').tail(1).index
    meta_df = mw_df.loc[latest_idx, ['Project', 'Region', 'Area', 'PM Name']].drop_duplicates('Project')

    avg_scores = mw_df.groupby('Project').agg(agg_cols).reset_index()
    # Round scores to nearest int for status mapping
    for dim in SCORE_DIMENSIONS:
        avg_scores[dim] = avg_scores[dim].round(1)
    avg_scores['Total Score'] = avg_scores['Total Score'].round(1)

    analysis_df = meta_df.merge(avg_scores, on='Project', how='inner')
    analysis_df['Pulse_Status'] = analysis_df['Total Score'].apply(get_pulse_status)
    analysis_df['Pulse_Color'] = analysis_df['Total Score'].apply(get_pulse_color)

    with scope_col2:
        st.caption(f"Showing average across {len(yw_pairs)} weeks | {len(analysis_df)} projects")
else:
    # Specific week selected
    yr_str, wk_str = week_scope.split('-W')
    sel_yr, sel_wk = int(yr_str), int(wk_str)
    analysis_df = mw_df[(mw_df['Year'] == sel_yr) & (mw_df['Wk'] == sel_wk)].copy()

    with scope_col2:
        st.caption(f"Week {week_scope} | {len(analysis_df)} projects")

if analysis_df.empty:
    st.warning("No data for the selected scope.")
    st.stop()

# Use analysis_df instead of filtered_df for all drill-down content
filtered_df = analysis_df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _dimension_heatmap_html(data_df, entity_col, entity_label='Entity'):
    """HTML heatmap table: rows = entities, columns = dimensions."""
    entities = sorted(data_df[entity_col].dropna().unique())

    html = (
        '<div class="matrix-container"><table class="matrix-table">'
        '<thead><tr>'
        f'<th style="text-align:left; min-width:120px;">{entity_label}</th>'
        '<th>Pulse</th>'
        '<th>N</th>'
    )
    for dim in SCORE_DIMENSIONS:
        short = dim.replace('PM Performance', 'PM Perf')
        html += f'<th>{short}</th>'
    html += '</tr></thead><tbody>'

    for entity in entities:
        edf = data_df[data_df[entity_col] == entity]
        avg_pulse = edf['Total Score'].mean()
        n_projects = edf['Project'].nunique()
        p_cls = pulse_css_class(avg_pulse)

        html += '<tr>'
        html += f'<td style="text-align:left; font-weight:600; color:#e2e8f0;">{entity}</td>'
        html += f'<td><span class="score-cell {p_cls}">{avg_pulse:.1f}</span></td>'
        html += f'<td style="color:#94a3b8;">{n_projects}</td>'

        for dim in SCORE_DIMENSIONS:
            val = edf[dim].mean()
            cls = score_css_class(val)
            html += f'<td><span class="score-cell {cls}">{val:.2f}</span></td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html


def _text_summary_panel(data_df, max_items=8):
    """Show pain points, comments, and resolution plans."""
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
            texts = data_df[col].dropna()
            texts = texts[texts.astype(str).str.strip() != '']
            if not texts.empty:
                found_any = True
                with st.expander(f"{icon} {col} ({len(texts)})", expanded=(col == 'Pain Points')):
                    for text in texts.head(max_items):
                        st.markdown(f"- {str(text)[:300]}")

    if not found_any:
        st.info("No text data available for this selection.")


def _comparison_bar_chart(data_df, group_col, title='Comparison'):
    """Horizontal bar chart: avg pulse by group, color by status."""
    agg = (
        data_df.groupby(group_col)
        .agg(AvgPulse=('Total Score', 'mean'), Count=('Project', 'nunique'))
        .reset_index()
        .sort_values('AvgPulse', ascending=True)
    )

    colors = [get_pulse_color(v) for v in agg['AvgPulse']]

    fig = go.Figure(go.Bar(
        y=agg[group_col],
        x=agg['AvgPulse'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f} ({c} proj)" for v, c in zip(agg['AvgPulse'], agg['Count'])],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=10),
    ))
    _apply_theme(fig)
    fig.update_layout(
        title=title,
        height=max(200, len(agg) * 35 + 80),
        margin=dict(l=120, r=80, t=40, b=30),
    )
    fig.update_xaxes(range=[0, 24], title_text='Avg Pulse Score')
    fig.add_vline(x=target, line_dash='dash', line_color='white', line_width=1,
                  annotation_text=f'Target ({target:.0f})',
                  annotation_font_color='#94a3b8')
    return fig


def _project_table_html(data_df, max_rows=30):
    """Styled project detail table with dimension scores."""
    cols = ['Project', 'PM Name'] + SCORE_DIMENSIONS + ['Total Score']
    available = [c for c in cols if c in data_df.columns]
    sorted_df = data_df[available].sort_values('Total Score', ascending=True).head(max_rows)

    html = (
        '<div class="matrix-container" style="max-height:400px; overflow-y:auto;">'
        '<table class="matrix-table"><thead><tr>'
    )
    for c in available:
        align = 'text-align:left;' if c in ('Project', 'PM Name') else ''
        label = c.replace('PM Performance', 'PM Perf')
        html += f'<th style="{align}">{label}</th>'
    html += '</tr></thead><tbody>'

    for _, r in sorted_df.iterrows():
        html += '<tr>'
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
            html += f'<td style="{align}">{display}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html


def _kpi_row(data_df):
    """Render a row of 4 KPI cards for the given data."""
    avg_pulse = data_df['Total Score'].mean()
    n_projects = data_df['Project'].nunique()
    green_count = len(data_df[data_df['Total Score'] >= 16])
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
        st.markdown(f"""
        <div class="kpi-container {'critical' if red_count > 0 else ''}">
            <p class="kpi-value {'red' if red_count > 0 else ''}">{red_count}</p>
            <p class="kpi-label">Red Projects</p>
        </div>
        """, unsafe_allow_html=True)


def _click_project_detail(click_df, label, drill_df, full_df, target_score):
    """Full project detail panel: KPIs, radar, trend, dimensions, text."""
    row = click_df.iloc[0]
    project_name = row.get('Project', label)
    status = get_pulse_status(row['Total Score'])
    color = STATUS_CONFIG[status]['color']

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

    # Dimension score cards
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

    # Radar + Trend side by side
    c1, c2 = st.columns(2)
    with c1:
        fig_radar = chart_radar(row)
        st.plotly_chart(fig_radar, use_container_width=True)
    with c2:
        proj_history = full_df[full_df['Project'] == project_name]
        if len(proj_history) > 1:
            fig_trend = chart_project_trend(full_df, project_name, target_score)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Only one week of data \u2014 no trend available.")

    # Dimension comparison vs region & portfolio averages
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

    # All text fields
    _text_summary_panel(click_df)


def _click_detail_panel(click_df, label):
    """Show inline detail when a chart element is clicked."""
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
        # Single project → radar chart + score comparison
        row = click_df.iloc[0]
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_radar = chart_radar(row)
            st.plotly_chart(fig_radar, use_container_width=True)
        with c2:
            comp_html = '<div class="matrix-container"><table class="matrix-table"><thead><tr>'
            comp_html += '<th style="text-align:left;">Dim</th><th>Score</th><th>Avg</th><th>Gap</th>'
            comp_html += '</tr></thead><tbody>'
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
        # Multiple projects → table + text
        st.markdown(_project_table_html(click_df, max_rows=10), unsafe_allow_html=True)
        _text_summary_panel(click_df, max_items=5)


# ============================================================================
# DRILL-DOWN NAVIGATION (cascading: Region → Area → Project)
# ============================================================================

nav_cols = st.columns([1, 1, 1.5])

with nav_cols[0]:
    regions = ['All Regions'] + sorted(filtered_df['Region'].dropna().unique().tolist())
    selected_region = st.selectbox('Region', regions, key='dd_region')

with nav_cols[1]:
    if selected_region != 'All Regions':
        areas_list = ['All Areas'] + sorted(
            filtered_df[filtered_df['Region'] == selected_region]['Area'].dropna().unique().tolist()
        )
    else:
        areas_list = ['All Areas'] + sorted(filtered_df['Area'].dropna().unique().tolist())
    selected_area = st.selectbox('Area', areas_list, key='dd_area')

with nav_cols[2]:
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

# Build drill-filtered data
drill_df = filtered_df.copy()
if selected_region != 'All Regions':
    drill_df = drill_df[drill_df['Region'] == selected_region]
if selected_area != 'All Areas':
    drill_df = drill_df[drill_df['Area'] == selected_area]
if selected_project != 'All Projects':
    drill_df = drill_df[drill_df['Project'] == selected_project]

# Breadcrumb
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
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Sunburst", "Treemap", "Sankey", "Icicle"])

with tab1:
    fig = chart_sunburst(drill_df)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="sunburst")

    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        label = pt.get('label', '') if isinstance(pt, dict) else getattr(pt, 'label', '')
        if label:
            # Check if clicked label is a dimension name
            if label in SCORE_DIMENSIONS:
                # Dimension leaf clicked — show dimension analysis
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
                    fig_dist = chart_dimension_distribution(drill_df, label)
                    st.plotly_chart(fig_dist, use_container_width=True)
                with dc2:
                    fig_bar = chart_dimension_by_region(drill_df, label, dim_color)
                    st.plotly_chart(fig_bar, use_container_width=True)
                # Show low scorers
                low = drill_df[drill_df[label] <= 1]
                if not low.empty:
                    st.markdown(f"**Projects scoring 0-1 on {label}** ({len(low)})")
                    show_cols = ['Project', 'Region', 'Area', 'PM Name', label, 'Total Score']
                    avail = [c for c in show_cols if c in low.columns]
                    st.dataframe(low[avail].sort_values(label), use_container_width=True, hide_index=True)
            else:
                # Region / Area / Project clicked
                click_df = drill_df[
                    (drill_df['Region'] == label) |
                    (drill_df['Area'] == label) |
                    (drill_df['Project'] == label)
                ]
                if not click_df.empty:
                    if click_df['Project'].nunique() == 1:
                        # Single project — show full project detail with dimensions
                        _click_project_detail(click_df, label, drill_df, df, target)
                    else:
                        _click_detail_panel(click_df, label)

with tab2:
    fig = chart_treemap(drill_df)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="treemap")

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
    fig = chart_sankey(drill_df)
    st.plotly_chart(fig, use_container_width=True, key="sankey")
    st.caption("Flow: Scoring Dimensions \u2192 Score Level \u2192 Pulse Status")

    # Dimension breakdown below Sankey
    st.markdown("**Dimension Deep Dive**")
    dim_selected = st.selectbox(
        'Select dimension to analyze',
        SCORE_DIMENSIONS,
        key='sankey_dim',
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_dist = chart_dimension_distribution(drill_df, dim_selected)
        st.plotly_chart(fig_dist, use_container_width=True)
    with sc2:
        dim_color = DIMENSION_COLORS.get(dim_selected, '#2563eb')
        fig_bar = chart_dimension_by_region(drill_df, dim_selected, dim_color)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Projects scoring 0-1 on this dimension
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
# ============================================================================
st.markdown("---")
st.markdown('<div class="section-title">Detail Panel</div>', unsafe_allow_html=True)

if selected_project != 'All Projects':
    # ── PROJECT LEVEL (deepest drill) ──
    proj_row = drill_df.iloc[0] if len(drill_df) > 0 else None
    if proj_row is not None:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            status = get_pulse_status(proj_row['Total Score'])
            color = STATUS_CONFIG[status]['color']
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

        # Radar + Trend side by side
        c1, c2 = st.columns(2)
        with c1:
            fig_radar = chart_radar(proj_row)
            st.plotly_chart(fig_radar, use_container_width=True)
        with c2:
            proj_history = df[df['Project'] == selected_project]
            if len(proj_history) > 1:
                fig_trend = chart_project_trend(df, selected_project, target)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Only one week of data \u2014 no trend available.")

        # Dimension scores vs region and portfolio averages
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

        # All text fields
        st.markdown("**Notes & Issues**")
        _text_summary_panel(drill_df)

elif selected_area != 'All Areas':
    # ── AREA LEVEL ──
    _kpi_row(drill_df)

    # Project comparison chart
    fig_comp = _comparison_bar_chart(drill_df, 'Project', f'Projects in {selected_area}')
    st.plotly_chart(fig_comp, use_container_width=True)

    # Dimension heatmap: Project x Dimension
    st.markdown("**Project \u00d7 Dimension Scores**")
    st.markdown(_dimension_heatmap_html(drill_df, 'Project', 'Project'),
                unsafe_allow_html=True)

    # Weakest dimensions
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

    # Text summary
    st.markdown("**Notes & Issues**")
    _text_summary_panel(drill_df)

elif selected_region != 'All Regions':
    # ── REGION LEVEL ──
    _kpi_row(drill_df)

    # Area comparison chart
    fig_comp = _comparison_bar_chart(drill_df, 'Area', f'Areas in {selected_region}')
    st.plotly_chart(fig_comp, use_container_width=True)

    # Dimension heatmap: Area x Dimension
    st.markdown("**Area \u00d7 Dimension Scores**")
    st.markdown(_dimension_heatmap_html(drill_df, 'Area', 'Area'),
                unsafe_allow_html=True)

    # Bottom projects in this region
    st.markdown("**Bottom Projects (by Total Score)**")
    st.markdown(_project_table_html(drill_df, max_rows=15), unsafe_allow_html=True)

    # Text summary
    st.markdown("**Notes & Issues**")
    _text_summary_panel(drill_df)

else:
    # ── PORTFOLIO LEVEL ──
    _kpi_row(drill_df)

    # Region comparison chart
    fig_comp = _comparison_bar_chart(drill_df, 'Region', 'Average Pulse by Region')
    st.plotly_chart(fig_comp, use_container_width=True)

    # Dimension heatmap: Region x Dimension
    st.markdown("**Region \u00d7 Dimension Scores**")
    st.markdown(_dimension_heatmap_html(drill_df, 'Region', 'Region'),
                unsafe_allow_html=True)

    # Status distribution
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

    # Bottom 10 projects
    st.markdown("**Bottom 10 Projects**")
    st.markdown(_project_table_html(drill_df, max_rows=10), unsafe_allow_html=True)
