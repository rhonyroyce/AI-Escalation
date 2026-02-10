"""
Pulse Dashboard - McKinsey-Grade Chart Library

All Plotly chart functions for the dashboard. Each returns a go.Figure.
Consistent dark theme via get_plotly_theme().
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.styles import (
    get_plotly_theme, get_pulse_status, get_pulse_color,
    SCORE_DIMENSIONS, STATUS_CONFIG, STATUS_ORDER,
    CONTINUOUS_COLOR_SCALE, COLOR_MIDPOINT, DISCRETE_COLOR_MAP,
    MCKINSEY_COLORS, AXIS_STYLE, DIMENSION_COLORS, REGION_LINE_COLORS,
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme + default axis styling to a figure."""
    fig.update_layout(**get_plotly_theme())
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ============================================================================
# KPI CARDS (returns HTML string)
# ============================================================================
def kpi_card_html(label: str, value: str, delta: str = "", css_class: str = "",
                  value_class: str = "") -> str:
    delta_html = ""
    if delta:
        is_positive = not delta.startswith('-')
        delta_class = "delta-positive" if is_positive else "delta-negative"
        delta_html = f'<p class="kpi-delta {delta_class}">{delta}</p>'

    return f"""
    <div class="kpi-container {css_class}">
        <p class="kpi-value {value_class}">{value}</p>
        <p class="kpi-label">{label}</p>
        {delta_html}
    </div>
    """


# ============================================================================
# VARIANCE BULLET CHART
# ============================================================================
def chart_variance_bullet(avg_score: float, target: float, stretch: float) -> go.Figure:
    """Bullet chart: actual avg pulse vs target and stretch markers."""
    fig = go.Figure()

    # Background ranges (poor → good)
    ranges = [14, 16, 20, 24]
    colors = ['rgba(239,68,68,0.2)', 'rgba(245,158,11,0.2)',
              'rgba(34,197,94,0.2)', 'rgba(5,150,105,0.2)']
    prev = 0
    for r, c in zip(ranges, colors):
        fig.add_trace(go.Bar(
            x=[r - prev], y=['Pulse'], orientation='h',
            base=prev, marker=dict(color=c),
            showlegend=False, hoverinfo='skip',
        ))
        prev = r

    # Actual value bar
    fig.add_trace(go.Bar(
        x=[avg_score], y=['Pulse'], orientation='h',
        marker=dict(color='#00BFFF', line=dict(width=0)),
        width=0.3, name='Actual',
        text=f'{avg_score:.1f}', textposition='outside',
        textfont=dict(color='#E0E0E0', size=14, family='Inter'),
    ))

    # Target marker
    fig.add_trace(go.Scatter(
        x=[target, target], y=[-0.4, 0.4],
        mode='lines', line=dict(color='white', width=3),
        name=f'Target ({target:.0f})',
    ))

    # Stretch marker
    fig.add_trace(go.Scatter(
        x=[stretch, stretch], y=[-0.3, 0.3],
        mode='lines', line=dict(color='#f59e0b', width=2, dash='dash'),
        name=f'Stretch ({stretch:.0f})',
    ))

    _apply_theme(fig)
    fig.update_layout(
        title='Portfolio Pulse vs Target',
        height=200,
        barmode='overlay',
        legend=dict(orientation='h', y=-0.3),
    )
    fig.update_xaxes(range=[0, 24], title_text='Score')
    fig.update_yaxes(visible=False)
    return fig


# ============================================================================
# TREND + FORECAST
# ============================================================================
def chart_trend_forecast(df: pd.DataFrame, target: float, n_forecast: int = 4) -> go.Figure:
    """Time series with target line and linear forecast."""
    # Aggregate by Year_Week
    weekly = (
        df.groupby(['Year', 'Wk', 'Year_Week'])
        .agg(avg_score=('Total Score', 'mean'), count=('Project', 'count'))
        .reset_index()
        .sort_values(['Year', 'Wk'])
    )

    fig = go.Figure()

    # Actual line
    fig.add_trace(go.Scatter(
        x=weekly['Year_Week'], y=weekly['avg_score'],
        mode='lines+markers', name='Actual',
        line=dict(color='#00BFFF', width=3),
        marker=dict(size=6),
    ))

    # Target line
    fig.add_hline(y=target, line_dash='dash', line_color='white',
                  annotation_text=f'Target ({target:.0f})',
                  annotation_font_color='white')

    # Forecast (linear regression on last 8 data points)
    if len(weekly) >= 4:
        n_points = min(8, len(weekly))
        recent = weekly.tail(n_points)
        x_num = np.arange(len(recent))
        y_vals = recent['avg_score'].values

        slope, intercept = np.polyfit(x_num, y_vals, 1)

        # Generate forecast points
        last_year = recent['Year'].iloc[-1]
        last_wk = recent['Wk'].iloc[-1]
        forecast_labels = []
        for i in range(1, n_forecast + 1):
            fw = last_wk + i
            fy = last_year
            if fw > 52:
                fw -= 52
                fy += 1
            forecast_labels.append(f"{fy}-W{fw:02d}")

        forecast_x = np.arange(len(recent), len(recent) + n_forecast)
        forecast_y = slope * forecast_x + intercept

        fig.add_trace(go.Scatter(
            x=forecast_labels, y=forecast_y,
            mode='lines+markers', name='Forecast',
            line=dict(color='#f59e0b', width=2, dash='dash'),
            marker=dict(size=5, symbol='diamond'),
        ))

    _apply_theme(fig)
    fig.update_layout(
        title='Pulse Score Trend & Forecast',
        xaxis_title='Week',
        yaxis_title='Avg Pulse Score',
        height=400,
    )
    fig.update_yaxes(range=[10, 24])
    return fig


# ============================================================================
# WATERFALL (Score Decomposition)
# ============================================================================
def chart_waterfall_decomposition(df: pd.DataFrame) -> go.Figure:
    """Waterfall: contribution of each dimension to the average Total Score."""
    means = df[SCORE_DIMENSIONS].mean()

    fig = go.Figure(go.Waterfall(
        x=list(means.index),
        y=list(means.values),
        measure=['relative'] * len(means),
        text=[f'{v:.2f}' for v in means.values],
        textposition='outside',
        textfont=dict(color='#E0E0E0'),
        connector=dict(line=dict(color='#334155', width=1)),
        increasing=dict(marker=dict(color='#22c55e')),
        decreasing=dict(marker=dict(color='#ef4444')),
        totals=dict(marker=dict(color='#0066CC')),
    ))

    # Add total bar
    fig.add_trace(go.Bar(
        x=['Total'], y=[means.sum()],
        marker=dict(color='#0066CC'),
        text=f'{means.sum():.1f}',
        textposition='outside',
        textfont=dict(color='#E0E0E0'),
        showlegend=False,
    ))

    _apply_theme(fig)
    fig.update_layout(
        title='Score Decomposition by Dimension',
        yaxis_title='Score Contribution',
        showlegend=False,
        height=400,
    )
    fig.update_yaxes(range=[0, max(means.sum() + 2, 24)])
    return fig


# ============================================================================
# PARETO CHART
# ============================================================================
def chart_pareto(df: pd.DataFrame, target: float, groupby_col: str = 'Area') -> go.Figure:
    """Pareto: count of below-target projects by group + cumulative line."""
    below_target = df[df['Total Score'] < target]
    if below_target.empty:
        fig = go.Figure()
        fig.update_layout(**get_plotly_theme(), title='No Below-Target Projects')
        return fig

    issues = below_target.groupby(groupby_col).size().sort_values(ascending=False)
    cumulative_pct = issues.cumsum() / issues.sum() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=issues.index, y=issues.values, name='Count',
               marker=dict(color='#ef4444')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=issues.index, y=cumulative_pct.values,
                   mode='lines+markers', name='Cumulative %',
                   line=dict(color='#f59e0b', width=2),
                   marker=dict(size=6)),
        secondary_y=True,
    )
    fig.add_hline(y=80, secondary_y=True, line_dash='dash',
                  line_color='#94a3b8', annotation_text='80%',
                  annotation_font_color='#94a3b8')

    fig.update_layout(
        **get_plotly_theme(),
        title=f'Pareto: Below-Target Projects by {groupby_col}',
        height=400,
    )
    fig.update_yaxes(title_text='Count', secondary_y=False, gridcolor='#1e293b')
    fig.update_yaxes(title_text='Cumulative %', secondary_y=True, range=[0, 105], gridcolor='#1e293b')
    return fig


# ============================================================================
# SUNBURST
# ============================================================================
def _dim_color(score):
    """Color for a single dimension score (0-3)."""
    if score >= 3:
        return '#059669'   # dark green
    elif score >= 2:
        return '#22c55e'   # green
    elif score >= 1:
        return '#f59e0b'   # yellow
    else:
        return '#ef4444'   # red


def chart_sunburst(df: pd.DataFrame) -> go.Figure:
    """Sunburst: Region -> Area -> Project -> Dimension (4 levels).

    Colors use two scales:
    - Dimensions (leaves): 0=red, 1=yellow, 2=green, 3=dark green
    - Projects/Areas/Regions: Total Score rules (Red <14, Yellow <16, Green <20, DG 20-24)
    """
    ids, labels, parents, values, colors, hovertexts = [], [], [], [], [], []

    # Root
    portfolio_avg = df['Total Score'].mean()
    ids.append('Portfolio')
    labels.append('Portfolio')
    parents.append('')
    values.append(0)
    colors.append(get_pulse_color(portfolio_avg))
    hovertexts.append(f'Avg Pulse: {portfolio_avg:.1f}')

    for region in sorted(df['Region'].dropna().unique()):
        rdf = df[df['Region'] == region]
        r_avg = rdf['Total Score'].mean()
        r_id = region

        ids.append(r_id)
        labels.append(region)
        parents.append('Portfolio')
        values.append(0)
        colors.append(get_pulse_color(r_avg))
        hovertexts.append(f'Avg Pulse: {r_avg:.1f} ({rdf["Project"].nunique()} proj)')

        for area in sorted(rdf['Area'].dropna().unique()):
            adf = rdf[rdf['Area'] == area]
            a_avg = adf['Total Score'].mean()
            a_id = f'{region}/{area}'

            ids.append(a_id)
            labels.append(area)
            parents.append(r_id)
            values.append(0)
            colors.append(get_pulse_color(a_avg))
            hovertexts.append(f'Avg Pulse: {a_avg:.1f} ({adf["Project"].nunique()} proj)')

            for _, row in adf.iterrows():
                project = row['Project']
                total = row['Total Score']
                p_id = f'{region}/{area}/{project}'

                ids.append(p_id)
                labels.append(project)
                parents.append(a_id)
                values.append(0)
                colors.append(get_pulse_color(total))
                hovertexts.append(f'Total: {int(total)} | PM: {row.get("PM Name", "—")}')

                for dim in SCORE_DIMENSIONS:
                    score = int(row.get(dim, 0))
                    d_id = f'{p_id}/{dim}'
                    short = dim.replace('PM Performance', 'PM Perf')

                    ids.append(d_id)
                    labels.append(short)
                    parents.append(p_id)
                    values.append(1)
                    colors.append(_dim_color(score))
                    hovertexts.append(f'{dim}: {score}/3')

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        hovertext=hovertexts,
        hoverinfo='label+text',
        branchvalues='remainder',
        maxdepth=3,
    ))
    fig.update_layout(**get_plotly_theme(), title='Portfolio Hierarchy', height=650)
    return fig


# ============================================================================
# TREEMAP
# ============================================================================
def chart_treemap(df: pd.DataFrame) -> go.Figure:
    agg = (
        df.groupby(['Region', 'Area'])
        .agg(Projects=('Project', 'nunique'), AvgPulse=('Total Score', 'mean'))
        .reset_index()
    )
    fig = px.treemap(
        agg,
        path=['Region', 'Area'],
        values='Projects',
        color='AvgPulse',
        color_continuous_scale=CONTINUOUS_COLOR_SCALE,
        color_continuous_midpoint=COLOR_MIDPOINT,
    )
    fig.update_layout(**get_plotly_theme(), title='Region / Area Treemap', height=600)
    return fig


# ============================================================================
# SANKEY (Dimension → Score Level → Pulse Status)
# ============================================================================
def chart_sankey(df: pd.DataFrame) -> go.Figure:
    """Sankey: Dimension → Score Level (Critical/At Risk/Healthy) → Pulse Status."""
    score_levels = ['Critical (0-1)', 'At Risk (2)', 'Healthy (3)']
    statuses = STATUS_ORDER  # Red, Yellow, Green, Dark Green

    # Build node labels
    nodes = list(SCORE_DIMENSIONS) + score_levels + statuses
    node_colors = (
        ['#0077B6'] * len(SCORE_DIMENSIONS) +
        ['#ef4444', '#f59e0b', '#22c55e'] +
        [STATUS_CONFIG[s]['color'] for s in statuses]
    )

    # Dimension → Score Level links
    source, target_idx, values = [], [], []
    for i, dim in enumerate(SCORE_DIMENSIONS):
        for score_val in df[dim].dropna():
            if score_val <= 1:
                level_idx = len(SCORE_DIMENSIONS) + 0
            elif score_val == 2:
                level_idx = len(SCORE_DIMENSIONS) + 1
            else:
                level_idx = len(SCORE_DIMENSIONS) + 2
            source.append(i)
            target_idx.append(level_idx)
            values.append(1)

    # Aggregate to avoid huge link counts
    link_counts = {}
    for s, t, v in zip(source, target_idx, values):
        key = (s, t)
        link_counts[key] = link_counts.get(key, 0) + v

    source_agg = [k[0] for k in link_counts]
    target_agg = [k[1] for k in link_counts]
    value_agg = list(link_counts.values())

    # Score Level → Pulse Status links
    status_offset = len(SCORE_DIMENSIONS) + len(score_levels)
    level_status_counts = {}
    for _, row in df.iterrows():
        status = get_pulse_status(row['Total Score'])
        status_idx = status_offset + statuses.index(status)

        dim_scores = [row[d] for d in SCORE_DIMENSIONS if pd.notna(row.get(d))]
        for s in dim_scores:
            if s <= 1:
                level_idx = len(SCORE_DIMENSIONS) + 0
            elif s == 2:
                level_idx = len(SCORE_DIMENSIONS) + 1
            else:
                level_idx = len(SCORE_DIMENSIONS) + 2
            key = (level_idx, status_idx)
            level_status_counts[key] = level_status_counts.get(key, 0) + 1

    source_agg.extend([k[0] for k in level_status_counts])
    target_agg.extend([k[1] for k in level_status_counts])
    value_agg.extend(list(level_status_counts.values()))

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,
            label=nodes,
            color=node_colors,
        ),
        link=dict(
            source=source_agg,
            target=target_agg,
            value=value_agg,
            color='rgba(255,255,255,0.08)',
        ),
    ))
    fig.update_layout(**get_plotly_theme(), title='Score Flow: Dimension → Level → Status', height=600)
    return fig


# ============================================================================
# ICICLE
# ============================================================================
def chart_icicle(df: pd.DataFrame) -> go.Figure:
    fig = px.icicle(
        df,
        path=['Region', 'Area', 'Pulse_Status'],
        color='Total Score',
        color_continuous_scale=CONTINUOUS_COLOR_SCALE,
        color_continuous_midpoint=COLOR_MIDPOINT,
    )
    fig.update_layout(**get_plotly_theme(), title='Icicle: Region → Area → Status', height=600)
    return fig


# ============================================================================
# SPARKLINES BY REGION
# ============================================================================
def chart_sparklines(df: pd.DataFrame) -> go.Figure:
    """Small multiples: one mini line per region."""
    regions = sorted(df['Region'].dropna().unique())
    n_regions = len(regions)
    if n_regions == 0:
        fig = go.Figure()
        fig.update_layout(**get_plotly_theme(), title='No Regions')
        return fig

    fig = make_subplots(
        rows=1, cols=n_regions,
        subplot_titles=regions,
        horizontal_spacing=0.05,
    )

    for i, region in enumerate(regions, 1):
        region_df = df[df['Region'] == region]
        weekly = (
            region_df.groupby(['Year', 'Wk'])
            .agg(avg=('Total Score', 'mean'))
            .reset_index()
            .sort_values(['Year', 'Wk'])
        )
        avg_score = weekly['avg'].mean()
        color = get_pulse_color(avg_score)

        fig.add_trace(
            go.Scatter(
                x=weekly['Wk'], y=weekly['avg'],
                mode='lines', line=dict(color=color, width=2),
                name=region, showlegend=False,
            ),
            row=1, col=i,
        )

    fig.update_layout(
        **get_plotly_theme(),
        height=250,
        title='Pulse Trend by Region',
    )
    for i in range(1, n_regions + 1):
        fig.update_yaxes(range=[10, 24], row=1, col=i, showticklabels=(i == 1), gridcolor='#1e293b')
        fig.update_xaxes(showticklabels=False, row=1, col=i, gridcolor='#1e293b')
    return fig


# ============================================================================
# 2x2 IMPACT-EFFORT MATRIX
# ============================================================================
def chart_impact_effort_matrix(df: pd.DataFrame, target: float) -> go.Figure:
    """Scatter plot: Impact (gap to target) vs Effort (dims < 2)."""
    plot_df = df.copy()
    plot_df['Impact'] = (target - plot_df['Total Score']).clip(lower=0)

    # Assign quadrants
    impact_mid = plot_df['Impact'].median() if len(plot_df) > 0 else 1
    effort_mid = plot_df['Effort'].median() if len(plot_df) > 0 else 1
    # Use at least 1 to avoid degenerate cases
    impact_mid = max(impact_mid, 0.5)
    effort_mid = max(effort_mid, 0.5)

    def quadrant(row):
        hi = row['Impact'] >= impact_mid
        he = row['Effort'] >= effort_mid
        if hi and not he:
            return 'Quick Wins'
        elif hi and he:
            return 'Major Projects'
        elif not hi and not he:
            return 'Fill-ins'
        else:
            return 'Deprioritize'

    plot_df['Quadrant'] = plot_df.apply(quadrant, axis=1)

    quad_colors = {
        'Quick Wins': '#22c55e',
        'Major Projects': '#f59e0b',
        'Fill-ins': '#0077B6',
        'Deprioritize': '#6D6E71',
    }

    fig = px.scatter(
        plot_df,
        x='Effort', y='Impact',
        color='Quadrant',
        color_discrete_map=quad_colors,
        hover_data=['Project', 'Region', 'Total Score'],
        title='Impact-Effort Prioritization Matrix',
    )

    # Quadrant lines
    fig.add_hline(y=impact_mid, line_dash='dot', line_color='#94a3b8', line_width=1)
    fig.add_vline(x=effort_mid, line_dash='dot', line_color='#94a3b8', line_width=1)

    fig.update_layout(
        **get_plotly_theme(),
        xaxis_title='Effort (# of Dimensions < 2)',
        yaxis_title='Impact (Gap to Target)',
        height=500,
    )
    return fig


# ============================================================================
# RADAR CHART (Single Project)
# ============================================================================
def chart_radar(project_row: pd.Series) -> go.Figure:
    """Radar/spider chart for a single project's 8 dimension scores."""
    dims = SCORE_DIMENSIONS
    values = [project_row.get(d, 0) for d in dims]
    values.append(values[0])  # close the polygon
    labels = dims + [dims[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        fillcolor=f'rgba(0, 191, 255, 0.15)',
        line=dict(color='#00BFFF', width=2),
        name=project_row.get('Project', ''),
    ))

    # Max possible ring
    fig.add_trace(go.Scatterpolar(
        r=[3] * (len(dims) + 1),
        theta=labels,
        fill='none',
        line=dict(color='rgba(255,255,255,0.1)', width=1, dash='dot'),
        showlegend=False,
    ))

    fig.update_layout(
        **get_plotly_theme(),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(range=[0, 3], gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
            angularaxis=dict(gridcolor='#1e293b', tickfont=dict(color='#E0E0E0')),
        ),
        title=f"Dimension Scores: {project_row.get('Project', '')}",
        height=400,
    )
    return fig


# ============================================================================
# PROJECT TREND (Historical)
# ============================================================================
def chart_project_trend(df: pd.DataFrame, project_name: str, target: float) -> go.Figure:
    """Historical trend line for a specific project."""
    proj_df = df[df['Project'] == project_name].sort_values(['Year', 'Wk'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=proj_df['Year_Week'], y=proj_df['Total Score'],
        mode='lines+markers',
        line=dict(color='#00BFFF', width=2),
        marker=dict(size=6),
        name=project_name,
    ))
    fig.add_hline(y=target, line_dash='dash', line_color='white',
                  annotation_text=f'Target ({target:.0f})')

    _apply_theme(fig)
    fig.update_layout(
        title=f'Trend: {project_name}',
        xaxis_title='Week',
        yaxis_title='Total Score',
        height=350,
    )
    fig.update_yaxes(range=[6, 24])
    return fig


# ============================================================================
# MINI SPARKLINE (inline SVG)
# ============================================================================
def mini_sparkline_svg(values, width: int = 70, height: int = 20,
                       color: str = "#60a5fa") -> str:
    """Generate inline SVG sparkline for use in HTML tables."""
    vals = [v for v in values if pd.notna(v)]
    if len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1

    points = []
    for i, v in enumerate(vals):
        x = (i / (len(vals) - 1)) * (width - 4) + 2
        y = height - 3 - ((v - mn) / rng) * (height - 6)
        points.append(f"{x},{y}")

    polyline = " ".join(points)
    last_x, last_y = points[-1].split(",")
    return (
        f'<svg width="{width}" height="{height}" style="display:block;">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2" fill="{color}"/>'
        f'</svg>'
    )


# ============================================================================
# PULSE RANKING CHART (Stacked Flow / Stream)
# ============================================================================
def chart_pulse_ranking(df: pd.DataFrame,
                        score_cols: list | None = None,
                        dim_colors: dict | None = None) -> go.Figure:
    """Stacked ranked flow chart showing dimension rankings week-over-week.

    Each dimension forms a horizontal band whose height = its avg score.
    Bands are stacked in rank order and smoothly interpolated between weeks.
    """
    if score_cols is None:
        score_cols = SCORE_DIMENSIONS
    if dim_colors is None:
        dim_colors = DIMENSION_COLORS

    # Group by Year+Wk for correct chronological order across year boundary
    weekly = (
        df.groupby(['Year', 'Wk'])[score_cols]
        .mean()
        .reset_index()
        .sort_values(['Year', 'Wk'])
    )

    if len(weekly) < 2:
        fig = go.Figure()
        fig.update_layout(**get_plotly_theme(),
                          title='Pulse Ranking (need ≥ 2 weeks of data)')
        return fig

    # Sequential integer index for spline interpolation
    n = len(weekly)
    x_idx = np.arange(n)
    # Labels for tick marks: "25-W26", "26-W01", etc.
    tick_labels = [
        f"{int(row['Year']) % 100}-W{int(row['Wk']):02d}"
        for _, row in weekly.iterrows()
    ]
    # Year_Week strings for drill-down customdata
    yw_labels = [
        f"{int(row['Year'])}-W{int(row['Wk']):02d}"
        for _, row in weekly.iterrows()
    ]

    # Build ranked stacks for each week
    week_stacks = []
    for _, row in weekly.iterrows():
        vals = {c: round(row[c], 2) for c in score_cols}
        sorted_dims = sorted(vals.items(), key=lambda x: x[1])
        cum_y = 0
        stack = {}
        for dim, val in sorted_dims:
            stack[dim] = {'y0': cum_y, 'y1': cum_y + val, 'val': val}
            cum_y += val
        week_stacks.append(stack)

    # Smooth interpolation
    try:
        from scipy.interpolate import make_interp_spline
        k = min(3, n - 1)
        x_smooth = np.linspace(0, n - 1, 300)
        use_spline = True
    except ImportError:
        x_smooth = np.linspace(0, n - 1, n * 10)
        use_spline = False

    fig = go.Figure()

    # Draw in order of avg value (smallest = bottom)
    avg_vals = {c: weekly[c].mean() for c in score_cols}
    draw_order = sorted(avg_vals, key=lambda x: avg_vals[x])

    for dim in draw_order:
        y0_pts = [week_stacks[i][dim]['y0'] for i in range(n)]
        y1_pts = [week_stacks[i][dim]['y1'] for i in range(n)]
        val_pts = [week_stacks[i][dim]['val'] for i in range(n)]

        if use_spline:
            spl_y0 = make_interp_spline(x_idx, y0_pts, k=k)
            spl_y1 = make_interp_spline(x_idx, y1_pts, k=k)
            y0_sm = spl_y0(x_smooth)
            y1_sm = spl_y1(x_smooth)
        else:
            y0_sm = np.interp(x_smooth, x_idx, y0_pts)
            y1_sm = np.interp(x_smooth, x_idx, y1_pts)

        # Filled band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_smooth, x_smooth[::-1]]),
            y=np.concatenate([y1_sm, y0_sm[::-1]]),
            fill='toself',
            fillcolor=dim_colors.get(dim, '#666'),
            line=dict(color='rgba(255,255,255,0.25)', width=0.5),
            name=dim,
            showlegend=True,
            hoverinfo='skip',
        ))

        # Clickable markers with values at band midpoint
        mid_pts = [(y0_pts[i] + y1_pts[i]) / 2 for i in range(n)]
        fig.add_trace(go.Scatter(
            x=list(x_idx),
            y=mid_pts,
            mode='markers+text',
            marker=dict(size=22, color=dim_colors.get(dim, '#666'),
                        opacity=0.01, symbol='square'),
            text=[f"{v:.2f}" for v in val_pts],
            textfont=dict(size=8, color='white'),
            textposition='middle center',
            name=dim,
            showlegend=False,
            customdata=[[dim, yw_labels[i], v]
                        for i, v in enumerate(val_pts)],
            hovertemplate=(
                f"<b>{dim}</b><br>"
                "%{customdata[1]}<br>"
                "Avg: %{customdata[2]:.2f}<extra></extra>"
            ),
        ))

    _apply_theme(fig)
    fig.update_layout(
        xaxis=dict(
            title='Week',
            tickmode='array',
            tickvals=list(x_idx),
            ticktext=tick_labels,
            range=[-0.5, n - 0.5],
            gridcolor='rgba(255,255,255,0.03)',
            tickangle=-90,
        ),
        yaxis=dict(
            range=[0, 20],
            dtick=10,
            gridcolor='rgba(255,255,255,0.05)',
            griddash='dot',
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            font=dict(size=10, color='#CBD5E1'),
        ),
        height=450,
        margin=dict(l=35, r=15, t=60, b=70),
        hovermode='closest',
    )
    return fig


# ============================================================================
# DRILL-DOWN: Dimension Bar Chart by Region
# ============================================================================
def chart_dimension_by_region(df: pd.DataFrame, dimension: str,
                              dim_color: str = '#2563eb') -> go.Figure:
    """Bar chart: average dimension score by region."""
    region_summary = (
        df.groupby('Region')
        .agg(Avg=(dimension, 'mean'))
        .reset_index()
        .sort_values('Avg')
    )

    fig = go.Figure(go.Bar(
        x=region_summary['Region'],
        y=region_summary['Avg'],
        marker_color=dim_color,
        text=[f"{v:.2f}" for v in region_summary['Avg']],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=11),
    ))
    _apply_theme(fig)
    fig.update_layout(height=220, margin=dict(l=30, r=15, t=10, b=30))
    fig.update_yaxes(range=[0, 3.2])
    return fig


# ============================================================================
# DRILL-DOWN: Dimension Score Distribution
# ============================================================================
def chart_dimension_distribution(df: pd.DataFrame,
                                 dimension: str) -> go.Figure:
    """Bar chart: score distribution (0/1/2/3) for a dimension."""
    dist = df[dimension].dropna().astype(int).value_counts().sort_index()
    dist_colors = {0: '#ef4444', 1: '#f97316', 2: '#3b82f6', 3: '#22c55e'}

    fig = go.Figure(go.Bar(
        x=[f"Score {int(s)}" for s in dist.index],
        y=dist.values,
        marker_color=[dist_colors.get(int(s), '#64748b') for s in dist.index],
        text=dist.values,
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=11),
    ))
    _apply_theme(fig)
    fig.update_layout(height=220, margin=dict(l=30, r=15, t=10, b=30))
    return fig


# ============================================================================
# DRILL-DOWN: Dimension Weekly Trend by Region
# ============================================================================
def chart_dimension_trend_by_region(df: pd.DataFrame,
                                     dimension: str,
                                     regions: list | None = None) -> go.Figure:
    """Line chart: weekly trend for a dimension, one line per region."""
    if regions is None:
        regions = sorted(df['Region'].dropna().unique())

    trend = df.groupby(['Wk', 'Region'])[dimension].mean().reset_index()
    fig = go.Figure()

    for region in regions:
        rd = trend[trend['Region'] == region].sort_values('Wk')
        if len(rd) > 0:
            fig.add_trace(go.Scatter(
                x=rd['Wk'], y=rd[dimension],
                mode='lines+markers',
                name=region,
                line=dict(color=REGION_LINE_COLORS.get(region, '#94a3b8'),
                          width=2),
                marker=dict(size=4),
            ))

    _apply_theme(fig)
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=15, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    font=dict(color='#cbd5e1', size=9)),
    )
    fig.update_yaxes(range=[0, 3.2])
    return fig
