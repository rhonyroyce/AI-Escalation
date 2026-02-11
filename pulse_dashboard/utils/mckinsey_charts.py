"""
Pulse Dashboard - McKinsey-Grade Chart Library
===============================================

This module is the sole charting back-end for the Pulse Dashboard.  Every public
function returns either a ``plotly.graph_objects.Figure`` (interactive chart) or
an HTML string (KPI cards, SVG sparklines) that Streamlit renders via
``st.plotly_chart()`` or ``st.markdown(..., unsafe_allow_html=True)``.

Design philosophy
-----------------
* **Dark theme first** - All figures are styled with a transparent background,
  light-gray text (Inter font), and subtle slate-gray grid lines so they blend
  into the dark Streamlit theme defined in ``utils/styles.py``.
* **McKinsey visual language** - Clean axes, sparing color, high data-ink ratio.
  Colors come from ``MCKINSEY_COLORS`` and the four-tier Pulse status palette
  (Red / Yellow / Green / Dark Green).
* **Consistency via ``_apply_theme()``** - A single private helper applies the
  Plotly layout dict *and* axis grid styling to every figure, so charts share
  identical typography, margins, and grid colors.

Color mapping conventions
-------------------------
There are three distinct color schemes used throughout:

1. **Pulse Status colors** (total score 0-24):
   - Red ``#ef4444`` -- Critical (score 1-13)
   - Yellow ``#f59e0b`` -- At Risk (score 14-15)
   - Green ``#22c55e`` -- On Track (score 16-19)
   - Dark Green ``#059669`` -- Exceptional (score 20-24)
   Accessed via ``get_pulse_color(score)`` from ``utils/styles.py``.

2. **Dimension colors** (per-dimension identity):
   Each of the 8 scoring dimensions (Design, IX, PAG, RF Opt, Field, CSAT,
   PM Performance, Potential) has a unique hue defined in ``DIMENSION_COLORS``.
   Used in the Pulse Ranking stacked-flow chart and dimension drill-downs.

3. **Dimension score colors** (individual dimension 0-3):
   - 0 = ``#ef4444`` (red)
   - 1 = ``#f59e0b`` (amber/yellow)
   - 2 = ``#22c55e`` (green)
   - 3 = ``#059669`` (dark green)
   Used in the Sunburst leaf nodes via the private ``_dim_color()`` helper.

4. **Region line colors** (per-region identity for trend lines):
   Central = blue, NE = green, South = yellow, West = pink.
   Defined in ``REGION_LINE_COLORS`` in ``utils/styles.py``.

Plotly patterns used
--------------------
* ``go.Figure(go.Trace(...))`` -- single-trace shorthand.
* ``fig.add_trace(go.Scatter(...))`` -- multi-trace composition.
* ``make_subplots(specs=...)`` -- dual-axis (Pareto) and small-multiple
  (sparklines) layouts.
* ``px.scatter / px.treemap / px.icicle`` -- Plotly Express convenience
  wrappers that accept DataFrame + column names.
* ``fig.update_layout(**get_plotly_theme())`` -- merges the global dark-theme
  dict into the figure layout.
* ``fig.update_xaxes(**AXIS_STYLE)`` / ``fig.update_yaxes(...)`` -- applies
  grid and zero-line colors consistently.

Imports from utils/styles.py
----------------------------
``get_plotly_theme``      Returns the dark-theme layout dict (transparent bg, Inter font, margins).
``get_pulse_status``      Maps a total score (0-24) to a status string ('Red'/'Yellow'/'Green'/'Dark Green').
``get_pulse_color``       Maps a total score (0-24) to its hex color.
``SCORE_DIMENSIONS``      List of the 8 dimension column names.
``STATUS_CONFIG``         Dict of status -> {color, label, range, min, max}.
``STATUS_ORDER``          Canonical ordering: ['Red', 'Yellow', 'Green', 'Dark Green'].
``CONTINUOUS_COLOR_SCALE``Four-color scale for continuous color mapping in treemaps/icicles.
``COLOR_MIDPOINT``        Midpoint value (16) for the continuous color scale.
``DISCRETE_COLOR_MAP``    Status -> hex color dict for categorical color assignments.
``MCKINSEY_COLORS``       Named palette (primary_blue, bright_blue, etc.).
``AXIS_STYLE``            Default axis grid styling dict.
``DIMENSION_COLORS``      Dimension name -> unique hex color for stacked charts.
``REGION_LINE_COLORS``    Region name -> hex color for multi-region trend lines.
"""

import pandas as pd
import numpy as np
import plotly.express as px                 # High-level Plotly API (scatter, treemap, icicle)
import plotly.graph_objects as go           # Low-level Plotly API (Figure, traces)
from plotly.subplots import make_subplots   # Subplot grid factory (used for dual-axis & small-multiples)

# Import the centralized style/theme constants from the companion styles module.
# See the module docstring above for a summary of what each import provides.
from utils.styles import (
    get_plotly_theme, get_pulse_status, get_pulse_color,
    SCORE_DIMENSIONS, STATUS_CONFIG, STATUS_ORDER,
    CONTINUOUS_COLOR_SCALE, COLOR_MIDPOINT, DISCRETE_COLOR_MAP,
    MCKINSEY_COLORS, AXIS_STYLE, DIMENSION_COLORS, REGION_LINE_COLORS,
)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the global dark theme and default axis grid styling to *fig*.

    This is the single point of consistency across all charts produced by this
    module.  It merges the layout dict returned by ``get_plotly_theme()``
    (transparent background, Inter font, standard margins) into the figure and
    then sets the x/y axis grid colors to the dark-slate tone defined in
    ``AXIS_STYLE`` (``#1e293b``).

    Parameters
    ----------
    fig : go.Figure
        Any Plotly figure object.

    Returns
    -------
    go.Figure
        The same figure, mutated in-place (also returned for chaining).
    """
    fig.update_layout(**get_plotly_theme())
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ============================================================================
# KPI CARDS (returns HTML string, not a Plotly figure)
# ============================================================================

def kpi_card_html(label: str, value: str, delta: str = "", css_class: str = "",
                  value_class: str = "") -> str:
    """Build a single KPI card as an HTML snippet for ``st.markdown()``.

    The card renders as a dark glass-morphism box with a large numeric value,
    a muted uppercase label underneath, and an optional delta indicator (up/down
    arrow-style text).

    Parameters
    ----------
    label : str
        Descriptive label displayed below the value (e.g. "Avg Pulse Score").
    value : str
        The hero metric string (e.g. "18.3" or "72%").
    delta : str, optional
        A change indicator string (e.g. "+2.1" or "-0.5").  If it starts with
        ``'-'`` it is styled red; otherwise green.
    css_class : str, optional
        Additional CSS class(es) for the outer ``<div>`` container (e.g.
        ``"critical"``, ``"warning"``, ``"success"``, ``"exceptional"``).
        These map to gradient backgrounds defined in the injected CSS.
    value_class : str, optional
        Additional CSS class for the ``<p class="kpi-value">`` element.  Used
        to apply status-colored gradients (e.g. ``"red"``, ``"green"``).

    Returns
    -------
    str
        An HTML string suitable for ``st.markdown(html, unsafe_allow_html=True)``.
    """
    # Build the optional delta sub-element only when a delta string is provided.
    delta_html = ""
    if delta:
        # Determine direction: any string NOT starting with '-' is treated as positive.
        is_positive = not delta.startswith('-')
        # Choose the CSS class that colors the delta green (positive) or red (negative).
        delta_class = "delta-positive" if is_positive else "delta-negative"
        delta_html = f'<p class="kpi-delta {delta_class}">{delta}</p>'

    # Assemble the card using CSS classes defined in utils/styles.py -> inject_css().
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
    """Horizontal bullet chart comparing the portfolio average Pulse Score
    against a target and a stretch goal.

    This is a McKinsey-style bullet chart with colored background ranges
    representing performance zones (poor to excellent) and two vertical marker
    lines for the target and stretch thresholds.

    Visual structure
    ----------------
    * **Background bars** (stacked horizontally, semi-transparent) represent
      four performance zones:
      - 0-14: red zone (critical)
      - 14-16: amber zone (at risk)
      - 16-20: green zone (on track)
      - 20-24: dark green zone (exceptional)
    * **Actual value bar** (bright blue ``#00BFFF``, narrow) shows the current
      portfolio average.
    * **Target line** (solid white vertical) marks the target threshold.
    * **Stretch line** (dashed amber vertical) marks the aspirational stretch
      goal.

    Parameters
    ----------
    avg_score : float
        Current portfolio average Pulse Score (0-24 scale).
    target : float
        Primary target score (displayed as a solid white vertical line).
    stretch : float
        Stretch/aspirational target (displayed as a dashed amber line).

    Returns
    -------
    go.Figure
        A compact (200px tall) horizontal bullet chart.
    """
    fig = go.Figure()

    # --- Background ranges (poor -> good) ---
    # Each range is a horizontal bar segment laid out sequentially using the
    # ``base`` parameter.  Semi-transparent colors visually encode the
    # performance zone without overwhelming the actual-value bar.
    ranges = [14, 16, 20, 24]                         # Upper bounds of each zone
    colors = ['rgba(239,68,68,0.2)',                   # Red zone    (0-14)
              'rgba(245,158,11,0.2)',                   # Amber zone  (14-16)
              'rgba(34,197,94,0.2)',                    # Green zone  (16-20)
              'rgba(5,150,105,0.2)']                    # Dark-green  (20-24)
    prev = 0  # Running left edge for each stacked segment
    for r, c in zip(ranges, colors):
        fig.add_trace(go.Bar(
            x=[r - prev],               # Width of this segment
            y=['Pulse'],                 # Single categorical y-axis value
            orientation='h',             # Horizontal bar
            base=prev,                   # Left edge of this segment
            marker=dict(color=c),
            showlegend=False,
            hoverinfo='skip',            # Background bars are non-interactive
        ))
        prev = r  # Advance left edge for the next segment

    # --- Actual value bar ---
    # A narrower bar (width=0.3) in bright blue overlaid on top of the
    # background ranges.  The text label is placed outside the bar end.
    fig.add_trace(go.Bar(
        x=[avg_score], y=['Pulse'], orientation='h',
        marker=dict(color='#00BFFF', line=dict(width=0)),
        width=0.3,                       # Narrower than background bars for layered effect
        name='Actual',
        text=f'{avg_score:.1f}',         # Numeric label at bar tip
        textposition='outside',
        textfont=dict(color='#E0E0E0', size=14, family='Inter'),
    ))

    # --- Target marker ---
    # A vertical white line spanning the y-axis at the target score.
    # Implemented as a two-point Scatter trace (line from y=-0.4 to y=0.4).
    fig.add_trace(go.Scatter(
        x=[target, target], y=[-0.4, 0.4],
        mode='lines', line=dict(color='white', width=3),
        name=f'Target ({target:.0f})',
    ))

    # --- Stretch marker ---
    # A dashed amber vertical line indicating the aspirational goal.
    fig.add_trace(go.Scatter(
        x=[stretch, stretch], y=[-0.3, 0.3],
        mode='lines', line=dict(color='#f59e0b', width=2, dash='dash'),
        name=f'Stretch ({stretch:.0f})',
    ))

    # Apply global dark theme, then override specific layout properties.
    _apply_theme(fig)
    fig.update_layout(
        title='Portfolio Pulse vs Target',
        height=200,                      # Compact vertical footprint
        barmode='overlay',               # Stack background + actual bar on same axis
        legend=dict(orientation='h', y=-0.3),  # Horizontal legend below chart
    )
    fig.update_xaxes(range=[0, 24], title_text='Score')
    fig.update_yaxes(visible=False)      # Hide the single-category y-axis
    return fig


# ============================================================================
# TREND + FORECAST
# ============================================================================

def chart_trend_forecast(df: pd.DataFrame, target: float, n_forecast: int = 4) -> go.Figure:
    """Portfolio-level Pulse Score time series with a target line and a linear
    forecast extension.

    The chart aggregates all projects by ``Year_Week`` to compute a weekly
    average Pulse Score, plots the historical line, and then fits a simple
    linear regression on the most recent 8 data points to project forward
    ``n_forecast`` weeks.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``Year``, ``Wk``, ``Year_Week``, ``Total Score``,
        ``Project``.  Typically the full multi-week dataset.
    target : float
        Target score rendered as a horizontal dashed white line.
    n_forecast : int, default 4
        Number of future weeks to extrapolate via linear regression.

    Returns
    -------
    go.Figure
        A 400px-tall line chart with actual trend, target line, and forecast.
    """
    # --- Aggregate to weekly averages ---
    # Group by (Year, Wk, Year_Week) to get one row per calendar week.
    # Year_Week is a pre-computed string like "2025-W26" used as the x-axis label.
    weekly = (
        df.groupby(['Year', 'Wk', 'Year_Week'])
        .agg(avg_score=('Total Score', 'mean'), count=('Project', 'count'))
        .reset_index()
        .sort_values(['Year', 'Wk'])      # Chronological order
    )

    fig = go.Figure()

    # --- Actual trend line ---
    # Bright-blue line with circular markers at each data point.
    fig.add_trace(go.Scatter(
        x=weekly['Year_Week'], y=weekly['avg_score'],
        mode='lines+markers', name='Actual',
        line=dict(color='#00BFFF', width=3),
        marker=dict(size=6),
    ))

    # --- Target reference line ---
    # Horizontal dashed white line with an annotation label.
    fig.add_hline(y=target, line_dash='dash', line_color='white',
                  annotation_text=f'Target ({target:.0f})',
                  annotation_font_color='white')

    # --- Linear forecast ---
    # Only computed when at least 4 historical weeks are available.
    if len(weekly) >= 4:
        # Use the last 8 data points (or fewer if less data exists) to fit the
        # regression, avoiding older data that may not represent the current trend.
        n_points = min(8, len(weekly))
        recent = weekly.tail(n_points)

        # Convert to a simple integer x-axis for np.polyfit (degree=1 = linear).
        x_num = np.arange(len(recent))
        y_vals = recent['avg_score'].values

        # Fit: y = slope * x + intercept
        slope, intercept = np.polyfit(x_num, y_vals, 1)

        # Generate future week labels ("YYYY-Www") by incrementing from the
        # last known week.  Handles year rollover at week 52.
        last_year = recent['Year'].iloc[-1]
        last_wk = recent['Wk'].iloc[-1]
        forecast_labels = []
        for i in range(1, n_forecast + 1):
            fw = last_wk + i
            fy = last_year
            if fw > 52:          # Simple year-boundary rollover
                fw -= 52
                fy += 1
            forecast_labels.append(f"{fy}-W{fw:02d}")

        # Extrapolate the regression line into the forecast period.
        forecast_x = np.arange(len(recent), len(recent) + n_forecast)
        forecast_y = slope * forecast_x + intercept

        # Plot the forecast as a dashed amber line with diamond markers to
        # visually distinguish it from the actual data.
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
    fig.update_yaxes(range=[10, 24])     # Fixed y-range to keep context across views
    return fig


# ============================================================================
# WATERFALL (Score Decomposition)
# ============================================================================

def chart_waterfall_decomposition(df: pd.DataFrame) -> go.Figure:
    """Waterfall chart decomposing the average Total Score into its 8 dimension
    contributions.

    Each bar shows one dimension's average score.  The bars stack cumulatively
    from left to right, with a final "Total" bar showing the sum.  This reveals
    which dimensions contribute most (or least) to the portfolio score.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 8 dimension columns listed in ``SCORE_DIMENSIONS``
        (e.g. 'Design', 'IX', 'PAG', ...).  Each value is 0-3.

    Returns
    -------
    go.Figure
        A 400px waterfall chart with dimension contributions and a total bar.
    """
    # Compute the mean of each dimension across all rows.
    means = df[SCORE_DIMENSIONS].mean()

    # The Waterfall trace uses ``measure='relative'`` for each bar, meaning
    # each bar's value is added to the running total (cumulative stacking).
    fig = go.Figure(go.Waterfall(
        x=list(means.index),                           # Dimension names on x-axis
        y=list(means.values),                          # Average score per dimension
        measure=['relative'] * len(means),             # All bars are relative (cumulative)
        text=[f'{v:.2f}' for v in means.values],       # Value labels on each bar
        textposition='outside',
        textfont=dict(color='#E0E0E0'),
        connector=dict(line=dict(color='#334155', width=1)),  # Subtle connector lines
        increasing=dict(marker=dict(color='#22c55e')),        # Green for positive contributions
        decreasing=dict(marker=dict(color='#ef4444')),        # Red for negative (unlikely here)
        totals=dict(marker=dict(color='#0066CC')),            # Blue for the implicit total
    ))

    # Add an explicit "Total" bar at the end using a separate Bar trace,
    # because the Waterfall trace does not automatically render a total column
    # when all measures are 'relative'.
    fig.add_trace(go.Bar(
        x=['Total'], y=[means.sum()],
        marker=dict(color='#0066CC'),       # McKinsey modern blue
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
    # Y-axis upper bound ensures headroom for text labels above the Total bar.
    fig.update_yaxes(range=[0, max(means.sum() + 2, 24)])
    return fig


# ============================================================================
# PARETO CHART
# ============================================================================

def chart_pareto(df: pd.DataFrame, target: float, groupby_col: str = 'Area') -> go.Figure:
    """Pareto chart of below-target projects grouped by a categorical column.

    Shows a bar chart (count of below-target projects per group) on the primary
    y-axis and a cumulative-percentage line on the secondary y-axis, with an
    80% threshold line to highlight the "vital few" groups that account for
    most issues (Pareto principle / 80-20 rule).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Total Score`` and the column specified by *groupby_col*.
    target : float
        Projects with ``Total Score < target`` are counted as below-target.
    groupby_col : str, default 'Area'
        The categorical column to group by (e.g. 'Area', 'Region', 'PM Name').

    Returns
    -------
    go.Figure
        A dual-axis 400px chart (bars + cumulative % line).  Returns an empty
        figure with a title if no projects are below target.
    """
    # Filter to only projects that fall below the target score.
    below_target = df[df['Total Score'] < target]
    if below_target.empty:
        # Early return: no issues to display.
        fig = go.Figure()
        fig.update_layout(**get_plotly_theme(), title='No Below-Target Projects')
        return fig

    # Count projects per group and sort descending (largest group first).
    issues = below_target.groupby(groupby_col).size().sort_values(ascending=False)
    # Compute running cumulative percentage for the Pareto line.
    cumulative_pct = issues.cumsum() / issues.sum() * 100

    # Create a subplot with a secondary y-axis for the cumulative % line.
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary axis: red bar chart of issue counts.
    fig.add_trace(
        go.Bar(x=issues.index, y=issues.values, name='Count',
               marker=dict(color='#ef4444')),        # Red = problems
        secondary_y=False,
    )
    # Secondary axis: amber cumulative-% line with markers.
    fig.add_trace(
        go.Scatter(x=issues.index, y=cumulative_pct.values,
                   mode='lines+markers', name='Cumulative %',
                   line=dict(color='#f59e0b', width=2),
                   marker=dict(size=6)),
        secondary_y=True,
    )
    # 80% threshold line: the classic Pareto cutoff.
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
    """Map a single dimension score (0-3 integer) to a traffic-light hex color.

    This is distinct from the Pulse Status color mapping (which operates on the
    0-24 Total Score).  Here we are coloring individual dimension values:

    * 3 -> ``#059669`` (dark green / exceptional)
    * 2 -> ``#22c55e`` (green / on track)
    * 1 -> ``#f59e0b`` (amber / at risk)
    * 0 -> ``#ef4444`` (red / critical)

    Used exclusively by ``chart_sunburst()`` to color the leaf-level dimension
    segments.

    Parameters
    ----------
    score : int or float
        A dimension score in the range 0-3.

    Returns
    -------
    str
        Hex color string.
    """
    if score >= 3:
        return '#059669'   # dark green - exceptional
    elif score >= 2:
        return '#22c55e'   # green - on track
    elif score >= 1:
        return '#f59e0b'   # amber/yellow - at risk
    else:
        return '#ef4444'   # red - critical


def chart_sunburst(df: pd.DataFrame) -> go.Figure:
    """Four-level sunburst chart: Portfolio -> Region -> Area -> Project -> Dimensions.

    This is the primary hierarchical drill-down visualization.  The innermost
    ring is the portfolio root; the outermost ring contains the 8 dimension
    scores for each project.

    Color logic
    -----------
    * **Inner rings** (Portfolio, Region, Area, Project): colored by the
      *Total Score* using ``get_pulse_color()`` -- the standard 4-tier Pulse
      status palette (Red/Yellow/Green/Dark Green).
    * **Outer ring** (dimension leaves): colored by the *individual dimension
      score* (0-3) using ``_dim_color()`` -- a separate 4-level traffic-light
      scale.

    This dual-scale approach lets viewers simultaneously assess overall project
    health (inner rings) and pinpoint specific dimension weaknesses (outer ring).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Region``, ``Area``, ``Project``, ``Total Score``,
        ``PM Name``, and the 8 ``SCORE_DIMENSIONS`` columns.

    Returns
    -------
    go.Figure
        A 650px-tall interactive sunburst chart with ``maxdepth=3`` (users can
        click to drill deeper).
    """
    # Parallel lists that define the sunburst hierarchy.  Plotly's Sunburst
    # trace requires flattened arrays of ids, labels, parents, values, and
    # colors (one entry per node in the tree).
    ids, labels, parents, values, colors, hovertexts = [], [], [], [], [], []

    # --- Root node: Portfolio ---
    portfolio_avg = df['Total Score'].mean()
    ids.append('Portfolio')
    labels.append('Portfolio')
    parents.append('')           # Empty string = root (no parent)
    values.append(0)             # Non-leaf nodes get value=0; leaves get 1
    colors.append(get_pulse_color(portfolio_avg))
    hovertexts.append(f'Avg Pulse: {portfolio_avg:.1f}')

    # --- Level 1: Regions ---
    for region in sorted(df['Region'].dropna().unique()):
        rdf = df[df['Region'] == region]
        r_avg = rdf['Total Score'].mean()
        r_id = region            # Unique ID = region name

        ids.append(r_id)
        labels.append(region)
        parents.append('Portfolio')
        values.append(0)
        colors.append(get_pulse_color(r_avg))
        hovertexts.append(f'Avg Pulse: {r_avg:.1f} ({rdf["Project"].nunique()} proj)')

        # --- Level 2: Areas within each Region ---
        for area in sorted(rdf['Area'].dropna().unique()):
            adf = rdf[rdf['Area'] == area]
            a_avg = adf['Total Score'].mean()
            a_id = f'{region}/{area}'   # Unique ID = "Region/Area"

            ids.append(a_id)
            labels.append(area)
            parents.append(r_id)
            values.append(0)
            colors.append(get_pulse_color(a_avg))
            hovertexts.append(f'Avg Pulse: {a_avg:.1f} ({adf["Project"].nunique()} proj)')

            # --- Level 3: Projects within each Area ---
            for _, row in adf.iterrows():
                project = row['Project']
                total = row['Total Score']
                p_id = f'{region}/{area}/{project}'   # Unique ID = full path

                ids.append(p_id)
                labels.append(project)
                parents.append(a_id)
                values.append(0)
                colors.append(get_pulse_color(total))
                hovertexts.append(f'Total: {int(total)} | PM: {row.get("PM Name", "—")}')

                # --- Level 4 (leaves): Dimension scores for each Project ---
                for dim in SCORE_DIMENSIONS:
                    score = int(row.get(dim, 0))
                    d_id = f'{p_id}/{dim}'             # Unique ID = full path + dim
                    # Shorten "PM Performance" for display space.
                    short = dim.replace('PM Performance', 'PM Perf')

                    ids.append(d_id)
                    labels.append(short)
                    parents.append(p_id)
                    values.append(1)                   # Leaf nodes carry value=1
                    colors.append(_dim_color(score))   # Dimension-level color (0-3)
                    hovertexts.append(f'{dim}: {score}/3')

    # Build the Sunburst figure from the flattened hierarchy arrays.
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),          # Per-node coloring
        hovertext=hovertexts,
        hoverinfo='label+text',
        branchvalues='remainder',            # Non-leaf value = sum of children minus own value
        maxdepth=3,                          # Initially show 3 levels; click to drill deeper
    ))
    fig.update_layout(**get_plotly_theme(), title='Portfolio Hierarchy', height=650)
    return fig


# ============================================================================
# TREEMAP
# ============================================================================

def chart_treemap(df: pd.DataFrame) -> go.Figure:
    """Treemap of Region -> Area, sized by project count and colored by average
    Pulse Score.

    Each rectangle represents a Region/Area combination.  The *area* of each
    rectangle is proportional to the number of projects, and the *color*
    encodes the average Pulse Score using the continuous 4-color scale
    (red -> amber -> green -> dark green) with a midpoint of 16.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Region``, ``Area``, ``Project``, ``Total Score``.

    Returns
    -------
    go.Figure
        A 600px Plotly Express treemap.
    """
    # Aggregate: count unique projects and compute mean Pulse Score per
    # Region/Area combination.
    agg = (
        df.groupby(['Region', 'Area'])
        .agg(Projects=('Project', 'nunique'), AvgPulse=('Total Score', 'mean'))
        .reset_index()
    )
    # px.treemap handles the hierarchical path decomposition automatically.
    # ``color_continuous_scale`` and ``color_continuous_midpoint`` create a
    # diverging color map centered at 16 (the Green/Yellow boundary).
    fig = px.treemap(
        agg,
        path=['Region', 'Area'],                   # Two-level hierarchy
        values='Projects',                          # Rectangle size = project count
        color='AvgPulse',                           # Color = avg pulse score
        color_continuous_scale=CONTINUOUS_COLOR_SCALE,
        color_continuous_midpoint=COLOR_MIDPOINT,    # 16 = On Track threshold
    )
    fig.update_layout(**get_plotly_theme(), title='Region / Area Treemap', height=600)
    return fig


# ============================================================================
# SANKEY (Dimension -> Score Level -> Pulse Status)
# ============================================================================

def chart_sankey(df: pd.DataFrame) -> go.Figure:
    """Three-column Sankey diagram: Dimension -> Score Level -> Pulse Status.

    Visualizes the flow of dimension scores through intermediate risk levels
    to the final project health status.

    Node columns
    -------------
    1. **Left (sources):** The 8 scoring dimensions (Design, IX, ...).
    2. **Middle:** Three aggregated score levels:
       - Critical (score 0-1)
       - At Risk (score 2)
       - Healthy (score 3)
    3. **Right (sinks):** The 4 Pulse status tiers (Red, Yellow, Green, Dark Green).

    Link logic
    ----------
    * **Dimension -> Score Level:** For each non-null dimension value in the
      dataset, one unit of flow goes from that dimension's node to the
      corresponding score level.  Links are aggregated (summed) to avoid
      rendering thousands of individual flows.
    * **Score Level -> Pulse Status:** For each project, each of its dimension
      scores contributes one unit of flow from the appropriate score level to
      the project's overall Pulse Status.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 8 ``SCORE_DIMENSIONS`` columns and ``Total Score``.

    Returns
    -------
    go.Figure
        A 600px Sankey diagram.
    """
    # Define the three categories of intermediate score levels.
    score_levels = ['Critical (0-1)', 'At Risk (2)', 'Healthy (3)']
    statuses = STATUS_ORDER  # ['Red', 'Yellow', 'Green', 'Dark Green']

    # --- Build the flat node list ---
    # Nodes are indexed by position in this list.  The order is:
    #   [0..7] = 8 dimensions
    #   [8..10] = 3 score levels
    #   [11..14] = 4 pulse statuses
    nodes = list(SCORE_DIMENSIONS) + score_levels + statuses
    node_colors = (
        ['#0077B6'] * len(SCORE_DIMENSIONS) +                  # Dimensions: secondary blue
        ['#ef4444', '#f59e0b', '#22c55e'] +                    # Score levels: red/amber/green
        [STATUS_CONFIG[s]['color'] for s in statuses]           # Statuses: from config
    )

    # --- Build Dimension -> Score Level links ---
    # For each individual dimension value, create a link from dimension node
    # index (i) to the appropriate score level node index.
    source, target_idx, values = [], [], []
    for i, dim in enumerate(SCORE_DIMENSIONS):
        for score_val in df[dim].dropna():
            # Map dimension score to the score-level node index:
            #   0-1 -> Critical (index = len(SCORE_DIMENSIONS) + 0)
            #   2   -> At Risk  (index = len(SCORE_DIMENSIONS) + 1)
            #   3   -> Healthy  (index = len(SCORE_DIMENSIONS) + 2)
            if score_val <= 1:
                level_idx = len(SCORE_DIMENSIONS) + 0
            elif score_val == 2:
                level_idx = len(SCORE_DIMENSIONS) + 1
            else:
                level_idx = len(SCORE_DIMENSIONS) + 2
            source.append(i)
            target_idx.append(level_idx)
            values.append(1)      # Each individual score = 1 unit of flow

    # Aggregate links: sum flows that share the same (source, target) pair.
    # This reduces potentially thousands of individual links to at most
    # 8 dimensions * 3 levels = 24 aggregated links.
    link_counts = {}
    for s, t, v in zip(source, target_idx, values):
        key = (s, t)
        link_counts[key] = link_counts.get(key, 0) + v

    source_agg = [k[0] for k in link_counts]
    target_agg = [k[1] for k in link_counts]
    value_agg = list(link_counts.values())

    # --- Build Score Level -> Pulse Status links ---
    # For each project (row), determine its Pulse Status, then create a link
    # from each of its dimension score levels to that status.
    status_offset = len(SCORE_DIMENSIONS) + len(score_levels)  # First status node index
    level_status_counts = {}
    for _, row in df.iterrows():
        status = get_pulse_status(row['Total Score'])
        status_idx = status_offset + statuses.index(status)

        # Iterate over each dimension score for this project.
        dim_scores = [row[d] for d in SCORE_DIMENSIONS if pd.notna(row.get(d))]
        for s in dim_scores:
            # Same score-level bucketing as above.
            if s <= 1:
                level_idx = len(SCORE_DIMENSIONS) + 0
            elif s == 2:
                level_idx = len(SCORE_DIMENSIONS) + 1
            else:
                level_idx = len(SCORE_DIMENSIONS) + 2
            key = (level_idx, status_idx)
            level_status_counts[key] = level_status_counts.get(key, 0) + 1

    # Append the second layer of links to the aggregated arrays.
    source_agg.extend([k[0] for k in level_status_counts])
    target_agg.extend([k[1] for k in level_status_counts])
    value_agg.extend(list(level_status_counts.values()))

    # Build the Sankey figure.
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,       # Visual spacing and node bar width
            label=nodes,
            color=node_colors,
        ),
        link=dict(
            source=source_agg,
            target=target_agg,
            value=value_agg,
            color='rgba(255,255,255,0.08)',   # Very subtle semi-transparent links
        ),
    ))
    fig.update_layout(**get_plotly_theme(), title='Score Flow: Dimension → Level → Status', height=600)
    return fig


# ============================================================================
# ICICLE
# ============================================================================

def chart_icicle(df: pd.DataFrame) -> go.Figure:
    """Icicle chart: Region -> Area -> Pulse Status.

    An icicle chart is a rectangular variant of the sunburst.  It shows the
    same hierarchical data but laid out as nested horizontal bars instead of
    concentric rings.  Color encodes the ``Total Score`` using the continuous
    4-color scale.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Region``, ``Area``, ``Pulse_Status``, ``Total Score``.

    Returns
    -------
    go.Figure
        A 600px Plotly Express icicle chart.
    """
    fig = px.icicle(
        df,
        path=['Region', 'Area', 'Pulse_Status'],          # Three-level hierarchy
        color='Total Score',                                # Continuous color mapping
        color_continuous_scale=CONTINUOUS_COLOR_SCALE,
        color_continuous_midpoint=COLOR_MIDPOINT,           # Midpoint at 16
    )
    fig.update_layout(**get_plotly_theme(), title='Icicle: Region → Area → Status', height=600)
    return fig


# ============================================================================
# SPARKLINES BY REGION
# ============================================================================

def chart_sparklines(df: pd.DataFrame) -> go.Figure:
    """Small-multiple sparklines: one mini trend line per region, laid out
    side by side.

    Each subplot shows the weekly average Pulse Score for one region.  The line
    color is determined by the region's overall average score using the Pulse
    status color mapping (``get_pulse_color``), providing an at-a-glance
    health indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Region``, ``Year``, ``Wk``, ``Total Score``.

    Returns
    -------
    go.Figure
        A compact 250px-tall figure with ``n_regions`` side-by-side subplots.
        Returns an empty figure if no regions are found.
    """
    regions = sorted(df['Region'].dropna().unique())
    n_regions = len(regions)
    if n_regions == 0:
        fig = go.Figure()
        fig.update_layout(**get_plotly_theme(), title='No Regions')
        return fig

    # Create a 1-row x N-column subplot grid, one column per region.
    fig = make_subplots(
        rows=1, cols=n_regions,
        subplot_titles=regions,            # Region names as subplot headers
        horizontal_spacing=0.05,           # Tight horizontal gaps
    )

    for i, region in enumerate(regions, 1):    # enumerate from 1 for Plotly's 1-indexed subplots
        region_df = df[df['Region'] == region]
        # Aggregate to weekly average within this region.
        weekly = (
            region_df.groupby(['Year', 'Wk'])
            .agg(avg=('Total Score', 'mean'))
            .reset_index()
            .sort_values(['Year', 'Wk'])
        )
        # Color the line based on the region's overall average score.
        avg_score = weekly['avg'].mean()
        color = get_pulse_color(avg_score)

        fig.add_trace(
            go.Scatter(
                x=weekly['Wk'], y=weekly['avg'],
                mode='lines', line=dict(color=color, width=2),
                name=region, showlegend=False,    # Legend is redundant with subplot titles
            ),
            row=1, col=i,
        )

    fig.update_layout(
        **get_plotly_theme(),
        height=250,
        title='Pulse Trend by Region',
    )
    # Shared y-axis range (10-24) so sparklines are visually comparable.
    # Only the leftmost subplot shows y-tick labels to save space.
    for i in range(1, n_regions + 1):
        fig.update_yaxes(range=[10, 24], row=1, col=i, showticklabels=(i == 1), gridcolor='#1e293b')
        fig.update_xaxes(showticklabels=False, row=1, col=i, gridcolor='#1e293b')
    return fig


# ============================================================================
# 2x2 IMPACT-EFFORT MATRIX
# ============================================================================

def chart_impact_effort_matrix(df: pd.DataFrame, target: float) -> go.Figure:
    """2x2 quadrant scatter plot: Impact (gap to target) vs Effort (number of
    weak dimensions).

    This is a classic McKinsey prioritization matrix.  Projects are classified
    into four quadrants:

    * **Quick Wins** (top-left): high impact, low effort -- fix these first.
    * **Major Projects** (top-right): high impact, high effort -- plan carefully.
    * **Fill-ins** (bottom-left): low impact, low effort -- tackle when convenient.
    * **Deprioritize** (bottom-right): low impact, high effort -- defer.

    The quadrant boundaries are drawn at the median Impact and median Effort
    values (with a floor of 0.5 to avoid degenerate splits).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Total Score``, ``Effort`` (pre-computed column: count
        of dimensions scoring < 2), ``Project``, ``Region``.
    target : float
        The target score used to compute the impact gap.

    Returns
    -------
    go.Figure
        A 500px scatter plot colored by quadrant assignment.
    """
    plot_df = df.copy()
    # Impact = how far below target the project is (clamped to >= 0).
    plot_df['Impact'] = (target - plot_df['Total Score']).clip(lower=0)

    # Compute quadrant boundary thresholds using the median.
    # Floor of 0.5 prevents all points collapsing into one quadrant when
    # median is 0 (e.g. when most projects are above target).
    impact_mid = plot_df['Impact'].median() if len(plot_df) > 0 else 1
    effort_mid = plot_df['Effort'].median() if len(plot_df) > 0 else 1
    impact_mid = max(impact_mid, 0.5)
    effort_mid = max(effort_mid, 0.5)

    def quadrant(row):
        """Classify a single project into one of four quadrants."""
        hi = row['Impact'] >= impact_mid    # High impact?
        he = row['Effort'] >= effort_mid    # High effort?
        if hi and not he:
            return 'Quick Wins'         # High impact, low effort
        elif hi and he:
            return 'Major Projects'     # High impact, high effort
        elif not hi and not he:
            return 'Fill-ins'           # Low impact, low effort
        else:
            return 'Deprioritize'       # Low impact, high effort

    plot_df['Quadrant'] = plot_df.apply(quadrant, axis=1)

    # Distinct color per quadrant for clear visual separation.
    quad_colors = {
        'Quick Wins': '#22c55e',       # Green - do these first
        'Major Projects': '#f59e0b',   # Amber - needs planning
        'Fill-ins': '#0077B6',         # Blue - nice to have
        'Deprioritize': '#6D6E71',     # Gray - defer
    }

    fig = px.scatter(
        plot_df,
        x='Effort', y='Impact',
        color='Quadrant',
        color_discrete_map=quad_colors,
        hover_data=['Project', 'Region', 'Total Score'],
        title='Impact-Effort Prioritization Matrix',
    )

    # Draw dotted quadrant divider lines at the median thresholds.
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
    """Radar (spider/polar) chart showing all 8 dimension scores for a single
    project.

    The chart overlays the project's scores on a polar grid with a max radius
    of 3, making it easy to spot strengths (spikes outward) and weaknesses
    (dips inward) at a glance.

    Parameters
    ----------
    project_row : pd.Series
        A single row from the DataFrame representing one project.  Must contain
        the 8 ``SCORE_DIMENSIONS`` columns and ``Project`` (for the title).

    Returns
    -------
    go.Figure
        A 400px polar/radar chart with the project polygon and a dotted max-ring.
    """
    dims = SCORE_DIMENSIONS
    # Extract dimension scores; default to 0 if missing.
    values = [project_row.get(d, 0) for d in dims]
    values.append(values[0])       # Close the polygon by repeating the first value
    labels = dims + [dims[0]]      # Matching label list (also closed)

    fig = go.Figure()
    # --- Project polygon ---
    # A filled polar area trace with semi-transparent bright-blue fill.
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',                                      # Fill the enclosed area
        fillcolor=f'rgba(0, 191, 255, 0.15)',               # Transparent blue fill
        line=dict(color='#00BFFF', width=2),                # Solid blue outline
        name=project_row.get('Project', ''),
    ))

    # --- Max-possible ring ---
    # A dotted ring at r=3 (the maximum score) for reference.
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
            bgcolor='rgba(0,0,0,0)',         # Transparent polar background
            radialaxis=dict(
                range=[0, 3],                # Dimension scores are 0-3
                gridcolor='#1e293b',         # Subtle radial grid
                tickfont=dict(color='#94a3b8'),
            ),
            angularaxis=dict(
                gridcolor='#1e293b',         # Subtle angular grid
                tickfont=dict(color='#E0E0E0'),
            ),
        ),
        title=f"Dimension Scores: {project_row.get('Project', '')}",
        height=400,
    )
    return fig


# ============================================================================
# PROJECT TREND (Historical)
# ============================================================================

def chart_project_trend(df: pd.DataFrame, project_name: str, target: float) -> go.Figure:
    """Historical Pulse Score trend line for a single named project.

    Filters the dataset to one project and plots its ``Total Score`` over time
    with a target reference line.

    Parameters
    ----------
    df : pd.DataFrame
        Full multi-week dataset with ``Project``, ``Year``, ``Wk``,
        ``Year_Week``, ``Total Score``.
    project_name : str
        Exact project name to filter on.
    target : float
        Target score shown as a dashed horizontal reference line.

    Returns
    -------
    go.Figure
        A 350px line chart for the specified project.
    """
    # Filter to the specific project and sort chronologically.
    proj_df = df[df['Project'] == project_name].sort_values(['Year', 'Wk'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=proj_df['Year_Week'], y=proj_df['Total Score'],
        mode='lines+markers',
        line=dict(color='#00BFFF', width=2),
        marker=dict(size=6),
        name=project_name,
    ))
    # Horizontal target reference line.
    fig.add_hline(y=target, line_dash='dash', line_color='white',
                  annotation_text=f'Target ({target:.0f})')

    _apply_theme(fig)
    fig.update_layout(
        title=f'Trend: {project_name}',
        xaxis_title='Week',
        yaxis_title='Total Score',
        height=350,
    )
    fig.update_yaxes(range=[6, 24])    # Generous range to accommodate low scores
    return fig


# ============================================================================
# MINI SPARKLINE (inline SVG -- not a Plotly figure)
# ============================================================================

def mini_sparkline_svg(values, width: int = 70, height: int = 20,
                       color: str = "#60a5fa") -> str:
    """Generate a tiny inline SVG sparkline for embedding in HTML tables.

    Unlike the Plotly sparklines produced by ``chart_sparklines()``, this
    function returns a raw ``<svg>`` string that can be inserted directly into
    HTML table cells via ``st.markdown()``.  This is much lighter-weight than
    a full Plotly figure and is used in the project-detail tables.

    The sparkline is an SVG ``<polyline>`` connecting the data points, with a
    small ``<circle>`` dot on the final (most recent) value.

    Coordinate mapping
    ------------------
    * **X**: evenly spaced across the SVG width, with 2px padding on each side.
    * **Y**: linearly scaled between the min and max values in the series,
      with 3px padding top/bottom.  Y is inverted (SVG y=0 is the top) so
      higher values appear higher visually.

    Parameters
    ----------
    values : iterable of float
        The time series values (NaN values are filtered out).
    width : int, default 70
        SVG viewport width in pixels.
    height : int, default 20
        SVG viewport height in pixels.
    color : str, default "#60a5fa"
        Stroke/fill color for the polyline and endpoint dot.

    Returns
    -------
    str
        An inline ``<svg>`` element string, or ``""`` if fewer than 2 non-null
        values are provided.
    """
    # Filter out NaN values to get a clean numeric series.
    vals = [v for v in values if pd.notna(v)]
    if len(vals) < 2:
        return ""       # Cannot draw a line with fewer than 2 points

    # Compute value range for Y-axis scaling.
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1     # Avoid division by zero for flat lines

    # Map each value to (x, y) pixel coordinates within the SVG viewport.
    points = []
    for i, v in enumerate(vals):
        # X: evenly spaced with 2px margin on each side.
        x = (i / (len(vals) - 1)) * (width - 4) + 2
        # Y: inverted (subtract from height) with 3px margin top and bottom.
        # This maps the max value to y=3 (near top) and min value to
        # y=height-3 (near bottom).
        y = height - 3 - ((v - mn) / rng) * (height - 6)
        points.append(f"{x},{y}")

    # Build the SVG polyline "points" attribute: "x1,y1 x2,y2 x3,y3 ..."
    polyline = " ".join(points)

    # Extract the coordinates of the last point for the endpoint dot.
    last_x, last_y = points[-1].split(",")

    # Return the complete inline SVG element.
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
    """Stacked ranked-flow chart showing dimension rankings week over week.

    This is the most visually complex chart in the library.  It renders each
    scoring dimension as a horizontal band whose height equals its average
    score for that week.  Bands are stacked in rank order (lowest-scoring
    dimension at the bottom) and smoothly interpolated between weeks using
    cubic splines (if scipy is available) or linear interpolation (fallback).

    The result resembles a streamgraph / stacked-area chart, but the vertical
    ordering can change from week to week as dimensions swap rank positions.

    Rendering approach
    ------------------
    1. **Aggregate**: Group by (Year, Wk), compute mean score per dimension.
    2. **Rank-stack**: For each week, sort dimensions by score (ascending).
       Build cumulative y-boundaries (y0, y1) for each dimension's band.
    3. **Interpolate**: Use scipy's ``make_interp_spline`` (cubic, k=3) to
       create 300-point smooth curves for each band's upper and lower
       boundaries.  Falls back to numpy ``interp`` if scipy is unavailable.
    4. **Draw filled bands**: Each band is a filled ``go.Scatter`` polygon
       trace (upper boundary concatenated with reversed lower boundary).
    5. **Overlay text markers**: Near-invisible square markers at band
       midpoints carry text labels showing the dimension's avg score that
       week, plus hover info with customdata.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Year``, ``Wk``, and the 8 ``SCORE_DIMENSIONS`` columns.
    score_cols : list of str, optional
        Override the dimension columns to use.  Defaults to ``SCORE_DIMENSIONS``.
    dim_colors : dict, optional
        Override the dimension -> hex color mapping.  Defaults to
        ``DIMENSION_COLORS`` from ``utils/styles.py``.

    Returns
    -------
    go.Figure
        A 450px stacked-flow chart.  Returns an empty figure with a message if
        fewer than 2 weeks of data are available.
    """
    # Use defaults from the styles module if not overridden.
    if score_cols is None:
        score_cols = SCORE_DIMENSIONS
    if dim_colors is None:
        dim_colors = DIMENSION_COLORS

    # --- Step 1: Aggregate weekly averages ---
    # Group by (Year, Wk) to handle year boundaries correctly (e.g. 2025-W52
    # followed by 2026-W01).  Sort chronologically.
    weekly = (
        df.groupby(['Year', 'Wk'])[score_cols]
        .mean()
        .reset_index()
        .sort_values(['Year', 'Wk'])
    )

    if len(weekly) < 2:
        # Need at least 2 weeks to draw meaningful bands.
        fig = go.Figure()
        fig.update_layout(**get_plotly_theme(),
                          title='Pulse Ranking (need >= 2 weeks of data)')
        return fig

    # --- Build sequential x-axis ---
    # Use integer indices for spline interpolation, then map back to labels
    # for the tick marks.
    n = len(weekly)
    x_idx = np.arange(n)

    # Short tick labels: "25-W26" (2-digit year).
    tick_labels = [
        f"{int(row['Year']) % 100}-W{int(row['Wk']):02d}"
        for _, row in weekly.iterrows()
    ]
    # Full Year_Week strings for hover customdata: "2025-W26".
    yw_labels = [
        f"{int(row['Year'])}-W{int(row['Wk']):02d}"
        for _, row in weekly.iterrows()
    ]

    # --- Step 2: Build rank-ordered stacks for each week ---
    # For each week, sort dimensions by their average score (ascending) and
    # compute cumulative y-boundaries so the lowest-scoring dimension sits
    # at the bottom of the stack.
    week_stacks = []
    for _, row in weekly.iterrows():
        vals = {c: round(row[c], 2) for c in score_cols}
        sorted_dims = sorted(vals.items(), key=lambda x: x[1])  # Sort by score, ascending
        cum_y = 0     # Running cumulative y position
        stack = {}
        for dim, val in sorted_dims:
            stack[dim] = {
                'y0': cum_y,           # Lower boundary of this band
                'y1': cum_y + val,     # Upper boundary of this band
                'val': val,            # Raw score value (for labels)
            }
            cum_y += val               # Advance for the next band
        week_stacks.append(stack)

    # --- Step 3: Set up smooth interpolation ---
    # Attempt to import scipy for cubic spline interpolation.  If unavailable,
    # fall back to numpy linear interpolation with 10x oversampling.
    try:
        from scipy.interpolate import make_interp_spline
        k = min(3, n - 1)                    # Spline degree: cubic or lower if too few points
        x_smooth = np.linspace(0, n - 1, 300)  # 300-point smooth curve
        use_spline = True
    except ImportError:
        x_smooth = np.linspace(0, n - 1, n * 10)  # Linear fallback: 10x oversampled
        use_spline = False

    fig = go.Figure()

    # --- Step 4: Draw filled bands ---
    # Process dimensions in order of their overall average (lowest first =
    # drawn first = rendered at the bottom).
    avg_vals = {c: weekly[c].mean() for c in score_cols}
    draw_order = sorted(avg_vals, key=lambda x: avg_vals[x])

    for dim in draw_order:
        # Extract the raw y-boundary arrays for this dimension across all weeks.
        y0_pts = [week_stacks[i][dim]['y0'] for i in range(n)]
        y1_pts = [week_stacks[i][dim]['y1'] for i in range(n)]
        val_pts = [week_stacks[i][dim]['val'] for i in range(n)]

        # Interpolate the boundary curves to create smooth transitions.
        if use_spline:
            spl_y0 = make_interp_spline(x_idx, y0_pts, k=k)
            spl_y1 = make_interp_spline(x_idx, y1_pts, k=k)
            y0_sm = spl_y0(x_smooth)
            y1_sm = spl_y1(x_smooth)
        else:
            y0_sm = np.interp(x_smooth, x_idx, y0_pts)
            y1_sm = np.interp(x_smooth, x_idx, y1_pts)

        # --- Filled band trace ---
        # Create a closed polygon by concatenating the upper boundary (left to
        # right) with the lower boundary (right to left, reversed).  The
        # ``fill='toself'`` directive fills the enclosed polygon.
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_smooth, x_smooth[::-1]]),
            y=np.concatenate([y1_sm, y0_sm[::-1]]),
            fill='toself',
            fillcolor=dim_colors.get(dim, '#666'),           # Dimension's unique color
            line=dict(color='rgba(255,255,255,0.25)', width=0.5),  # Subtle white border
            name=dim,
            showlegend=True,
            hoverinfo='skip',            # Hover is handled by the marker trace below
        ))

        # --- Step 5: Overlay text markers at band midpoints ---
        # Nearly invisible square markers (opacity=0.01) carry the value text
        # and provide hover targets with customdata for drill-down.
        mid_pts = [(y0_pts[i] + y1_pts[i]) / 2 for i in range(n)]
        fig.add_trace(go.Scatter(
            x=list(x_idx),
            y=mid_pts,
            mode='markers+text',
            marker=dict(
                size=22,
                color=dim_colors.get(dim, '#666'),
                opacity=0.01,            # Nearly invisible -- just a hover hit target
                symbol='square',
            ),
            text=[f"{v:.2f}" for v in val_pts],     # Score labels at band centers
            textfont=dict(size=8, color='white'),
            textposition='middle center',
            name=dim,
            showlegend=False,
            # customdata carries [dimension_name, year_week_label, score_value]
            # for each point, enabling downstream click/hover handlers.
            customdata=[[dim, yw_labels[i], v]
                        for i, v in enumerate(val_pts)],
            hovertemplate=(
                f"<b>{dim}</b><br>"
                "%{customdata[1]}<br>"
                "Avg: %{customdata[2]:.2f}<extra></extra>"
            ),
        ))

    # --- Final layout ---
    _apply_theme(fig)
    fig.update_layout(
        xaxis=dict(
            title='Week',
            tickmode='array',                        # Explicit tick positions
            tickvals=list(x_idx),
            ticktext=tick_labels,                    # "25-W26", etc.
            range=[-0.5, n - 0.5],                  # Slight padding on edges
            gridcolor='rgba(255,255,255,0.03)',      # Very faint vertical grid
            tickangle=-90,                           # Vertical tick labels
        ),
        yaxis=dict(
            range=[0, 20],                           # Max stack height ~24 but typically ~16-20
            dtick=10,                                # Grid lines every 10 units
            gridcolor='rgba(255,255,255,0.05)',
            griddash='dot',
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            font=dict(size=10, color='#CBD5E1'),
        ),
        height=450,
        margin=dict(l=35, r=15, t=60, b=70),        # Extra bottom margin for rotated labels
        hovermode='closest',
    )
    return fig


# ============================================================================
# DRILL-DOWN: Dimension Bar Chart by Region
# ============================================================================

def chart_dimension_by_region(df: pd.DataFrame, dimension: str,
                              dim_color: str = '#2563eb') -> go.Figure:
    """Horizontal bar chart showing the average score for a single dimension,
    broken down by region.

    Used in the dimension drill-down panel to compare how different regions
    perform on a specific dimension.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Region`` and the column named by *dimension*.
    dimension : str
        The dimension column name (e.g. 'Design', 'CSAT').
    dim_color : str, default '#2563eb'
        Bar fill color.  Typically overridden by the caller with the
        dimension's color from ``DIMENSION_COLORS``.

    Returns
    -------
    go.Figure
        A compact 220px bar chart sorted ascending by average score.
    """
    # Compute the mean dimension score per region.
    region_summary = (
        df.groupby('Region')
        .agg(Avg=(dimension, 'mean'))
        .reset_index()
        .sort_values('Avg')           # Sorted ascending so weakest region is leftmost
    )

    fig = go.Figure(go.Bar(
        x=region_summary['Region'],
        y=region_summary['Avg'],
        marker_color=dim_color,
        text=[f"{v:.2f}" for v in region_summary['Avg']],   # Value labels above bars
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=11),
    ))
    _apply_theme(fig)
    fig.update_layout(height=220, margin=dict(l=30, r=15, t=10, b=30))
    fig.update_yaxes(range=[0, 3.2])   # Dimension scores are 0-3; 3.2 gives label headroom
    return fig


# ============================================================================
# DRILL-DOWN: Dimension Score Distribution
# ============================================================================

def chart_dimension_distribution(df: pd.DataFrame,
                                 dimension: str) -> go.Figure:
    """Bar chart showing the score distribution (0, 1, 2, 3) for a single
    dimension across all projects.

    Each bar represents a discrete score level, colored by a traffic-light
    scheme: 0=red, 1=orange, 2=blue, 3=green.  The chart reveals how many
    projects scored at each level, helping identify systemic weaknesses.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the column named by *dimension*.
    dimension : str
        The dimension column name (e.g. 'RF Opt', 'PM Performance').

    Returns
    -------
    go.Figure
        A compact 220px bar chart with one bar per score level (0-3).
    """
    # Count occurrences of each score value, sorted by score.
    dist = df[dimension].dropna().astype(int).value_counts().sort_index()

    # Color mapping: each discrete score level gets a distinct color.
    dist_colors = {0: '#ef4444', 1: '#f97316', 2: '#3b82f6', 3: '#22c55e'}

    fig = go.Figure(go.Bar(
        x=[f"Score {int(s)}" for s in dist.index],     # "Score 0", "Score 1", etc.
        y=dist.values,
        marker_color=[dist_colors.get(int(s), '#64748b') for s in dist.index],
        text=dist.values,                               # Count labels above each bar
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
    """Multi-line chart: weekly average of a single dimension, one line per region.

    Each region gets a distinctly colored line (from ``REGION_LINE_COLORS``)
    so viewers can compare regional trends for a specific dimension over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ``Wk``, ``Region``, and the column named by *dimension*.
    dimension : str
        The dimension column name (e.g. 'Field', 'Potential').
    regions : list of str, optional
        Specific regions to include.  Defaults to all unique regions in the data.

    Returns
    -------
    go.Figure
        A compact 220px multi-line trend chart.
    """
    if regions is None:
        regions = sorted(df['Region'].dropna().unique())

    # Compute the weekly average of this dimension per region.
    trend = df.groupby(['Wk', 'Region'])[dimension].mean().reset_index()
    fig = go.Figure()

    for region in regions:
        rd = trend[trend['Region'] == region].sort_values('Wk')
        if len(rd) > 0:
            fig.add_trace(go.Scatter(
                x=rd['Wk'], y=rd[dimension],
                mode='lines+markers',
                name=region,
                # Use the region's assigned color; fall back to muted gray.
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
    fig.update_yaxes(range=[0, 3.2])   # Dimension scores are 0-3
    return fig
