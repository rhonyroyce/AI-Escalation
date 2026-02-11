"""
Advanced financial visualizations for dashboards and reports.

This module provides all Plotly-based interactive financial charts used by
the Streamlit dashboard (streamlit_app.py). Unlike the matplotlib-based charts
in visualization/chart_generator.py and visualization/advanced_charts.py
(which produce static PNGs for the Excel report), these Plotly charts are
designed for interactive web-based exploration with hover tooltips, drill-down
capabilities, and responsive layouts.

Architecture Overview:
    - All functions are stateless (no class) and return go.Figure or pd.DataFrame
    - Each function accepts either a metrics dict/object or a DataFrame
    - Charts use the 'plotly_white' template for clean, professional appearance
    - Dark-themed charts (ROI opportunity) support dashboard dark mode

Integration Points:
    - Called by: escalation_ai/dashboard/streamlit_app.py (Financial Analysis page)
    - Data from: escalation_ai/financial/metrics.py (FinancialMetrics calculations)
    - Also called by: Pulse dashboard (via imports in ML_DigitalTwins_App_Full)

Chart Functions (in order of appearance):

    1. create_financial_waterfall() - Waterfall: Base Cost -> Risk Adjustments -> Total
       Uses go.Waterfall with relative/total measures

    2. create_roi_opportunity_chart() - Bubble scatter: Investment vs Annual Savings
       Break-even diagonal line, ROI-sized bubbles, dark theme

    3. create_cost_avoidance_breakdown() - Sunburst: hierarchical cost avoidance
       Categories: Recurring, Preventable, Knowledge Sharing, Automation

    4. create_cost_trend_forecast() - Line+markers with trend extrapolation
       Monthly aggregation, linear trend, 6-month forecast with confidence band

    5. create_efficiency_scorecard() - Horizontal gauge bars (0-100 scale)
       Red-Orange-Yellow-Green colorscale for Cost Efficiency, Cost/Hour, Prevention Rate

    6. create_category_cost_comparison() - Side-by-side subplots
       Total cost and average cost per ticket by AI_Category

    7. create_financial_kpi_cards() - Returns list of KPI dicts (not a chart)
       6 cards: Total Cost, Avg Cost/Ticket, ROI Opportunity, Efficiency Score,
       Revenue at Risk, Cost Avoidance

    8. create_cost_concentration_chart() - Pareto chart (defined TWICE, see note)
       Bar + cumulative line with 80% reference, dual y-axis

    9. create_financial_forecast_chart() - Forecast with risk scenarios
       Confidence interval band, horizontal lines for best/expected/worst case

    10. create_engineer_cost_efficiency_matrix() - Quadrant scatter
        4 quadrants: High Efficiency, Low Cost Slow, Fast High Cost, Needs Improvement

    11. create_insights_table() - Formats insights list into display DataFrame

    12. create_subcategory_financial_breakdown() - Grouped horizontal bars
        Category-colored bars at the sub-category level

    13. create_subcategory_cost_treemap() - Treemap with category/subcategory hierarchy
        Click-to-drill-down, RdYlGn_r colorscale by cost

    14. create_category_financial_summary_table() - Summary DataFrame
        Category x Sub-Category aggregation with currency formatting

Note: create_cost_concentration_chart is defined TWICE in this file (lines ~531
and ~766). The second definition silently overwrites the first. Both implementations
are identical Pareto charts. This is likely a copy-paste artifact.

Provides comprehensive financial charts including:
- ROI waterfall charts
- Cost trend analysis
- Efficiency scorecards
- Financial forecast charts
- Cost avoidance breakdowns
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def _create_empty_forecast_figure(message: str = "No data available") -> go.Figure:
    """
    Create an empty Plotly figure with a centered message.

    Used as a fallback when forecast data is unavailable or invalid.
    Displays a gray text annotation in the center of a blank chart area.

    Args:
        message: Text to display in the empty chart (e.g., "No date column found")

    Returns:
        go.Figure with centered annotation text, 500px height, plotly_white template
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title="Cost Trend & Forecast",
        height=500,
        template="plotly_white"
    )
    return fig


def create_financial_waterfall(metrics_dict: Dict) -> go.Figure:
    """
    Create a waterfall chart showing financial impact breakdown.

    Visualizes how different cost components build up from base cost to total
    impact. Uses Plotly's native go.Waterfall trace type which automatically
    handles floating bar positioning.

    Waterfall Flow:
        Base Cost -> Recurring Issues -> Preventable -> Customer Impact -> SLA Risk -> Total Impact

    Measure Types:
        - 'relative': Each bar is added to/subtracted from the running total
        - 'total': Shows the absolute running total (used for the final bar)

    Color Convention:
        - Red (#d62728): Cost increases
        - Green (#2ca02c): Cost decreases/savings
        - Blue (#1f77b4): Total/summary bars

    Customer Impact Calculation:
        customer = customer_impact_cost - base_cost * 0.5
        This computes the incremental customer impact above the baseline 50%
        (assumes half of base cost is already customer-facing).

    Args:
        metrics_dict: Dict from FinancialMetrics with keys:
            total_cost, recurring_issue_cost, preventable_cost,
            customer_impact_cost, sla_penalty_exposure

    Returns:
        go.Figure with waterfall chart, 500px height
    """
    # Build waterfall stage labels
    x_labels = [
        'Base Cost',
        'Recurring Issues',
        'Preventable',
        'Customer Impact',
        'SLA Risk',
        'Total Impact'
    ]

    # Plotly waterfall measure types: 'relative' adds incrementally,
    # 'total' shows the absolute running sum
    measures = ['relative', 'relative', 'relative', 'relative', 'relative', 'total']

    # Extract cost components from the metrics dictionary
    base_cost = metrics_dict.get('total_cost', 0)
    recurring = metrics_dict.get('recurring_issue_cost', 0)
    preventable = metrics_dict.get('preventable_cost', 0)
    # Customer impact is incremental: subtract half the base cost that is already
    # accounted for in the base cost figure
    customer = metrics_dict.get('customer_impact_cost', 0) - base_cost * 0.5
    sla = metrics_dict.get('sla_penalty_exposure', 0)

    # Y-values for each waterfall stage
    y_values = [
        base_cost,
        recurring,
        preventable,
        customer,
        sla,
        base_cost + recurring + preventable + customer + sla  # Total (absolute)
    ]

    fig = go.Figure(go.Waterfall(
        name="Financial Impact",
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=[f"${v:,.0f}" for v in y_values],  # Dollar-formatted labels
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},  # Gray connector lines
        increasing={"marker": {"color": "#d62728"}},  # Red for cost increases
        decreasing={"marker": {"color": "#2ca02c"}},  # Green for savings
        totals={"marker": {"color": "#1f77b4"}}       # Blue for total bars
    ))

    fig.update_layout(
        title="Financial Impact Waterfall Analysis",
        showlegend=False,
        height=500,
        yaxis_title="Cost ($)",
        xaxis_title="",
        template="plotly_white"
    )

    return fig


def create_roi_opportunity_chart(roi_data: Dict) -> go.Figure:
    """
    Create ROI opportunity visualization showing investment vs annual savings.

    Plots the top 5 cost reduction opportunities as bubbles on an
    investment-vs-savings scatter chart. Points above the break-even diagonal
    represent positive-ROI opportunities (savings exceed investment).

    Visual Elements:
        - Bubbles: Sized by ROI percentage, colored by ROI on Greens colorscale
        - Break-even line: Gray dashed diagonal (investment == savings)
        - Annotations: Category labels with dark background for readability
        - Dark theme: Transparent background for dashboard dark mode integration

    Bubble Size Calculation:
        size = max(30, min(roi_pct / 20, 80))
        This clamps bubble diameter between 30px and 80px.

    Args:
        roi_data: Dict with 'top_opportunities' list, each containing:
            category, investment_required, annual_savings, roi_percentage, incident_count

    Returns:
        go.Figure with bubble scatter chart, or empty figure if no data
    """
    if not roi_data.get('top_opportunities'):
        return go.Figure()

    opportunities = roi_data['top_opportunities'][:5]  # Top 5 only

    # Extract arrays from the opportunities list
    categories = [opp['category'] for opp in opportunities]
    investments = [opp['investment_required'] for opp in opportunities]
    savings = [opp['annual_savings'] for opp in opportunities]
    roi_pcts = [opp['roi_percentage'] for opp in opportunities]
    incidents = [opp['incident_count'] for opp in opportunities]

    fig = go.Figure()

    # Add opportunity bubbles with rich hover info.
    # No text labels on the chart itself to avoid overlap -- labels are
    # added as separate annotations below.
    fig.add_trace(
        go.Scatter(
            x=investments,
            y=savings,
            mode='markers',
            marker=dict(
                # Bubble size proportional to ROI%, clamped to [30, 80] px range
                size=[max(30, min(roi / 20, 80)) for roi in roi_pcts],
                color=roi_pcts,
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title="ROI %"),
                line=dict(width=2, color='white')  # White border for contrast
            ),
            text=categories,
            customdata=list(zip(categories, roi_pcts, incidents)),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Investment: $%{x:,.0f}<br>'
                'Annual Savings: $%{y:,.0f}<br>'
                'ROI: %{customdata[1]:.0f}%<br>'
                'Incidents: %{customdata[2]}<extra></extra>'
            ),
            name="Opportunities"
        )
    )

    # Break-even diagonal line: where investment equals savings.
    # Points above this line have positive ROI.
    max_val = max(max(investments), max(savings)) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            name='Break-even',
            showlegend=False,
            hoverinfo='skip'
        )
    )

    # Add text annotations for each bubble, positioned to minimize overlap.
    # Alternating ax/ay offsets push labels to different sides.
    positions = ['top right', 'top left', 'bottom right', 'bottom left', 'top center']
    for i, (cat, x, y) in enumerate(zip(categories, investments, savings)):
        # Truncate long category names to prevent layout issues
        short_cat = cat[:20] + '...' if len(cat) > 20 else cat
        fig.add_annotation(
            x=x, y=y,
            text=f"<b>{short_cat}</b>",
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='#666',
            ax=30 if i % 2 == 0 else -30,     # Alternate left/right offset
            ay=-30 if i < 2 else 30,            # Alternate up/down offset
            font=dict(size=10, color='#E0E0E0'),
            bgcolor='rgba(0,0,0,0.7)',          # Dark background for readability
            borderpad=3
        )

    # Dark-themed layout for dashboard integration
    fig.update_layout(
        title=dict(text='Investment vs Annual Savings', font=dict(size=16)),
        xaxis_title="Investment Required ($)",
        yaxis_title="Annual Savings ($)",
        showlegend=False,
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',    # Transparent paper background
        plot_bgcolor='rgba(0,0,0,0)',      # Transparent plot background
        font=dict(family='Inter', color='#E0E0E0')  # Light text for dark mode
    )

    # Format axes as currency with subtle grid lines
    fig.update_xaxes(tickprefix='$', tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(tickprefix='$', tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)')

    return fig


def create_cost_avoidance_breakdown(avoidance_data: Dict) -> go.Figure:
    """
    Create sunburst chart showing cost avoidance opportunities.

    A hierarchical sunburst with "Total Avoidance" at the center and four
    avoidance categories as outer segments. Each segment's size is proportional
    to its savings potential. Hovering shows dollar amounts.

    Hierarchy:
        Total Avoidance (center)
            -> Recurring Issues
            -> Preventable Categories
            -> Knowledge Sharing
            -> Automation

    Only categories with positive values are included (zero-value categories
    are filtered out to avoid empty segments).

    Args:
        avoidance_data: Dict with keys:
            total_avoidance, recurring_issues, preventable_categories,
            knowledge_sharing, automation

    Returns:
        go.Figure with sunburst chart, 500px height
    """
    # Build the hierarchical data arrays for go.Sunburst.
    # labels/parents/values define the tree structure.
    labels = ['Total Avoidance']
    parents = ['']  # Root node has empty parent
    values = [avoidance_data['total_avoidance']]
    colors = ['#2ca02c']  # Green for the root node

    # Add category segments as children of "Total Avoidance"
    categories = [
        ('Recurring Issues', avoidance_data['recurring_issues']),
        ('Preventable Categories', avoidance_data['preventable_categories']),
        ('Knowledge Sharing', avoidance_data['knowledge_sharing']),
        ('Automation', avoidance_data['automation'])
    ]

    for cat_name, cat_value in categories:
        if cat_value > 0:  # Only include categories with positive savings
            labels.append(cat_name)
            parents.append('Total Avoidance')
            values.append(cat_value)
            colors.append(None)  # Let Plotly auto-color from Greens scale

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",  # Values represent totals, not just the leaf contribution
        marker=dict(
            colors=colors,
            colorscale='Greens'
        ),
        hovertemplate='<b>%{label}</b><br>Potential Savings: $%{value:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Cost Avoidance Opportunities: ${avoidance_data['total_avoidance']:,.0f}",
        height=500,
        template="plotly_white"
    )

    return fig


def create_cost_trend_forecast(df: pd.DataFrame, forecasts: Dict) -> go.Figure:
    """
    Create cost trend chart with forecast projection.

    Combines three visual layers:
        1. Historical line: Actual monthly costs from the data
        2. Trend line: Linear regression fit through historical data
        3. Forecast projection: 6-month forward projection with confidence interval

    Data Processing:
        - Tries multiple date column names to find a valid datetime source
        - Aggregates Financial_Impact by month using pd.to_period('M')
        - Linear trend via np.polyfit(degree=1) requires >= 3 data points

    Forecast Band:
        The confidence interval is shown as a filled area between upper_bound
        and lower_bound from the forecasts dict. The fill uses Plotly's 'toself'
        pattern with reversed x-coordinates to create a closed polygon.

    Args:
        df: Escalation DataFrame with Financial_Impact and a date column
        forecasts: Dict from FinancialMetrics.forecast with keys:
            monthly_projection (list of {month, projected_cost, upper_bound, lower_bound}),
            trend (str: 'increasing', 'decreasing', 'stable')

    Returns:
        go.Figure with line chart, or empty forecast figure on error
    """
    # Verify required column exists
    if 'Financial_Impact' not in df.columns:
        logger.warning("Missing Financial_Impact column for trend forecast")
        return _create_empty_forecast_figure("Missing financial data")

    try:
        df_temp = df.copy()

        # Try multiple date column names to find a valid datetime source.
        # Different data sources may use different column naming conventions.
        date_col = None
        for col in ['tickets_data_issue_datetime', 'Issue_Date', 'Issue Date',
                    'Created_Date', 'Date', 'Timestamp', 'tickets_data_resolution_datetime']:
            if col in df_temp.columns:
                date_col = col
                break

        if not date_col:
            return _create_empty_forecast_figure("No date column found")

        # Parse dates and filter out invalid entries
        df_temp['Issue_Date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) == 0:
            return _create_empty_forecast_figure("No valid dates in data")

        # Aggregate Financial_Impact by month
        df_temp['Month'] = df_temp['Issue_Date'].dt.to_period('M')
        monthly_costs = df_temp.groupby('Month')['Financial_Impact'].sum()

        if len(monthly_costs) == 0:
            return _create_empty_forecast_figure("No monthly data available")

        # Convert Period index to timestamps for Plotly compatibility
        dates = [pd.Period(m).to_timestamp() for m in monthly_costs.index]
        values = monthly_costs.values

        fig = go.Figure()

        # Layer 1: Historical actual cost line with markers
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Actual Cost',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))

        # Layer 2: Linear trend line (requires at least 3 data points)
        if len(dates) >= 3:
            # Linear regression: fit a degree-1 polynomial to the time series
            z = np.polyfit(range(len(values)), values, 1)
            p = np.poly1d(z)
            trend_values = p(range(len(values)))

            fig.add_trace(go.Scatter(
                x=dates,
                y=trend_values,
                mode='lines',
                name='Trend',
                line=dict(color='orange', width=2, dash='dash')
            ))

        # Layer 3: Forward forecast projection (if available in forecasts dict)
        if forecasts.get('monthly_projection'):
            last_date = max(dates)
            # Generate 6 future month timestamps
            forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
            forecast_values = [proj['projected_cost'] for proj in forecasts['monthly_projection'][:6]]
            upper_bounds = [proj['upper_bound'] for proj in forecasts['monthly_projection'][:6]]
            lower_bounds = [proj['lower_bound'] for proj in forecasts['monthly_projection'][:6]]

            # Forecast center line (dotted with diamond markers)
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#2ca02c', width=2, dash='dot'),
                marker=dict(size=6, symbol='diamond')
            ))

            # Confidence interval band (filled polygon).
            # Plotly 'toself' fill requires x-coordinates to trace forward then backward,
            # with y-coordinates being upper bounds then reversed lower bounds.
            fig.add_trace(go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(44, 160, 44, 0.2)',  # Semi-transparent green
                line=dict(color='rgba(255,255,255,0)'),  # Invisible border
                name='Forecast Range',
                showlegend=True
            ))

        fig.update_layout(
            title=f"Cost Trend & Forecast ({forecasts.get('trend', 'stable').title()} trend)",
            xaxis_title="Month",
            yaxis_title="Cost ($)",
            height=500,
            template="plotly_white",
            hovermode='x unified'  # Show all traces at same x on hover
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating cost trend chart: {e}")
        return go.Figure()


def create_efficiency_scorecard(efficiency_data: Dict, metrics) -> go.Figure:
    """
    Create efficiency scorecard with horizontal gauge-style bars.

    Displays three key efficiency metrics as horizontal bars on a 0-100 scale,
    color-coded using a Red-Orange-Yellow-Green gradient to indicate health.

    Metrics Displayed:
        1. Cost Efficiency: Direct from metrics.cost_efficiency_score
        2. Cost per Hour: Inverse mapping from cost_per_resolution_hour
           Formula: min(100, 1000 / cost_per_hour) -- lower cost = higher score
        3. Prevention Rate: Percentage of costs that are NOT preventable
           Formula: (1 - preventable_cost / total_cost) * 100

    Color Scale Thresholds:
        - Red: 0-40
        - Orange: 40-60
        - Yellow: 60-80
        - Green: 80-100

    Args:
        efficiency_data: Dict (currently unused, metrics object provides data)
        metrics: FinancialMetrics object with attributes:
            cost_efficiency_score, cost_per_resolution_hour,
            preventable_cost, total_cost

    Returns:
        go.Figure with horizontal bar chart, 400px height
    """
    fig = go.Figure()

    # Define the three scorecard metrics: (name, value, max_value)
    scorecard_metrics = [
        ('Cost Efficiency', metrics.cost_efficiency_score, 100),
        ('Cost per Hour', min(100, 1000 / metrics.cost_per_resolution_hour) if metrics.cost_per_resolution_hour > 0 else 0, 100),
        ('Prevention Rate', min(100, (1 - metrics.preventable_cost / metrics.total_cost) * 100) if metrics.total_cost > 0 else 0, 100),
    ]

    categories = [m[0] for m in scorecard_metrics]
    values = [m[1] for m in scorecard_metrics]
    max_vals = [m[2] for m in scorecard_metrics]

    # Create horizontal gauge-style bars with continuous color mapping
    fig.add_trace(go.Bar(
        y=categories,
        x=values,
        orientation='h',
        text=[f"{v:.0f}" for v in values],
        textposition='outside',
        marker=dict(
            color=values,
            colorscale=[
                [0, '#d62728'],      # Red (0-40)
                [0.4, '#ff7f0e'],    # Orange (40-60)
                [0.6, '#ffdd57'],    # Yellow (60-80)
                [0.8, '#2ca02c']     # Green (80-100)
            ],
            cmin=0,
            cmax=100
        ),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.0f}/100<extra></extra>'
    ))

    fig.update_layout(
        title="Cost Efficiency Scorecard",
        xaxis=dict(range=[0, 110], title="Score (0-100)"),  # Extra space for text labels
        yaxis_title="",
        height=400,
        template="plotly_white",
        showlegend=False
    )

    return fig


def create_category_cost_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Create side-by-side comparison of cost metrics by category.

    Uses Plotly make_subplots to show two bar charts side by side:
        - Left panel: Total cost per category (sorted descending)
        - Right panel: Average cost per ticket per category (same sort order)

    This dual view helps distinguish between categories that are expensive
    because of volume (many tickets) vs. categories where individual tickets
    are expensive (high per-ticket cost).

    Data Aggregation:
        Groups by AI_Category, computes sum, mean, count of Financial_Impact.
        Sorted by total cost descending.

    Args:
        df: Escalation DataFrame with AI_Category and Financial_Impact columns

    Returns:
        go.Figure with two subplot panels, or empty figure if columns missing
    """
    if 'AI_Category' not in df.columns or 'Financial_Impact' not in df.columns:
        return go.Figure()

    # Aggregate financial metrics by category
    category_stats = df.groupby('AI_Category').agg({
        'Financial_Impact': ['sum', 'mean', 'count']
    }).round(2)

    category_stats.columns = ['Total_Cost', 'Avg_Cost', 'Count']
    category_stats = category_stats.sort_values('Total_Cost', ascending=False)

    # Create 1x2 subplot layout with shared x-axis categories
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Cost by Category', 'Average Cost per Ticket'),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.15
    )

    # Left panel: Total cost per category (blue bars)
    fig.add_trace(
        go.Bar(
            x=category_stats.index,
            y=category_stats['Total_Cost'],
            marker_color='#1f77b4',
            text=[f"${v:,.0f}" for v in category_stats['Total_Cost']],
            textposition='outside',
            textangle=0,
            name='Total Cost',
            cliponaxis=False  # Allow text labels to extend beyond plot area
        ),
        row=1, col=1
    )

    # Right panel: Average cost per ticket (orange bars)
    fig.add_trace(
        go.Bar(
            x=category_stats.index,
            y=category_stats['Avg_Cost'],
            marker_color='#ff7f0e',
            text=[f"${v:,.0f}" for v in category_stats['Avg_Cost']],
            textposition='outside',
            textangle=0,
            name='Avg Cost',
            cliponaxis=False
        ),
        row=1, col=2
    )

    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text="Total Cost ($)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Cost ($)", row=1, col=2)

    # Add 15% headroom above bars for text labels
    max_total = category_stats['Total_Cost'].max()
    max_avg = category_stats['Avg_Cost'].max()

    fig.update_yaxes(range=[0, max_total * 1.15], row=1, col=1)
    fig.update_yaxes(range=[0, max_avg * 1.15], row=1, col=2)

    fig.update_layout(
        title_text="Category Cost Analysis",
        showlegend=False,
        height=500,
        template="plotly_white",
        margin=dict(t=100, b=100, l=60, r=60)  # Extra bottom margin for rotated labels
    )

    return fig


def create_financial_kpi_cards(metrics) -> List[Dict[str, any]]:
    """
    Generate financial KPI card data for the Streamlit dashboard.

    Unlike other functions in this module that return go.Figure objects, this
    returns a list of dictionaries that Streamlit renders using st.metric()
    or custom card components.

    KPI Cards (6 total):
        1. Total Cost: Absolute total with benchmark comparison delta
        2. Avg Cost/Ticket: Per-ticket average with median for context
        3. ROI Opportunity: Total prevention-based savings potential
        4. Efficiency Score: 0-100 score vs 75 target
        5. Revenue at Risk: Total cost * 2.5x multiplier
        6. Cost Avoidance: Total potential savings from improvements

    Each card dict contains:
        - title: Display name
        - value: Formatted string (e.g., "$12,345")
        - delta: Numeric delta from target (positive = bad for costs)
        - delta_text: Human-readable delta description
        - trend: Direction indicator ('up', 'down', 'neutral')
        - icon: Emoji icon for visual identification

    Args:
        metrics: FinancialMetrics object with all calculated financial attributes

    Returns:
        List of 6 KPI dictionaries for dashboard rendering
    """
    kpis = [
        {
            'title': 'Total Cost',
            'value': f"${metrics.total_cost:,.0f}",
            'delta': metrics.cost_vs_benchmark,
            'delta_text': f"${abs(metrics.cost_vs_benchmark):,.0f} {'over' if metrics.cost_vs_benchmark > 0 else 'under'} target",
            'trend': 'down' if metrics.cost_vs_benchmark < 0 else 'up',
            'icon': '💰'
        },
        {
            'title': 'Avg Cost/Ticket',
            'value': f"${metrics.avg_cost_per_ticket:,.0f}",
            'delta': None,
            'delta_text': f"Median: ${metrics.median_cost:,.0f}",
            'trend': 'neutral',
            'icon': '📊'
        },
        {
            'title': 'ROI Opportunity',
            'value': f"${metrics.roi_opportunity:,.0f}",
            'delta': None,
            'delta_text': f"From prevention",
            'trend': 'up',
            'icon': '📈'
        },
        {
            'title': 'Efficiency Score',
            'value': f"{metrics.cost_efficiency_score:.0f}/100",
            'delta': metrics.cost_efficiency_score - 75,  # Target benchmark = 75
            'delta_text': f"{'Above' if metrics.cost_efficiency_score > 75 else 'Below'} target",
            'trend': 'up' if metrics.cost_efficiency_score > 75 else 'down',
            'icon': '⚡'
        },
        {
            'title': 'Revenue at Risk',
            'value': f"${metrics.revenue_at_risk:,.0f}",
            'delta': None,
            'delta_text': '2.5x cost multiplier',
            'trend': 'neutral',
            'icon': '⚠️'
        },
        {
            'title': 'Cost Avoidance',
            'value': f"${metrics.cost_avoidance_potential:,.0f}",
            'delta': None,
            'delta_text': 'Potential savings',
            'trend': 'up',
            'icon': '💡'
        }
    ]

    return kpis


def create_cost_concentration_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create Pareto chart showing cost concentration (80/20 analysis).

    Combines a bar chart of individual ticket costs (sorted descending) with
    a cumulative percentage line to visualize the Pareto principle: typically
    ~20% of tickets account for ~80% of total cost.

    Visual Elements:
        - Primary y-axis (left): Bar chart of individual ticket costs
        - Secondary y-axis (right): Cumulative percentage line (0-100%)
        - Green dashed horizontal line at 80% cumulative threshold
        - Shows top 50 tickets only (for readability)

    Data Processing:
        1. Sort DataFrame by Financial_Impact descending
        2. Compute cumulative sum and cumulative percentage
        3. Compute ticket rank percentage

    Dual Y-Axis Pattern:
        Uses make_subplots(specs=[[{"secondary_y": True}]]) to create a chart
        with independent left (cost $) and right (cumulative %) y-axes.

    Note: This function is defined TWICE in this file. The second definition
    (below) silently overwrites this one. Both are identical.

    Args:
        df: Escalation DataFrame with Financial_Impact column

    Returns:
        go.Figure with Pareto chart, or empty figure if column missing
    """
    if 'Financial_Impact' not in df.columns:
        return go.Figure()

    # Sort tickets by cost descending and compute cumulative metrics
    sorted_df = df.sort_values('Financial_Impact', ascending=False).reset_index(drop=True)
    sorted_df['Cumulative_Cost'] = sorted_df['Financial_Impact'].cumsum()
    sorted_df['Cumulative_Pct'] = sorted_df['Cumulative_Cost'] / sorted_df['Financial_Impact'].sum() * 100
    sorted_df['Ticket_Pct'] = (sorted_df.index + 1) / len(sorted_df) * 100

    # Create dual y-axis layout
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary axis: Individual ticket cost bars (top 50 only)
    fig.add_trace(
        go.Bar(
            x=sorted_df.index[:50],
            y=sorted_df['Financial_Impact'][:50],
            name='Cost per Ticket',
            marker_color='#1f77b4',
            opacity=0.7
        ),
        secondary_y=False
    )

    # Secondary axis: Cumulative percentage line
    fig.add_trace(
        go.Scatter(
            x=sorted_df.index[:50],
            y=sorted_df['Cumulative_Pct'][:50],
            name='Cumulative %',
            line=dict(color='#d62728', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )

    # 80% reference line (the Pareto threshold)
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="green",
        annotation_text="80% Target",
        secondary_y=True
    )

    fig.update_xaxes(title_text="Ticket Rank (sorted by cost)")
    fig.update_yaxes(title_text="Cost ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)

    fig.update_layout(
        title="Cost Concentration Analysis (Pareto)",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )

    return fig


def create_financial_forecast_chart(forecasts: Dict) -> go.Figure:
    """
    Create financial forecast visualization with risk scenarios.

    Displays a forward-looking cost projection with:
        - Confidence interval band (upper/lower bounds as filled area)
        - Expected forecast center line (blue with markers)
        - Risk scenario horizontal lines (best/expected/worst case annual rates
          divided by 12 to get monthly values)

    Scenario Colors:
        - Best Case: Green (#2ca02c)
        - Expected: Orange (#ff7f0e)
        - Worst Case: Red (#d62728)

    This chart differs from create_cost_trend_forecast() in that it shows
    ONLY the forecast (no historical data), and includes risk scenario
    reference lines.

    Args:
        forecasts: Dict with keys:
            monthly_projection: list of {month, projected_cost, upper_bound, lower_bound}
            risk_scenarios: {best_case, expected, worst_case} (annual values)
            confidence: str ('low', 'medium', 'high')

    Returns:
        go.Figure with forecast chart, or empty figure if no projection data
    """
    if not forecasts.get('monthly_projection'):
        return go.Figure()

    # Extract forecast data arrays
    months = [proj['month'] for proj in forecasts['monthly_projection']]
    projected = [proj['projected_cost'] for proj in forecasts['monthly_projection']]
    upper = [proj['upper_bound'] for proj in forecasts['monthly_projection']]
    lower = [proj['lower_bound'] for proj in forecasts['monthly_projection']]

    fig = go.Figure()

    # Confidence interval band (filled polygon between upper and lower bounds).
    # The 'toself' fill creates a closed shape by tracing x forward with upper bounds,
    # then x backward with lower bounds.
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',  # Semi-transparent blue
        line=dict(color='rgba(255,255,255,0)'),  # Invisible border
        name='Forecast Range',
        showlegend=True
    ))

    # Expected forecast center line
    fig.add_trace(go.Scatter(
        x=months,
        y=projected,
        mode='lines+markers',
        name='Expected',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # Risk scenario horizontal reference lines.
    # Annual values are divided by 12 to show monthly rate equivalents.
    if forecasts.get('risk_scenarios'):
        scenarios = forecasts['risk_scenarios']
        monthly_scenarios = {
            'Best Case': scenarios['best_case'] / 12,
            'Expected': scenarios['expected'] / 12,
            'Worst Case': scenarios['worst_case'] / 12
        }

        colors = {'Best Case': '#2ca02c', 'Expected': '#ff7f0e', 'Worst Case': '#d62728'}

        for scenario_name, monthly_value in monthly_scenarios.items():
            fig.add_hline(
                y=monthly_value,
                line_dash="dash",
                line_color=colors[scenario_name],
                annotation_text=f"{scenario_name}: ${monthly_value:,.0f}/mo",
                annotation_position="right"
            )

    fig.update_layout(
        title=f"Financial Forecast - {forecasts.get('confidence', 'medium').title()} Confidence",
        xaxis_title="Month",
        yaxis_title="Projected Cost ($)",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )

    return fig


def create_engineer_cost_efficiency_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Create scatter matrix: Engineer efficiency vs cost (quadrant analysis).

    Plots each engineer as a bubble positioned by their average resolution time
    (x-axis) and average cost per ticket (y-axis). Bubble size represents ticket
    volume. Engineers are classified into four quadrants based on median splits.

    Quadrant Classification:
        - High Efficiency (green): Below median cost AND below median time
        - Low Cost, Slow (orange): Below median cost, above median time
        - Fast, High Cost (purple): Above median cost, below median time
        - Needs Improvement (red): Above median cost AND above median time

    Filtering:
        Engineers with fewer than 3 tickets are excluded to ensure
        meaningful averages.

    Required Columns:
        - Engineer_Assigned: Engineer identifier
        - Financial_Impact: Cost per ticket
        - Resolution_Days: Resolution time in days

    Visual Elements:
        - Colored bubbles by quadrant assignment
        - Median crosshair lines (dashed gray)
        - Hover tooltip with name, cost, resolution time, ticket count
        - Bubble size = ticket_count * 1.5

    Args:
        df: Escalation DataFrame with engineer, cost, and resolution data

    Returns:
        go.Figure with quadrant scatter chart, 600px height, or empty figure
    """
    if 'Engineer_Assigned' not in df.columns or 'Financial_Impact' not in df.columns:
        return go.Figure()

    # Calculate per-engineer aggregated metrics
    engineer_metrics = df.groupby('Engineer_Assigned').agg({
        'Financial_Impact': ['mean', 'sum', 'count'],
        'Resolution_Days': 'mean'
    }).reset_index()

    engineer_metrics.columns = ['Engineer', 'Avg_Cost', 'Total_Cost', 'Ticket_Count', 'Avg_Resolution_Days']
    engineer_metrics = engineer_metrics[engineer_metrics['Ticket_Count'] >= 3]  # Min 3 tickets

    if len(engineer_metrics) == 0:
        return go.Figure()

    # Calculate medians for quadrant boundary lines
    cost_median = engineer_metrics['Avg_Cost'].median()
    time_median = engineer_metrics['Avg_Resolution_Days'].median()

    # Classify each engineer into one of four quadrants based on median splits
    def assign_quadrant(row):
        if row['Avg_Cost'] < cost_median and row['Avg_Resolution_Days'] < time_median:
            return 'High Efficiency'        # Fast AND cheap
        elif row['Avg_Cost'] < cost_median:
            return 'Low Cost, Slow'          # Cheap but slow
        elif row['Avg_Resolution_Days'] < time_median:
            return 'Fast, High Cost'         # Fast but expensive
        else:
            return 'Needs Improvement'       # Slow AND expensive

    engineer_metrics['Quadrant'] = engineer_metrics.apply(assign_quadrant, axis=1)

    # Quadrant-to-color mapping
    color_map = {
        'High Efficiency': '#2ca02c',      # Green
        'Low Cost, Slow': '#ff7f0e',       # Orange
        'Fast, High Cost': '#9467bd',      # Purple
        'Needs Improvement': '#d62728'     # Red
    }

    fig = go.Figure()

    # Plot each quadrant as a separate trace for legend grouping
    for quadrant in engineer_metrics['Quadrant'].unique():
        subset = engineer_metrics[engineer_metrics['Quadrant'] == quadrant]

        fig.add_trace(go.Scatter(
            x=subset['Avg_Resolution_Days'],
            y=subset['Avg_Cost'],
            mode='markers',
            name=quadrant,
            text=subset['Engineer'],
            customdata=subset['Ticket_Count'],
            marker=dict(
                size=subset['Ticket_Count'] * 1.5,  # Bubble size proportional to volume
                color=color_map.get(quadrant, 'gray'),
                line=dict(width=2, color='white'),   # White border for contrast
                opacity=0.8
            ),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Avg Cost: $%{y:,.0f}<br>' +
                'Avg Resolution: %{x:.1f} days<br>' +
                'Tickets: %{customdata}<extra></extra>'
            )
        ))

    # Add median reference lines to show quadrant boundaries
    fig.add_vline(x=time_median, line_dash="dash", line_color="gray", annotation_text="Median Time")
    fig.add_hline(y=cost_median, line_dash="dash", line_color="gray", annotation_text="Median Cost")

    fig.update_layout(
        title="Engineer Efficiency Matrix",
        xaxis_title="Average Resolution Time (days)",
        yaxis_title="Average Cost per Ticket ($)",
        height=600,
        template="plotly_white"
    )

    return fig


def create_insights_table(insights: List[Dict]) -> pd.DataFrame:
    """
    Convert financial insights list to a formatted DataFrame for display.

    Transforms a list of insight dictionaries (from FinancialMetrics analysis)
    into a presentation-ready DataFrame with formatted columns suitable for
    rendering in Streamlit dataframes or HTML tables.

    Column Mapping:
        priority -> Priority (uppercased)
        type -> Category (underscores replaced with spaces, title case)
        title -> Insight
        description -> Details
        recommendation -> Recommendation
        potential_savings -> Potential Savings (formatted as $X,XXX)

    Args:
        insights: List of dicts, each with keys: priority, type, title,
                 description, recommendation, potential_savings

    Returns:
        Formatted pd.DataFrame for display, or empty DataFrame if no insights
    """
    if not insights:
        return pd.DataFrame()

    df_insights = pd.DataFrame(insights)

    # Format for display with clean column names
    display_df = pd.DataFrame({
        'Priority': [i['priority'].upper() for i in insights],
        'Category': [i['type'].replace('_', ' ').title() for i in insights],
        'Insight': [i['title'] for i in insights],
        'Details': [i['description'] for i in insights],
        'Recommendation': [i['recommendation'] for i in insights],
        'Potential Savings': [f"${i.get('potential_savings', 0):,.0f}" for i in insights]
    })

    return display_df


def create_cost_concentration_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create Pareto chart showing cost concentration (80/20 analysis).

    NOTE: This is a DUPLICATE definition. An identical function with the same
    name exists earlier in this file. This definition silently overwrites the
    first one. Both implementations are functionally identical.

    See the earlier definition for full documentation.

    Args:
        df: Escalation DataFrame with Financial_Impact column

    Returns:
        go.Figure with Pareto chart (dual y-axis: cost bars + cumulative % line)
    """
    if 'Financial_Impact' not in df.columns:
        return go.Figure()

    # Sort by cost descending and compute cumulative metrics
    sorted_df = df.sort_values('Financial_Impact', ascending=False).reset_index(drop=True)
    sorted_df['Cumulative_Cost'] = sorted_df['Financial_Impact'].cumsum()
    sorted_df['Cumulative_Pct'] = sorted_df['Cumulative_Cost'] / sorted_df['Financial_Impact'].sum() * 100
    sorted_df['Ticket_Pct'] = (sorted_df.index + 1) / len(sorted_df) * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart: individual ticket costs (top 50)
    fig.add_trace(
        go.Bar(
            x=sorted_df.index[:50],  # Show top 50
            y=sorted_df['Financial_Impact'][:50],
            name='Cost per Ticket',
            marker_color='#1f77b4',
            opacity=0.7
        ),
        secondary_y=False
    )

    # Line chart: cumulative percentage
    fig.add_trace(
        go.Scatter(
            x=sorted_df.index[:50],
            y=sorted_df['Cumulative_Pct'][:50],
            name='Cumulative %',
            line=dict(color='#d62728', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )

    # 80% Pareto reference line
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="green",
        annotation_text="80% Target",
        secondary_y=True
    )

    fig.update_xaxes(title_text="Ticket Rank (sorted by cost)")
    fig.update_yaxes(title_text="Cost ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)

    fig.update_layout(
        title="Cost Concentration Analysis (Pareto)",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )

    return fig


# =============================================================================
# SUB-CATEGORY FINANCIAL DRILL-DOWN CHARTS
#
# These charts provide deeper analysis at the AI_Sub_Category level, allowing
# stakeholders to identify specific sub-category cost drivers within each
# parent AI_Category. Used on the Financial Analysis page of the dashboard.
# =============================================================================

def create_subcategory_financial_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Create financial breakdown by sub-category with category grouping.

    Interactive horizontal bar chart showing financial impact at the sub-category
    level, color-coded by parent category. If AI_Sub_Category is not available,
    falls back to a category-only view.

    Data Handling:
        - If Financial_Impact column is missing, estimates all tickets at $850 each
        - Groups by (AI_Category, AI_Sub_Category) for the primary view
        - Falls back to AI_Category-only grouping if sub-categories unavailable

    Color Map:
        Pre-defined colors for known categories:
            Scheduling & Planning: #1f77b4 (blue)
            Documentation & Reporting: #ff7f0e (orange)
            Validation & QA: #2ca02c (green)
            Process Compliance: #d62728 (red)
            Configuration & Data Mismatch: #9467bd (purple)
            Site Readiness: #8c564b (brown)
            Communication & Response: #e377c2 (pink)
            Nesting & Tool Errors: #7f7f7f (gray)
        Unknown categories default to #17becf (cyan).

    Layout:
        - Horizontal bars sorted by total cost ascending (highest at top)
        - Stacked bar mode groups sub-categories within categories
        - Legend placed horizontally below the chart
        - Left margin of 200px to accommodate long sub-category names

    Args:
        df: Escalation DataFrame with AI_Category and optionally AI_Sub_Category

    Returns:
        go.Figure with grouped horizontal bar chart, 600px height
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Sub-Category Financial Breakdown", height=500)
        return fig

    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if not cost_col:
        # Estimate costs at $850 per ticket when Financial_Impact is not available
        df = df.copy()
        df['Financial_Impact'] = 850
        cost_col = 'Financial_Impact'

    if sub_cat_col:
        # --- Primary view: Category + Sub-Category grouping ---

        # Aggregate by category and sub-category
        subcat_stats = df.groupby(['AI_Category', sub_cat_col]).agg({
            cost_col: ['sum', 'mean', 'count']
        }).round(2)
        subcat_stats.columns = ['Total_Cost', 'Avg_Cost', 'Count']
        subcat_stats = subcat_stats.reset_index()
        subcat_stats = subcat_stats.sort_values('Total_Cost', ascending=True)

        fig = go.Figure()

        # Get unique categories and assign pre-defined colors
        categories = subcat_stats['AI_Category'].unique()
        color_map = {
            'Scheduling & Planning': '#1f77b4',
            'Documentation & Reporting': '#ff7f0e',
            'Validation & QA': '#2ca02c',
            'Process Compliance': '#d62728',
            'Configuration & Data Mismatch': '#9467bd',
            'Site Readiness': '#8c564b',
            'Communication & Response': '#e377c2',
            'Nesting & Tool Errors': '#7f7f7f'
        }

        # Add a trace per category, each containing its sub-category bars
        for cat in categories:
            cat_data = subcat_stats[subcat_stats['AI_Category'] == cat]
            fig.add_trace(go.Bar(
                y=cat_data[sub_cat_col],
                x=cat_data['Total_Cost'],
                name=cat[:20] + '..' if len(cat) > 20 else cat,  # Truncate long names
                orientation='h',
                marker_color=color_map.get(cat, '#17becf'),
                text=[f"${v:,.0f}" for v in cat_data['Total_Cost']],
                textposition='outside',
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    f'Category: {cat}<br>'
                    'Total Cost: $%{x:,.0f}<br>'
                    '<extra></extra>'
                )
            ))

        fig.update_layout(
            title="Financial Impact by Sub-Category",
            xaxis_title="Total Financial Impact ($)",
            yaxis_title="",
            height=600,
            template="plotly_white",
            barmode='stack',  # Stack sub-categories within each category group
            legend=dict(orientation='h', y=-0.2),  # Horizontal legend below chart
            margin=dict(l=200, r=80, t=80, b=100)  # Extra left margin for long names
        )

        fig.update_xaxes(tickprefix='$', tickformat=',.0f')

    else:
        # --- Fallback view: Category-only (no sub-categories available) ---

        cat_stats = df.groupby('AI_Category')[cost_col].agg(['sum', 'mean', 'count'])
        cat_stats.columns = ['Total_Cost', 'Avg_Cost', 'Count']
        cat_stats = cat_stats.sort_values('Total_Cost', ascending=True)

        fig = go.Figure(go.Bar(
            y=cat_stats.index,
            x=cat_stats['Total_Cost'],
            orientation='h',
            marker_color='#1f77b4',
            text=[f"${v:,.0f}" for v in cat_stats['Total_Cost']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Total Cost: $%{x:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title="Financial Impact by Category<br><span style='font-size:12px'>(Sub-category data not available)</span>",
            xaxis_title="Total Financial Impact ($)",
            yaxis_title="",
            height=500,
            template="plotly_white",
            margin=dict(l=200, r=80, t=80, b=40)
        )

        fig.update_xaxes(tickprefix='$', tickformat=',.0f')

    return fig


def create_subcategory_cost_treemap(df: pd.DataFrame) -> go.Figure:
    """
    Create hierarchical treemap showing financial impact by category and sub-category.

    Plotly's go.Treemap creates an interactive drill-down visualization where:
        - Top level: AI_Category rectangles sized by total cost
        - Nested level: AI_Sub_Category rectangles within each category
        - Click on a category to drill down into its sub-categories
        - Use the pathbar (breadcrumb) to navigate back up

    Hierarchy Construction:
        The ids/labels/parents/values arrays define a tree structure:
        - Root nodes: Each AI_Category with parent='' (empty string)
        - Leaf nodes: Each AI_Sub_Category with parent=AI_Category
        - IDs use "Category/SubCategory" format to ensure uniqueness

    Color Encoding:
        RdYlGn_r (reversed Red-Yellow-Green) colorscale maps cost values,
        so higher costs appear red and lower costs appear green. A colorbar
        is displayed for reference.

    Filtering:
        Sub-categories named "General" are excluded as they typically represent
        an unclassified catch-all that adds noise to the visualization.

    Args:
        df: Escalation DataFrame with AI_Category, optionally AI_Sub_Category

    Returns:
        go.Figure with interactive treemap, 600px height
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5, showarrow=False)
        return fig

    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if not cost_col:
        # Estimate costs at $850 per ticket when Financial_Impact is not available
        df = df.copy()
        df['Financial_Impact'] = 850
        cost_col = 'Financial_Impact'

    # Build hierarchy data arrays for go.Treemap
    ids = []
    labels = []
    parents = []
    values = []

    # Add category-level nodes (root level)
    for cat in df['AI_Category'].unique():
        cat_df = df[df['AI_Category'] == cat]
        cat_cost = cat_df[cost_col].sum()

        ids.append(cat)
        labels.append(cat)
        parents.append('')  # Root nodes have empty parent
        values.append(cat_cost)

        # Add sub-category nodes as children (if sub-category column exists)
        if sub_cat_col:
            for sub_cat in cat_df[sub_cat_col].unique():
                # Skip "General" catch-all sub-category to reduce noise
                if sub_cat and sub_cat != 'General':
                    sub_df = cat_df[cat_df[sub_cat_col] == sub_cat]
                    sub_cost = sub_df[cost_col].sum()

                    # Use "Category/SubCategory" as unique ID to avoid collisions
                    # when the same sub-category name appears under multiple categories
                    ids.append(f"{cat}/{sub_cat}")
                    labels.append(sub_cat)
                    parents.append(cat)
                    values.append(sub_cost)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',  # Parent values include children (not additive)
        textinfo='label+value+percent parent',  # Show name, cost, and % of parent
        marker=dict(
            colors=values,
            colorscale='RdYlGn_r',  # Red=high cost, Green=low cost
            showscale=True,
            colorbar=dict(title='Cost ($)')
        ),
        hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.0f}<br>%{percentParent:.1%} of parent<extra></extra>',
        pathbar=dict(visible=True)  # Show breadcrumb navigation bar
    ))

    # Calculate total cost for display in the title
    total_cost = sum(v for v, p in zip(values, parents) if p == '')

    fig.update_layout(
        title=f"Financial Impact Drill-Down<br><span style='font-size:14px'>Total: ${total_cost:,.0f} | Click to explore sub-categories</span>",
        height=600,
        template="plotly_white"
    )

    return fig


def create_category_financial_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary table with category and sub-category financial metrics.

    Creates a presentation-ready DataFrame aggregating financial impact at the
    category and sub-category level. Currency columns are pre-formatted as strings
    (e.g., "$1,234") for direct display in Streamlit dataframes or HTML tables.

    Columns Returned:
        - Category: AI_Category name
        - Sub-Category: AI_Sub_Category name (or "All" if sub-categories unavailable)
        - Total Impact: Sum of Financial_Impact, formatted as currency
        - Avg Impact: Mean Financial_Impact, formatted as currency
        - Ticket Count: Number of tickets in the group

    Sorting:
        Sorted by total impact descending. A temporary '_sort' column is used
        to sort the currency-formatted strings numerically, then dropped.

    Args:
        df: Escalation DataFrame with AI_Category, optionally AI_Sub_Category

    Returns:
        Formatted pd.DataFrame for display, or empty DataFrame if no categories
    """
    if 'AI_Category' not in df.columns:
        return pd.DataFrame()

    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if not cost_col:
        # Estimate costs at $850 per ticket when Financial_Impact is not available
        df = df.copy()
        df['Financial_Impact'] = 850
        cost_col = 'Financial_Impact'

    # Build the summary table
    if sub_cat_col:
        # Group by both category and sub-category
        summary = df.groupby(['AI_Category', sub_cat_col]).agg({
            cost_col: ['sum', 'mean', 'count']
        }).round(2)
        summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
        summary = summary.reset_index()
        summary = summary.rename(columns={
            'AI_Category': 'Category',
            sub_cat_col: 'Sub-Category'
        })
    else:
        # Group by category only (no sub-category breakdown available)
        summary = df.groupby('AI_Category').agg({
            cost_col: ['sum', 'mean', 'count']
        }).round(2)
        summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
        summary = summary.reset_index()
        summary = summary.rename(columns={'AI_Category': 'Category'})
        summary['Sub-Category'] = 'All'  # Placeholder when sub-categories unavailable

    # Format currency columns as display strings
    summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f'${x:,.0f}')
    summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f'${x:,.0f}')

    # Sort by total impact descending.
    # Since Total Impact is now a formatted string, we need to parse it back to
    # numeric for sorting, then drop the temporary sort column.
    summary['_sort'] = summary['Total Impact'].str.replace('[$,]', '', regex=True).astype(float)
    summary = summary.sort_values('_sort', ascending=False).drop('_sort', axis=1)

    return summary
