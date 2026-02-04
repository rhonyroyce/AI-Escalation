"""
Advanced financial visualizations for dashboards and reports.

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
    """Create an empty figure with a message."""
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

    Shows: Base Cost → Risk Adjustments → Final Impact
    """
    # Build waterfall data
    x_labels = [
        'Base Cost',
        'Recurring Issues',
        'Preventable',
        'Customer Impact',
        'SLA Risk',
        'Total Impact'
    ]

    measures = ['relative', 'relative', 'relative', 'relative', 'relative', 'total']

    base_cost = metrics_dict.get('total_cost', 0)
    recurring = metrics_dict.get('recurring_issue_cost', 0)
    preventable = metrics_dict.get('preventable_cost', 0)
    customer = metrics_dict.get('customer_impact_cost', 0) - base_cost * 0.5  # Incremental
    sla = metrics_dict.get('sla_penalty_exposure', 0)

    y_values = [
        base_cost,
        recurring,
        preventable,
        customer,
        sla,
        base_cost + recurring + preventable + customer + sla
    ]

    fig = go.Figure(go.Waterfall(
        name="Financial Impact",
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=[f"${v:,.0f}" for v in y_values],
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#d62728"}},  # Red for costs
        decreasing={"marker": {"color": "#2ca02c"}},  # Green for savings
        totals={"marker": {"color": "#1f77b4"}}  # Blue for total
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
    Create ROI opportunity visualization showing investment vs returns.
    """
    if not roi_data.get('top_opportunities'):
        return go.Figure()

    opportunities = roi_data['top_opportunities'][:5]

    # Scatter: Investment vs Savings (single chart, cleaner)
    categories = [opp['category'] for opp in opportunities]
    investments = [opp['investment_required'] for opp in opportunities]
    savings = [opp['annual_savings'] for opp in opportunities]
    roi_pcts = [opp['roi_percentage'] for opp in opportunities]
    incidents = [opp['incident_count'] for opp in opportunities]

    fig = go.Figure()

    # Add bubbles with hover info (no text labels to avoid overlap)
    fig.add_trace(
        go.Scatter(
            x=investments,
            y=savings,
            mode='markers',
            marker=dict(
                size=[max(30, min(roi / 20, 80)) for roi in roi_pcts],
                color=roi_pcts,
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title="ROI %"),
                line=dict(width=2, color='white')
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

    # Add diagonal line (break-even)
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

    # Add annotations for each point (positioned to avoid overlap)
    positions = ['top right', 'top left', 'bottom right', 'bottom left', 'top center']
    for i, (cat, x, y) in enumerate(zip(categories, investments, savings)):
        # Truncate long category names
        short_cat = cat[:20] + '...' if len(cat) > 20 else cat
        fig.add_annotation(
            x=x, y=y,
            text=f"<b>{short_cat}</b>",
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='#666',
            ax=30 if i % 2 == 0 else -30,
            ay=-30 if i < 2 else 30,
            font=dict(size=10, color='#E0E0E0'),
            bgcolor='rgba(0,0,0,0.7)',
            borderpad=3
        )

    fig.update_layout(
        title=dict(text='Investment vs Annual Savings', font=dict(size=16)),
        xaxis_title="Investment Required ($)",
        yaxis_title="Annual Savings ($)",
        showlegend=False,
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#E0E0E0')
    )

    # Format axes as currency
    fig.update_xaxes(tickprefix='$', tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(tickprefix='$', tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)')

    return fig


def create_cost_avoidance_breakdown(avoidance_data: Dict) -> go.Figure:
    """Create sunburst chart showing cost avoidance opportunities."""
    labels = ['Total Avoidance']
    parents = ['']
    values = [avoidance_data['total_avoidance']]
    colors = ['#2ca02c']

    # Add categories
    categories = [
        ('Recurring Issues', avoidance_data['recurring_issues']),
        ('Preventable Categories', avoidance_data['preventable_categories']),
        ('Knowledge Sharing', avoidance_data['knowledge_sharing']),
        ('Automation', avoidance_data['automation'])
    ]

    for cat_name, cat_value in categories:
        if cat_value > 0:
            labels.append(cat_name)
            parents.append('Total Avoidance')
            values.append(cat_value)
            colors.append(None)

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
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
    """Create cost trend chart with forecast projection."""
    # Check for Financial_Impact column
    if 'Financial_Impact' not in df.columns:
        logger.warning("Missing Financial_Impact column for trend forecast")
        return _create_empty_forecast_figure("Missing financial data")

    try:
        df_temp = df.copy()

        # Try multiple date column names
        date_col = None
        for col in ['tickets_data_issue_datetime', 'Issue_Date', 'Issue Date',
                    'Created_Date', 'Date', 'Timestamp', 'tickets_data_resolution_datetime']:
            if col in df_temp.columns:
                date_col = col
                break

        if not date_col:
            return _create_empty_forecast_figure("No date column found")

        df_temp['Issue_Date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) == 0:
            return _create_empty_forecast_figure("No valid dates in data")

        # Group by month
        df_temp['Month'] = df_temp['Issue_Date'].dt.to_period('M')
        monthly_costs = df_temp.groupby('Month')['Financial_Impact'].sum()

        if len(monthly_costs) == 0:
            return _create_empty_forecast_figure("No monthly data available")

        # Convert to datetime for plotting
        dates = [pd.Period(m).to_timestamp() for m in monthly_costs.index]
        values = monthly_costs.values

        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Actual Cost',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))

        # Trend line
        if len(dates) >= 3:
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

        # Forecast projection
        if forecasts.get('monthly_projection'):
            last_date = max(dates)
            forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
            forecast_values = [proj['projected_cost'] for proj in forecasts['monthly_projection'][:6]]
            upper_bounds = [proj['upper_bound'] for proj in forecasts['monthly_projection'][:6]]
            lower_bounds = [proj['lower_bound'] for proj in forecasts['monthly_projection'][:6]]

            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#2ca02c', width=2, dash='dot'),
                marker=dict(size=6, symbol='diamond')
            ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(44, 160, 44, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Forecast Range',
                showlegend=True
            ))

        fig.update_layout(
            title=f"Cost Trend & Forecast ({forecasts.get('trend', 'stable').title()} trend)",
            xaxis_title="Month",
            yaxis_title="Cost ($)",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating cost trend chart: {e}")
        return go.Figure()


def create_efficiency_scorecard(efficiency_data: Dict, metrics) -> go.Figure:
    """Create efficiency scorecard with key metrics."""
    fig = go.Figure()

    # Metrics to display
    scorecard_metrics = [
        ('Cost Efficiency', metrics.cost_efficiency_score, 100),
        ('Cost per Hour', min(100, 1000 / metrics.cost_per_resolution_hour) if metrics.cost_per_resolution_hour > 0 else 0, 100),
        ('Prevention Rate', min(100, (1 - metrics.preventable_cost / metrics.total_cost) * 100) if metrics.total_cost > 0 else 0, 100),
    ]

    categories = [m[0] for m in scorecard_metrics]
    values = [m[1] for m in scorecard_metrics]
    max_vals = [m[2] for m in scorecard_metrics]

    # Create gauge-style bars
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
        xaxis=dict(range=[0, 110], title="Score (0-100)"),
        yaxis_title="",
        height=400,
        template="plotly_white",
        showlegend=False
    )

    return fig


def create_category_cost_comparison(df: pd.DataFrame) -> go.Figure:
    """Create comparison of cost per ticket by category."""
    if 'AI_Category' not in df.columns or 'Financial_Impact' not in df.columns:
        return go.Figure()

    category_stats = df.groupby('AI_Category').agg({
        'Financial_Impact': ['sum', 'mean', 'count']
    }).round(2)

    category_stats.columns = ['Total_Cost', 'Avg_Cost', 'Count']
    category_stats = category_stats.sort_values('Total_Cost', ascending=False)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Cost by Category', 'Average Cost per Ticket'),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.15
    )

    # Total cost
    fig.add_trace(
        go.Bar(
            x=category_stats.index,
            y=category_stats['Total_Cost'],
            marker_color='#1f77b4',
            text=[f"${v:,.0f}" for v in category_stats['Total_Cost']],
            textposition='outside',
            textangle=0,
            name='Total Cost',
            cliponaxis=False
        ),
        row=1, col=1
    )

    # Average cost
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

    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text="Total Cost ($)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Cost ($)", row=1, col=2)

    # Add more space at top for labels
    max_total = category_stats['Total_Cost'].max()
    max_avg = category_stats['Avg_Cost'].max()

    fig.update_yaxes(range=[0, max_total * 1.15], row=1, col=1)
    fig.update_yaxes(range=[0, max_avg * 1.15], row=1, col=2)

    fig.update_layout(
        title_text="Category Cost Analysis",
        showlegend=False,
        height=500,
        template="plotly_white",
        margin=dict(t=100, b=100, l=60, r=60)
    )

    return fig


def create_financial_kpi_cards(metrics) -> List[Dict[str, any]]:
    """
    Generate financial KPI cards data for dashboard.

    Returns list of KPI dictionaries with metric, value, delta, and trend.
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
            'delta': metrics.cost_efficiency_score - 75,  # Target = 75
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
    """
    if 'Financial_Impact' not in df.columns:
        return go.Figure()

    # Sort by cost descending
    sorted_df = df.sort_values('Financial_Impact', ascending=False).reset_index(drop=True)
    sorted_df['Cumulative_Cost'] = sorted_df['Financial_Impact'].cumsum()
    sorted_df['Cumulative_Pct'] = sorted_df['Cumulative_Cost'] / sorted_df['Financial_Impact'].sum() * 100
    sorted_df['Ticket_Pct'] = (sorted_df.index + 1) / len(sorted_df) * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart: individual costs
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

    # Add 80% reference line
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
    """Create financial forecast visualization with scenarios."""
    if not forecasts.get('monthly_projection'):
        return go.Figure()

    months = [proj['month'] for proj in forecasts['monthly_projection']]
    projected = [proj['projected_cost'] for proj in forecasts['monthly_projection']]
    upper = [proj['upper_bound'] for proj in forecasts['monthly_projection']]
    lower = [proj['lower_bound'] for proj in forecasts['monthly_projection']]

    fig = go.Figure()

    # Forecast range (confidence interval)
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Forecast Range',
        showlegend=True
    ))

    # Expected forecast
    fig.add_trace(go.Scatter(
        x=months,
        y=projected,
        mode='lines+markers',
        name='Expected',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # Add risk scenarios as horizontal lines
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
    Create scatter matrix: Engineer efficiency vs cost.

    X-axis: Average resolution time
    Y-axis: Average cost per ticket
    Size: Number of tickets handled
    """
    if 'Engineer_Assigned' not in df.columns or 'Financial_Impact' not in df.columns:
        return go.Figure()

    # Calculate metrics per engineer
    engineer_metrics = df.groupby('Engineer_Assigned').agg({
        'Financial_Impact': ['mean', 'sum', 'count'],
        'Resolution_Days': 'mean'
    }).reset_index()

    engineer_metrics.columns = ['Engineer', 'Avg_Cost', 'Total_Cost', 'Ticket_Count', 'Avg_Resolution_Days']
    engineer_metrics = engineer_metrics[engineer_metrics['Ticket_Count'] >= 3]  # Min 3 tickets

    if len(engineer_metrics) == 0:
        return go.Figure()

    # Calculate quartiles for coloring
    cost_median = engineer_metrics['Avg_Cost'].median()
    time_median = engineer_metrics['Avg_Resolution_Days'].median()

    # Assign quadrants
    def assign_quadrant(row):
        if row['Avg_Cost'] < cost_median and row['Avg_Resolution_Days'] < time_median:
            return 'High Efficiency'
        elif row['Avg_Cost'] < cost_median:
            return 'Low Cost, Slow'
        elif row['Avg_Resolution_Days'] < time_median:
            return 'Fast, High Cost'
        else:
            return 'Needs Improvement'

    engineer_metrics['Quadrant'] = engineer_metrics.apply(assign_quadrant, axis=1)

    color_map = {
        'High Efficiency': '#2ca02c',
        'Low Cost, Slow': '#ff7f0e',
        'Fast, High Cost': '#9467bd',
        'Needs Improvement': '#d62728'
    }

    fig = go.Figure()

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
                size=subset['Ticket_Count'] * 1.5,
                color=color_map.get(quadrant, 'gray'),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Avg Cost: $%{y:,.0f}<br>' +
                'Avg Resolution: %{x:.1f} days<br>' +
                'Tickets: %{customdata}<extra></extra>'
            )
        ))

    # Add median lines
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
    """Convert financial insights to DataFrame for display."""
    if not insights:
        return pd.DataFrame()

    df_insights = pd.DataFrame(insights)

    # Format for display
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
    """
    if 'Financial_Impact' not in df.columns:
        return go.Figure()

    # Sort by cost descending
    sorted_df = df.sort_values('Financial_Impact', ascending=False).reset_index(drop=True)
    sorted_df['Cumulative_Cost'] = sorted_df['Financial_Impact'].cumsum()
    sorted_df['Cumulative_Pct'] = sorted_df['Cumulative_Cost'] / sorted_df['Financial_Impact'].sum() * 100
    sorted_df['Ticket_Pct'] = (sorted_df.index + 1) / len(sorted_df) * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart: individual costs
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

    # Add 80% reference line
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
# =============================================================================

def create_subcategory_financial_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Create financial breakdown by sub-category with category grouping.
    Interactive chart showing costs at the sub-category level.
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Sub-Category Financial Breakdown", height=500)
        return fig

    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if not cost_col:
        # Estimate costs
        df = df.copy()
        df['Financial_Impact'] = 850
        cost_col = 'Financial_Impact'

    if sub_cat_col:
        # Aggregate by category and sub-category
        subcat_stats = df.groupby(['AI_Category', sub_cat_col]).agg({
            cost_col: ['sum', 'mean', 'count']
        }).round(2)
        subcat_stats.columns = ['Total_Cost', 'Avg_Cost', 'Count']
        subcat_stats = subcat_stats.reset_index()
        subcat_stats = subcat_stats.sort_values('Total_Cost', ascending=True)

        # Create grouped bar chart
        fig = go.Figure()

        # Get unique categories for color mapping
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

        for cat in categories:
            cat_data = subcat_stats[subcat_stats['AI_Category'] == cat]
            fig.add_trace(go.Bar(
                y=cat_data[sub_cat_col],
                x=cat_data['Total_Cost'],
                name=cat[:20] + '..' if len(cat) > 20 else cat,
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
            barmode='stack',
            legend=dict(orientation='h', y=-0.2),
            margin=dict(l=200, r=80, t=80, b=100)
        )

        fig.update_xaxes(tickprefix='$', tickformat=',.0f')

    else:
        # Fallback to category-only view
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
    Allows drill-down from category to sub-category level.
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5, showarrow=False)
        return fig

    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if not cost_col:
        df = df.copy()
        df['Financial_Impact'] = 850
        cost_col = 'Financial_Impact'

    # Build hierarchy data
    ids = []
    labels = []
    parents = []
    values = []

    # Add categories
    for cat in df['AI_Category'].unique():
        cat_df = df[df['AI_Category'] == cat]
        cat_cost = cat_df[cost_col].sum()

        ids.append(cat)
        labels.append(cat)
        parents.append('')
        values.append(cat_cost)

        # Add sub-categories if available
        if sub_cat_col:
            for sub_cat in cat_df[sub_cat_col].unique():
                if sub_cat and sub_cat != 'General':
                    sub_df = cat_df[cat_df[sub_cat_col] == sub_cat]
                    sub_cost = sub_df[cost_col].sum()

                    ids.append(f"{cat}/{sub_cat}")
                    labels.append(sub_cat)
                    parents.append(cat)
                    values.append(sub_cost)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        textinfo='label+value+percent parent',
        marker=dict(
            colors=values,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='Cost ($)')
        ),
        hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.0f}<br>%{percentParent:.1%} of parent<extra></extra>',
        pathbar=dict(visible=True)
    ))

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
    Returns a DataFrame suitable for display in dashboards.
    """
    if 'AI_Category' not in df.columns:
        return pd.DataFrame()

    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if not cost_col:
        df = df.copy()
        df['Financial_Impact'] = 850
        cost_col = 'Financial_Impact'

    # Build summary
    if sub_cat_col:
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
        summary = df.groupby('AI_Category').agg({
            cost_col: ['sum', 'mean', 'count']
        }).round(2)
        summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
        summary = summary.reset_index()
        summary = summary.rename(columns={'AI_Category': 'Category'})
        summary['Sub-Category'] = 'All'

    # Format currency columns
    summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f'${x:,.0f}')
    summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f'${x:,.0f}')

    # Sort by total impact (need to convert back for sorting)
    summary['_sort'] = summary['Total Impact'].str.replace('[$,]', '', regex=True).astype(float)
    summary = summary.sort_values('_sort', ascending=False).drop('_sort', axis=1)

    return summary
