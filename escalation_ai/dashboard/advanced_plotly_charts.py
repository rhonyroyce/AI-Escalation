"""
Advanced Plotly Charts for Streamlit Dashboard

High-value interactive visualizations:
- SLA Compliance Funnel
- Engineer Efficiency Quadrant
- Cost Avoidance Waterfall
- Time-of-Day Heatmap
- Executive Scorecard
- Resolution Consistency Analysis
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


def create_plotly_theme():
    """Get consistent Plotly theme settings."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#E0E0E0'),
        margin=dict(l=40, r=40, t=50, b=40),
    )


# =============================================================================
# SLA COMPLIANCE FUNNEL
# =============================================================================

def chart_sla_funnel(df: pd.DataFrame,
                     datetime_col: str = 'tickets_data_issue_datetime',
                     close_col: str = 'tickets_data_close_datetime') -> go.Figure:
    """
    Interactive SLA Compliance Funnel showing resolution rates at key thresholds.
    """
    # Calculate resolution times
    df_temp = df.copy()
    
    if datetime_col in df_temp.columns and close_col in df_temp.columns:
        df_temp['open_dt'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
        df_temp['close_dt'] = pd.to_datetime(df_temp[close_col], errors='coerce')
        df_temp['resolution_hours'] = (df_temp['close_dt'] - df_temp['open_dt']).dt.total_seconds() / 3600
        valid = df_temp.dropna(subset=['resolution_hours'])
    elif 'Predicted_Resolution_Days' in df_temp.columns:
        valid = df_temp.dropna(subset=['Predicted_Resolution_Days'])
        valid['resolution_hours'] = valid['Predicted_Resolution_Days'] * 24
    else:
        # Generate sample data
        np.random.seed(42)
        valid = pd.DataFrame({'resolution_hours': np.random.exponential(48, len(df))})
    
    total = len(valid)
    if total == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    # Calculate funnel stages
    thresholds = [
        ('< 4 hours (Urgent)', 0, 4),
        ('4-24 hours (Critical)', 4, 24),
        ('24-48 hours (High)', 24, 48),
        ('48h-7 days (Standard)', 48, 168),
        ('> 7 days (Aged)', 168, float('inf'))
    ]
    
    counts = []
    for label, min_h, max_h in thresholds:
        if max_h == float('inf'):
            count = (valid['resolution_hours'] > min_h).sum()
        else:
            count = ((valid['resolution_hours'] > min_h) & (valid['resolution_hours'] <= max_h)).sum()
        counts.append(count)
    
    labels = [t[0] for t in thresholds]
    percentages = [c / total * 100 for c in counts]
    
    # Colors from green to red
    colors = ['#28A745', '#7CB342', '#FFC107', '#FF9800', '#DC3545']
    
    fig = go.Figure()
    
    # Create funnel bars
    for i, (label, count, pct, color) in enumerate(zip(labels, counts, percentages, colors)):
        fig.add_trace(go.Bar(
            y=[label],
            x=[pct],
            orientation='h',
            marker=dict(color=color, line=dict(color='white', width=2)),
            text=f'{count:,} ({pct:.1f}%)',
            textposition='inside',
            textfont=dict(size=12, color='white'),
            hovertemplate=f'<b>{label}</b><br>Count: {count:,}<br>Percentage: {pct:.1f}%<extra></extra>',
            showlegend=False
        ))
    
    # SLA target line (80% within 48 hours)
    resolved_48h = sum(percentages[:3])
    sla_target = 80
    
    fig.add_vline(x=sla_target, line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"SLA Target ({sla_target}%)")
    
    status = '✓ Meeting' if resolved_48h >= sla_target else '✗ Below'
    status_color = '#28A745' if resolved_48h >= sla_target else '#DC3545'
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(
            text=f'SLA Compliance Funnel<br><span style="font-size:14px;color:{status_color}">{status} Target: {resolved_48h:.1f}% resolved within 48h</span>',
            font=dict(size=18)
        ),
        xaxis_title='Percentage of Tickets',
        yaxis=dict(autorange='reversed'),
        height=450,
        bargap=0.3
    )
    
    return fig


# =============================================================================
# ENGINEER EFFICIENCY QUADRANT
# =============================================================================

def chart_engineer_quadrant(df: pd.DataFrame,
                            engineer_col: str = 'Engineer') -> go.Figure:
    """
    Interactive 2x2 Quadrant showing Speed vs Quality for engineers.
    """
    # Find engineer column
    eng_col = None
    for col in [engineer_col, 'tickets_data_engineer_name', 'Engineer_Name']:
        if col in df.columns:
            eng_col = col
            break
    
    if eng_col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Engineer data not available<br>Ensure engineer_name column exists",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16)
        )
        fig.update_layout(**create_plotly_theme(), height=500)
        return fig
    
    # Calculate metrics per engineer
    resolution_col = 'Predicted_Resolution_Days'
    recurrence_col = 'AI_Recurrence_Probability' if 'AI_Recurrence_Probability' in df.columns else 'AI_Recurrence_Risk'
    
    agg_dict = {eng_col: 'count'}
    if resolution_col in df.columns:
        agg_dict[resolution_col] = 'mean'
    if recurrence_col in df.columns:
        agg_dict[recurrence_col] = 'mean'
    
    eng_stats = df.groupby(eng_col).agg(agg_dict)
    eng_stats.columns = ['ticket_count', 'avg_resolution', 'recurrence_rate']
    eng_stats = eng_stats[eng_stats['ticket_count'] >= 3]  # Min 3 tickets
    
    if len(eng_stats) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for quadrant analysis", x=0.5, y=0.5)
        return fig
    
    # Calculate quality score (inverse of recurrence)
    eng_stats['quality_score'] = (1 - eng_stats['recurrence_rate']) * 100
    
    # Medians for quadrant lines
    x_median = eng_stats['avg_resolution'].median()
    y_median = eng_stats['quality_score'].median()
    
    # Determine quadrant for each engineer
    def get_quadrant(row):
        if row['avg_resolution'] <= x_median and row['quality_score'] >= y_median:
            return 'Fast & Clean ⭐'
        elif row['avg_resolution'] > x_median and row['quality_score'] >= y_median:
            return 'Slow but Thorough 🐢'
        elif row['avg_resolution'] <= x_median and row['quality_score'] < y_median:
            return 'Fast but Sloppy ⚡'
        else:
            return 'Needs Support 🆘'
    
    eng_stats['quadrant'] = eng_stats.apply(get_quadrant, axis=1)
    
    # Color mapping
    color_map = {
        'Fast & Clean ⭐': '#28A745',
        'Slow but Thorough 🐢': '#17A2B8',
        'Fast but Sloppy ⚡': '#FFC107',
        'Needs Support 🆘': '#DC3545'
    }
    
    fig = go.Figure()
    
    # Add quadrant backgrounds
    x_max = eng_stats['avg_resolution'].max() * 1.2
    y_min = eng_stats['quality_score'].min() * 0.9
    
    # Quadrant rectangles
    quadrants = [
        (0, x_median, y_median, 100, 'rgba(40, 167, 69, 0.1)'),  # Fast & Clean
        (x_median, x_max, y_median, 100, 'rgba(23, 162, 184, 0.1)'),  # Slow but Thorough
        (0, x_median, y_min, y_median, 'rgba(255, 193, 7, 0.1)'),  # Fast but Sloppy
        (x_median, x_max, y_min, y_median, 'rgba(220, 53, 69, 0.1)')  # Needs Support
    ]
    
    for x0, x1, y0, y1, color in quadrants:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, line=dict(width=0)
        )
    
    # Quadrant labels
    quadrant_labels = [
        (x_median/2, (y_median + 100)/2, '⭐ Fast & Clean', '#28A745'),
        ((x_median + x_max)/2, (y_median + 100)/2, '🐢 Slow but Thorough', '#17A2B8'),
        (x_median/2, (y_min + y_median)/2, '⚡ Fast but Sloppy', '#FFC107'),
        ((x_median + x_max)/2, (y_min + y_median)/2, '🆘 Needs Support', '#DC3545')
    ]
    
    for x, y, text, color in quadrant_labels:
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=False,
            font=dict(size=14, color=color), opacity=0.7
        )
    
    # Scatter plot for engineers
    for quadrant, color in color_map.items():
        mask = eng_stats['quadrant'] == quadrant
        subset = eng_stats[mask]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset['avg_resolution'],
                y=subset['quality_score'],
                mode='markers+text',
                marker=dict(
                    size=subset['ticket_count'] * 3 + 10,
                    color=color,
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                text=subset.index,
                textposition='top center',
                textfont=dict(size=9),
                name=quadrant,
                hovertemplate='<b>%{text}</b><br>' +
                              'Resolution: %{x:.1f} days<br>' +
                              'Quality Score: %{y:.1f}%<br>' +
                              f'Tickets: %{{marker.size:.0f}}<extra></extra>'
            ))
    
    # Median lines
    fig.add_hline(y=y_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig.add_vline(x=x_median, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Engineer Efficiency Quadrant', font=dict(size=18)),
        xaxis_title='Average Resolution Time (days) →',
        yaxis_title='Quality Score (100 - Recurrence %) →',
        height=550,
        showlegend=True,
        legend=dict(orientation='h', y=-0.15)
    )
    
    return fig


# =============================================================================
# COST AVOIDANCE WATERFALL
# =============================================================================

def chart_cost_waterfall(df: pd.DataFrame, cost_col: str = 'Financial_Impact') -> go.Figure:
    """
    Interactive Waterfall showing path from current costs to achievable target.
    """
    # Calculate base costs
    if cost_col in df.columns:
        total_cost = df[cost_col].sum()
    else:
        total_cost = len(df) * 850
    
    # Calculate potential savings
    recurrence_rate = df['AI_Recurrence_Probability'].mean() if 'AI_Recurrence_Probability' in df.columns else 0.2
    recurrence_savings = total_cost * recurrence_rate * 0.5
    
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    resolution_savings = total_cost * 0.15 * (avg_resolution / 5)
    
    if 'AI_Category' in df.columns and cost_col in df.columns:
        top_cat_cost = df.groupby('AI_Category')[cost_col].sum().max()
        category_savings = top_cat_cost * 0.3
    else:
        category_savings = total_cost * 0.08
    
    process_savings = total_cost * 0.05
    
    # Build waterfall
    target = total_cost - recurrence_savings - resolution_savings - category_savings - process_savings
    
    fig = go.Figure(go.Waterfall(
        name="Cost Analysis",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["Current<br>Total Cost", "Reduce<br>Recurrence", "Faster<br>Resolution", 
           "Category<br>Focus", "Process<br>Improvement", "Achievable<br>Target"],
        y=[total_cost, -recurrence_savings, -resolution_savings, -category_savings, -process_savings, 0],
        text=[f"${total_cost/1000:.0f}K", f"-${recurrence_savings/1000:.0f}K", 
              f"-${resolution_savings/1000:.0f}K", f"-${category_savings/1000:.0f}K",
              f"-${process_savings/1000:.0f}K", f"${target/1000:.0f}K"],
        textposition="outside",
        textfont=dict(size=11, color='white'),
        connector=dict(line=dict(color="rgba(255,255,255,0.3)", width=2)),
        decreasing=dict(marker=dict(color="#28A745")),
        increasing=dict(marker=dict(color="#DC3545")),
        totals=dict(marker=dict(color="#0066CC"))
    ))
    
    total_savings = total_cost - target
    savings_pct = (total_savings / total_cost) * 100
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(
            text=f'Cost Avoidance Waterfall<br><span style="font-size:14px;color:#28A745">Potential Savings: ${total_savings/1000:.0f}K ({savings_pct:.0f}%)</span>',
            font=dict(size=18)
        ),
        yaxis_title='Cost ($)',
        height=450,
        showlegend=False
    )
    
    # Format y-axis
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    
    return fig


# =============================================================================
# TIME-OF-DAY HEATMAP
# =============================================================================

def chart_time_heatmap(df: pd.DataFrame,
                       datetime_col: str = 'tickets_data_issue_datetime') -> go.Figure:
    """
    Interactive heatmap showing escalation patterns by day and hour.
    """
    if datetime_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Datetime data not available", x=0.5, y=0.5)
        fig.update_layout(**create_plotly_theme(), height=450)
        return fig
    
    df_temp = df.copy()
    df_temp['datetime'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
    df_temp = df_temp.dropna(subset=['datetime'])
    
    if len(df_temp) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No valid datetime data", x=0.5, y=0.5)
        return fig
    
    # Extract day and hour
    df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
    df_temp['hour'] = df_temp['datetime'].dt.hour
    
    # Create pivot
    pivot = df_temp.pivot_table(
        values='Strategic_Friction_Score' if 'Strategic_Friction_Score' in df_temp.columns else datetime_col,
        index='hour',
        columns='day_of_week',
        aggfunc='count' if 'Strategic_Friction_Score' not in df_temp.columns else 'sum',
        fill_value=0
    )
    
    # Ensure all hours and days
    pivot = pivot.reindex(index=range(24), columns=range(7), fill_value=0)
    
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hour_labels = [f'{h:02d}:00' for h in range(24)]
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=day_labels,
        y=hour_labels,
        colorscale='YlOrRd',
        hovertemplate='<b>%{x} at %{y}</b><br>Count: %{z}<extra></extra>'
    ))
    
    # Add shift boundary lines
    for shift_hour in [6, 14, 22]:
        fig.add_hline(y=shift_hour, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    
    # Find peak
    max_idx = np.unravel_index(pivot.values.argmax(), pivot.values.shape)
    peak_hour = max_idx[0]
    peak_day = max_idx[1]
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(
            text=f'Escalation Time Pattern Analysis<br><span style="font-size:14px">Peak: {day_labels[peak_day]} at {peak_hour:02d}:00</span>',
            font=dict(size=18)
        ),
        xaxis_title='Day of Week',
        yaxis_title='Hour of Day',
        height=500
    )
    
    return fig


# =============================================================================
# AGING TICKETS ANALYSIS
# =============================================================================

def chart_aging_analysis(df: pd.DataFrame,
                         datetime_col: str = 'tickets_data_issue_datetime') -> go.Figure:
    """
    Ticket aging distribution with highlighting for aged tickets.
    """
    if datetime_col not in df.columns:
        # Use prediction if available
        if 'Predicted_Resolution_Days' in df.columns:
            df_temp = df.copy()
            df_temp['age_days'] = df_temp['Predicted_Resolution_Days']
        else:
            fig = go.Figure()
            fig.add_annotation(text="Datetime data not available", x=0.5, y=0.5)
            fig.update_layout(**create_plotly_theme(), height=400)
            return fig
    else:
        df_temp = df.copy()
        df_temp['datetime'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
        df_temp = df_temp.dropna(subset=['datetime'])
        df_temp['age_days'] = (datetime.now() - df_temp['datetime']).dt.days
    
    # Age buckets
    buckets = [
        ('0-7 days', 0, 7, '#28A745'),
        ('8-14 days', 8, 14, '#7CB342'),
        ('15-30 days', 15, 30, '#FFC107'),
        ('31-60 days', 31, 60, '#FF9800'),
        ('60+ days', 61, 9999, '#DC3545')
    ]
    
    counts = []
    for label, min_d, max_d, color in buckets:
        count = ((df_temp['age_days'] >= min_d) & (df_temp['age_days'] <= max_d)).sum()
        counts.append(count)
    
    labels = [b[0] for b in buckets]
    colors = [b[3] for b in buckets]
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker=dict(color=colors, line=dict(color='white', width=2)),
        text=counts,
        textposition='outside',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>Tickets: %{y}<extra></extra>'
    ))
    
    aged_count = sum(counts[3:])
    aged_pct = (aged_count / sum(counts)) * 100 if sum(counts) > 0 else 0
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(
            text=f'Ticket Aging Distribution<br><span style="font-size:14px;color:#DC3545">{aged_count} tickets ({aged_pct:.1f}%) aged beyond 30 days</span>',
            font=dict(size=18)
        ),
        xaxis_title='Age Category',
        yaxis_title='Number of Tickets',
        height=400
    )
    
    return fig


# =============================================================================
# EXECUTIVE SCORECARD GAUGES
# =============================================================================

def chart_health_gauge(df: pd.DataFrame) -> go.Figure:
    """
    Overall operational health score gauge.
    """
    # Calculate metrics
    recurrence_rate = df['AI_Recurrence_Probability'].mean() * 100 if 'AI_Recurrence_Probability' in df.columns else 15
    resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    critical_pct = (df['tickets_data_severity'] == 'Critical').mean() * 100 if 'tickets_data_severity' in df.columns else 10
    
    # Health score calculation
    health_score = max(0, min(100, 100 - (recurrence_rate * 1.5) - (resolution_days * 5) - (critical_pct * 0.5)))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        number={'suffix': '', 'font': {'size': 48}},
        delta={'reference': 75, 'relative': False},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#0066CC'},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(220, 53, 69, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(40, 167, 69, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#00BFFF', 'width': 4},
                'thickness': 0.75,
                'value': health_score
            }
        }
    ))
    
    status = 'Healthy' if health_score >= 70 else 'At Risk' if health_score >= 40 else 'Critical'
    status_color = '#28A745' if health_score >= 70 else '#FFC107' if health_score >= 40 else '#DC3545'
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(
            text=f'Operational Health Score<br><span style="font-size:14px;color:{status_color}">{status}</span>',
            font=dict(size=16)
        ),
        height=300
    )
    
    return fig


# =============================================================================
# RESOLUTION CONSISTENCY ANALYSIS
# =============================================================================

def chart_resolution_consistency(df: pd.DataFrame) -> go.Figure:
    """
    Shows variance in resolution approaches across similar categories.
    """
    if 'AI_Category' not in df.columns or 'Predicted_Resolution_Days' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category and resolution data required", x=0.5, y=0.5)
        fig.update_layout(**create_plotly_theme(), height=400)
        return fig
    
    # Calculate resolution stats per category
    cat_stats = df.groupby('AI_Category')['Predicted_Resolution_Days'].agg(['mean', 'std', 'count'])
    cat_stats = cat_stats[cat_stats['count'] >= 5].sort_values('std', ascending=False)
    
    if len(cat_stats) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for consistency analysis", x=0.5, y=0.5)
        return fig
    
    # Coefficient of variation (relative consistency)
    cat_stats['cv'] = (cat_stats['std'] / cat_stats['mean']) * 100
    cat_stats = cat_stats.head(10)  # Top 10 most variable
    
    fig = go.Figure()
    
    # Error bars showing variability
    fig.add_trace(go.Bar(
        y=cat_stats.index,
        x=cat_stats['mean'],
        orientation='h',
        marker=dict(
            color=cat_stats['cv'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='Variability %')
        ),
        error_x=dict(type='data', array=cat_stats['std'], color='rgba(255,255,255,0.5)'),
        text=[f'{m:.1f} ± {s:.1f} days (CV: {cv:.0f}%)' 
              for m, s, cv in zip(cat_stats['mean'], cat_stats['std'], cat_stats['cv'])],
        textposition='outside',
        textfont=dict(size=9),
        hovertemplate='<b>%{y}</b><br>Mean: %{x:.1f} days<br>Std Dev: %{error_x.array:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#E0E0E0'),
        title=dict(text='Resolution Consistency by Category', font=dict(size=18)),
        xaxis_title='Average Resolution Time (days)',
        yaxis_title='',
        height=450,
        margin=dict(l=180, r=80, t=60, b=40)
    )
    
    return fig


# =============================================================================
# RECURRENCE PATTERN NETWORK (Simplified)
# =============================================================================

def chart_recurrence_patterns(df: pd.DataFrame) -> go.Figure:
    """
    Sankey diagram showing recurrence flow from categories to outcomes.
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5)
        return fig

    # Get recurrence risk column
    rec_col = 'AI_Recurrence_Probability' if 'AI_Recurrence_Probability' in df.columns else 'AI_Recurrence_Risk'

    if rec_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Recurrence data required", x=0.5, y=0.5)
        return fig

    # Categorize recurrence risk
    df_temp = df.copy()
    df_temp['rec_level'] = pd.cut(
        df_temp[rec_col],
        bins=[0, 0.2, 0.5, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Create flow data
    flow_data = df_temp.groupby(['AI_Category', 'rec_level']).size().reset_index(name='count')
    flow_data = flow_data.dropna()

    # Get unique categories and risk levels
    categories = flow_data['AI_Category'].unique().tolist()
    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']

    # Create node labels
    all_nodes = categories + risk_levels

    # Create source, target, value lists for Sankey
    source = []
    target = []
    value = []

    for _, row in flow_data.iterrows():
        cat_idx = all_nodes.index(row['AI_Category'])
        risk_idx = all_nodes.index(row['rec_level'])
        source.append(cat_idx)
        target.append(risk_idx)
        value.append(row['count'])

    # Colors
    node_colors = ['#0066CC'] * len(categories) + ['#28A745', '#FFC107', '#DC3545']

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(100, 150, 200, 0.4)'
        )
    ))

    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Category to Recurrence Risk Flow', font=dict(size=18)),
        height=500
    )

    return fig


# =============================================================================
# CATEGORY/SUB-CATEGORY DRILL-DOWN CHARTS
# =============================================================================

def chart_category_sunburst(df: pd.DataFrame) -> go.Figure:
    """
    Interactive Sunburst chart for Category → Sub-Category drill-down.
    Click on a category to see its sub-categories with ticket counts and financial data.
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5)
        fig.update_layout(**create_plotly_theme(), height=500)
        return fig

    # Check for sub-category column
    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    # Build hierarchy data
    labels = ['All Escalations']
    parents = ['']
    values = [len(df)]
    colors = ['#003366']
    custom_data = [{'count': len(df), 'cost': df[cost_col].sum() if cost_col else 0}]

    # Color palette for categories
    cat_colors = {
        'Scheduling & Planning': '#1f77b4',
        'Documentation & Reporting': '#ff7f0e',
        'Validation & QA': '#2ca02c',
        'Process Compliance': '#d62728',
        'Configuration & Data Mismatch': '#9467bd',
        'Site Readiness': '#8c564b',
        'Communication & Response': '#e377c2',
        'Nesting & Tool Errors': '#7f7f7f',
        'Unclassified': '#bcbd22'
    }

    # Add categories
    cat_counts = df['AI_Category'].value_counts()
    for cat in cat_counts.index:
        cat_df = df[df['AI_Category'] == cat]
        cat_count = len(cat_df)
        cat_cost = cat_df[cost_col].sum() if cost_col else 0

        labels.append(cat)
        parents.append('All Escalations')
        values.append(cat_count)
        colors.append(cat_colors.get(cat, '#17becf'))
        custom_data.append({'count': cat_count, 'cost': cat_cost})

        # Add sub-categories if available
        if sub_cat_col and sub_cat_col in df.columns:
            sub_counts = cat_df[sub_cat_col].value_counts()
            for sub_cat in sub_counts.index:
                if sub_cat and sub_cat != 'General':
                    sub_df = cat_df[cat_df[sub_cat_col] == sub_cat]
                    sub_count = len(sub_df)
                    sub_cost = sub_df[cost_col].sum() if cost_col else 0

                    labels.append(sub_cat)
                    parents.append(cat)
                    values.append(sub_count)
                    # Lighter shade for sub-categories
                    base_color = cat_colors.get(cat, '#17becf')
                    colors.append(base_color + '99')  # Add transparency
                    custom_data.append({'count': sub_count, 'cost': sub_cost})

    # Build hover text
    hover_texts = []
    for data in custom_data:
        if cost_col:
            hover_texts.append(f"Tickets: {data['count']:,}<br>Financial Impact: ${data['cost']:,.0f}")
        else:
            hover_texts.append(f"Tickets: {data['count']:,}")

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        marker=dict(colors=colors),
        hovertext=hover_texts,
        hoverinfo='label+text+percent parent',
        insidetextorientation='radial'
    ))

    fig.update_layout(
        **create_plotly_theme(),
        title=dict(
            text='Category & Sub-Category Drill-Down<br><span style="font-size:12px">Click to expand categories</span>',
            font=dict(size=18)
        ),
        height=550
    )

    return fig


def chart_category_treemap(df: pd.DataFrame) -> go.Figure:
    """
    Interactive Treemap for Category → Sub-Category hierarchy.
    Shows proportional sizing by ticket count with financial impact on hover.
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5)
        fig.update_layout(**create_plotly_theme(), height=500)
        return fig

    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    # Build data for treemap
    ids = []
    labels = []
    parents = []
    values = []
    costs = []

    # Add categories and sub-categories
    for cat in df['AI_Category'].unique():
        cat_df = df[df['AI_Category'] == cat]
        cat_count = len(cat_df)
        cat_cost = cat_df[cost_col].sum() if cost_col else 0

        ids.append(cat)
        labels.append(cat)
        parents.append('')
        values.append(cat_count)
        costs.append(cat_cost)

        # Add sub-categories
        if sub_cat_col and sub_cat_col in df.columns:
            for sub_cat in cat_df[sub_cat_col].unique():
                if sub_cat and sub_cat != 'General':
                    sub_df = cat_df[cat_df[sub_cat_col] == sub_cat]
                    sub_count = len(sub_df)
                    sub_cost = sub_df[cost_col].sum() if cost_col else 0

                    ids.append(f"{cat}/{sub_cat}")
                    labels.append(sub_cat)
                    parents.append(cat)
                    values.append(sub_count)
                    costs.append(sub_cost)

    # Create hover text with financial data
    hover_texts = []
    for i, (label, count, cost) in enumerate(zip(labels, values, costs)):
        if cost_col:
            avg_cost = cost / count if count > 0 else 0
            hover_texts.append(
                f"<b>{label}</b><br>"
                f"Tickets: {count:,}<br>"
                f"Total Impact: ${cost:,.0f}<br>"
                f"Avg per Ticket: ${avg_cost:,.0f}"
            )
        else:
            hover_texts.append(f"<b>{label}</b><br>Tickets: {count:,}")

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        hovertext=hover_texts,
        hoverinfo='text',
        textinfo='label+value+percent parent',
        marker=dict(
            colors=costs if cost_col else values,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='Financial<br>Impact ($)')
        ),
        pathbar=dict(visible=True)
    ))

    fig.update_layout(
        **create_plotly_theme(),
        title=dict(
            text='Category Treemap with Sub-Category Drill-Down<br><span style="font-size:12px">Click to zoom into categories</span>',
            font=dict(size=18)
        ),
        height=550
    )

    return fig


def chart_subcategory_breakdown(df: pd.DataFrame, selected_category: str = None) -> go.Figure:
    """
    Detailed sub-category breakdown for a selected category.
    Shows ticket count, financial impact, and recurrence rate for each sub-category.
    """
    if 'AI_Category' not in df.columns or 'AI_Sub_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category and Sub-Category data required", x=0.5, y=0.5)
        fig.update_layout(**create_plotly_theme(), height=450)
        return fig

    # Filter to selected category if provided
    if selected_category:
        df_filtered = df[df['AI_Category'] == selected_category].copy()
        title_text = f'Sub-Category Breakdown: {selected_category}'
    else:
        df_filtered = df.copy()
        title_text = 'Sub-Category Breakdown (All Categories)'

    if len(df_filtered) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for category: {selected_category}", x=0.5, y=0.5)
        return fig

    # Aggregate by sub-category
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None
    rec_col = 'AI_Recurrence_Probability' if 'AI_Recurrence_Probability' in df.columns else None

    agg_dict = {'AI_Sub_Category': 'count'}
    if cost_col:
        agg_dict[cost_col] = 'sum'
    if rec_col:
        agg_dict[rec_col] = 'mean'

    sub_stats = df_filtered.groupby('AI_Sub_Category').agg(agg_dict)
    sub_stats.columns = ['Count'] + (['Total_Cost'] if cost_col else []) + (['Recurrence'] if rec_col else [])
    sub_stats = sub_stats.sort_values('Count', ascending=False).head(15)

    fig = make_subplots(
        rows=1, cols=3 if cost_col and rec_col else (2 if cost_col or rec_col else 1),
        subplot_titles=['Ticket Count', 'Financial Impact ($)', 'Recurrence Rate (%)'][:3 if cost_col and rec_col else (2 if cost_col or rec_col else 1)],
        horizontal_spacing=0.12
    )

    # Ticket count bars
    fig.add_trace(
        go.Bar(
            y=sub_stats.index,
            x=sub_stats['Count'],
            orientation='h',
            marker=dict(color='#0066CC'),
            text=sub_stats['Count'],
            textposition='auto',
            name='Count',
            hovertemplate='<b>%{y}</b><br>Tickets: %{x:,}<extra></extra>'
        ),
        row=1, col=1
    )

    col_idx = 2

    # Financial impact bars
    if cost_col and 'Total_Cost' in sub_stats.columns:
        fig.add_trace(
            go.Bar(
                y=sub_stats.index,
                x=sub_stats['Total_Cost'],
                orientation='h',
                marker=dict(color='#DC3545'),
                text=[f'${v:,.0f}' for v in sub_stats['Total_Cost']],
                textposition='auto',
                name='Cost',
                hovertemplate='<b>%{y}</b><br>Impact: $%{x:,.0f}<extra></extra>'
            ),
            row=1, col=col_idx
        )
        col_idx += 1

    # Recurrence rate bars
    if rec_col and 'Recurrence' in sub_stats.columns:
        fig.add_trace(
            go.Bar(
                y=sub_stats.index,
                x=sub_stats['Recurrence'] * 100,
                orientation='h',
                marker=dict(
                    color=sub_stats['Recurrence'] * 100,
                    colorscale='RdYlGn_r',
                    showscale=False
                ),
                text=[f'{v:.0f}%' for v in sub_stats['Recurrence'] * 100],
                textposition='auto',
                name='Recurrence',
                hovertemplate='<b>%{y}</b><br>Recurrence: %{x:.1f}%<extra></extra>'
            ),
            row=1, col=col_idx
        )

    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text=title_text, font=dict(size=18)),
        height=500,
        showlegend=False
    )

    return fig


def chart_category_financial_drilldown(df: pd.DataFrame) -> go.Figure:
    """
    Financial analysis with category/sub-category drill-down.
    Bar chart showing financial metrics by category with expandable sub-categories.
    """
    if 'AI_Category' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Category data required", x=0.5, y=0.5)
        return fig

    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None
    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None

    if not cost_col:
        # Generate estimated costs
        df = df.copy()
        df['Financial_Impact'] = 850  # Default cost per escalation
        cost_col = 'Financial_Impact'

    # Category-level aggregation
    cat_stats = df.groupby('AI_Category').agg({
        cost_col: ['sum', 'mean', 'count']
    }).round(2)
    cat_stats.columns = ['Total_Cost', 'Avg_Cost', 'Count']
    cat_stats = cat_stats.sort_values('Total_Cost', ascending=True)

    fig = go.Figure()

    # Main category bars
    fig.add_trace(go.Bar(
        y=cat_stats.index,
        x=cat_stats['Total_Cost'],
        orientation='h',
        name='Total Financial Impact',
        marker=dict(color='#0066CC', line=dict(color='white', width=1)),
        text=[f"${v:,.0f}" for v in cat_stats['Total_Cost']],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Total Impact: $%{x:,.0f}<br>'
            f'Tickets: %{{customdata[0]:,}}<br>'
            f'Avg/Ticket: $%{{customdata[1]:,.0f}}<extra></extra>'
        ),
        customdata=list(zip(cat_stats['Count'], cat_stats['Avg_Cost']))
    ))

    fig.update_layout(
        **{
            **create_plotly_theme(),
            'margin': dict(l=200, r=80, t=80, b=40),
        },
        title=dict(
            text='Financial Impact by Category<br><span style="font-size:12px">Hover for details • Use sunburst/treemap for sub-category drill-down</span>',
            font=dict(size=18)
        ),
        xaxis_title='Total Financial Impact ($)',
        yaxis_title='',
        height=450,
        showlegend=False
    )

    # Format x-axis as currency
    fig.update_xaxes(tickprefix='$', tickformat=',.0f')

    return fig


def chart_subcategory_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comparison table DataFrame for sub-categories.
    Returns a styled DataFrame that can be displayed in Streamlit.
    """
    if 'AI_Category' not in df.columns or 'AI_Sub_Category' not in df.columns:
        return pd.DataFrame()

    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None
    rec_col = 'AI_Recurrence_Probability' if 'AI_Recurrence_Probability' in df.columns else None

    # Aggregate data
    agg_dict = {
        'AI_Sub_Category': 'count'
    }
    if cost_col:
        agg_dict[cost_col] = ['sum', 'mean']
    if rec_col:
        agg_dict[rec_col] = 'mean'

    sub_stats = df.groupby(['AI_Category', 'AI_Sub_Category']).agg(agg_dict)
    sub_stats.columns = ['_'.join(col).strip('_') for col in sub_stats.columns.values]
    sub_stats = sub_stats.reset_index()

    # Rename columns for clarity
    rename_map = {
        'AI_Category': 'Category',
        'AI_Sub_Category': 'Sub-Category',
        'AI_Sub_Category_count': 'Tickets'
    }
    if cost_col:
        rename_map[f'{cost_col}_sum'] = 'Total Impact'
        rename_map[f'{cost_col}_mean'] = 'Avg Impact'
    if rec_col:
        rename_map[f'{rec_col}_mean'] = 'Recurrence %'

    sub_stats = sub_stats.rename(columns=rename_map)

    # Format columns
    if 'Total Impact' in sub_stats.columns:
        sub_stats['Total Impact'] = sub_stats['Total Impact'].apply(lambda x: f'${x:,.0f}')
    if 'Avg Impact' in sub_stats.columns:
        sub_stats['Avg Impact'] = sub_stats['Avg Impact'].apply(lambda x: f'${x:,.0f}')
    if 'Recurrence %' in sub_stats.columns:
        sub_stats['Recurrence %'] = sub_stats['Recurrence %'].apply(lambda x: f'{x*100:.1f}%')

    # Sort by category then by tickets
    sub_stats = sub_stats.sort_values(['Category', 'Tickets'], ascending=[True, False])

    return sub_stats
