"""
Advanced financial metrics and analysis for escalation costs.

This module provides:
- ROI and cost avoidance calculations
- Financial trend analysis and forecasting
- Cost efficiency and benchmark metrics
- Risk-adjusted financial exposure
- Cost optimization recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinancialMetrics:
    """Container for comprehensive financial metrics."""

    # Core metrics
    total_cost: float = 0.0
    total_tickets: int = 0
    avg_cost_per_ticket: float = 0.0
    median_cost: float = 0.0

    # Distribution metrics
    cost_by_category: Dict[str, float] = field(default_factory=dict)
    cost_by_severity: Dict[str, float] = field(default_factory=dict)
    cost_by_engineer: Dict[str, float] = field(default_factory=dict)
    cost_by_lob: Dict[str, float] = field(default_factory=dict)

    # Efficiency metrics
    cost_per_resolution_hour: float = 0.0
    high_cost_tickets_count: int = 0
    high_cost_percentage: float = 0.0
    cost_concentration_ratio: float = 0.0  # Top 20% tickets / Total cost

    # ROI metrics
    recurring_issue_cost: float = 0.0
    preventable_cost: float = 0.0
    roi_opportunity: float = 0.0
    cost_avoidance_potential: float = 0.0

    # Trend metrics
    monthly_cost_trend: Dict[str, float] = field(default_factory=dict)
    cost_velocity: float = 0.0  # Rate of cost change
    cost_forecast_30d: float = 0.0
    cost_forecast_90d: float = 0.0

    # Risk metrics
    risk_adjusted_exposure: float = 0.0
    recurrence_exposure: float = 0.0
    critical_cost_ratio: float = 0.0

    # Benchmark metrics
    target_cost_per_ticket: float = 500.0
    cost_efficiency_score: float = 0.0  # 0-100 score
    cost_vs_benchmark: float = 0.0

    # Business impact
    revenue_at_risk: float = 0.0
    customer_impact_cost: float = 0.0
    sla_penalty_exposure: float = 0.0
    opportunity_cost: float = 0.0


def calculate_financial_metrics(df: pd.DataFrame) -> FinancialMetrics:
    """
    Calculate comprehensive financial metrics from escalation data.

    Args:
        df: DataFrame with Financial_Impact and other escalation columns

    Returns:
        FinancialMetrics object with all calculated metrics
    """
    metrics = FinancialMetrics()

    if df.empty or 'Financial_Impact' not in df.columns:
        logger.warning("No financial data available")
        return metrics

    # Core metrics
    metrics.total_cost = df['Financial_Impact'].sum()
    metrics.total_tickets = len(df)
    metrics.avg_cost_per_ticket = metrics.total_cost / metrics.total_tickets if metrics.total_tickets > 0 else 0
    metrics.median_cost = df['Financial_Impact'].median()

    # Distribution metrics
    if 'AI_Category' in df.columns:
        metrics.cost_by_category = df.groupby('AI_Category')['Financial_Impact'].sum().to_dict()

    if 'Severity_Norm' in df.columns:
        metrics.cost_by_severity = df.groupby('Severity_Norm')['Financial_Impact'].sum().to_dict()

    if 'Engineer_Assigned' in df.columns:
        metrics.cost_by_engineer = df.groupby('Engineer_Assigned')['Financial_Impact'].sum().to_dict()

    if 'LOB' in df.columns:
        metrics.cost_by_lob = df.groupby('LOB')['Financial_Impact'].sum().to_dict()

    # Efficiency metrics
    if 'Resolution_Days' in df.columns:
        total_hours = df['Resolution_Days'].sum() * 24
        metrics.cost_per_resolution_hour = metrics.total_cost / total_hours if total_hours > 0 else 0

    # High cost tickets (top 10%)
    high_cost_threshold = df['Financial_Impact'].quantile(0.9)
    metrics.high_cost_tickets_count = (df['Financial_Impact'] >= high_cost_threshold).sum()
    metrics.high_cost_percentage = (metrics.high_cost_tickets_count / metrics.total_tickets * 100) if metrics.total_tickets > 0 else 0

    # Cost concentration (80/20 rule)
    sorted_costs = df['Financial_Impact'].sort_values(ascending=False)
    top_20_percent_count = int(len(sorted_costs) * 0.2)
    top_20_cost = sorted_costs.head(top_20_percent_count).sum()
    metrics.cost_concentration_ratio = (top_20_cost / metrics.total_cost) if metrics.total_cost > 0 else 0

    # ROI metrics
    metrics.recurring_issue_cost = _calculate_recurring_cost(df)
    metrics.preventable_cost = _calculate_preventable_cost(df)
    metrics.roi_opportunity = metrics.preventable_cost * 0.8  # 80% of preventable cost
    metrics.cost_avoidance_potential = _calculate_cost_avoidance(df)

    # Trend metrics
    if 'Issue_Date' in df.columns:
        metrics.monthly_cost_trend = _calculate_monthly_trend(df)
        metrics.cost_velocity = _calculate_cost_velocity(df)
        metrics.cost_forecast_30d = _forecast_costs(df, days=30)
        metrics.cost_forecast_90d = _forecast_costs(df, days=90)

    # Risk metrics
    metrics.risk_adjusted_exposure = _calculate_risk_exposure(df)
    metrics.recurrence_exposure = _calculate_recurrence_exposure(df)

    if 'Severity_Norm' in df.columns:
        critical_cost = df[df['Severity_Norm'] == 'Critical']['Financial_Impact'].sum()
        metrics.critical_cost_ratio = (critical_cost / metrics.total_cost) if metrics.total_cost > 0 else 0

    # Benchmark metrics
    metrics.cost_efficiency_score = _calculate_efficiency_score(df, metrics)
    metrics.cost_vs_benchmark = metrics.avg_cost_per_ticket - metrics.target_cost_per_ticket

    # Business impact
    metrics.revenue_at_risk = metrics.total_cost * 2.5  # 2.5x multiplier for downstream impact
    metrics.customer_impact_cost = _calculate_customer_impact(df)
    metrics.sla_penalty_exposure = _calculate_sla_penalty(df)
    metrics.opportunity_cost = metrics.total_cost * 0.35  # 35% opportunity cost

    logger.info(f"✓ Calculated comprehensive financial metrics: ${metrics.total_cost:,.2f} total")

    return metrics


def _calculate_recurring_cost(df: pd.DataFrame) -> float:
    """Calculate cost of recurring issues."""
    if 'AI_Recurrence_Risk' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    try:
        recurrence_risk = pd.to_numeric(df['AI_Recurrence_Risk'], errors='coerce').fillna(0)
        # Issues with >30% recurrence probability
        recurring_mask = recurrence_risk > 0.3
        return df[recurring_mask]['Financial_Impact'].sum()
    except Exception:
        return 0.0


def _calculate_preventable_cost(df: pd.DataFrame) -> float:
    """Calculate cost of preventable issues."""
    if 'AI_Category' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    # Categories that are typically preventable
    preventable_categories = [
        'Process & Documentation',
        'Communication & Coordination',
        'Configuration & Integration',
        'Contractor & Vendor Issues'
    ]

    preventable_mask = df['AI_Category'].isin(preventable_categories)
    return df[preventable_mask]['Financial_Impact'].sum()


def _calculate_cost_avoidance(df: pd.DataFrame) -> float:
    """Calculate potential cost avoidance through root cause fixes."""
    if 'Similar_Tickets_Found' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    try:
        # Tickets with similar historical issues
        similar_count = pd.to_numeric(df['Similar_Tickets_Found'], errors='coerce').fillna(0)
        repeat_mask = similar_count > 2

        # Cost avoidance = cost of repeat issues if root cause was fixed
        return df[repeat_mask]['Financial_Impact'].sum() * 0.7  # 70% avoidable
    except Exception:
        return 0.0


def _calculate_monthly_trend(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate monthly cost trends."""
    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        df_temp['Month'] = df_temp['Issue_Date'].dt.to_period('M').astype(str)
        monthly_costs = df_temp.groupby('Month')['Financial_Impact'].sum().to_dict()

        return monthly_costs
    except Exception:
        return {}


def _calculate_cost_velocity(df: pd.DataFrame) -> float:
    """Calculate rate of cost change ($ per day)."""
    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) < 2:
            return 0.0

        df_temp = df_temp.sort_values('Issue_Date')
        df_temp['Days_From_Start'] = (df_temp['Issue_Date'] - df_temp['Issue_Date'].min()).dt.days

        # Linear regression for cost velocity
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(df_temp['Days_From_Start'], df_temp['Financial_Impact'], 1)

        return float(p.coef[1])  # Slope = cost per day
    except Exception:
        return 0.0


def _forecast_costs(df: pd.DataFrame, days: int = 30) -> float:
    """Forecast future costs based on historical trends."""
    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) < 5:
            # Not enough data, use average
            return df['Financial_Impact'].mean() * days

        # Calculate average cost per day
        date_range = (df_temp['Issue_Date'].max() - df_temp['Issue_Date'].min()).days
        if date_range == 0:
            return 0.0

        total_cost = df_temp['Financial_Impact'].sum()
        cost_per_day = total_cost / date_range

        return cost_per_day * days
    except Exception:
        return 0.0


def _calculate_risk_exposure(df: pd.DataFrame) -> float:
    """Calculate risk-adjusted financial exposure."""
    if 'Financial_Impact' not in df.columns:
        return 0.0

    risk_weights = {
        'Critical': 1.0,
        'High': 0.7,
        'Major': 0.7,
        'Medium': 0.4,
        'Minor': 0.4,
        'Low': 0.2
    }

    if 'Severity_Norm' not in df.columns:
        return df['Financial_Impact'].sum()

    total_exposure = 0.0
    for severity, weight in risk_weights.items():
        mask = df['Severity_Norm'] == severity
        total_exposure += df[mask]['Financial_Impact'].sum() * weight

    return total_exposure


def _calculate_recurrence_exposure(df: pd.DataFrame) -> float:
    """Calculate financial exposure from recurrence risk."""
    if 'AI_Recurrence_Risk' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    try:
        recurrence_risk = pd.to_numeric(df['AI_Recurrence_Risk'], errors='coerce').fillna(0)
        financial_impact = pd.to_numeric(df['Financial_Impact'], errors='coerce').fillna(0)

        return (recurrence_risk * financial_impact).sum()
    except Exception:
        return 0.0


def _calculate_customer_impact(df: pd.DataFrame) -> float:
    """Calculate customer-facing impact costs."""
    if 'Origin_Norm' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    # Customer-facing issues have higher business impact
    customer_facing = ['External', 'Customer']
    mask = df['Origin_Norm'].isin(customer_facing)

    # 1.5x multiplier for customer impact
    return df[mask]['Financial_Impact'].sum() * 1.5


def _calculate_sla_penalty(df: pd.DataFrame) -> float:
    """Estimate SLA penalty exposure."""
    if 'Severity_Norm' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    # Critical issues often have SLA penalties
    critical_mask = df['Severity_Norm'] == 'Critical'
    critical_cost = df[critical_mask]['Financial_Impact'].sum()

    # Assume 20% SLA penalty rate for critical issues
    return critical_cost * 0.2


def _calculate_efficiency_score(df: pd.DataFrame, metrics: FinancialMetrics) -> float:
    """
    Calculate cost efficiency score (0-100).

    Higher score = better efficiency
    """
    score = 100.0

    # Penalty for high average cost
    if metrics.avg_cost_per_ticket > metrics.target_cost_per_ticket:
        cost_ratio = metrics.avg_cost_per_ticket / metrics.target_cost_per_ticket
        score -= min(30, (cost_ratio - 1) * 20)  # Max 30 point penalty

    # Penalty for high cost concentration
    if metrics.cost_concentration_ratio > 0.8:  # Worse than 80/20
        score -= (metrics.cost_concentration_ratio - 0.8) * 50

    # Penalty for high recurring cost
    if metrics.total_cost > 0:
        recurring_ratio = metrics.recurring_issue_cost / metrics.total_cost
        score -= recurring_ratio * 20  # Max 20 point penalty

    # Penalty for high critical cost ratio
    score -= metrics.critical_cost_ratio * 20  # Max 20 point penalty

    return max(0, min(100, score))


def calculate_roi_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate ROI opportunities from fixing root causes.

    Returns:
        Dictionary with ROI analysis including:
        - Investment required
        - Expected savings
        - ROI percentage
        - Payback period
        - Top opportunities
    """
    roi_analysis = {
        'total_investment_required': 0.0,
        'expected_annual_savings': 0.0,
        'roi_percentage': 0.0,
        'payback_months': 0.0,
        'top_opportunities': []
    }

    if 'Financial_Impact' not in df.columns or 'AI_Category' not in df.columns:
        return roi_analysis

    # Calculate by category
    category_costs = df.groupby('AI_Category')['Financial_Impact'].agg(['sum', 'count', 'mean'])
    category_costs = category_costs.sort_values('sum', ascending=False)

    # Top opportunities = categories with highest total cost and multiple incidents
    for category in category_costs.index[:5]:
        total_cost = category_costs.loc[category, 'sum']
        count = int(category_costs.loc[category, 'count'])
        avg_cost = category_costs.loc[category, 'mean']

        if count >= 3:  # Multiple incidents = pattern
            # Investment = 10x average ticket cost for root cause fix
            investment = avg_cost * 10

            # Savings = 80% of recurring issues prevented
            annual_savings = total_cost * 4 * 0.8  # Extrapolate to year, 80% prevention

            # ROI
            roi_pct = ((annual_savings - investment) / investment * 100) if investment > 0 else 0
            payback_months = (investment / (annual_savings / 12)) if annual_savings > 0 else float('inf')

            roi_analysis['top_opportunities'].append({
                'category': category,
                'total_cost': total_cost,
                'incident_count': count,
                'avg_cost': avg_cost,
                'investment_required': investment,
                'annual_savings': annual_savings,
                'roi_percentage': roi_pct,
                'payback_months': payback_months
            })

    # Aggregate
    if roi_analysis['top_opportunities']:
        roi_analysis['total_investment_required'] = sum(opp['investment_required'] for opp in roi_analysis['top_opportunities'])
        roi_analysis['expected_annual_savings'] = sum(opp['annual_savings'] for opp in roi_analysis['top_opportunities'])

        if roi_analysis['total_investment_required'] > 0:
            roi_analysis['roi_percentage'] = (
                (roi_analysis['expected_annual_savings'] - roi_analysis['total_investment_required']) /
                roi_analysis['total_investment_required'] * 100
            )
            roi_analysis['payback_months'] = (
                roi_analysis['total_investment_required'] /
                (roi_analysis['expected_annual_savings'] / 12)
            )

    return roi_analysis


def calculate_cost_avoidance(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate potential cost avoidance opportunities."""
    avoidance = {
        'recurring_issues': 0.0,
        'preventable_categories': 0.0,
        'knowledge_sharing': 0.0,
        'automation': 0.0,
        'total_avoidance': 0.0
    }

    if 'Financial_Impact' not in df.columns:
        return avoidance

    # Recurring issues
    avoidance['recurring_issues'] = _calculate_recurring_cost(df) * 0.8

    # Preventable categories
    avoidance['preventable_categories'] = _calculate_preventable_cost(df) * 0.6

    # Knowledge sharing (tickets with similar historical issues)
    avoidance['knowledge_sharing'] = _calculate_cost_avoidance(df)

    # Automation potential (repetitive categories)
    if 'AI_Category' in df.columns:
        automatable_categories = [
            'Configuration & Integration',
            'OSS/NMS & Systems',
            'Process & Documentation'
        ]
        auto_mask = df['AI_Category'].isin(automatable_categories)
        avoidance['automation'] = df[auto_mask]['Financial_Impact'].sum() * 0.5

    avoidance['total_avoidance'] = sum([
        avoidance['recurring_issues'],
        avoidance['preventable_categories'],
        avoidance['knowledge_sharing'],
        avoidance['automation']
    ])

    return avoidance


def calculate_efficiency_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate cost efficiency and performance metrics."""
    efficiency = {
        'cost_per_hour': 0.0,
        'cost_per_engineer': {},
        'cost_per_category': {},
        'engineer_efficiency_scores': {},
        'category_efficiency_scores': {},
        'outliers': []
    }

    if 'Financial_Impact' not in df.columns:
        return efficiency

    # Cost per resolution hour
    if 'Resolution_Days' in df.columns:
        total_hours = df['Resolution_Days'].sum() * 24
        efficiency['cost_per_hour'] = df['Financial_Impact'].sum() / total_hours if total_hours > 0 else 0

    # Cost per engineer
    if 'Engineer_Assigned' in df.columns:
        efficiency['cost_per_engineer'] = df.groupby('Engineer_Assigned')['Financial_Impact'].mean().to_dict()

        # Engineer efficiency scores (lower cost = higher score)
        avg_cost = df['Financial_Impact'].mean()
        for engineer, cost in efficiency['cost_per_engineer'].items():
            ratio = avg_cost / cost if cost > 0 else 1.0
            efficiency['engineer_efficiency_scores'][engineer] = min(100, ratio * 100)

    # Cost per category
    if 'AI_Category' in df.columns:
        efficiency['cost_per_category'] = df.groupby('AI_Category')['Financial_Impact'].mean().to_dict()

    # Find high-cost outliers
    threshold = df['Financial_Impact'].quantile(0.95)
    outliers = df[df['Financial_Impact'] >= threshold]

    for idx, row in outliers.iterrows():
        efficiency['outliers'].append({
            'id': row.get('Ticket_ID', idx),
            'cost': row['Financial_Impact'],
            'category': row.get('AI_Category', 'Unknown'),
            'severity': row.get('Severity_Norm', 'Unknown')
        })

    return efficiency


def calculate_financial_forecasts(df: pd.DataFrame, periods: int = 12) -> Dict[str, Any]:
    """
    Generate financial forecasts and projections.

    Args:
        df: Escalation data
        periods: Number of months to forecast

    Returns:
        Forecast data including trends, projections, and confidence intervals
    """
    forecasts = {
        'monthly_projection': [],
        'annual_projection': 0.0,
        'trend': 'stable',  # 'increasing', 'decreasing', 'stable'
        'confidence': 'low',  # 'high', 'medium', 'low'
        'risk_scenarios': {}
    }

    if 'Issue_Date' not in df.columns or 'Financial_Impact' not in df.columns:
        return forecasts

    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) < 3:
            forecasts['confidence'] = 'low'
            return forecasts

        # Calculate monthly historical data
        df_temp['Month'] = df_temp['Issue_Date'].dt.to_period('M')
        monthly_data = df_temp.groupby('Month')['Financial_Impact'].sum()

        # Calculate trend
        if len(monthly_data) >= 3:
            recent_avg = monthly_data.tail(3).mean()
            older_avg = monthly_data.head(len(monthly_data) - 3).mean()

            if recent_avg > older_avg * 1.1:
                forecasts['trend'] = 'increasing'
            elif recent_avg < older_avg * 0.9:
                forecasts['trend'] = 'decreasing'
            else:
                forecasts['trend'] = 'stable'

        # Simple moving average forecast
        avg_monthly_cost = monthly_data.mean()
        std_monthly_cost = monthly_data.std()

        for i in range(1, periods + 1):
            forecasts['monthly_projection'].append({
                'month': i,
                'projected_cost': avg_monthly_cost,
                'lower_bound': max(0, avg_monthly_cost - std_monthly_cost),
                'upper_bound': avg_monthly_cost + std_monthly_cost
            })

        forecasts['annual_projection'] = avg_monthly_cost * 12

        # Confidence level
        cv = (std_monthly_cost / avg_monthly_cost) if avg_monthly_cost > 0 else 1.0
        if cv < 0.3:
            forecasts['confidence'] = 'high'
        elif cv < 0.6:
            forecasts['confidence'] = 'medium'
        else:
            forecasts['confidence'] = 'low'

        # Risk scenarios
        forecasts['risk_scenarios'] = {
            'best_case': avg_monthly_cost * 12 * 0.8,  # 20% reduction
            'expected': avg_monthly_cost * 12,
            'worst_case': avg_monthly_cost * 12 * 1.3  # 30% increase
        }

    except Exception as e:
        logger.warning(f"Error calculating forecasts: {e}")

    return forecasts


def generate_financial_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generate actionable financial insights and recommendations.

    Returns:
        List of insights with priority, type, and recommendation
    """
    insights = []

    if df.empty or 'Financial_Impact' not in df.columns:
        return insights

    metrics = calculate_financial_metrics(df)
    roi_data = calculate_roi_metrics(df)

    # High cost concentration
    if metrics.cost_concentration_ratio > 0.85:
        insights.append({
            'priority': 'high',
            'type': 'cost_concentration',
            'title': 'High Cost Concentration Detected',
            'description': f'{metrics.cost_concentration_ratio*100:.0f}% of costs come from top 20% of tickets',
            'recommendation': 'Focus root cause analysis on high-cost tickets for maximum impact',
            'potential_savings': metrics.total_cost * 0.3
        })

    # High recurring issue cost
    if metrics.recurring_issue_cost > metrics.total_cost * 0.3:
        insights.append({
            'priority': 'high',
            'type': 'recurring_issues',
            'title': 'Significant Recurring Issue Costs',
            'description': f'${metrics.recurring_issue_cost:,.0f} in recurring issues',
            'recommendation': 'Implement preventive measures for high-recurrence categories',
            'potential_savings': metrics.recurring_issue_cost * 0.7
        })

    # ROI opportunities
    if roi_data['top_opportunities']:
        top_roi = roi_data['top_opportunities'][0]
        if top_roi['roi_percentage'] > 200:
            insights.append({
                'priority': 'high',
                'type': 'roi_opportunity',
                'title': f'High ROI Opportunity: {top_roi["category"]}',
                'description': f'{top_roi["roi_percentage"]:.0f}% ROI, {top_roi["payback_months"]:.1f} month payback',
                'recommendation': f'Invest ${top_roi["investment_required"]:,.0f} to save ${top_roi["annual_savings"]:,.0f}/year',
                'potential_savings': top_roi['annual_savings']
            })

    # Low efficiency score
    if metrics.cost_efficiency_score < 60:
        insights.append({
            'priority': 'medium',
            'type': 'efficiency',
            'title': 'Cost Efficiency Below Target',
            'description': f'Efficiency score: {metrics.cost_efficiency_score:.0f}/100',
            'recommendation': 'Review high-cost categories and implement process improvements',
            'potential_savings': metrics.preventable_cost * 0.5
        })

    # Preventable cost
    if metrics.preventable_cost > metrics.total_cost * 0.25:
        insights.append({
            'priority': 'medium',
            'type': 'preventable_cost',
            'title': 'High Preventable Cost',
            'description': f'${metrics.preventable_cost:,.0f} in preventable categories',
            'recommendation': 'Improve processes and documentation to prevent recurring issues',
            'potential_savings': metrics.preventable_cost * 0.6
        })

    # Increasing cost trend
    forecasts = calculate_financial_forecasts(df)
    if forecasts['trend'] == 'increasing':
        insights.append({
            'priority': 'high',
            'type': 'cost_trend',
            'title': 'Increasing Cost Trend',
            'description': f'Projected annual cost: ${forecasts["annual_projection"]:,.0f}',
            'recommendation': 'Immediate action required to reverse cost growth trend',
            'potential_savings': forecasts['annual_projection'] * 0.25
        })

    # Sort by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    insights.sort(key=lambda x: priority_order.get(x['priority'], 3))

    return insights
