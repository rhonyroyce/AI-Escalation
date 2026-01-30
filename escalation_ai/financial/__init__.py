"""
Financial metrics and analysis module.
Provides advanced financial calculations, ROI analysis, and cost optimization insights.
"""

from .metrics import (
    FinancialMetrics,
    calculate_financial_metrics,
    calculate_roi_metrics,
    calculate_cost_avoidance,
    calculate_efficiency_metrics,
    calculate_financial_forecasts,
    generate_financial_insights
)

from .visualizations import (
    create_financial_waterfall,
    create_roi_opportunity_chart,
    create_cost_avoidance_breakdown,
    create_cost_trend_forecast,
    create_efficiency_scorecard,
    create_category_cost_comparison,
    create_engineer_cost_efficiency_matrix,
    create_financial_kpi_cards,
    create_insights_table,
    create_cost_concentration_chart
)

__all__ = [
    'FinancialMetrics',
    'calculate_financial_metrics',
    'calculate_roi_metrics',
    'calculate_cost_avoidance',
    'calculate_efficiency_metrics',
    'calculate_financial_forecasts',
    'generate_financial_insights',
    'create_financial_waterfall',
    'create_roi_opportunity_chart',
    'create_cost_avoidance_breakdown',
    'create_cost_trend_forecast',
    'create_efficiency_scorecard',
    'create_category_cost_comparison',
    'create_engineer_cost_efficiency_matrix',
    'create_financial_kpi_cards',
    'create_insights_table',
    'create_cost_concentration_chart'
]
