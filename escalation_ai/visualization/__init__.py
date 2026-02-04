"""
Visualization module for McKinsey-style executive charts.
"""

from .chart_generator import ChartGenerator
from .advanced_charts import AdvancedChartGenerator
from .chart_insights import ChartInsightAnalyzer, get_chart_analyzer, analyze_chart_image
from ..core.config import MC_BLUE

__all__ = [
    'ChartGenerator',
    'AdvancedChartGenerator',
    'ChartInsightAnalyzer',
    'get_chart_analyzer',
    'analyze_chart_image',
    'MC_BLUE',
]
