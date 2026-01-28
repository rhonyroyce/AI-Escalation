"""
Visualization module for McKinsey-style executive charts.
"""

from .chart_generator import ChartGenerator
from .advanced_charts import AdvancedChartGenerator
from ..core.config import MC_BLUE

__all__ = [
    'ChartGenerator',
    'AdvancedChartGenerator',
    'MC_BLUE',
]
