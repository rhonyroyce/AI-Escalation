"""
Category Drift Detection Module.

Detects shifts in escalation categories over time using statistical 
tests and trend analysis to identify emerging patterns or declining issues.
"""

from .category_drift import (
    CategoryDriftDetector,
    DriftResult,
    DriftType,
    detect_category_drift,
    compare_periods,
    get_emerging_categories,
    get_declining_categories
)

__all__ = [
    'CategoryDriftDetector',
    'DriftResult',
    'DriftType',
    'detect_category_drift',
    'compare_periods',
    'get_emerging_categories',
    'get_declining_categories'
]
