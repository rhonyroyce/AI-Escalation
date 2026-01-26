"""
Models module for Escalation AI.

Contains data models, ML predictors, and analysis components.
"""

# Data models are always available
from .data_models import (
    TicketData,
    AnalysisResult,
    SimilarTicketMatch,
    ResolutionTimePrediction,
    validate_dataframe,
    normalize_severity,
    normalize_origin,
    calculate_resolution_days
)

__all__ = [
    # Data models
    'TicketData',
    'AnalysisResult', 
    'SimilarTicketMatch',
    'ResolutionTimePrediction',
    'validate_dataframe',
    'normalize_severity',
    'normalize_origin',
    'calculate_resolution_days',
]
