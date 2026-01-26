"""
Escalation AI - AI-powered telecom escalation analysis with McKinsey-style reporting.

This package provides comprehensive analysis of telecom escalation tickets including:
- AI-powered classification using semantic embeddings
- Strategic friction scoring
- Recidivism detection and pattern analysis
- Similar ticket finding with resolution comparison
- ML-based resolution time prediction
- Executive-quality visualization and reporting
"""

__version__ = "2.2.0"
__author__ = "Escalation AI Team"

# Core imports
from .core.config import *
from .core.ai_engine import OllamaBrain
from .core.utils import clean_text, validate_columns
from .core.gpu_utils import (
    is_gpu_available,
    get_gpu_info,
    init_rapids,
    clear_gpu_memory,
)

# Classification and Scoring
from .classification import classify_rows, get_anchor_centroids
from .scoring import calculate_strategic_friction

# Feedback modules
from .feedback import FeedbackLearning, PriceCatalog

# Predictors
from .predictors import (
    RecurrencePredictor,
    SimilarTicketFinder,
    ResolutionTimePredictor,
    apply_recurrence_predictions,
    apply_similar_ticket_analysis,
    apply_resolution_time_prediction
)

# Pipeline and Reports
from .pipeline import EscalationPipeline, main_pipeline
from .reports import generate_report

# Visualization
from .visualization import ChartGenerator

# Alerting
from .alerting import (
    SmartThresholdCalculator,
    ThresholdConfig,
    AlertLevel,
    calculate_dynamic_thresholds,
    check_threshold_breach,
    get_adaptive_limits
)

# Analysis
from .analysis import (
    CategoryDriftDetector,
    DriftResult,
    DriftType,
    detect_category_drift,
    compare_periods,
    get_emerging_categories,
    get_declining_categories
)

__all__ = [
    # Core
    'OllamaBrain',
    'clean_text',
    'validate_columns',
    
    # Classification & Scoring
    'classify_rows',
    'get_anchor_centroids',
    'calculate_strategic_friction',
    
    # Feedback
    'FeedbackLearning',
    'PriceCatalog',
    
    # Predictors
    'RecurrencePredictor',
    'SimilarTicketFinder',
    'ResolutionTimePredictor',
    'apply_recurrence_predictions',
    'apply_similar_ticket_analysis',
    'apply_resolution_time_prediction',
    
    # Pipeline & Reports
    'EscalationPipeline',
    'main_pipeline',
    'generate_report',
    
    # Visualization
    'ChartGenerator',
    
    # GPU Utilities
    'is_gpu_available',
    'get_gpu_info',
    'init_rapids',
    'clear_gpu_memory',
    
    # Alerting
    'SmartThresholdCalculator',
    'ThresholdConfig',
    'AlertLevel',
    'calculate_dynamic_thresholds',
    'check_threshold_breach',
    'get_adaptive_limits',
    
    # Analysis
    'CategoryDriftDetector',
    'DriftResult',
    'DriftType',
    'detect_category_drift',
    'compare_periods',
    'get_emerging_categories',
    'get_declining_categories',
    
    # Metadata
    '__version__',
]
