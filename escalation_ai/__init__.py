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
from .core.config import (
    # AI model selection
    EMBED_MODEL,
    GEN_MODEL,
    DETECTED_VRAM_GB,
    OLLAMA_BASE_URL,
    VISION_MODEL,
    VISION_MODEL_TIMEOUT,
    USE_GPU,
    # Scoring weights & thresholds
    WEIGHTS,
    MIN_CLASSIFICATION_CONFIDENCE,
    # Classification taxonomy
    ANCHORS,
    SUB_CATEGORIES,
    CATEGORY_KEYWORDS,
    ROOT_CAUSE_CATEGORIES,
    # Column mappings
    COL_CATEGORY,
    COL_CLOSE_DATE,
    COL_DATETIME,
    COL_ENGINEER,
    COL_IMPACT,
    COL_LESSON_STATUS,
    COL_LESSON_TITLE,
    COL_LOB,
    COL_ORIGIN,
    COL_RECURRENCE_RISK,
    COL_RESOLUTION_DATE,
    COL_RESOLUTION_NOTES,
    COL_ROOT_CAUSE,
    COL_SEVERITY,
    COL_SUMMARY,
    COL_TYPE,
    REQUIRED_COLUMNS,
    # File paths & output
    FEEDBACK_FILE,
    FEEDBACK_WEIGHT,
    PLOT_DIR,
    PRICE_CATALOG_FILE,
    RECURRENCE_ENCODERS_PATH,
    RECURRENCE_MODEL_PATH,
    RESOLUTION_MODEL_PATH,
    SIMILARITY_FEEDBACK_PATH,
    # Financial & reporting
    DEFAULT_HOURLY_RATE,
    MC_BLUE,
    REPORT_TITLE,
    REPORT_VERSION,
    # Logging
    logger,
)
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
