"""
Core module for Escalation AI.

Contains configuration, AI engine, GPU utilities, and base utilities.
"""

from escalation_ai.core.config import (
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
from escalation_ai.core.ai_engine import OllamaBrain, check_ollama_server, check_models
from escalation_ai.core.utils import clean_text, validate_columns, extract_keywords, calculate_keyword_overlap
from escalation_ai.core.gpu_utils import (
    is_gpu_available,
    get_gpu_info,
    init_rapids,
    GPURandomForestClassifier,
    GPURandomForestRegressor,
    GPUSimilaritySearch,
    cosine_similarity_gpu,
    batch_cosine_similarity_gpu,
    clear_gpu_memory,
    get_gpu_memory_usage,
)

__all__ = [
    # AI Engine
    'OllamaBrain',
    'check_ollama_server', 
    'check_models',
    # Utils
    'clean_text',
    'validate_columns',
    'extract_keywords',
    'calculate_keyword_overlap',
    # GPU Utils
    'is_gpu_available',
    'get_gpu_info',
    'init_rapids',
    'GPURandomForestClassifier',
    'GPURandomForestRegressor',
    'GPUSimilaritySearch',
    'cosine_similarity_gpu',
    'batch_cosine_similarity_gpu',
    'clear_gpu_memory',
    'get_gpu_memory_usage',
]
