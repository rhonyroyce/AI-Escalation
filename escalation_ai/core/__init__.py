"""
Core module for Escalation AI.

Contains configuration, AI engine, GPU utilities, and base utilities.
"""

from escalation_ai.core.config import *
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
