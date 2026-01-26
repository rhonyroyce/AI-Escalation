"""
Core module for Escalation AI.

Contains configuration, AI engine, and base utilities.
"""

from escalation_ai.core.config import *
from escalation_ai.core.ai_engine import OllamaBrain, check_ollama_server, check_models
from escalation_ai.core.utils import clean_text, validate_columns, extract_keywords, calculate_keyword_overlap

__all__ = [
    'OllamaBrain',
    'check_ollama_server', 
    'check_models',
    'clean_text',
    'validate_columns',
    'extract_keywords',
    'calculate_keyword_overlap',
]
