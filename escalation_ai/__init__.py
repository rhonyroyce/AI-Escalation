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

from escalation_ai.core.config import *
from escalation_ai.core.ai_engine import OllamaBrain
from escalation_ai.pipeline import main_pipeline

__all__ = [
    'main_pipeline',
    'OllamaBrain',
    '__version__',
]
