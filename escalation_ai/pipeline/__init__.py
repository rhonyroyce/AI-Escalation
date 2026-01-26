"""
Pipeline module for Escalation AI.

Contains the main orchestration logic for the 7-phase analysis pipeline.
"""

from .orchestrator import (
    EscalationPipeline,
    main_pipeline,
    validate_data_quality,
    check_ollama_server,
    check_models
)

__all__ = [
    'EscalationPipeline',
    'main_pipeline',
    'validate_data_quality',
    'check_ollama_server',
    'check_models'
]
