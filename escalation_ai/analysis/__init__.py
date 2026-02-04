"""
Analysis modules for escalation AI.

Includes:
- Category Drift Detection
- Lessons Learned Effectiveness Analysis
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

from .lessons_learned import (
    LessonsLearnedAnalyzer,
    LearningGrade,
    LearningScore,
    get_lessons_analyzer,
    analyze_lessons_learned
)

__all__ = [
    # Category Drift
    'CategoryDriftDetector',
    'DriftResult',
    'DriftType',
    'detect_category_drift',
    'compare_periods',
    'get_emerging_categories',
    'get_declining_categories',
    # Lessons Learned
    'LessonsLearnedAnalyzer',
    'LearningGrade',
    'LearningScore',
    'get_lessons_analyzer',
    'analyze_lessons_learned',
]
