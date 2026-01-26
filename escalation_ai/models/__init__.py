"""
Models module for Escalation AI.

Contains ML models for prediction, classification, and analysis.
"""

from escalation_ai.models.recurrence_predictor import RecurrencePredictor
from escalation_ai.models.similar_ticket_finder import SimilarTicketFinder
from escalation_ai.models.resolution_predictor import ResolutionTimePredictor
from escalation_ai.models.feedback_learning import FeedbackLearning
from escalation_ai.models.price_catalog import PriceCatalog

__all__ = [
    'RecurrencePredictor',
    'SimilarTicketFinder', 
    'ResolutionTimePredictor',
    'FeedbackLearning',
    'PriceCatalog',
]
