"""
Predictors module for Escalation AI.

Contains ML-based predictive models:
- RecurrencePredictor: Predicts ticket recurrence probability
- SimilarTicketFinder: Finds semantically similar historical tickets
- ResolutionTimePredictor: Predicts resolution time based on patterns
"""

from .recurrence import RecurrencePredictor, apply_recurrence_predictions
from .similar_tickets import SimilarTicketFinder, apply_similar_ticket_analysis
from .resolution_time import ResolutionTimePredictor, apply_resolution_time_prediction
from .vector_store import TicketVectorStore, FAISS_AVAILABLE

__all__ = [
    'RecurrencePredictor',
    'SimilarTicketFinder',
    'ResolutionTimePredictor',
    'TicketVectorStore',
    'FAISS_AVAILABLE',
    'apply_recurrence_predictions',
    'apply_similar_ticket_analysis',
    'apply_resolution_time_prediction'
]
