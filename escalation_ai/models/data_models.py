"""
Data models and validation for escalation analysis.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TicketData:
    """Represents a single escalation ticket with all relevant fields."""
    id: str
    summary: str
    severity: str = "Default"
    type: str = ""
    origin: str = ""
    impact: str = "None"
    category: str = "Unclassified"
    engineer: str = "Unknown"
    lob: str = "Unknown"
    issue_date: Optional[datetime] = None
    close_date: Optional[datetime] = None
    resolution_notes: str = ""
    root_cause: str = ""
    recurrence_risk: str = "Unknown"
    
    # Computed fields
    ai_category: str = "Unclassified"
    ai_confidence: float = 0.0
    strategic_friction_score: float = 0.0
    financial_impact: float = 0.0
    risk_tier: str = "Low"
    
    # Similar ticket analysis
    similar_ticket_count: int = 0
    best_match_similarity: float = 0.0
    resolution_consistency: str = "Unknown"
    
    # Resolution time prediction
    predicted_resolution_days: float = 0.0
    expected_resolution_days: float = 0.0
    actual_resolution_days: float = 0.0


@dataclass
class AnalysisResult:
    """Container for all analysis results."""
    df: pd.DataFrame
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    engineer_patterns: Dict[str, Any] = field(default_factory=dict)
    recidivism_data: Dict[str, Any] = field(default_factory=dict)
    financial_summary: Dict[str, float] = field(default_factory=dict)
    ai_synthesis: str = ""
    charts: List[str] = field(default_factory=list)
    
    @property
    def total_friction_score(self) -> float:
        """Calculate total weighted friction score."""
        if 'Strategic_Friction_Score' in self.df.columns:
            return self.df['Strategic_Friction_Score'].sum()
        return 0.0
    
    @property
    def total_financial_impact(self) -> float:
        """Calculate total financial impact."""
        if 'Financial_Impact' in self.df.columns:
            return self.df['Financial_Impact'].sum()
        return 0.0
    
    @property
    def critical_count(self) -> int:
        """Count critical severity issues."""
        if 'Severity_Norm' in self.df.columns:
            return (self.df['Severity_Norm'] == 'Critical').sum()
        return 0
    
    @property
    def external_ratio(self) -> float:
        """Calculate ratio of external-facing issues."""
        if 'Origin_Norm' in self.df.columns:
            total = len(self.df)
            external = (self.df['Origin_Norm'] == 'External').sum()
            return external / total if total > 0 else 0.0
        return 0.0


@dataclass  
class SimilarTicketMatch:
    """Represents a similarity match between two tickets."""
    source_id: str
    match_id: str
    similarity_score: float
    match_type: str  # 'semantic', 'keyword', 'category'
    source_summary: str
    match_summary: str
    source_resolution_days: Optional[float] = None
    match_resolution_days: Optional[float] = None
    confidence: str = "Medium"  # High, Medium, Low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'match_id': self.match_id,
            'similarity_score': self.similarity_score,
            'match_type': self.match_type,
            'source_summary': self.source_summary[:200],
            'match_summary': self.match_summary[:200],
            'source_resolution_days': self.source_resolution_days,
            'match_resolution_days': self.match_resolution_days,
            'confidence': self.confidence
        }


@dataclass
class ResolutionTimePrediction:
    """Result of resolution time prediction."""
    ticket_id: str
    predicted_days: float
    confidence_interval: tuple = (0.0, 0.0)  # (lower, upper)
    prediction_method: str = "model"  # 'model', 'category_avg', 'fallback'
    similar_tickets_used: int = 0
    human_expected_days: Optional[float] = None
    
    @property
    def prediction_quality(self) -> str:
        """Assess quality of prediction."""
        if self.prediction_method == "model" and self.similar_tickets_used >= 3:
            return "High"
        elif self.prediction_method == "category_avg":
            return "Medium"
        return "Low"


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> tuple:
    """
    Validate that a DataFrame has required columns.
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return (len(missing) == 0, missing)


def normalize_severity(severity: str) -> str:
    """Normalize severity values to standard format."""
    if pd.isna(severity):
        return "Default"
    
    severity_map = {
        'critical': 'Critical',
        'crit': 'Critical',
        'p1': 'Critical',
        'major': 'Major',
        'high': 'Major',
        'p2': 'Major',
        'minor': 'Minor',
        'medium': 'Minor',
        'p3': 'Minor',
        'low': 'Low',
        'p4': 'Low',
        'info': 'Low',
    }
    
    severity_lower = str(severity).lower().strip()
    return severity_map.get(severity_lower, str(severity).title().strip())


def normalize_origin(origin: str) -> str:
    """Normalize origin values to standard format."""
    if pd.isna(origin):
        return "Unknown"
    
    origin_map = {
        'external': 'External',
        'customer': 'External',
        'vendor': 'External',
        'internal': 'Internal',
        'amdocs': 'Internal',
        'team': 'Internal',
    }
    
    origin_lower = str(origin).lower().strip()
    return origin_map.get(origin_lower, str(origin).title().strip())


def calculate_resolution_days(issue_date, close_date) -> Optional[float]:
    """Calculate resolution time in days."""
    if pd.isna(issue_date) or pd.isna(close_date):
        return None
    
    try:
        issue_dt = pd.to_datetime(issue_date)
        close_dt = pd.to_datetime(close_date)
        delta = (close_dt - issue_dt).total_seconds() / (24 * 3600)
        return max(0, delta)  # Ensure non-negative
    except Exception:
        return None
