"""
Data models and validation for escalation analysis.

This module defines the **schema layer** for the Escalation AI pipeline.  It
provides typed dataclasses that describe the shape of every significant data
entity flowing through the system, plus utility functions for data
normalisation and validation.

Role in the pipeline
--------------------
While the pipeline itself operates primarily on ``pandas.DataFrame`` columns
(for speed and flexibility), these dataclasses serve three purposes:

1. **Documentation** -- they formally describe what fields exist on a ticket,
   what an analysis result contains, and what a similarity match or resolution
   prediction looks like.

2. **Serialisation** -- the ``to_dict()`` methods and ``@dataclass`` structure
   make it easy to convert entities to JSON or insert them into structured
   reports.

3. **Validation & normalisation** -- helper functions like
   ``validate_dataframe``, ``normalize_severity``, and ``normalize_origin``
   are used during data loading (Phase 0) to clean raw input data into a
   consistent format before the analysis phases begin.

Dataclass hierarchy
-------------------
::

    TicketData
        Individual escalation ticket with raw fields (id, summary, severity,
        etc.) and computed fields (ai_category, financial_impact, etc.)

    AnalysisResult
        Container for the full pipeline output: the enriched DataFrame plus
        aggregate statistics, chart paths, and the AI-generated synthesis.

    SimilarTicketMatch
        A single similarity match between two tickets (source + match),
        produced by Phase 5.

    ResolutionTimePrediction
        A single resolution-time prediction for one ticket, produced by
        Phase 6.

Normalisation conventions
-------------------------
- **Severity** is normalised to one of: ``Critical``, ``Major``, ``Minor``,
  ``Low``, or ``Default`` (via ``normalize_severity``).  Common aliases like
  ``"P1"``, ``"crit"``, ``"high"`` are mapped to their canonical forms.

- **Origin** is normalised to ``External``, ``Internal``, or ``Unknown``
  (via ``normalize_origin``).  Domain-specific aliases like ``"Amdocs"``
  (an internal team name) and ``"vendor"`` / ``"customer"`` are mapped
  accordingly.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============================================================================
# TICKET DATA MODEL
# ============================================================================

@dataclass
class TicketData:
    """Represents a single escalation ticket with all relevant fields.

    This dataclass mirrors the columns of the working DataFrame.  It is
    divided into three groups:

    **Raw fields** (populated during data loading):
        ``id``, ``summary``, ``severity``, ``type``, ``origin``, ``impact``,
        ``category``, ``engineer``, ``lob``, ``issue_date``, ``close_date``,
        ``resolution_notes``, ``root_cause``, ``recurrence_risk``.

    **AI-computed fields** (populated by pipeline phases 1-6):
        ``ai_category``, ``ai_confidence``, ``strategic_friction_score``,
        ``financial_impact``, ``risk_tier``.

    **Similarity & prediction fields** (populated by phases 5-6):
        ``similar_ticket_count``, ``best_match_similarity``,
        ``resolution_consistency``, ``predicted_resolution_days``,
        ``expected_resolution_days``, ``actual_resolution_days``.
    """
    # ---- Raw input fields ----
    id: str                                  # Unique ticket identifier (e.g. "ESC-1234")
    summary: str                             # Free-text description of the issue
    severity: str = "Default"                # Raw severity string (normalised later)
    type: str = ""                           # Ticket type (e.g. "Incident", "Problem")
    origin: str = ""                         # Where the ticket came from (e.g. "External")
    impact: str = "None"                     # Business impact descriptor
    category: str = "Unclassified"           # Original manual category (if any)
    engineer: str = "Unknown"                # Assigned engineer name
    lob: str = "Unknown"                     # Line of Business
    issue_date: Optional[datetime] = None    # When the ticket was opened
    close_date: Optional[datetime] = None    # When the ticket was closed (None if open)
    resolution_notes: str = ""               # Free-text resolution description
    root_cause: str = ""                     # Root cause (if identified)
    recurrence_risk: str = "Unknown"         # Human-assessed recurrence risk

    # ---- AI-computed fields (populated by pipeline phases) ----
    ai_category: str = "Unclassified"        # Phase 1: hybrid classifier output
    ai_confidence: float = 0.0               # Phase 1: classification confidence [0, 1]
    strategic_friction_score: float = 0.0    # Phase 2: composite friction score
    financial_impact: float = 0.0            # Financial cost estimate in dollars
    risk_tier: str = "Low"                   # Derived risk tier (Low/Medium/High/Critical)

    # ---- Similar ticket analysis (Phase 5) ----
    similar_ticket_count: int = 0            # Number of similar resolved tickets found
    best_match_similarity: float = 0.0       # Cosine similarity of best match [0, 1]
    resolution_consistency: str = "Unknown"  # Whether similar tickets had consistent resolutions

    # ---- Resolution time prediction (Phase 6) ----
    predicted_resolution_days: float = 0.0   # ML model prediction (days)
    expected_resolution_days: float = 0.0    # Human expectation (from feedback file)
    actual_resolution_days: float = 0.0      # Actual resolution time (if ticket is closed)


# ============================================================================
# ANALYSIS RESULT CONTAINER
# ============================================================================

@dataclass
class AnalysisResult:
    """Container for all analysis results produced by the pipeline.

    This is the top-level "output object" that bundles the enriched DataFrame
    with all aggregate metrics, chart file paths, and the AI-generated
    executive synthesis.  It is used by the report generator to produce the
    final Excel workbook.

    Attributes:
        df: The enriched DataFrame containing all original and computed columns.
        summary_stats: Aggregate statistics (ticket counts, averages, etc.).
        category_distribution: Mapping of AI_Category to ticket count.
        engineer_patterns: Per-engineer performance metrics.
        recidivism_data: Recidivism analysis results (repeat counts, etc.).
        financial_summary: Aggregate financial metrics.
        ai_synthesis: The LLM-generated executive summary text.
        charts: List of file paths to generated chart images.
    """
    df: pd.DataFrame
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    engineer_patterns: Dict[str, Any] = field(default_factory=dict)
    recidivism_data: Dict[str, Any] = field(default_factory=dict)
    financial_summary: Dict[str, float] = field(default_factory=dict)
    ai_synthesis: str = ""
    charts: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties that derive key metrics directly from the
    # enriched DataFrame, avoiding the need to recompute them elsewhere.
    # ------------------------------------------------------------------

    @property
    def total_friction_score(self) -> float:
        """Calculate total weighted friction score across all tickets.

        Returns the sum of the ``Strategic_Friction_Score`` column, which
        is the primary "pain index" used in the executive summary.
        """
        if 'Strategic_Friction_Score' in self.df.columns:
            return self.df['Strategic_Friction_Score'].sum()
        return 0.0

    @property
    def total_financial_impact(self) -> float:
        """Calculate total financial impact across all tickets.

        Returns the sum of the ``Financial_Impact`` column (in dollars).
        """
        if 'Financial_Impact' in self.df.columns:
            return self.df['Financial_Impact'].sum()
        return 0.0

    @property
    def critical_count(self) -> int:
        """Count tickets with ``Critical`` severity.

        This is a headline metric used in the executive summary's
        "Situation Overview" section.
        """
        if 'Severity_Norm' in self.df.columns:
            return (self.df['Severity_Norm'] == 'Critical').sum()
        return 0

    @property
    def external_ratio(self) -> float:
        """Calculate the ratio of external-facing issues to total issues.

        External issues (originating from customers or vendors) typically
        carry higher business risk than internal issues.

        Returns:
            Float in [0, 1] representing the proportion of external tickets.
        """
        if 'Origin_Norm' in self.df.columns:
            total = len(self.df)
            external = (self.df['Origin_Norm'] == 'External').sum()
            return external / total if total > 0 else 0.0
        return 0.0


# ============================================================================
# SIMILAR TICKET MATCH MODEL
# ============================================================================

@dataclass
class SimilarTicketMatch:
    """Represents a similarity match between two tickets.

    Produced by **Phase 5** (Similar Ticket Analysis) when a k-NN search
    finds a resolved ticket that is semantically close to an open ticket.

    The match includes both ticket summaries (truncated to 200 chars for
    readability), the similarity score, and optionally the resolution times
    of both tickets (enabling resolution-time comparison).

    Attributes:
        source_id: ID of the ticket being analysed.
        match_id: ID of the similar resolved ticket found.
        similarity_score: Cosine similarity [0, 1] between the two embeddings.
        match_type: How the match was found (``'semantic'``, ``'keyword'``,
            or ``'category'``).
        source_summary: First 200 chars of the source ticket's description.
        match_summary: First 200 chars of the matched ticket's description.
        source_resolution_days: Resolution time of the source (if closed).
        match_resolution_days: Resolution time of the matched ticket.
        confidence: Qualitative confidence label (``'High'``, ``'Medium'``,
            ``'Low'``).
    """
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
        """Convert to a flat dictionary for DataFrame / JSON serialisation.

        Summaries are truncated to 200 characters to keep the output compact
        when serialised into Excel cells or JSON payloads.

        Returns:
            Dict with all match attributes.
        """
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


# ============================================================================
# RESOLUTION TIME PREDICTION MODEL
# ============================================================================

@dataclass
class ResolutionTimePrediction:
    """Result of resolution time prediction for a single ticket.

    Produced by **Phase 6** (Resolution Time Prediction) and captures not
    just the predicted value but also metadata about how the prediction was
    made (model-based, category average, or fallback) and a confidence
    interval.

    Attributes:
        ticket_id: The ticket this prediction applies to.
        predicted_days: Predicted resolution time in days.
        confidence_interval: (lower, upper) bounds for the prediction.
        prediction_method: How the prediction was derived:
            - ``'model'``        -- Random Forest regressor.
            - ``'category_avg'`` -- average of resolved tickets in the same
              category (fallback when model has insufficient training data).
            - ``'fallback'``     -- global average (last resort).
        similar_tickets_used: Number of similar resolved tickets that informed
            the prediction.
        human_expected_days: Human-provided expected resolution time (from the
            resolution feedback file), if available.
    """
    ticket_id: str
    predicted_days: float
    confidence_interval: tuple = (0.0, 0.0)  # (lower_bound, upper_bound)
    prediction_method: str = "model"  # 'model', 'category_avg', 'fallback'
    similar_tickets_used: int = 0
    human_expected_days: Optional[float] = None

    @property
    def prediction_quality(self) -> str:
        """Assess the qualitative quality of this prediction.

        Quality heuristic:
        - ``'High'``   -- model-based with >= 3 similar tickets for support.
        - ``'Medium'`` -- category-average fallback.
        - ``'Low'``    -- global fallback or model with few similar tickets.

        Returns:
            One of ``'High'``, ``'Medium'``, or ``'Low'``.
        """
        if self.prediction_method == "model" and self.similar_tickets_used >= 3:
            return "High"
        elif self.prediction_method == "category_avg":
            return "Medium"
        return "Low"


# ============================================================================
# DATAFRAME VALIDATION UTILITY
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> tuple:
    """
    Validate that a DataFrame has the required columns.

    Used as a guard at the start of pipeline phases to ensure that upstream
    phases have produced the expected columns before downstream processing
    begins.

    Args:
        df: The DataFrame to validate.
        required_columns: List of column names that must be present.

    Returns:
        Tuple of ``(is_valid: bool, missing_columns: List[str])``.
        ``is_valid`` is ``True`` when all required columns are present.
    """
    missing = [col for col in required_columns if col not in df.columns]
    return (len(missing) == 0, missing)


# ============================================================================
# DATA NORMALISATION FUNCTIONS
# ============================================================================

def normalize_severity(severity: str) -> str:
    """Normalize severity values to a standard four-level format.

    The raw data may use a variety of labels for severity (e.g. ``"P1"``,
    ``"crit"``, ``"high"``, ``"major"``).  This function maps them all to
    one of four canonical levels:

    - ``Critical`` -- P1 / crit / critical
    - ``Major``    -- P2 / high / major
    - ``Minor``    -- P3 / medium / minor
    - ``Low``      -- P4 / info / low
    - ``Default``  -- used when the input is NaN or unrecognised

    Args:
        severity: Raw severity string from the input data.

    Returns:
        Canonical severity string.
    """
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
    # Return the mapped value, or title-case the original if not in the map
    return severity_map.get(severity_lower, str(severity).title().strip())


def normalize_origin(origin: str) -> str:
    """Normalize origin values to a standard two-level format.

    Maps various origin labels to either ``External`` (customer-facing) or
    ``Internal`` (within the organisation).  This normalisation is important
    for the friction scoring algorithm, which weights external issues higher
    than internal ones.

    Known aliases:
    - ``External`` <- ``"external"``, ``"customer"``, ``"vendor"``
    - ``Internal`` <- ``"internal"``, ``"amdocs"`` (internal team), ``"team"``

    Args:
        origin: Raw origin string from the input data.

    Returns:
        Canonical origin string (``'External'``, ``'Internal'``, or
        ``'Unknown'``).
    """
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
    """Calculate resolution time in days between issue opening and closing.

    Used during data loading to compute the ``actual_resolution_days`` field,
    which serves as the ground-truth label for the Phase 6 resolution-time
    prediction model.

    The result is clamped to a minimum of 0 to handle any data-entry errors
    where the close date precedes the issue date.

    Args:
        issue_date: Timestamp (or parseable string) when the ticket was opened.
        close_date: Timestamp (or parseable string) when the ticket was closed.

    Returns:
        Float number of days (>= 0), or ``None`` if either date is missing
        or unparseable.
    """
    if pd.isna(issue_date) or pd.isna(close_date):
        return None

    try:
        issue_dt = pd.to_datetime(issue_date)
        close_dt = pd.to_datetime(close_date)
        delta = (close_dt - issue_dt).total_seconds() / (24 * 3600)
        return max(0, delta)  # Ensure non-negative
    except Exception:
        return None
