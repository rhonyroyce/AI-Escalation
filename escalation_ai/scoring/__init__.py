"""
Strategic Friction Scoring Engine.

=== PURPOSE ===
This module is the heart of the Escalation AI risk-ranking pipeline.  It
takes a raw DataFrame of escalation tickets (already loaded and cleaned by
upstream code) and produces a fully enriched DataFrame with:

  - Strategic_Friction_Score  -- the primary risk-ranking metric
  - Financial_Impact          -- dollar-denominated cost per ticket
  - Risk_Tier                 -- Critical / High / Medium / Low bucket
  - Engineer accountability   -- per-engineer issue count & flags
  - Aging status              -- how stale each ticket is
  - Root cause classification -- 7-bucket root-cause from free text
  - PM recurrence risk        -- normalised recurrence probability
  - Priority score            -- friction x recency for action queues
  - Recommended actions       -- human-readable next steps

=== SCORING FORMULA ===
The core formula is a McKinsey-style multiplicative model:

    Score = Base_Severity x Type_Multiplier x Origin_Multiplier x Impact_Multiplier

Where:
  - Base_Severity:    Critical=100, Major=50, Minor=10, Default=5
  - Type_Multiplier:  Escalations=1.5, Concerns=1.0, Lessons Learned=0.0
  - Origin_Multiplier: External=2.5, Internal=1.0
  - Impact_Multiplier: High=2.0, Low=1.1, None=1.0

This produces a score range of 0 (lessons learned) to 750
(Critical + Escalation + External + High Impact).

=== DATA FLOW ===
  Input:  pd.DataFrame with raw ticket columns (COL_SEVERITY, COL_TYPE, etc.)
          plus AI-generated columns (AI_Category, AI_Sub_Category) from the
          classifier module.
  Steps:
    1. Normalise severity/type/origin/impact strings to title-case for
       consistent lookup against the WEIGHTS dictionary.
    2. Load the price catalog (price_catalog.xlsx) and compute per-ticket
       financial impact using category + sub-category + severity + origin.
    3. Apply the multiplicative scoring formula row-by-row.
    4. Layer on enrichment passes: risk tiers, engineer tracking, aging,
       human-error flags, root-cause classification, recurrence risk, and
       finally priority scoring + recommended actions.
  Output: Enriched pd.DataFrame ready for report generation and dashboards.

Multi-variable risk scoring based on McKinsey framework.
"""

import logging
import pandas as pd
from typing import Dict

from ..core.config import (
    WEIGHTS, COL_SEVERITY, COL_TYPE, COL_ORIGIN, COL_IMPACT,
    COL_SUMMARY, COL_ENGINEER, COL_DATETIME, COL_ROOT_CAUSE,
    COL_RECURRENCE_RISK, ROOT_CAUSE_CATEGORIES, REQUIRED_COLUMNS
)
from ..feedback.price_catalog import get_price_catalog
from ..core.utils import validate_columns

logger = logging.getLogger(__name__)


def calculate_strategic_friction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multi-variable risk scoring to the dataframe.

    This is the main entry point for the scoring pipeline.  It orchestrates
    every enrichment step in sequence, ensuring that downstream steps can
    rely on columns created by upstream steps (e.g. ``_add_priority_and_actions``
    depends on ``Risk_Tier`` created by ``_add_risk_tier``).

    Scoring formula:
        Score = Base_Severity x Type_Multiplier x Origin_Multiplier x Impact_Multiplier

    Args:
        df: Input DataFrame containing at least the REQUIRED_COLUMNS
            (COL_SEVERITY, COL_TYPE, COL_ORIGIN).  Should also contain
            AI_Category and AI_Sub_Category from the classification stage.

    Returns:
        DataFrame with the following new/modified columns:
        - Strategic_Friction_Score, Financial_Impact, Financial_Impact_Source
        - Risk_Tier, Engineer, Engineer_Issue_Count, Engineer_Total_Friction,
          Engineer_Flag
        - Issue_Date, Days_Since_Issue, Aging_Status
        - Is_Human_Error, Root_Cause_Category, Root_Cause_Original
        - PM_Recurrence_Risk, PM_Recurrence_Risk_Norm
        - Priority_Score, Action_Required
    """
    logger.info("[Strategic Engine] Applying Multi-Variable Risk Scoring...")

    # Work on a copy to avoid mutating the caller's DataFrame
    df = df.copy()

    # Validate that the minimum required columns exist before proceeding
    validate_columns(df, REQUIRED_COLUMNS)

    # ------------------------------------------------------------------
    # Step 1: Normalise raw column values to title-case strings
    # ------------------------------------------------------------------
    # Title-casing ensures that 'critical', 'CRITICAL', and 'Critical' all
    # map to the same WEIGHTS key.  .str.strip() removes stray whitespace.
    # If a column is missing, a safe default is used so scoring still works.
    df['Severity_Norm'] = df[COL_SEVERITY].astype(str).str.title().str.strip() if COL_SEVERITY in df.columns else 'Default'
    df['Type_Norm'] = df[COL_TYPE].astype(str).str.title().str.strip() if COL_TYPE in df.columns else ''
    df['Origin_Norm'] = df[COL_ORIGIN].astype(str).str.title().str.strip() if COL_ORIGIN in df.columns else ''
    df['Impact_Norm'] = df[COL_IMPACT].fillna('None').astype(str).str.title().str.strip() if COL_IMPACT in df.columns else 'None'

    # ------------------------------------------------------------------
    # Step 2: Load price catalog for financial impact calculation
    # ------------------------------------------------------------------
    # The price catalog is always reloaded to pick up any mid-session edits
    # an analyst may have made to price_catalog.xlsx (e.g. via the Streamlit
    # dashboard's feedback panel).
    price_catalog = get_price_catalog()
    price_catalog.load_catalog()  # Always reload to ensure latest values from price_catalog.xlsx

    def get_score(row):
        """Calculate strategic friction score for a single row.

        Implements the multiplicative formula:
            base x type_mult x origin_mult x impact_mult

        Each multiplier defaults to a neutral value (1.0 for multipliers,
        5 for base) if the lookup key is not found, ensuring graceful
        handling of unexpected data values.
        """
        # 1. Base Score -- driven by the severity field
        #    Critical=100, Major=50, Minor=10, Default=5
        base = WEIGHTS['BASE_SEVERITY'].get(row['Severity_Norm'], 5)

        # 2. Type Multiplier -- Escalations are 1.5x (customer-visible),
        #    Concerns are 1.0x (default), Lessons Learned are 0.0x (not risks)
        m_type = 1.0
        if 'Escalation' in row['Type_Norm']:
            m_type = WEIGHTS['TYPE_MULTIPLIER']['Escalations']
        elif 'Lesson' in row['Type_Norm']:
            m_type = 0.0  # Lessons are not risks

        # 3. Origin Multiplier -- External issues carry 2.5x weight due to
        #    customer/vendor visibility and SLA exposure
        m_origin = WEIGHTS['ORIGIN_MULTIPLIER'].get(row['Origin_Norm'], 1.0)

        # 4. Impact Multiplier -- High=2.0x, Low=1.1x, None=1.0x
        m_impact = 1.0
        if 'High' in row['Impact_Norm']:
            m_impact = WEIGHTS['IMPACT_MULTIPLIER']['High']

        # Final score: product of all four dimensions
        return base * m_type * m_origin * m_impact

    def get_financial_impact(row):
        """Calculate financial impact for a single row using the price catalog.

        Passes the AI-assigned category, sub-category, severity, origin, and
        description into the price catalog's costing engine.  The catalog
        applies its priority-based lookup (keyword > sub-category > category
        > fallback) and returns a dict with cost breakdown and audit trail.
        """
        category = row.get('AI_Category', 'Unclassified')
        sub_category = row.get('AI_Sub_Category', '')
        severity = row['Severity_Norm']
        origin = row['Origin_Norm']
        description = str(row.get(COL_SUMMARY, ''))

        # Delegate to the price catalog's costing engine
        impact = price_catalog.calculate_financial_impact(
            category=category,
            sub_category=sub_category if pd.notna(sub_category) else '',
            severity=severity,
            origin=origin,
            description=description,
        )
        return impact

    # ------------------------------------------------------------------
    # Step 3: Apply scoring formula to every row
    # ------------------------------------------------------------------
    df['Strategic_Friction_Score'] = df.apply(get_score, axis=1)

    # ------------------------------------------------------------------
    # Step 4: Compute financial impact per ticket
    # ------------------------------------------------------------------
    # Each row returns a dict; we extract 'total_impact' (dollar amount)
    # and 'source' (audit trail showing which catalog lookup was used).
    impact_results = df.apply(get_financial_impact, axis=1)
    df['Financial_Impact'] = impact_results.apply(lambda r: r['total_impact'])
    df['Financial_Impact_Source'] = impact_results.apply(lambda r: r['source'])
    logger.info(f"  → Total estimated financial impact: ${df['Financial_Impact'].sum():,.2f}")
    logger.info(f"  → Impact sources: {df['Financial_Impact_Source'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Step 5: Layer on enrichment passes (order matters!)
    # ------------------------------------------------------------------
    df = _add_risk_tier(df)                  # Needs: Strategic_Friction_Score
    df = _add_engineer_accountability(df)    # Needs: Type_Norm, Strategic_Friction_Score
    df = _add_aging_status(df)               # Needs: COL_DATETIME
    df = _add_human_error_flags(df)          # Needs: Origin_Norm, Type_Norm
    df = _add_root_cause_classification(df)  # Needs: COL_ROOT_CAUSE, updates Is_Human_Error
    df = _add_pm_recurrence_risk(df)         # Needs: COL_RECURRENCE_RISK
    df = _add_priority_and_actions(df)       # Needs: Risk_Tier, Is_Human_Error, Engineer_Flag, Days_Since_Issue

    return df


def _add_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk tier based on friction score.

    Bins the continuous Strategic_Friction_Score into four ordinal tiers
    for use in dashboards and filters:

        Score >= 150  ->  Critical   (top priority, immediate review)
        Score >= 75   ->  High       (significant risk, schedule review)
        Score >= 25   ->  Medium     (moderate risk, monitor)
        Score < 25    ->  Low        (low risk, routine handling)

    These thresholds were calibrated against the score distribution in the
    original 300+ ticket dataset to produce a roughly 10/25/40/25 split.

    Args:
        df: DataFrame with 'Strategic_Friction_Score' column.

    Returns:
        DataFrame with new 'Risk_Tier' column.
    """
    def get_risk_tier(score):
        if score >= 150:
            return "Critical"
        elif score >= 75:
            return "High"
        elif score >= 25:
            return "Medium"
        else:
            return "Low"

    df['Risk_Tier'] = df['Strategic_Friction_Score'].apply(get_risk_tier)
    return df


def _add_engineer_accountability(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineer tracking columns for accountability analysis.

    For each engineer, this computes:
    - Engineer_Issue_Count:     number of escalation/concern tickets assigned
    - Engineer_Total_Friction:  sum of friction scores across their tickets
    - Engineer_Flag:            emoji flag for repeat offenders

    Flag thresholds:
    - 5+ issues  -> "Repeat Offender" (red flag)
    - 3+ issues  -> "Multiple Issues" (yellow flag)
    - <3 issues  -> no flag

    Only Escalations and Concerns count toward the tally; Lessons Learned
    are excluded because they are informational, not performance-related.

    Args:
        df: DataFrame with COL_ENGINEER, Type_Norm, Strategic_Friction_Score.

    Returns:
        DataFrame with Engineer, Engineer_Issue_Count, Engineer_Total_Friction,
        and Engineer_Flag columns.
    """
    if COL_ENGINEER in df.columns:
        # Clean engineer names: fill blanks with 'Unknown', strip whitespace
        df['Engineer'] = df[COL_ENGINEER].fillna('Unknown').astype(str).str.strip()

        # Count only escalations and concerns (not lessons learned) per engineer
        engineer_counts = df[df['Type_Norm'].isin(['Escalations', 'Concerns'])].groupby('Engineer').size()
        df['Engineer_Issue_Count'] = df['Engineer'].map(engineer_counts).fillna(0).astype(int)

        # Sum friction scores per engineer (again, only escalations/concerns)
        engineer_friction = df[df['Type_Norm'].isin(['Escalations', 'Concerns'])].groupby('Engineer')['Strategic_Friction_Score'].sum()
        df['Engineer_Total_Friction'] = df['Engineer'].map(engineer_friction).fillna(0)

        # Flag engineers who appear frequently -- signals potential training or
        # performance issues that management should review
        df['Engineer_Flag'] = df['Engineer_Issue_Count'].apply(
            lambda x: '🔴 Repeat Offender' if x >= 5 else ('🟡 Multiple Issues' if x >= 3 else '')
        )

        logger.info(f"  → Engineer accountability tracked for {df['Engineer'].nunique()} unique engineers")

    return df


def _add_aging_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add issue aging tracking.

    Computes how many days have elapsed since each ticket was raised and
    assigns an aging status label:

        >30 days  ->  Red    (overdue, should have been resolved)
        >14 days  ->  Yellow (aging, needs attention)
        0-14 days ->  Green  (recent, within normal SLA window)
        unknown   ->  Unknown (date could not be parsed)

    Aging status feeds into both the Priority_Score calculation (older
    tickets get a recency boost) and the SLA compliance dashboards.

    Args:
        df: DataFrame with COL_DATETIME column.

    Returns:
        DataFrame with Issue_Date, Days_Since_Issue, and Aging_Status columns.
    """
    if COL_DATETIME in df.columns:
        # Parse the raw datetime string; coerce unparseable values to NaT
        df['Issue_Date'] = pd.to_datetime(df[COL_DATETIME], errors='coerce')

        # Days elapsed since the issue was opened (negative = future date = data error)
        df['Days_Since_Issue'] = (pd.Timestamp.now() - df['Issue_Date']).dt.days
        df['Days_Since_Issue'] = df['Days_Since_Issue'].fillna(-1).astype(int)

        # Assign aging status labels for dashboard colour-coding
        df['Aging_Status'] = df['Days_Since_Issue'].apply(
            lambda x: '🔴 >30 days' if x > 30 else ('🟡 >14 days' if x > 14 else ('🟢 Recent' if x >= 0 else 'Unknown'))
        )

    return df


def _add_human_error_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add human error flag based on origin and type.

    A ticket is flagged as a human error if it is:
    1. Internal origin (i.e. caused by the organisation's own staff), AND
    2. Classified as an Escalation or Concern (not a Lesson Learned).

    This is a first-pass heuristic; it is refined later by
    ``_add_root_cause_classification()`` which can override based on the
    free-text root-cause field (e.g. marking 'External Party' items).

    Args:
        df: DataFrame with Origin_Norm and Type_Norm columns.

    Returns:
        DataFrame with Is_Human_Error column ('Yes' / 'No').
    """
    df['Is_Human_Error'] = (
        (df['Origin_Norm'] == 'Internal') &
        (df['Type_Norm'].isin(['Escalations', 'Concerns']))
    ).map({True: 'Yes', False: 'No'})

    return df


def _add_root_cause_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Add root cause classification from free-text root cause field.

    Scans the ticket's root-cause text against the ROOT_CAUSE_CATEGORIES
    dictionary (7 buckets: Human Error, External Party, Process Gap,
    System/Technical, Training Gap, Communication, Resource).  First
    keyword match wins.

    After classification, the Is_Human_Error flag is updated:
    - Root cause = 'Human Error'    -> Is_Human_Error = 'Yes'
    - Root cause = 'External Party' -> Is_Human_Error = 'External'
    This refinement overrides the initial heuristic from
    ``_add_human_error_flags()``.

    Args:
        df: DataFrame with COL_ROOT_CAUSE column (optional; if missing,
            all rows are labelled 'Unclassified').

    Returns:
        DataFrame with Root_Cause_Category, Root_Cause_Original columns,
        and potentially updated Is_Human_Error values.
    """
    # Initialise with defaults; will be overwritten if root cause data exists
    df['Root_Cause_Category'] = 'Unclassified'
    df['Root_Cause_Original'] = ''

    if COL_ROOT_CAUSE in df.columns:
        # Preserve the original free-text for audit purposes
        df['Root_Cause_Original'] = df[COL_ROOT_CAUSE].fillna('').astype(str).str.strip()

        def classify_root_cause(root_cause_text):
            """Map free-text root cause to one of 7 standardised categories.

            Iterates through ROOT_CAUSE_CATEGORIES in definition order.
            First keyword match wins.  Returns 'Other' if text is present
            but no keywords match; returns 'Unclassified' if text is empty.
            """
            if pd.isna(root_cause_text) or not root_cause_text:
                return 'Unclassified'
            text_lower = str(root_cause_text).lower()

            for category, keywords in ROOT_CAUSE_CATEGORIES.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return category
            return 'Other'

        df['Root_Cause_Category'] = df['Root_Cause_Original'].apply(classify_root_cause)

        # Refine Is_Human_Error flag based on root-cause classification
        # This overrides the initial heuristic for rows with explicit root cause
        human_error_mask = df['Root_Cause_Category'] == 'Human Error'
        df.loc[human_error_mask, 'Is_Human_Error'] = 'Yes'

        external_mask = df['Root_Cause_Category'] == 'External Party'
        df.loc[external_mask, 'Is_Human_Error'] = 'External'

        logger.info(f"  → Root cause classified: {df['Root_Cause_Category'].value_counts().to_dict()}")

    return df


def _add_pm_recurrence_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add PM recurrence risk normalisation.

    The source data contains a free-text recurrence risk field filled by
    project managers (e.g. 'High', 'yes', 'Likely', 'maybe', 'Low').
    This function normalises these into three buckets: High / Medium / Low,
    with 'Unknown' as fallback.

    Mapping:
        high, yes, likely, probable, very high  ->  High
        medium, moderate, possible, maybe       ->  Medium
        low, no, unlikely, none, very low, minimal -> Low
        anything else / blank                   ->  Unknown

    These normalised values are used by the recurrence exposure calculations
    in the financial metrics module.

    Args:
        df: DataFrame with COL_RECURRENCE_RISK column (optional).

    Returns:
        DataFrame with PM_Recurrence_Risk (original) and
        PM_Recurrence_Risk_Norm (normalised) columns.
    """
    # Initialise defaults for when the column is absent
    df['PM_Recurrence_Risk'] = 'Unknown'
    df['PM_Recurrence_Risk_Norm'] = 'Unknown'

    if COL_RECURRENCE_RISK in df.columns:
        df['PM_Recurrence_Risk'] = df[COL_RECURRENCE_RISK].fillna('Unknown').astype(str).str.strip()

        def normalize_recurrence_risk(risk_text):
            """Normalise free-text recurrence risk to High/Medium/Low/Unknown."""
            if pd.isna(risk_text) or not risk_text:
                return 'Unknown'
            text_lower = str(risk_text).lower().strip()

            if text_lower in ['high', 'yes', 'likely', 'probable', 'very high']:
                return 'High'
            elif text_lower in ['medium', 'moderate', 'possible', 'maybe']:
                return 'Medium'
            elif text_lower in ['low', 'no', 'unlikely', 'none', 'very low', 'minimal']:
                return 'Low'
            else:
                return 'Unknown'

        df['PM_Recurrence_Risk_Norm'] = df['PM_Recurrence_Risk'].apply(normalize_recurrence_risk)
        logger.info(f"  → PM Recurrence risk: {df['PM_Recurrence_Risk_Norm'].value_counts().to_dict()}")

    return df


def _add_priority_and_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Add priority score and action required columns.

    Priority Score:
        Starts with the Strategic_Friction_Score and applies a recency boost.
        Older tickets get a LOWER boost (they're stale), while recent tickets
        get a HIGHER boost.  The formula is:

            Priority = Friction * (1 + (30 - clamp(Days, 0, 30)) / 100)

        This means a ticket from today gets a +30% boost, a 15-day-old ticket
        gets a +15% boost, and a 30+-day-old ticket gets +0% boost.  The
        intent is to prioritise fresh high-friction items for immediate action
        while letting stale items naturally decay in the action queue.

    Action Required:
        A pipe-delimited list of recommended next steps based on the ticket's
        risk profile:
        - Critical or High risk tier  -> 'Immediate Review'
        - Human error confirmed       -> 'Training Review'
        - Engineer is a repeat offender -> 'Performance Discussion'
        - Confirmed repeat issue       -> 'Process Fix Required'
        - None of the above            -> 'Monitor'

    Args:
        df: DataFrame with Risk_Tier, Is_Human_Error, Engineer_Flag,
            Learning_Status, Days_Since_Issue, Strategic_Friction_Score.

    Returns:
        DataFrame with Priority_Score and Action_Required columns.
    """
    # --- Priority Score ---
    # Start from the raw friction score
    df['Priority_Score'] = df['Strategic_Friction_Score']

    if 'Days_Since_Issue' in df.columns:
        # Apply recency boost: newer issues get up to +30% priority bump
        # clamp Days_Since_Issue to [0, 30] to prevent negative boosts
        df['Priority_Score'] = df['Priority_Score'] * (1 + (30 - df['Days_Since_Issue'].clip(0, 30)) / 100)

    # --- Action Required ---
    def get_action_required(row):
        """Determine recommended actions based on multi-dimensional risk profile."""
        actions = []

        # High/Critical risk tier -> immediate management attention
        if row['Risk_Tier'] in ['Critical', 'High']:
            actions.append('Immediate Review')

        # Human error -> potential training need
        if row.get('Is_Human_Error') == 'Yes':
            actions.append('Training Review')

        # Repeat-offender engineer -> performance management intervention
        if row.get('Engineer_Flag') and 'Repeat' in str(row.get('Engineer_Flag', '')):
            actions.append('Performance Discussion')

        # Confirmed repeat issue (from similarity engine) -> systemic fix needed
        if row.get('Learning_Status') == 'Confirmed Repeat':
            actions.append('Process Fix Required')

        # Default: no specific action, just monitor
        return ' | '.join(actions) if actions else 'Monitor'

    df['Action_Required'] = df.apply(get_action_required, axis=1)

    return df
