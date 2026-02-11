"""
Advanced Financial Metrics and Analysis for Escalation Costs.

=== PURPOSE ===
This module translates escalation ticket data into financial language that
executives and finance teams can act on.  While the scoring engine
(escalation_ai.scoring) produces a dimensionless "friction score," this
module produces dollar-denominated metrics: total cost, ROI opportunities,
cost avoidance potential, forecasts, and actionable insights.

=== ARCHITECTURE ===
All per-ticket financial impact values are computed upstream by the price
catalog (escalation_ai.feedback.price_catalog) and stored in the DataFrame's
``Financial_Impact`` column.  This module aggregates, trends, forecasts, and
interprets those per-ticket values.

Business multipliers (revenue_at_risk, customer_impact, sla_penalty,
prevention_rate, cost_avoidance_rate, opportunity_cost) are loaded from the
"Business Multipliers" sheet of price_catalog.xlsx via ``_get_biz_multiplier()``.
This keeps every financial assumption editable in the spreadsheet without
touching code.

=== KEY CALCULATIONS ===
1. **FinancialMetrics dataclass** -- comprehensive container holding 30+
   metrics across core, distribution, efficiency, ROI, trend, risk, benchmark,
   and business-impact dimensions.

2. **calculate_financial_metrics()** -- fills the dataclass from the DataFrame.
   Computes cost concentration (Pareto/80-20), high-cost ticket identification,
   recurring/preventable cost segmentation, and risk-adjusted exposure.

3. **calculate_roi_metrics()** -- identifies the top 5 categories by total cost,
   annualises them based on the actual data date range (not an arbitrary
   multiplier), and projects ROI and payback period for root-cause fixes.

4. **calculate_cost_avoidance()** -- estimates savings from four levers:
   recurring issue elimination, preventable category reduction, knowledge
   sharing, and automation of repetitive categories.

5. **calculate_financial_forecasts()** -- simple moving-average forecast with
   confidence intervals and trend detection (increasing / decreasing / stable).

6. **generate_financial_insights()** -- produces prioritised, human-readable
   insight cards for the executive dashboard.

=== DATA FLOW ===
  Input:  Enriched pd.DataFrame from the scoring engine, containing at
          minimum ``Financial_Impact`` and typically also ``AI_Category``,
          ``Severity_Norm``, ``Issue_Date``, ``Origin_Norm``, etc.
  Output: Python dicts / dataclasses / lists consumed by:
          - The Streamlit dashboard (real-time display)
          - The Excel report generator (Strategic_Report.xlsx)
          - The chart generation modules (waterfall, funnel, etc.)

This module provides:
- ROI and cost avoidance calculations
- Financial trend analysis and forecasting
- Cost efficiency and benchmark metrics
- Risk-adjusted financial exposure
- Cost optimization recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from ..feedback.price_catalog import get_price_catalog

logger = logging.getLogger(__name__)


def _get_biz_multiplier(name: str, default: float) -> float:
    """Get a business multiplier from price_catalog.xlsx, falling back to default.

    Business multipliers are stored in the "Business Multipliers" sheet of
    price_catalog.xlsx and control how raw escalation costs are translated into
    broader financial exposure metrics (revenue at risk, opportunity cost, etc.).

    This helper provides a safe accessor: if the catalog is not yet loaded or
    the requested multiplier does not exist, it returns the provided default
    rather than raising an exception.

    Args:
        name:    The multiplier name as it appears in the spreadsheet's
                 ``Multiplier_Name`` column (case-insensitive).
        default: Value to return if the multiplier is not found.

    Returns:
        The multiplier value from the catalog, or ``default`` on any failure.
    """
    try:
        catalog = get_price_catalog()
        if not catalog.is_loaded:
            catalog.load_catalog()
        return catalog.get_business_multiplier(name, default)
    except Exception:
        return default


@dataclass
class FinancialMetrics:
    """Container for comprehensive financial metrics.

    This dataclass groups 30+ financial indicators into logical sections.
    It is populated by ``calculate_financial_metrics()`` and consumed by
    dashboard widgets, report generators, and the insights engine.

    All monetary values are in USD.  Ratios are expressed as floats (0.0-1.0)
    unless otherwise noted.  Scores are on a 0-100 scale.
    """

    # === Core metrics ===
    # Basic aggregates over all tickets in the analysis window.
    total_cost: float = 0.0            # Sum of Financial_Impact across all tickets
    total_tickets: int = 0             # Count of tickets analysed
    avg_cost_per_ticket: float = 0.0   # total_cost / total_tickets
    median_cost: float = 0.0           # 50th percentile of Financial_Impact

    # === Distribution metrics ===
    # Cost broken down by key dimensions for drill-down analysis.
    cost_by_category: Dict[str, float] = field(default_factory=dict)   # AI_Category -> total $
    cost_by_severity: Dict[str, float] = field(default_factory=dict)   # Severity_Norm -> total $
    cost_by_engineer: Dict[str, float] = field(default_factory=dict)   # Engineer name -> total $
    cost_by_lob: Dict[str, float] = field(default_factory=dict)        # Line of Business -> total $

    # === Efficiency metrics ===
    # How efficiently are escalations being resolved?
    cost_per_resolution_hour: float = 0.0   # $ spent per hour of resolution work
    high_cost_tickets_count: int = 0        # Number of tickets in top 10% by cost
    high_cost_percentage: float = 0.0       # high_cost_tickets_count / total_tickets * 100
    cost_concentration_ratio: float = 0.0   # Fraction of total cost from top 20% of tickets (Pareto)

    # === ROI metrics ===
    # Quantifies the financial opportunity from fixing root causes.
    recurring_issue_cost: float = 0.0       # $ from issues with >30% recurrence probability
    preventable_cost: float = 0.0           # $ from categories deemed preventable
    roi_opportunity: float = 0.0            # preventable_cost x prevention_rate
    cost_avoidance_potential: float = 0.0   # $ saveable by fixing repeat patterns

    # === Trend metrics ===
    # Time-series analysis for financial planning.
    monthly_cost_trend: Dict[str, float] = field(default_factory=dict)  # "YYYY-MM" -> monthly $
    cost_velocity: float = 0.0       # Rate of cost change in $/day (slope of linear fit)
    cost_forecast_30d: float = 0.0   # Projected total cost over next 30 days
    cost_forecast_90d: float = 0.0   # Projected total cost over next 90 days

    # === Risk metrics ===
    # Probability-weighted financial exposure.
    risk_adjusted_exposure: float = 0.0   # Sum of (Financial_Impact x severity_weight) per ticket
    recurrence_exposure: float = 0.0      # Sum of (Financial_Impact x recurrence_probability)
    critical_cost_ratio: float = 0.0      # Fraction of total cost from Critical-severity tickets

    # === Benchmark metrics ===
    # Comparison against target performance.
    target_cost_per_ticket: float = 500.0   # Aspirational benchmark for avg cost ($500 default)
    cost_efficiency_score: float = 0.0      # Composite 0-100 score (100 = most efficient)
    cost_vs_benchmark: float = 0.0          # avg_cost - target_cost (positive = over target)

    # === Business impact ===
    # Broader organisational impact derived from multipliers in price_catalog.xlsx.
    revenue_at_risk: float = 0.0        # total_cost x revenue_at_risk multiplier
    customer_impact_cost: float = 0.0   # Cost of external/customer-facing escalations
    sla_penalty_exposure: float = 0.0   # Estimated SLA penalties for critical tickets
    opportunity_cost: float = 0.0       # total_cost x opportunity_cost multiplier


def calculate_financial_metrics(df: pd.DataFrame) -> FinancialMetrics:
    """
    Calculate comprehensive financial metrics from escalation data.

    This is the main aggregation function.  It takes the enriched DataFrame
    (with per-ticket Financial_Impact already computed) and produces a full
    FinancialMetrics object covering cost distribution, efficiency, ROI,
    trends, risk exposure, benchmarks, and business impact.

    Args:
        df: DataFrame with ``Financial_Impact`` and other enrichment columns
            (AI_Category, Severity_Norm, Issue_Date, Origin_Norm, etc.).

    Returns:
        FinancialMetrics object with all calculated metrics.  Returns a
        zero-filled object if the DataFrame is empty or lacks the required
        Financial_Impact column.
    """
    metrics = FinancialMetrics()

    if df.empty or 'Financial_Impact' not in df.columns:
        logger.warning("No financial data available")
        return metrics

    # ------------------------------------------------------------------
    # Core metrics: basic aggregates
    # ------------------------------------------------------------------
    metrics.total_cost = df['Financial_Impact'].sum()
    metrics.total_tickets = len(df)
    metrics.avg_cost_per_ticket = metrics.total_cost / metrics.total_tickets if metrics.total_tickets > 0 else 0
    metrics.median_cost = df['Financial_Impact'].median()

    # ------------------------------------------------------------------
    # Distribution metrics: cost by dimension
    # ------------------------------------------------------------------
    # These power the "cost breakdown" charts in the dashboard.
    if 'AI_Category' in df.columns:
        metrics.cost_by_category = df.groupby('AI_Category')['Financial_Impact'].sum().to_dict()

    if 'Severity_Norm' in df.columns:
        metrics.cost_by_severity = df.groupby('Severity_Norm')['Financial_Impact'].sum().to_dict()

    if 'Engineer_Assigned' in df.columns:
        metrics.cost_by_engineer = df.groupby('Engineer_Assigned')['Financial_Impact'].sum().to_dict()

    if 'LOB' in df.columns:
        metrics.cost_by_lob = df.groupby('LOB')['Financial_Impact'].sum().to_dict()

    # ------------------------------------------------------------------
    # Efficiency metrics
    # ------------------------------------------------------------------
    # Cost per resolution hour: how much does each hour of resolution work cost?
    if 'Resolution_Days' in df.columns:
        total_hours = df['Resolution_Days'].sum() * 24  # Convert days to hours
        metrics.cost_per_resolution_hour = metrics.total_cost / total_hours if total_hours > 0 else 0

    # High cost tickets: those in the top 10% by Financial_Impact.
    # These are the "vital few" that drive a disproportionate share of total cost.
    high_cost_threshold = df['Financial_Impact'].quantile(0.9)
    metrics.high_cost_tickets_count = (df['Financial_Impact'] >= high_cost_threshold).sum()
    metrics.high_cost_percentage = (metrics.high_cost_tickets_count / metrics.total_tickets * 100) if metrics.total_tickets > 0 else 0

    # Cost concentration ratio (Pareto / 80-20 analysis):
    # What fraction of total cost comes from the top 20% of tickets?
    # A ratio > 0.8 means costs are highly concentrated = easier to target.
    sorted_costs = df['Financial_Impact'].sort_values(ascending=False)
    top_20_percent_count = int(len(sorted_costs) * 0.2)
    top_20_cost = sorted_costs.head(top_20_percent_count).sum()
    metrics.cost_concentration_ratio = (top_20_cost / metrics.total_cost) if metrics.total_cost > 0 else 0

    # ------------------------------------------------------------------
    # ROI metrics: quantifying the savings opportunity
    # ------------------------------------------------------------------
    # prevention_rate: what fraction of preventable costs can realistically
    # be eliminated? Default 0.8 (80%) from price_catalog.xlsx Business Multipliers.
    prevention_rate = _get_biz_multiplier('prevention_rate', 0.8)

    # Recurring issue cost: total $ from tickets with >30% AI-predicted recurrence risk
    metrics.recurring_issue_cost = _calculate_recurring_cost(df)

    # Preventable cost: total $ from categories that are inherently preventable
    # (process, communication, config, scheduling, documentation failures)
    metrics.preventable_cost = _calculate_preventable_cost(df)

    # ROI opportunity: the realistic savings if preventive measures are implemented
    metrics.roi_opportunity = metrics.preventable_cost * prevention_rate

    # Cost avoidance: savings from fixing root causes of repeat issues
    metrics.cost_avoidance_potential = _calculate_cost_avoidance(df)

    # ------------------------------------------------------------------
    # Trend metrics: time-series analysis
    # ------------------------------------------------------------------
    if 'Issue_Date' in df.columns:
        metrics.monthly_cost_trend = _calculate_monthly_trend(df)
        metrics.cost_velocity = _calculate_cost_velocity(df)
        metrics.cost_forecast_30d = _forecast_costs(df, days=30)
        metrics.cost_forecast_90d = _forecast_costs(df, days=90)

    # ------------------------------------------------------------------
    # Risk metrics: probability-weighted exposure
    # ------------------------------------------------------------------
    # Risk-adjusted exposure weights each ticket's cost by its severity level
    metrics.risk_adjusted_exposure = _calculate_risk_exposure(df)

    # Recurrence exposure: each ticket's cost weighted by recurrence probability
    metrics.recurrence_exposure = _calculate_recurrence_exposure(df)

    # Critical cost ratio: what fraction of total cost comes from Critical severity?
    if 'Severity_Norm' in df.columns:
        critical_cost = df[df['Severity_Norm'] == 'Critical']['Financial_Impact'].sum()
        metrics.critical_cost_ratio = (critical_cost / metrics.total_cost) if metrics.total_cost > 0 else 0

    # ------------------------------------------------------------------
    # Benchmark metrics
    # ------------------------------------------------------------------
    # Composite efficiency score (0-100) with penalty deductions
    metrics.cost_efficiency_score = _calculate_efficiency_score(df, metrics)

    # How far the average ticket cost is from the target benchmark
    metrics.cost_vs_benchmark = metrics.avg_cost_per_ticket - metrics.target_cost_per_ticket

    # ------------------------------------------------------------------
    # Business impact: broader organisational exposure
    # ------------------------------------------------------------------
    # All multipliers are loaded from price_catalog.xlsx "Business Multipliers" sheet
    # so finance teams can tune them without code changes.

    # Revenue at risk: escalation costs cascade into downstream revenue loss.
    # Default multiplier 2.5x means $1 of escalation cost implies $2.50 of
    # revenue exposure (customer churn, SLA credits, rework downstream).
    metrics.revenue_at_risk = metrics.total_cost * _get_biz_multiplier('revenue_at_risk', 2.5)

    # Customer impact: extra cost weight for external/customer-facing escalations
    metrics.customer_impact_cost = _calculate_customer_impact(df)

    # SLA penalty: estimated contractual penalties for unresolved Critical tickets
    metrics.sla_penalty_exposure = _calculate_sla_penalty(df)

    # Opportunity cost: resources spent on escalations could have been productive
    metrics.opportunity_cost = metrics.total_cost * _get_biz_multiplier('opportunity_cost', 0.35)

    logger.info(f"✓ Calculated comprehensive financial metrics: ${metrics.total_cost:,.2f} total")

    return metrics


def _calculate_recurring_cost(df: pd.DataFrame) -> float:
    """Calculate cost of recurring issues.

    Recurring issues are those where the AI-predicted recurrence probability
    (``AI_Recurrence_Risk``) exceeds 30%.  These represent the "chronic pain"
    that organisations should invest in fixing permanently.

    Args:
        df: DataFrame with AI_Recurrence_Risk and Financial_Impact columns.

    Returns:
        Total dollar cost of recurring issues, or 0.0 if data is unavailable.
    """
    if 'AI_Recurrence_Risk' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    try:
        recurrence_risk = pd.to_numeric(df['AI_Recurrence_Risk'], errors='coerce').fillna(0)
        # Threshold: >30% probability is considered "likely to recur"
        recurring_mask = recurrence_risk > 0.3
        return df[recurring_mask]['Financial_Impact'].sum()
    except Exception:
        return 0.0


def _calculate_preventable_cost(df: pd.DataFrame) -> float:
    """Calculate cost of preventable issues.

    "Preventable" categories are those where the root cause is typically
    a human process failure that better training, tooling, or SOPs could
    eliminate.  The five preventable categories from the 8-category taxonomy:

    1. Process Compliance      -- SOP violations, skipped steps
    2. Communication & Response -- delayed replies, missing updates
    3. Configuration & Data Mismatch -- port matrix, RET, TAC errors
    4. Scheduling & Planning   -- TI scheduling, premature scheduling
    5. Documentation & Reporting -- missing snapshots, wrong attachments

    The remaining three categories (Validation & QA, Site Readiness,
    Nesting & Tool Errors) are excluded because they often involve
    external dependencies or infrastructure state that is harder to prevent
    through process changes alone.

    Args:
        df: DataFrame with AI_Category and Financial_Impact columns.

    Returns:
        Total dollar cost of tickets in preventable categories.
    """
    if 'AI_Category' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    # Categories that are typically preventable (matches 8-category AI classification)
    preventable_categories = [
        'Process Compliance',
        'Communication & Response',
        'Configuration & Data Mismatch',
        'Scheduling & Planning',
        'Documentation & Reporting',
    ]

    preventable_mask = df['AI_Category'].isin(preventable_categories)
    return df[preventable_mask]['Financial_Impact'].sum()


def _calculate_cost_avoidance(df: pd.DataFrame) -> float:
    """Calculate potential cost avoidance through root cause fixes.

    Looks at tickets that have 3+ similar historical tickets (indicating a
    pattern) and estimates how much could be saved if the root cause were
    fixed.  The cost_avoidance_rate multiplier (default 0.7 = 70%) is
    loaded from price_catalog.xlsx.

    Logic: if a ticket has >2 similar historical tickets, it is a systemic
    pattern.  Fixing the root cause would avoid ~70% of the cost for all
    such tickets going forward.

    Args:
        df: DataFrame with Similar_Tickets_Found and Financial_Impact columns.

    Returns:
        Estimated dollar savings from root-cause fixes.
    """
    if 'Similar_Tickets_Found' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    try:
        similar_count = pd.to_numeric(df['Similar_Tickets_Found'], errors='coerce').fillna(0)
        # Threshold: >2 similar tickets = confirmed pattern worth investing in
        repeat_mask = similar_count > 2

        # Avoidance rate from price_catalog.xlsx (default 70%)
        avoidance_rate = _get_biz_multiplier('cost_avoidance_rate', 0.7)
        return df[repeat_mask]['Financial_Impact'].sum() * avoidance_rate
    except Exception:
        return 0.0


def _calculate_monthly_trend(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate monthly cost trends.

    Groups tickets by calendar month and sums Financial_Impact to produce
    a time series for trend charts and forecasting.

    Args:
        df: DataFrame with Issue_Date and Financial_Impact columns.

    Returns:
        Dict mapping month strings ("YYYY-MM") to total cost for that month.
        Empty dict on failure.
    """
    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        # Group by calendar month period and sum costs
        df_temp['Month'] = df_temp['Issue_Date'].dt.to_period('M').astype(str)
        monthly_costs = df_temp.groupby('Month')['Financial_Impact'].sum().to_dict()

        return monthly_costs
    except Exception:
        return {}


def _calculate_cost_velocity(df: pd.DataFrame) -> float:
    """Calculate rate of cost change ($ per day).

    Fits a degree-1 polynomial (linear regression) to the scatter plot of
    (days_from_start, Financial_Impact) across all tickets.  The slope
    coefficient represents the average change in per-ticket cost over time.

    A positive velocity means per-ticket costs are increasing (bad).
    A negative velocity means costs are decreasing (good).
    Near-zero means stable.

    Args:
        df: DataFrame with Issue_Date and Financial_Impact columns.

    Returns:
        Slope of the linear fit ($/day), or 0.0 if insufficient data.
    """
    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) < 2:
            return 0.0

        df_temp = df_temp.sort_values('Issue_Date')
        # Convert dates to integer days from the earliest ticket
        df_temp['Days_From_Start'] = (df_temp['Issue_Date'] - df_temp['Issue_Date'].min()).dt.days

        # Linear regression: Financial_Impact ~ Days_From_Start
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(df_temp['Days_From_Start'], df_temp['Financial_Impact'], 1)

        # coef[1] is the slope (cost change per day)
        return float(p.coef[1])  # Slope = cost per day
    except Exception:
        return 0.0


def _forecast_costs(df: pd.DataFrame, days: int = 30) -> float:
    """Forecast future costs based on historical trends.

    Uses a simple cost-per-day extrapolation:
        forecast = (total_cost / data_span_days) x forecast_days

    If fewer than 5 data points exist, falls back to average cost per day.
    This is intentionally simple; the forecasting confidence level reported
    by ``calculate_financial_forecasts()`` communicates the reliability.

    Args:
        df:   DataFrame with Issue_Date and Financial_Impact columns.
        days: Number of days to forecast into the future.

    Returns:
        Projected total cost for the specified period, or 0.0 if
        insufficient data.
    """
    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) < 5:
            # Not enough data for a meaningful trend; use simple average
            return df['Financial_Impact'].mean() * days

        # Calculate average cost per day over the data's time span
        date_range = (df_temp['Issue_Date'].max() - df_temp['Issue_Date'].min()).days
        if date_range == 0:
            return 0.0  # All tickets on same day; cannot extrapolate

        total_cost = df_temp['Financial_Impact'].sum()
        cost_per_day = total_cost / date_range

        return cost_per_day * days
    except Exception:
        return 0.0


def _calculate_risk_exposure(df: pd.DataFrame) -> float:
    """Calculate risk-adjusted financial exposure.

    Weights each ticket's Financial_Impact by a severity-dependent risk
    factor.  Critical tickets are counted at full face value (1.0x), while
    lower severities are discounted because they are less likely to cause
    downstream business damage:

        Critical: 1.0x  (full exposure)
        High/Major: 0.7x
        Medium/Minor: 0.4x
        Low: 0.2x

    The result represents the probability-weighted financial exposure --
    useful for risk reserves and insurance discussions.

    Args:
        df: DataFrame with Financial_Impact and optionally Severity_Norm.

    Returns:
        Risk-weighted total exposure in dollars.
    """
    if 'Financial_Impact' not in df.columns:
        return 0.0

    # Severity-to-risk-weight mapping
    risk_weights = {
        'Critical': 1.0,   # Full exposure -- these issues have real business impact
        'High': 0.7,       # Significant but not always catastrophic
        'Major': 0.7,      # Same weight as High (alternate naming convention)
        'Medium': 0.4,     # Moderate probability of downstream impact
        'Minor': 0.4,      # Same weight as Medium (alternate naming)
        'Low': 0.2         # Minimal expected impact
    }

    if 'Severity_Norm' not in df.columns:
        # Without severity data, return un-weighted total as conservative estimate
        return df['Financial_Impact'].sum()

    # Weight each ticket's cost by its severity risk factor and sum
    total_exposure = 0.0
    for severity, weight in risk_weights.items():
        mask = df['Severity_Norm'] == severity
        total_exposure += df[mask]['Financial_Impact'].sum() * weight

    return total_exposure


def _calculate_recurrence_exposure(df: pd.DataFrame) -> float:
    """Calculate financial exposure from recurrence risk.

    For each ticket, multiplies its Financial_Impact by its AI-predicted
    recurrence probability.  The result represents the expected cost of
    the issue happening again.

    Formula per ticket: Financial_Impact x AI_Recurrence_Risk
    This module sums across all tickets.

    Args:
        df: DataFrame with AI_Recurrence_Risk and Financial_Impact columns.

    Returns:
        Total expected recurrence cost in dollars.
    """
    if 'AI_Recurrence_Risk' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    try:
        recurrence_risk = pd.to_numeric(df['AI_Recurrence_Risk'], errors='coerce').fillna(0)
        financial_impact = pd.to_numeric(df['Financial_Impact'], errors='coerce').fillna(0)

        # Element-wise product: each ticket's cost x its recurrence probability
        return (recurrence_risk * financial_impact).sum()
    except Exception:
        return 0.0


def _calculate_customer_impact(df: pd.DataFrame) -> float:
    """Calculate customer-facing impact costs.

    External and customer-originated escalations carry a premium because
    they affect customer satisfaction, brand reputation, and potentially
    trigger SLA penalties.  The premium multiplier (default 1.5x) is
    loaded from price_catalog.xlsx.

    Args:
        df: DataFrame with Origin_Norm and Financial_Impact columns.

    Returns:
        Dollar cost of customer-facing escalations with premium applied.
    """
    if 'Origin_Norm' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    # Filter for customer/external-facing tickets
    customer_facing = ['External', 'Customer']
    mask = df['Origin_Norm'].isin(customer_facing)

    # Apply customer impact premium from price_catalog.xlsx
    multiplier = _get_biz_multiplier('customer_impact', 1.5)
    return df[mask]['Financial_Impact'].sum() * multiplier


def _calculate_sla_penalty(df: pd.DataFrame) -> float:
    """Estimate SLA penalty exposure.

    Only Critical-severity tickets are considered for SLA penalties because
    lower severities typically have more lenient SLA terms.  The penalty
    rate (default 20%) is loaded from price_catalog.xlsx.

    Formula: SLA_Penalty = Critical_Ticket_Cost x penalty_rate

    Args:
        df: DataFrame with Severity_Norm and Financial_Impact columns.

    Returns:
        Estimated SLA penalty exposure in dollars.
    """
    if 'Severity_Norm' not in df.columns or 'Financial_Impact' not in df.columns:
        return 0.0

    # Only critical-severity tickets trigger SLA penalties
    critical_mask = df['Severity_Norm'] == 'Critical'
    critical_cost = df[critical_mask]['Financial_Impact'].sum()

    # Penalty rate from price_catalog.xlsx (default 20%)
    penalty_rate = _get_biz_multiplier('sla_penalty', 0.2)
    return critical_cost * penalty_rate


def _calculate_efficiency_score(df: pd.DataFrame, metrics: FinancialMetrics) -> float:
    """
    Calculate cost efficiency score (0-100).

    Starts at 100 (perfect efficiency) and deducts penalty points for
    four inefficiency indicators:

    1. **High average cost** (max -30 points):
       Penalty if avg_cost > target_cost.  Scaled by how far over target.
    2. **High cost concentration** (max -10 points):
       Penalty if Pareto ratio > 0.8 (worse than 80/20 rule).
    3. **High recurring cost** (max -20 points):
       Penalty proportional to recurring cost as fraction of total.
    4. **High critical cost ratio** (max -20 points):
       Penalty proportional to critical-severity share of total cost.

    Higher score = better efficiency.  Score is clamped to [0, 100].

    Args:
        df:      The analysis DataFrame (unused directly but available).
        metrics: Partially-filled FinancialMetrics with the values needed.

    Returns:
        Efficiency score as a float in range [0.0, 100.0].
    """
    score = 100.0

    # Penalty 1: Average cost exceeds the target benchmark
    if metrics.avg_cost_per_ticket > metrics.target_cost_per_ticket:
        cost_ratio = metrics.avg_cost_per_ticket / metrics.target_cost_per_ticket
        score -= min(30, (cost_ratio - 1) * 20)  # Max 30 point penalty

    # Penalty 2: Cost concentration worse than 80/20 rule
    # (i.e. top 20% of tickets account for >80% of cost)
    if metrics.cost_concentration_ratio > 0.8:  # Worse than 80/20
        score -= (metrics.cost_concentration_ratio - 0.8) * 50

    # Penalty 3: High proportion of costs from recurring issues
    if metrics.total_cost > 0:
        recurring_ratio = metrics.recurring_issue_cost / metrics.total_cost
        score -= recurring_ratio * 20  # Max 20 point penalty

    # Penalty 4: High proportion of costs from critical-severity tickets
    score -= metrics.critical_cost_ratio * 20  # Max 20 point penalty

    # Clamp to valid range
    return max(0, min(100, score))


def calculate_roi_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate ROI opportunities from fixing root causes.

    Identifies the top 5 cost-driving categories and, for each one with
    3+ incidents (a confirmed pattern), estimates:
    - Investment required: 10x the average ticket cost for that category
      (covers root-cause analysis, process redesign, training, tooling)
    - Annual savings: total_cost x annualization_factor x prevention_rate
    - ROI percentage: (savings - investment) / investment x 100
    - Payback period in months: investment / (monthly savings)

    The annualisation factor is computed from the actual data date range
    (e.g. if data spans 90 days, factor = 365/90 = 4.06x) rather than
    using an arbitrary multiplier.  This avoids over/under-stating
    savings when the analysis window varies.

    Prevention rate and cost avoidance rate are loaded from
    price_catalog.xlsx Business Multipliers sheet.

    Args:
        df: Enriched DataFrame with Financial_Impact, AI_Category, and
            optionally Issue_Date.

    Returns:
        Dictionary with ROI analysis including:
        - total_investment_required: sum of all category investments
        - expected_annual_savings:   sum of all category annual savings
        - roi_percentage:            aggregate ROI
        - payback_months:            aggregate payback period
        - data_span_days:            actual date range in the data
        - annualization_factor:      365 / data_span_days
        - top_opportunities:         list of per-category ROI breakdowns
    """
    roi_analysis = {
        'total_investment_required': 0.0,
        'expected_annual_savings': 0.0,
        'roi_percentage': 0.0,
        'payback_months': 0.0,
        'data_span_days': 0,
        'annualization_factor': 1.0,
        'top_opportunities': []
    }

    if 'Financial_Impact' not in df.columns or 'AI_Category' not in df.columns:
        return roi_analysis

    # ------------------------------------------------------------------
    # Step 1: Calculate annualisation factor from the actual data range
    # ------------------------------------------------------------------
    # This converts observed costs into annual projections proportionally.
    # E.g. 90 days of data -> factor = 365/90 = 4.06x
    annualization_factor = 1.0
    if 'Issue_Date' in df.columns:
        dates = pd.to_datetime(df['Issue_Date'], errors='coerce').dropna()
        if len(dates) >= 2:
            data_span_days = (dates.max() - dates.min()).days
            if data_span_days > 0:
                annualization_factor = 365.0 / data_span_days
                roi_analysis['data_span_days'] = data_span_days
            else:
                annualization_factor = 1.0  # All same day - can't extrapolate
        else:
            annualization_factor = 1.0  # Not enough data points
    roi_analysis['annualization_factor'] = round(annualization_factor, 2)

    # Load the expected prevention success rate (default 80%)
    prevention_rate = _get_biz_multiplier('prevention_rate', 0.8)

    # ------------------------------------------------------------------
    # Step 2: Rank categories by total cost
    # ------------------------------------------------------------------
    category_costs = df.groupby('AI_Category')['Financial_Impact'].agg(['sum', 'count', 'mean'])
    category_costs = category_costs.sort_values('sum', ascending=False)

    # ------------------------------------------------------------------
    # Step 3: Build ROI case for top 5 categories with 3+ incidents
    # ------------------------------------------------------------------
    # Only categories with 3+ incidents are included because fewer
    # incidents do not constitute a pattern worth investing in.
    for category in category_costs.index[:5]:
        total_cost = category_costs.loc[category, 'sum']
        count = int(category_costs.loc[category, 'count'])
        avg_cost = category_costs.loc[category, 'mean']

        if count >= 3:  # Multiple incidents = pattern
            # Investment heuristic: 10x the average ticket cost covers
            # root-cause analysis, process redesign, training, and tooling
            investment = avg_cost * 10

            # Annual savings: annualise the observed cost and apply prevention rate
            annual_savings = total_cost * annualization_factor * prevention_rate

            # Classic ROI formula: (gain - cost) / cost x 100
            roi_pct = ((annual_savings - investment) / investment * 100) if investment > 0 else 0

            # Payback: how many months until the investment is recovered
            payback_months = (investment / (annual_savings / 12)) if annual_savings > 0 else float('inf')

            roi_analysis['top_opportunities'].append({
                'category': category,
                'total_cost': total_cost,
                'incident_count': count,
                'avg_cost': avg_cost,
                'investment_required': investment,
                'annual_savings': annual_savings,
                'roi_percentage': roi_pct,
                'payback_months': payback_months
            })

    # ------------------------------------------------------------------
    # Step 4: Aggregate across all identified opportunities
    # ------------------------------------------------------------------
    if roi_analysis['top_opportunities']:
        roi_analysis['total_investment_required'] = sum(opp['investment_required'] for opp in roi_analysis['top_opportunities'])
        roi_analysis['expected_annual_savings'] = sum(opp['annual_savings'] for opp in roi_analysis['top_opportunities'])

        if roi_analysis['total_investment_required'] > 0:
            # Aggregate ROI across all opportunities
            roi_analysis['roi_percentage'] = (
                (roi_analysis['expected_annual_savings'] - roi_analysis['total_investment_required']) /
                roi_analysis['total_investment_required'] * 100
            )
            # Aggregate payback period
            roi_analysis['payback_months'] = (
                roi_analysis['total_investment_required'] /
                (roi_analysis['expected_annual_savings'] / 12)
            )

    return roi_analysis


def calculate_cost_avoidance(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate potential cost avoidance opportunities across four levers.

    Each lever represents a different strategy for reducing escalation costs:

    1. **Recurring issues** (80% avoidable):
       Fix root causes of issues that keep coming back.
    2. **Preventable categories** (60% avoidable):
       Improve processes in categories that are inherently preventable.
    3. **Knowledge sharing** (variable):
       Share learnings from similar historical tickets to prevent repeats.
    4. **Automation** (50% avoidable):
       Automate repetitive categories (Scheduling, Documentation, Process
       Compliance, Nesting) where tool-based solutions can replace manual work.

    These percentages are conservative estimates.  The total_avoidance is the
    sum across all four levers (note: there is intentional overlap between
    levers -- this represents the "maximum opportunity envelope").

    Args:
        df: Enriched DataFrame with Financial_Impact and AI_Category columns.

    Returns:
        Dict with per-lever avoidance amounts and total_avoidance.
    """
    avoidance = {
        'recurring_issues': 0.0,
        'preventable_categories': 0.0,
        'knowledge_sharing': 0.0,
        'automation': 0.0,
        'total_avoidance': 0.0
    }

    if 'Financial_Impact' not in df.columns:
        return avoidance

    # Lever 1: Eliminate recurring issues (80% of recurring cost is avoidable)
    avoidance['recurring_issues'] = _calculate_recurring_cost(df) * 0.8

    # Lever 2: Fix preventable categories (60% of preventable cost is avoidable)
    avoidance['preventable_categories'] = _calculate_preventable_cost(df) * 0.6

    # Lever 3: Knowledge sharing (savings from fixing repeat patterns)
    avoidance['knowledge_sharing'] = _calculate_cost_avoidance(df)

    # Lever 4: Automation potential for repetitive categories
    # These four categories involve structured, rule-based work that is
    # amenable to tool-based automation (50% cost reduction estimate)
    if 'AI_Category' in df.columns:
        automatable_categories = [
            'Scheduling & Planning',       # Automated TI scheduling checks
            'Documentation & Reporting',   # Template-based report generation
            'Process Compliance',          # Automated workflow enforcement
            'Nesting & Tool Errors'        # Automated nesting validation
        ]
        auto_mask = df['AI_Category'].isin(automatable_categories)
        avoidance['automation'] = df[auto_mask]['Financial_Impact'].sum() * 0.5

    # Total avoidance: sum of all four levers
    # Note: levers can overlap (a recurring issue in a preventable category
    # would be counted in both levers), so total represents the maximum envelope
    avoidance['total_avoidance'] = sum([
        avoidance['recurring_issues'],
        avoidance['preventable_categories'],
        avoidance['knowledge_sharing'],
        avoidance['automation']
    ])

    return avoidance


def calculate_efficiency_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate cost efficiency and performance metrics.

    Provides per-engineer and per-category cost analysis, engineer
    efficiency scores, and identifies statistical outlier tickets.

    Engineer efficiency score formula:
        score = min(100, (avg_cost / engineer_avg_cost) x 100)
    Engineers with below-average cost get scores > 100 (capped at 100).
    Engineers with above-average cost get scores < 100.

    Args:
        df: Enriched DataFrame with Financial_Impact and optionally
            Resolution_Days, Engineer_Assigned, AI_Category columns.

    Returns:
        Dict with cost_per_hour, cost_per_engineer, cost_per_category,
        engineer_efficiency_scores, category_efficiency_scores, and outliers.
    """
    efficiency = {
        'cost_per_hour': 0.0,
        'cost_per_engineer': {},
        'cost_per_category': {},
        'engineer_efficiency_scores': {},
        'category_efficiency_scores': {},
        'outliers': []
    }

    if 'Financial_Impact' not in df.columns:
        return efficiency

    # Overall cost per resolution hour
    if 'Resolution_Days' in df.columns:
        total_hours = df['Resolution_Days'].sum() * 24  # Days -> hours
        efficiency['cost_per_hour'] = df['Financial_Impact'].sum() / total_hours if total_hours > 0 else 0

    # Per-engineer average cost
    if 'Engineer_Assigned' in df.columns:
        efficiency['cost_per_engineer'] = df.groupby('Engineer_Assigned')['Financial_Impact'].mean().to_dict()

        # Engineer efficiency scores: inverse relationship with cost
        # Lower cost = higher efficiency score
        avg_cost = df['Financial_Impact'].mean()
        for engineer, cost in efficiency['cost_per_engineer'].items():
            # Ratio: global_avg / engineer_avg.  If engineer costs less than
            # average, ratio > 1, score > 100 (capped).  If more expensive,
            # ratio < 1, score < 100.
            ratio = avg_cost / cost if cost > 0 else 1.0
            efficiency['engineer_efficiency_scores'][engineer] = min(100, ratio * 100)

    # Per-category average cost
    if 'AI_Category' in df.columns:
        efficiency['cost_per_category'] = df.groupby('AI_Category')['Financial_Impact'].mean().to_dict()

    # Identify high-cost outliers (top 5% by Financial_Impact)
    # These are the tickets that deserve individual investigation
    threshold = df['Financial_Impact'].quantile(0.95)
    outliers = df[df['Financial_Impact'] >= threshold]

    for idx, row in outliers.iterrows():
        efficiency['outliers'].append({
            'id': row.get('Ticket_ID', idx),
            'cost': row['Financial_Impact'],
            'category': row.get('AI_Category', 'Unknown'),
            'severity': row.get('Severity_Norm', 'Unknown')
        })

    return efficiency


def calculate_financial_forecasts(df: pd.DataFrame, periods: int = 12) -> Dict[str, Any]:
    """
    Generate financial forecasts and projections.

    Uses a simple moving-average approach:
    1. Compute monthly historical cost data from Issue_Date + Financial_Impact.
    2. Calculate mean and standard deviation of monthly costs.
    3. Project forward for ``periods`` months using the mean as the baseline
       and the standard deviation as the confidence interval.
    4. Detect trend direction by comparing the most recent 3 months against
       older months (>10% higher = increasing, >10% lower = decreasing).
    5. Assess forecast confidence using the coefficient of variation (CV):
       CV < 0.3 = high, 0.3-0.6 = medium, >0.6 = low confidence.
    6. Produce three risk scenarios: best case (-20%), expected, worst (+30%).

    Args:
        df:      Enriched DataFrame with Issue_Date and Financial_Impact.
        periods: Number of months to forecast (default 12 = one year).

    Returns:
        Dict with monthly_projection (list of dicts with projected_cost,
        lower_bound, upper_bound), annual_projection, trend direction,
        confidence level, and risk_scenarios (best/expected/worst).
    """
    forecasts = {
        'monthly_projection': [],
        'annual_projection': 0.0,
        'trend': 'stable',  # 'increasing', 'decreasing', 'stable'
        'confidence': 'low',  # 'high', 'medium', 'low'
        'risk_scenarios': {}
    }

    if 'Issue_Date' not in df.columns or 'Financial_Impact' not in df.columns:
        return forecasts

    try:
        df_temp = df.copy()
        df_temp['Issue_Date'] = pd.to_datetime(df_temp['Issue_Date'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Issue_Date'])

        if len(df_temp) < 3:
            # Fewer than 3 data points: cannot establish a meaningful trend
            forecasts['confidence'] = 'low'
            return forecasts

        # ------------------------------------------------------------------
        # Step 1: Build monthly cost history
        # ------------------------------------------------------------------
        df_temp['Month'] = df_temp['Issue_Date'].dt.to_period('M')
        monthly_data = df_temp.groupby('Month')['Financial_Impact'].sum()

        # ------------------------------------------------------------------
        # Step 2: Detect trend direction
        # ------------------------------------------------------------------
        # Compare recent 3 months' average vs older months' average.
        # >10% increase = "increasing", >10% decrease = "decreasing"
        if len(monthly_data) >= 3:
            recent_avg = monthly_data.tail(3).mean()
            older_avg = monthly_data.head(len(monthly_data) - 3).mean()

            if recent_avg > older_avg * 1.1:
                forecasts['trend'] = 'increasing'
            elif recent_avg < older_avg * 0.9:
                forecasts['trend'] = 'decreasing'
            else:
                forecasts['trend'] = 'stable'

        # ------------------------------------------------------------------
        # Step 3: Simple moving-average forecast
        # ------------------------------------------------------------------
        avg_monthly_cost = monthly_data.mean()
        std_monthly_cost = monthly_data.std()

        for i in range(1, periods + 1):
            forecasts['monthly_projection'].append({
                'month': i,
                'projected_cost': avg_monthly_cost,
                # Confidence bands: mean +/- 1 standard deviation
                'lower_bound': max(0, avg_monthly_cost - std_monthly_cost),
                'upper_bound': avg_monthly_cost + std_monthly_cost
            })

        # Annual projection = 12 months at the average monthly rate
        forecasts['annual_projection'] = avg_monthly_cost * 12

        # ------------------------------------------------------------------
        # Step 4: Assess forecast confidence
        # ------------------------------------------------------------------
        # Coefficient of variation (CV): lower = more predictable = higher confidence
        cv = (std_monthly_cost / avg_monthly_cost) if avg_monthly_cost > 0 else 1.0
        if cv < 0.3:
            forecasts['confidence'] = 'high'    # Very consistent monthly costs
        elif cv < 0.6:
            forecasts['confidence'] = 'medium'  # Moderate variability
        else:
            forecasts['confidence'] = 'low'     # High variability, forecast unreliable

        # ------------------------------------------------------------------
        # Step 5: Risk scenarios for executive planning
        # ------------------------------------------------------------------
        forecasts['risk_scenarios'] = {
            'best_case': avg_monthly_cost * 12 * 0.8,  # 20% reduction
            'expected': avg_monthly_cost * 12,
            'worst_case': avg_monthly_cost * 12 * 1.3  # 30% increase
        }

    except Exception as e:
        logger.warning(f"Error calculating forecasts: {e}")

    return forecasts


def generate_financial_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generate actionable financial insights and recommendations.

    Analyses the data across five dimensions and produces prioritised,
    human-readable insight cards for the executive dashboard:

    1. **Cost Concentration** (high priority):
       Fires when >85% of costs come from top 20% of tickets.
       Recommendation: focus root-cause analysis on the vital few.

    2. **Recurring Issue Costs** (high priority):
       Fires when recurring issues exceed 30% of total cost.
       Recommendation: implement preventive measures.

    3. **ROI Opportunity** (high priority):
       Fires when the top ROI opportunity exceeds 200% return.
       Recommendation: invest in root-cause fix for the top category.

    4. **Low Efficiency Score** (medium priority):
       Fires when efficiency score drops below 60/100.
       Recommendation: review high-cost categories for process improvements.

    5. **High Preventable Cost** (medium priority):
       Fires when preventable categories exceed 25% of total cost.
       Recommendation: improve processes and documentation.

    6. **Increasing Cost Trend** (high priority):
       Fires when the trend analysis detects increasing costs.
       Recommendation: immediate action to reverse growth.

    Each insight includes a ``potential_savings`` estimate to help
    prioritise investment decisions.

    Args:
        df: Enriched DataFrame with Financial_Impact and other columns.

    Returns:
        List of insight dicts, sorted by priority (high first), each
        containing: priority, type, title, description, recommendation,
        and potential_savings.
    """
    insights = []

    if df.empty or 'Financial_Impact' not in df.columns:
        return insights

    # Compute the metrics and ROI data needed for insight generation
    metrics = calculate_financial_metrics(df)
    roi_data = calculate_roi_metrics(df)

    # --- Insight 1: High cost concentration ---
    # If >85% of costs come from top 20% of tickets, the "vital few" pattern
    # is strong and targeted intervention will have outsized impact.
    if metrics.cost_concentration_ratio > 0.85:
        insights.append({
            'priority': 'high',
            'type': 'cost_concentration',
            'title': 'High Cost Concentration Detected',
            'description': f'{metrics.cost_concentration_ratio*100:.0f}% of costs come from top 20% of tickets',
            'recommendation': 'Focus root cause analysis on high-cost tickets for maximum impact',
            'potential_savings': metrics.total_cost * 0.3
        })

    # --- Insight 2: High recurring issue costs ---
    # Recurring issues are the most actionable cost driver because they
    # represent known, fixable patterns.
    if metrics.recurring_issue_cost > metrics.total_cost * 0.3:
        insights.append({
            'priority': 'high',
            'type': 'recurring_issues',
            'title': 'Significant Recurring Issue Costs',
            'description': f'${metrics.recurring_issue_cost:,.0f} in recurring issues',
            'recommendation': 'Implement preventive measures for high-recurrence categories',
            'potential_savings': metrics.recurring_issue_cost * 0.7
        })

    # --- Insight 3: High-ROI opportunity ---
    # If the top category has >200% ROI, it is a no-brainer investment.
    if roi_data['top_opportunities']:
        top_roi = roi_data['top_opportunities'][0]
        if top_roi['roi_percentage'] > 200:
            insights.append({
                'priority': 'high',
                'type': 'roi_opportunity',
                'title': f'High ROI Opportunity: {top_roi["category"]}',
                'description': f'{top_roi["roi_percentage"]:.0f}% ROI, {top_roi["payback_months"]:.1f} month payback',
                'recommendation': f'Invest ${top_roi["investment_required"]:,.0f} to save ${top_roi["annual_savings"]:,.0f}/year',
                'potential_savings': top_roi['annual_savings']
            })

    # --- Insight 4: Low efficiency score ---
    # Below 60/100 indicates systemic inefficiency that needs attention.
    if metrics.cost_efficiency_score < 60:
        insights.append({
            'priority': 'medium',
            'type': 'efficiency',
            'title': 'Cost Efficiency Below Target',
            'description': f'Efficiency score: {metrics.cost_efficiency_score:.0f}/100',
            'recommendation': 'Review high-cost categories and implement process improvements',
            'potential_savings': metrics.preventable_cost * 0.5
        })

    # --- Insight 5: High preventable cost ---
    # If >25% of costs are in preventable categories, there is significant
    # room for process improvement.
    if metrics.preventable_cost > metrics.total_cost * 0.25:
        insights.append({
            'priority': 'medium',
            'type': 'preventable_cost',
            'title': 'High Preventable Cost',
            'description': f'${metrics.preventable_cost:,.0f} in preventable categories',
            'recommendation': 'Improve processes and documentation to prevent recurring issues',
            'potential_savings': metrics.preventable_cost * 0.6
        })

    # --- Insight 6: Increasing cost trend ---
    # Rising costs demand immediate action before they compound.
    forecasts = calculate_financial_forecasts(df)
    if forecasts['trend'] == 'increasing':
        insights.append({
            'priority': 'high',
            'type': 'cost_trend',
            'title': 'Increasing Cost Trend',
            'description': f'Projected annual cost: ${forecasts["annual_projection"]:,.0f}',
            'recommendation': 'Immediate action required to reverse cost growth trend',
            'potential_savings': forecasts['annual_projection'] * 0.25
        })

    # Sort insights: high priority first, then medium, then low
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    insights.sort(key=lambda x: priority_order.get(x['priority'], 3))

    return insights
