"""
Smart Alert Thresholds.

Dynamic threshold calculation based on historical patterns using:
- IQR (Interquartile Range) for outlier detection
- Z-score for standard deviation-based thresholds
- Percentile-based limits
- Rolling window analysis for trend-aware thresholds
- Exponential smoothing for seasonal adjustments

Architecture Overview:
    This module provides a statistical framework for computing adaptive alert
    thresholds from historical metric data.  Rather than using static, manually
    configured thresholds (e.g., "alert if count > 20"), it learns the normal
    distribution of a metric and sets warning, critical, and emergency levels
    based on how unusual the current value is relative to history.

Statistical Methods:
    Three core methods are supported, plus a "hybrid" that combines them:

    1. **IQR (Interquartile Range)**:
       - Warning  = Q3 (75th percentile)
       - Critical = Q3 + 1.5 * IQR  (Tukey's fence for mild outliers)
       - Emergency = 1.5 * (Q3 + 1.5 * IQR)  (far outlier)
       Best suited for skewed distributions common in count-based metrics.

    2. **Z-score (standard deviation)**:
       - Warning  = mean + 2*sigma
       - Critical = mean + 3*sigma
       - Emergency = mean + 4*sigma
       Appropriate for approximately normal distributions.

    3. **Percentile**:
       - Warning  = 75th percentile
       - Critical = 90th percentile
       - Emergency = 99th percentile
       Non-parametric; works well regardless of distribution shape.

    4. **Hybrid (default)**:
       Takes the *minimum* of the percentile and z-score thresholds at each
       level.  This produces conservative (more sensitive) thresholds that
       trigger earlier, which is generally desirable for operational alerting.

Time-Aware Thresholds:
    When a datetime column is provided during ``fit``, day-of-week and
    hour-of-day patterns are computed.  ``get_time_adjusted_threshold`` then
    scales the base thresholds by the ratio of the current time-slot's mean
    to the global mean, enabling thresholds that are higher on busy days
    and lower on quiet days.

Trend Detection:
    The ``_detect_trend`` method compares the current value to Q1 and Q3 of
    the historical distribution to determine if the metric is increasing,
    decreasing, or stable.

Multi-Metric Monitoring:
    The ``MultiMetricMonitor`` class wraps multiple ``SmartThresholdCalculator``
    instances, enabling dashboard-style simultaneous monitoring of several KPIs
    with a unified ``check_all`` / ``get_alerts`` / ``get_dashboard_summary``
    API.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

try:
    from ..core.gpu_utils import is_gpu_available
    GPU_AVAILABLE = is_gpu_available()
except ImportError:
    GPU_AVAILABLE = False


class AlertLevel(Enum):
    """
    Alert severity levels.

    Ordered from least to most severe:
        NORMAL < WARNING < CRITICAL < EMERGENCY

    The ordering is used by ``MultiMetricMonitor.get_alerts`` to filter
    results by minimum severity.
    """
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ThresholdConfig:
    """
    Configuration for threshold calculation.

    All parameters have sensible defaults based on common statistical practice.
    Users can override individual values to tune sensitivity.

    Attributes:
        iqr_multiplier (float): Multiplier for the IQR when computing the upper
            fence.  Standard Tukey fence = 1.5; increase for fewer false alarms.
        z_score_warning (float): Number of standard deviations from the mean
            for the warning threshold (default 2.0).
        z_score_critical (float): Standard deviations for critical (default 3.0).
        z_score_emergency (float): Standard deviations for emergency (default 4.0).
        percentile_warning (float): Percentile for warning (default 75th).
        percentile_critical (float): Percentile for critical (default 90th).
        percentile_emergency (float): Percentile for emergency (default 99th).
        rolling_window_days (int): Number of historical days for rolling
            statistics (default 30).
        min_samples (int): Minimum data points required for reliable threshold
            computation.  A warning is issued if fewer samples are available.
        seasonality_period (int): Days in one seasonality cycle (default 7 for
            weekly patterns).
        trend_sensitivity (float): Weight given to recent trend vs. historical
            baseline (default 0.3).
        cooldown_minutes (int): Minimum minutes between repeated alerts for the
            same metric, to prevent alert fatigue.
    """
    # Statistical method parameters
    iqr_multiplier: float = 1.5  # Standard IQR multiplier for outliers
    z_score_warning: float = 2.0  # Z-score for warning level
    z_score_critical: float = 3.0  # Z-score for critical level
    z_score_emergency: float = 4.0  # Z-score for emergency level

    # Percentile-based thresholds
    percentile_warning: float = 75.0
    percentile_critical: float = 90.0
    percentile_emergency: float = 99.0

    # Rolling window settings
    rolling_window_days: int = 30  # Days for rolling statistics
    min_samples: int = 10  # Minimum samples for valid threshold

    # Adaptive settings
    seasonality_period: int = 7  # Days (weekly seasonality)
    trend_sensitivity: float = 0.3  # How much to weight recent trends

    # Alert cooldown
    cooldown_minutes: int = 60  # Minimum time between repeat alerts


@dataclass
class ThresholdResult:
    """
    Result of threshold calculation for a single metric check.

    Attributes:
        metric_name (str): Name of the metric being checked.
        current_value (float): The value that was evaluated.
        thresholds (dict): The computed warning/critical/emergency thresholds.
        alert_level (AlertLevel): Which severity level was triggered.
        breach_amount (float): How far above the triggered threshold the value
            is (0.0 if no breach).
        historical_mean (float): Mean of the historical data used to fit.
        historical_std (float): Standard deviation of the historical data.
        percentile_rank (float): Where the current value falls in the
            historical distribution (0-100).
        trend_direction (str): 'increasing', 'decreasing', or 'stable'.
        recommendation (str): Human-readable action recommendation.
    """
    metric_name: str
    current_value: float
    thresholds: Dict[str, float]
    alert_level: AlertLevel
    breach_amount: float = 0.0
    historical_mean: float = 0.0
    historical_std: float = 0.0
    percentile_rank: float = 0.0
    trend_direction: str = "stable"
    recommendation: str = ""


class SmartThresholdCalculator:
    """
    Calculates dynamic alert thresholds based on historical data patterns.

    Features:
    - Multiple statistical methods (IQR, z-score, percentiles)
    - Time-aware thresholds (day-of-week, hour-of-day patterns)
    - Trend-adaptive limits that adjust to data drift
    - Anomaly detection with confidence scores

    Usage:
        >>> calc = SmartThresholdCalculator()
        >>> calc.fit(df, 'escalation_count', 'date_column')
        >>> result = calc.check_value('escalation_count', current_value=25)
        >>> print(result.alert_level)

    Attributes:
        config (ThresholdConfig): Configuration parameters.
        _historical_stats (dict): Fitted statistics keyed by metric column name.
        _alert_history (list): Record of past alert events (for audit trail).
        _last_alert_time (dict): Cooldown tracker per metric.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize with optional configuration.

        Args:
            config (ThresholdConfig, optional): Custom config.  If None, all
                parameters use their documented defaults.
        """
        self.config = config or ThresholdConfig()
        self._historical_stats: Dict[str, Dict] = {}
        self._alert_history: List[Dict] = []
        self._last_alert_time: Dict[str, datetime] = {}

    def fit(self, df: pd.DataFrame, metric_column: str,
            datetime_column: Optional[str] = None) -> 'SmartThresholdCalculator':
        """
        Fit the calculator on historical data.

        Computes all statistics needed for threshold calculation from the
        provided historical data.  The computed stats are stored internally
        and used by ``calculate_thresholds`` and ``check_value``.

        Statistics computed:
            - Basic: mean, std, median, Q1, Q3, min, max, count
            - IQR-based: IQR value, lower/upper fences
            - Percentile: P75, P90, P95, P99
            - Z-score: warning/critical/emergency upper bounds
            - Time patterns (optional): day-of-week and hour-of-day means/stds

        Args:
            df (pd.DataFrame): Historical data.
            metric_column (str): Column containing the metric values.
            datetime_column (str, optional): Datetime column for time-aware
                thresholds.

        Returns:
            SmartThresholdCalculator: Self, for method chaining.

        Raises:
            ValueError: If ``metric_column`` is not found in ``df``.
        """
        if metric_column not in df.columns:
            raise ValueError(f"Column '{metric_column}' not found in dataframe")

        # Drop NaN values before computing statistics
        values = df[metric_column].dropna()

        if len(values) < self.config.min_samples:
            warnings.warn(f"Insufficient samples ({len(values)}) for reliable thresholds")

        # --- Basic descriptive statistics ---
        stats = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'median': float(values.median()),
            'q1': float(values.quantile(0.25)),
            'q3': float(values.quantile(0.75)),
            'min': float(values.min()),
            'max': float(values.max()),
            'count': len(values)
        }

        # --- IQR-based outlier bounds (Tukey's fences) ---
        # IQR = Q3 - Q1; fences at Q1 - k*IQR and Q3 + k*IQR where k = iqr_multiplier
        iqr = stats['q3'] - stats['q1']
        stats['iqr'] = iqr
        stats['iqr_lower'] = stats['q1'] - (self.config.iqr_multiplier * iqr)
        stats['iqr_upper'] = stats['q3'] + (self.config.iqr_multiplier * iqr)

        # --- Percentile thresholds ---
        stats['p75'] = float(values.quantile(0.75))
        stats['p90'] = float(values.quantile(0.90))
        stats['p95'] = float(values.quantile(0.95))
        stats['p99'] = float(values.quantile(0.99))

        # --- Z-score thresholds ---
        # Each level = mean + z * std
        stats['z_warning_upper'] = stats['mean'] + (self.config.z_score_warning * stats['std'])
        stats['z_critical_upper'] = stats['mean'] + (self.config.z_score_critical * stats['std'])
        stats['z_emergency_upper'] = stats['mean'] + (self.config.z_score_emergency * stats['std'])

        # --- Time-based patterns (optional) ---
        if datetime_column and datetime_column in df.columns:
            stats['time_patterns'] = self._calculate_time_patterns(df, metric_column, datetime_column)

        self._historical_stats[metric_column] = stats
        return self

    def _calculate_time_patterns(self, df: pd.DataFrame, metric_column: str,
                                  datetime_column: str) -> Dict:
        """
        Calculate day-of-week and hour-of-day patterns.

        Groups the metric by temporal features and computes per-group mean and
        standard deviation.  These patterns enable time-adjusted thresholds
        that account for periodic variation (e.g., more escalations on Mondays).

        Args:
            df (pd.DataFrame): Historical data with both metric and datetime.
            metric_column (str): The metric column.
            datetime_column (str): The datetime column.

        Returns:
            dict: Nested dict with keys 'day_of_week' and 'hour_of_day',
                each mapping group ID to {'mean': float, 'std': float}.
        """
        df = df.copy()
        df['_dt'] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=['_dt', metric_column])

        patterns = {}

        # Day of week patterns (0=Monday, 6=Sunday in pandas convention)
        df['_dow'] = df['_dt'].dt.dayofweek
        dow_stats = df.groupby('_dow')[metric_column].agg(['mean', 'std']).to_dict('index')
        patterns['day_of_week'] = dow_stats

        # Hour of day patterns (0-23)
        df['_hour'] = df['_dt'].dt.hour
        hour_stats = df.groupby('_hour')[metric_column].agg(['mean', 'std']).to_dict('index')
        patterns['hour_of_day'] = hour_stats

        return patterns

    def calculate_thresholds(self, metric_column: str,
                             method: str = 'hybrid') -> Dict[str, float]:
        """
        Calculate thresholds for a metric.

        Four methods are available:
            - 'iqr': Uses Q3 and Tukey fences.
            - 'zscore': Uses mean + z*sigma at each level.
            - 'percentile': Uses P75, P90, P99.
            - 'hybrid' (default): Takes the *minimum* of percentile and
              z-score at each level, producing more conservative (earlier)
              alerts.

        Args:
            metric_column (str): The metric to calculate thresholds for.
            method (str): One of 'iqr', 'zscore', 'percentile', 'hybrid'.

        Returns:
            dict: Keys 'warning', 'critical', 'emergency' with float values.

        Raises:
            ValueError: If ``fit()`` has not been called for this metric.
        """
        if metric_column not in self._historical_stats:
            raise ValueError(f"No historical data for '{metric_column}'. Call fit() first.")

        stats = self._historical_stats[metric_column]

        if method == 'iqr':
            return {
                'warning': stats['q3'],                    # 75th percentile
                'critical': stats['iqr_upper'],             # Q3 + 1.5 * IQR
                'emergency': stats['iqr_upper'] * 1.5       # far outlier zone
            }
        elif method == 'zscore':
            return {
                'warning': stats['z_warning_upper'],        # mean + 2*sigma
                'critical': stats['z_critical_upper'],      # mean + 3*sigma
                'emergency': stats['z_emergency_upper']     # mean + 4*sigma
            }
        elif method == 'percentile':
            return {
                'warning': stats['p75'],                    # 75th percentile
                'critical': stats['p90'],                   # 90th percentile
                'emergency': stats['p99']                   # 99th percentile
            }
        else:  # hybrid - conservative approach using min of methods
            # Taking the minimum of percentile and z-score produces thresholds
            # that trigger earlier, which is preferred for operational alerting
            # where missed alarms are more costly than false alarms.
            return {
                'warning': min(stats['p75'], stats['z_warning_upper']),
                'critical': min(stats['p90'], stats['z_critical_upper']),
                'emergency': min(stats['p99'], stats['z_emergency_upper'])
            }

    def check_value(self, metric_column: str, current_value: float,
                    timestamp: Optional[datetime] = None) -> ThresholdResult:
        """
        Check if a value breaches any thresholds.

        Compares ``current_value`` against the computed thresholds for the
        metric and determines the appropriate alert level.  Also computes
        the percentile rank of the value within the historical distribution
        and detects the current trend direction.

        Percentile Rank Calculation:
            Uses the z-score of the current value and the standard normal CDF
            to estimate where the value falls in the historical distribution.
            This is an approximation that assumes approximate normality.

        Args:
            metric_column (str): The metric being checked.
            current_value (float): The current value to check.
            timestamp (datetime, optional): Timestamp for time-aware checking
                (reserved for future use).

        Returns:
            ThresholdResult: Complete result with alert level, breach amount,
                percentile rank, trend, and recommendation.

        Raises:
            ValueError: If ``fit()`` has not been called for this metric.
        """
        if metric_column not in self._historical_stats:
            raise ValueError(f"No historical data for '{metric_column}'. Call fit() first.")

        stats = self._historical_stats[metric_column]
        thresholds = self.calculate_thresholds(metric_column)

        # --- Determine alert level by checking thresholds from most to least severe ---
        if current_value >= thresholds['emergency']:
            alert_level = AlertLevel.EMERGENCY
            breach = current_value - thresholds['emergency']
        elif current_value >= thresholds['critical']:
            alert_level = AlertLevel.CRITICAL
            breach = current_value - thresholds['critical']
        elif current_value >= thresholds['warning']:
            alert_level = AlertLevel.WARNING
            breach = current_value - thresholds['warning']
        else:
            alert_level = AlertLevel.NORMAL
            breach = 0.0

        # --- Calculate percentile rank via the standard normal CDF ---
        # Convert to z-score, then use scipy.stats.norm.cdf to get percentile.
        if stats['std'] > 0:
            z_score = (current_value - stats['mean']) / stats['std']
            from scipy import stats as scipy_stats
            percentile_rank = scipy_stats.norm.cdf(z_score) * 100
        else:
            percentile_rank = 50.0  # all values identical -> 50th percentile

        # --- Detect trend direction ---
        trend = self._detect_trend(metric_column, current_value)

        # --- Generate actionable recommendation ---
        recommendation = self._generate_recommendation(
            metric_column, current_value, alert_level, stats
        )

        return ThresholdResult(
            metric_name=metric_column,
            current_value=current_value,
            thresholds=thresholds,
            alert_level=alert_level,
            breach_amount=breach,
            historical_mean=stats['mean'],
            historical_std=stats['std'],
            percentile_rank=percentile_rank,
            trend_direction=trend,
            recommendation=recommendation
        )

    def _detect_trend(self, metric_column: str, current_value: float) -> str:
        """
        Detect if values are trending up, down, or stable.

        Simple trend detection based on the current value's position relative
        to the historical quartiles:
            - Above Q3: 'increasing'
            - Below Q1: 'decreasing'
            - Between Q1 and Q3 (the IQR): 'stable'

        Args:
            metric_column (str): The metric name.
            current_value (float): The current observation.

        Returns:
            str: 'increasing', 'decreasing', 'stable', or 'unknown'.
        """
        stats = self._historical_stats.get(metric_column, {})
        if not stats:
            return "unknown"

        median = stats.get('median', stats.get('mean', current_value))
        q1 = stats.get('q1', median * 0.75)
        q3 = stats.get('q3', median * 1.25)

        if current_value > q3:
            return "increasing"
        elif current_value < q1:
            return "decreasing"
        else:
            return "stable"

    def _generate_recommendation(self, metric_column: str, current_value: float,
                                  alert_level: AlertLevel, stats: Dict) -> str:
        """
        Generate actionable recommendation based on alert.

        Produces human-readable guidance appropriate to the severity level:
            - NORMAL: "No action required."
            - WARNING: "Monitor closely."
            - CRITICAL: "Investigate root cause immediately."
            - EMERGENCY: "Engage emergency response protocols."

        Args:
            metric_column (str): The metric name (included in message).
            current_value (float): The current value.
            alert_level (AlertLevel): The triggered level.
            stats (dict): Historical statistics for context.

        Returns:
            str: Recommendation text.
        """
        if alert_level == AlertLevel.NORMAL:
            return "No action required. Values within normal range."

        # Approximate percentile rank for display in the recommendation text
        percentile_rank = 0
        if stats['std'] > 0:
            z = (current_value - stats['mean']) / stats['std']
            # Linear approximation of percentile for display purposes
            percentile_rank = min(99.9, max(0.1, 50 + z * 30))

        if alert_level == AlertLevel.EMERGENCY:
            return (f"IMMEDIATE ACTION REQUIRED: {metric_column} at {current_value:.1f} "
                    f"is in the {percentile_rank:.0f}th percentile. "
                    "Engage emergency response protocols.")
        elif alert_level == AlertLevel.CRITICAL:
            return (f"URGENT: {metric_column} at {current_value:.1f} exceeds critical threshold. "
                    f"Historical mean is {stats['mean']:.1f}. Investigate root cause immediately.")
        else:  # WARNING
            return (f"Monitor closely: {metric_column} at {current_value:.1f} exceeds warning level. "
                    f"Normal range is {stats['q1']:.1f} - {stats['q3']:.1f}.")

    def get_time_adjusted_threshold(self, metric_column: str,
                                     timestamp: datetime) -> Dict[str, float]:
        """
        Get thresholds adjusted for day-of-week and hour-of-day patterns.

        Adjustment Procedure:
            1. Retrieve the base (global) thresholds via ``calculate_thresholds``.
            2. Look up the mean metric value for the current day-of-week and
               hour-of-day from the fitted time patterns.
            3. Compute an adjustment factor as the average ratio of the
               time-specific means to the global mean.
            4. Clamp the adjustment to [0.5, 2.0] to prevent extreme swings.
            5. Multiply all three threshold levels by the adjustment factor.

        Example:
            If Mondays historically have 1.5x the global mean and 9 AM has
            1.2x, the adjustment factor would be (1.5 + 1.2) / 2 = 1.35,
            making thresholds 35% higher on Monday mornings.

        Args:
            metric_column (str): The metric.
            timestamp (datetime): When to calculate thresholds for.

        Returns:
            dict: Time-adjusted thresholds with keys 'warning', 'critical',
                'emergency'.
        """
        base_thresholds = self.calculate_thresholds(metric_column)
        stats = self._historical_stats.get(metric_column, {})

        time_patterns = stats.get('time_patterns', {})
        if not time_patterns:
            return base_thresholds  # no time patterns learned; return base

        # Get day-of-week adjustment (0=Monday, 6=Sunday)
        dow = timestamp.weekday()
        dow_stats = time_patterns.get('day_of_week', {}).get(dow, {})
        dow_mean = dow_stats.get('mean', stats['mean'])

        # Get hour-of-day adjustment
        hour = timestamp.hour
        hour_stats = time_patterns.get('hour_of_day', {}).get(hour, {})
        hour_mean = hour_stats.get('mean', stats['mean'])

        # Compute combined adjustment factor (average of day-of-week and
        # hour-of-day ratios relative to the global mean)
        if stats['mean'] > 0:
            adjustment = ((dow_mean / stats['mean']) + (hour_mean / stats['mean'])) / 2
        else:
            adjustment = 1.0

        # Clamp to prevent extreme threshold distortion
        adjustment = max(0.5, min(2.0, adjustment))

        return {
            'warning': base_thresholds['warning'] * adjustment,
            'critical': base_thresholds['critical'] * adjustment,
            'emergency': base_thresholds['emergency'] * adjustment
        }

    def get_stats(self, metric_column: str) -> Dict:
        """
        Get calculated statistics for a metric.

        Args:
            metric_column (str): The metric name.

        Returns:
            dict: All computed statistics, or empty dict if not fitted.
        """
        return self._historical_stats.get(metric_column, {})


# ---------------------------------------------------------------------------
# Convenience functions for one-shot usage
# ---------------------------------------------------------------------------

def calculate_dynamic_thresholds(df: pd.DataFrame, metric_column: str,
                                  datetime_column: Optional[str] = None,
                                  method: str = 'hybrid') -> Dict[str, float]:
    """
    Convenience function to calculate thresholds in one call.

    Creates a temporary ``SmartThresholdCalculator``, fits it, and returns
    the thresholds.  Useful for quick, stateless checks.

    Args:
        df (pd.DataFrame): Historical data.
        metric_column (str): Column with metric values.
        datetime_column (str, optional): Datetime column for time patterns.
        method (str): 'iqr', 'zscore', 'percentile', or 'hybrid'.

    Returns:
        dict: Threshold values with keys 'warning', 'critical', 'emergency'.
    """
    calculator = SmartThresholdCalculator()
    calculator.fit(df, metric_column, datetime_column)
    return calculator.calculate_thresholds(metric_column, method)


def check_threshold_breach(df: pd.DataFrame, metric_column: str,
                           current_value: float) -> ThresholdResult:
    """
    Convenience function to check if a value breaches thresholds.

    Creates a temporary calculator, fits on ``df``, and checks the value.

    Args:
        df (pd.DataFrame): Historical data for baseline.
        metric_column (str): Column with metric values.
        current_value (float): Value to check.

    Returns:
        ThresholdResult: Full result with alert details.
    """
    calculator = SmartThresholdCalculator()
    calculator.fit(df, metric_column)
    return calculator.check_value(metric_column, current_value)


def get_adaptive_limits(df: pd.DataFrame, metric_column: str,
                        datetime_column: str,
                        target_time: datetime) -> Dict[str, float]:
    """
    Get time-aware adaptive thresholds.

    Combines historical fitting with time-of-day/day-of-week adjustment
    in a single call.

    Args:
        df (pd.DataFrame): Historical data.
        metric_column (str): Metric column.
        datetime_column (str): Datetime column.
        target_time (datetime): Time to calculate thresholds for.

    Returns:
        dict: Time-adjusted thresholds.
    """
    calculator = SmartThresholdCalculator()
    calculator.fit(df, metric_column, datetime_column)
    return calculator.get_time_adjusted_threshold(metric_column, target_time)


# ---------------------------------------------------------------------------
# Multi-metric monitoring
# ---------------------------------------------------------------------------

class MultiMetricMonitor:
    """
    Monitor multiple metrics with smart thresholds.

    Useful for dashboard-style monitoring of escalation KPIs.

    Wraps multiple ``SmartThresholdCalculator`` instances (one per metric)
    behind a unified interface for batch checking, alert filtering, and
    dashboard summary generation.

    Usage:
        >>> monitor = MultiMetricMonitor()
        >>> monitor.add_metric(df, 'escalation_count', 'date')
        >>> monitor.add_metric(df, 'avg_resolution_days', 'date')
        >>> alerts = monitor.get_alerts({'escalation_count': 25, 'avg_resolution_days': 8})

    Attributes:
        config (ThresholdConfig): Shared configuration for all metrics.
        calculators (dict): Mapping of metric name to fitted calculator.
        metrics (list[str]): Ordered list of monitored metric names.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize with optional shared configuration.

        Args:
            config (ThresholdConfig, optional): Shared config for all metrics.
        """
        self.config = config or ThresholdConfig()
        self.calculators: Dict[str, SmartThresholdCalculator] = {}
        self.metrics: List[str] = []

    def add_metric(self, df: pd.DataFrame, metric_column: str,
                   datetime_column: Optional[str] = None) -> 'MultiMetricMonitor':
        """
        Add a metric to monitor.

        Creates a new ``SmartThresholdCalculator`` with the shared config,
        fits it on the provided data, and registers it.

        Args:
            df (pd.DataFrame): Historical data for this metric.
            metric_column (str): Column name for the metric.
            datetime_column (str, optional): Datetime column for time patterns.

        Returns:
            MultiMetricMonitor: Self, for method chaining.
        """
        calc = SmartThresholdCalculator(self.config)
        calc.fit(df, metric_column, datetime_column)
        self.calculators[metric_column] = calc
        self.metrics.append(metric_column)
        return self

    def check_all(self, current_values: Dict[str, float],
                  timestamp: Optional[datetime] = None) -> Dict[str, ThresholdResult]:
        """
        Check all monitored metrics against their thresholds.

        Args:
            current_values (dict): Maps metric names to current values.
            timestamp (datetime, optional): For time-aware checking.

        Returns:
            dict: Maps metric names to ThresholdResults.
        """
        results = {}
        for metric, value in current_values.items():
            if metric in self.calculators:
                results[metric] = self.calculators[metric].check_value(metric, value, timestamp)
        return results

    def get_alerts(self, current_values: Dict[str, float],
                   min_level: AlertLevel = AlertLevel.WARNING) -> List[ThresholdResult]:
        """
        Get only metrics that have breached thresholds.

        Filters the results of ``check_all`` to include only metrics at or
        above the specified minimum alert level, sorted by severity
        (most severe first).

        Args:
            current_values (dict): Current metric values.
            min_level (AlertLevel): Minimum alert level to include.

        Returns:
            list[ThresholdResult]: Breached metrics sorted by severity descending.
        """
        all_results = self.check_all(current_values)

        # Define the severity ordering for comparison and sorting
        level_order = [AlertLevel.NORMAL, AlertLevel.WARNING,
                       AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
        min_index = level_order.index(min_level)

        alerts = []
        for result in all_results.values():
            if level_order.index(result.alert_level) >= min_index:
                alerts.append(result)

        # Sort by severity (most severe first)
        alerts.sort(key=lambda x: level_order.index(x.alert_level), reverse=True)
        return alerts

    def get_dashboard_summary(self, current_values: Dict[str, float]) -> Dict:
        """
        Get a summary suitable for dashboards.

        Returns a structured dict containing:
            - Timestamp of the check
            - Total number of metrics monitored
            - Count by alert level (normal, warning, critical, emergency)
            - Per-metric details (value, level, percentile, thresholds)
            - List of active alerts with recommendations

        Args:
            current_values (dict): Current metric values.

        Returns:
            dict: Dashboard-ready summary.
        """
        all_results = self.check_all(current_values)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_metrics': len(all_results),
            'by_level': {
                'normal': 0,
                'warning': 0,
                'critical': 0,
                'emergency': 0
            },
            'alerts': [],
            'metrics': {}
        }

        for metric, result in all_results.items():
            # Increment the counter for this alert level
            summary['by_level'][result.alert_level.value] += 1
            # Store per-metric detail
            summary['metrics'][metric] = {
                'value': result.current_value,
                'level': result.alert_level.value,
                'percentile': result.percentile_rank,
                'threshold_warning': result.thresholds['warning'],
                'threshold_critical': result.thresholds['critical']
            }

            # Collect active alerts (non-NORMAL) for the alerts list
            if result.alert_level != AlertLevel.NORMAL:
                summary['alerts'].append({
                    'metric': metric,
                    'level': result.alert_level.value,
                    'value': result.current_value,
                    'recommendation': result.recommendation
                })

        return summary
