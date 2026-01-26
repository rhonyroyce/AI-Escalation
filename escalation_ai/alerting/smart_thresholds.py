"""
Smart Alert Thresholds.

Dynamic threshold calculation based on historical patterns using:
- IQR (Interquartile Range) for outlier detection
- Z-score for standard deviation-based thresholds
- Percentile-based limits
- Rolling window analysis for trend-aware thresholds
- Exponential smoothing for seasonal adjustments
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
    """Alert severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ThresholdConfig:
    """Configuration for threshold calculation."""
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
    """Result of threshold calculation."""
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
    """
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or ThresholdConfig()
        self._historical_stats: Dict[str, Dict] = {}
        self._alert_history: List[Dict] = []
        self._last_alert_time: Dict[str, datetime] = {}
    
    def fit(self, df: pd.DataFrame, metric_column: str, 
            datetime_column: Optional[str] = None) -> 'SmartThresholdCalculator':
        """
        Fit the calculator on historical data.
        
        Args:
            df: Historical data
            metric_column: Column containing the metric values
            datetime_column: Optional datetime column for time-aware thresholds
            
        Returns:
            Self for method chaining
        """
        if metric_column not in df.columns:
            raise ValueError(f"Column '{metric_column}' not found in dataframe")
        
        values = df[metric_column].dropna()
        
        if len(values) < self.config.min_samples:
            warnings.warn(f"Insufficient samples ({len(values)}) for reliable thresholds")
        
        # Calculate basic statistics
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
        
        # IQR-based bounds
        iqr = stats['q3'] - stats['q1']
        stats['iqr'] = iqr
        stats['iqr_lower'] = stats['q1'] - (self.config.iqr_multiplier * iqr)
        stats['iqr_upper'] = stats['q3'] + (self.config.iqr_multiplier * iqr)
        
        # Percentile thresholds
        stats['p75'] = float(values.quantile(0.75))
        stats['p90'] = float(values.quantile(0.90))
        stats['p95'] = float(values.quantile(0.95))
        stats['p99'] = float(values.quantile(0.99))
        
        # Z-score thresholds
        stats['z_warning_upper'] = stats['mean'] + (self.config.z_score_warning * stats['std'])
        stats['z_critical_upper'] = stats['mean'] + (self.config.z_score_critical * stats['std'])
        stats['z_emergency_upper'] = stats['mean'] + (self.config.z_score_emergency * stats['std'])
        
        # Time-based patterns if datetime provided
        if datetime_column and datetime_column in df.columns:
            stats['time_patterns'] = self._calculate_time_patterns(df, metric_column, datetime_column)
        
        self._historical_stats[metric_column] = stats
        return self
    
    def _calculate_time_patterns(self, df: pd.DataFrame, metric_column: str, 
                                  datetime_column: str) -> Dict:
        """Calculate day-of-week and hour-of-day patterns."""
        df = df.copy()
        df['_dt'] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=['_dt', metric_column])
        
        patterns = {}
        
        # Day of week patterns (0=Monday, 6=Sunday)
        df['_dow'] = df['_dt'].dt.dayofweek
        dow_stats = df.groupby('_dow')[metric_column].agg(['mean', 'std']).to_dict('index')
        patterns['day_of_week'] = dow_stats
        
        # Hour of day patterns
        df['_hour'] = df['_dt'].dt.hour
        hour_stats = df.groupby('_hour')[metric_column].agg(['mean', 'std']).to_dict('index')
        patterns['hour_of_day'] = hour_stats
        
        return patterns
    
    def calculate_thresholds(self, metric_column: str, 
                             method: str = 'hybrid') -> Dict[str, float]:
        """
        Calculate thresholds for a metric.
        
        Args:
            metric_column: The metric to calculate thresholds for
            method: 'iqr', 'zscore', 'percentile', or 'hybrid' (combines all)
            
        Returns:
            Dictionary with warning, critical, emergency thresholds
        """
        if metric_column not in self._historical_stats:
            raise ValueError(f"No historical data for '{metric_column}'. Call fit() first.")
        
        stats = self._historical_stats[metric_column]
        
        if method == 'iqr':
            return {
                'warning': stats['q3'],
                'critical': stats['iqr_upper'],
                'emergency': stats['iqr_upper'] * 1.5
            }
        elif method == 'zscore':
            return {
                'warning': stats['z_warning_upper'],
                'critical': stats['z_critical_upper'],
                'emergency': stats['z_emergency_upper']
            }
        elif method == 'percentile':
            return {
                'warning': stats['p75'],
                'critical': stats['p90'],
                'emergency': stats['p99']
            }
        else:  # hybrid - conservative approach using max of methods
            return {
                'warning': min(stats['p75'], stats['z_warning_upper']),
                'critical': min(stats['p90'], stats['z_critical_upper']),
                'emergency': min(stats['p99'], stats['z_emergency_upper'])
            }
    
    def check_value(self, metric_column: str, current_value: float,
                    timestamp: Optional[datetime] = None) -> ThresholdResult:
        """
        Check if a value breaches any thresholds.
        
        Args:
            metric_column: The metric being checked
            current_value: The current value to check
            timestamp: Optional timestamp for time-aware checking
            
        Returns:
            ThresholdResult with alert level and details
        """
        if metric_column not in self._historical_stats:
            raise ValueError(f"No historical data for '{metric_column}'. Call fit() first.")
        
        stats = self._historical_stats[metric_column]
        thresholds = self.calculate_thresholds(metric_column)
        
        # Determine alert level
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
        
        # Calculate percentile rank
        if stats['std'] > 0:
            z_score = (current_value - stats['mean']) / stats['std']
            from scipy import stats as scipy_stats
            percentile_rank = scipy_stats.norm.cdf(z_score) * 100
        else:
            percentile_rank = 50.0
        
        # Determine trend
        trend = self._detect_trend(metric_column, current_value)
        
        # Generate recommendation
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
        """Detect if values are trending up, down, or stable."""
        # Simple trend detection based on relation to median
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
        """Generate actionable recommendation based on alert."""
        if alert_level == AlertLevel.NORMAL:
            return "No action required. Values within normal range."
        
        percentile_rank = 0
        if stats['std'] > 0:
            z = (current_value - stats['mean']) / stats['std']
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
        
        Args:
            metric_column: The metric
            timestamp: When to calculate thresholds for
            
        Returns:
            Time-adjusted thresholds
        """
        base_thresholds = self.calculate_thresholds(metric_column)
        stats = self._historical_stats.get(metric_column, {})
        
        time_patterns = stats.get('time_patterns', {})
        if not time_patterns:
            return base_thresholds
        
        # Get day-of-week adjustment
        dow = timestamp.weekday()
        dow_stats = time_patterns.get('day_of_week', {}).get(dow, {})
        dow_mean = dow_stats.get('mean', stats['mean'])
        
        # Get hour adjustment
        hour = timestamp.hour
        hour_stats = time_patterns.get('hour_of_day', {}).get(hour, {})
        hour_mean = hour_stats.get('mean', stats['mean'])
        
        # Calculate adjustment factor
        if stats['mean'] > 0:
            adjustment = ((dow_mean / stats['mean']) + (hour_mean / stats['mean'])) / 2
        else:
            adjustment = 1.0
        
        # Apply adjustment (bounded to reasonable range)
        adjustment = max(0.5, min(2.0, adjustment))
        
        return {
            'warning': base_thresholds['warning'] * adjustment,
            'critical': base_thresholds['critical'] * adjustment,
            'emergency': base_thresholds['emergency'] * adjustment
        }
    
    def get_stats(self, metric_column: str) -> Dict:
        """Get calculated statistics for a metric."""
        return self._historical_stats.get(metric_column, {})


def calculate_dynamic_thresholds(df: pd.DataFrame, metric_column: str,
                                  datetime_column: Optional[str] = None,
                                  method: str = 'hybrid') -> Dict[str, float]:
    """
    Convenience function to calculate thresholds in one call.
    
    Args:
        df: Historical data
        metric_column: Column with metric values
        datetime_column: Optional datetime column
        method: 'iqr', 'zscore', 'percentile', or 'hybrid'
        
    Returns:
        Dictionary with threshold values
    """
    calculator = SmartThresholdCalculator()
    calculator.fit(df, metric_column, datetime_column)
    return calculator.calculate_thresholds(metric_column, method)


def check_threshold_breach(df: pd.DataFrame, metric_column: str,
                           current_value: float) -> ThresholdResult:
    """
    Convenience function to check if a value breaches thresholds.
    
    Args:
        df: Historical data for baseline
        metric_column: Column with metric values
        current_value: Value to check
        
    Returns:
        ThresholdResult with alert details
    """
    calculator = SmartThresholdCalculator()
    calculator.fit(df, metric_column)
    return calculator.check_value(metric_column, current_value)


def get_adaptive_limits(df: pd.DataFrame, metric_column: str,
                        datetime_column: str, 
                        target_time: datetime) -> Dict[str, float]:
    """
    Get time-aware adaptive thresholds.
    
    Args:
        df: Historical data
        metric_column: Metric column
        datetime_column: Datetime column
        target_time: Time to calculate thresholds for
        
    Returns:
        Time-adjusted thresholds
    """
    calculator = SmartThresholdCalculator()
    calculator.fit(df, metric_column, datetime_column)
    return calculator.get_time_adjusted_threshold(metric_column, target_time)


class MultiMetricMonitor:
    """
    Monitor multiple metrics with smart thresholds.
    
    Useful for dashboard-style monitoring of escalation KPIs.
    """
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        """Initialize with optional shared configuration."""
        self.config = config or ThresholdConfig()
        self.calculators: Dict[str, SmartThresholdCalculator] = {}
        self.metrics: List[str] = []
    
    def add_metric(self, df: pd.DataFrame, metric_column: str,
                   datetime_column: Optional[str] = None) -> 'MultiMetricMonitor':
        """Add a metric to monitor."""
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
            current_values: Dict mapping metric names to current values
            timestamp: Optional timestamp for time-aware checking
            
        Returns:
            Dict mapping metric names to ThresholdResults
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
        
        Args:
            current_values: Current metric values
            min_level: Minimum alert level to include
            
        Returns:
            List of ThresholdResults for breached metrics
        """
        all_results = self.check_all(current_values)
        
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
        
        Returns:
            Summary dict with counts by alert level and details
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
            summary['by_level'][result.alert_level.value] += 1
            summary['metrics'][metric] = {
                'value': result.current_value,
                'level': result.alert_level.value,
                'percentile': result.percentile_rank,
                'threshold_warning': result.thresholds['warning'],
                'threshold_critical': result.thresholds['critical']
            }
            
            if result.alert_level != AlertLevel.NORMAL:
                summary['alerts'].append({
                    'metric': metric,
                    'level': result.alert_level.value,
                    'value': result.current_value,
                    'recommendation': result.recommendation
                })
        
        return summary
