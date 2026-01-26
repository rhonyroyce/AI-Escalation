"""
Category Drift Detection.

Detects shifts in escalation categories over time using:
- Chi-square tests for distribution changes
- Jensen-Shannon divergence for similarity
- Trend analysis with Mann-Kendall test
- Rolling window comparison
- Emerging/declining category identification
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import Counter
import warnings

try:
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import jensenshannon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class DriftType(Enum):
    """Types of category drift detected."""
    NONE = "none"
    EMERGING = "emerging"  # Category becoming more frequent
    DECLINING = "declining"  # Category becoming less frequent
    SHIFT = "shift"  # Overall distribution change
    VOLATILE = "volatile"  # High variance, unstable


@dataclass
class DriftResult:
    """Result of drift detection for a category."""
    category: str
    drift_type: DriftType
    severity: float  # 0-1, how significant the drift is
    current_rate: float  # Current proportion
    baseline_rate: float  # Historical proportion
    change_pct: float  # Percentage change
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    p_value: Optional[float] = None  # Statistical significance
    confidence: float = 0.0  # Confidence in the detection
    recommendation: str = ""


@dataclass
class DistributionComparison:
    """Comparison of two distributions."""
    period1_name: str
    period2_name: str
    chi_square_stat: float
    chi_square_pvalue: float
    js_divergence: float  # Jensen-Shannon divergence (0=identical, 1=completely different)
    significant_drift: bool
    drifted_categories: List[str]
    summary: str


class CategoryDriftDetector:
    """
    Detects shifts in escalation category distributions over time.
    
    Features:
    - Statistical tests for distribution changes
    - Trend detection for individual categories
    - Time-windowed comparisons
    - Emerging/declining pattern identification
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 min_samples_per_period: int = 30,
                 drift_threshold: float = 0.2):
        """
        Initialize the detector.
        
        Args:
            significance_level: P-value threshold for statistical tests
            min_samples_per_period: Minimum samples needed for valid comparison
            drift_threshold: Minimum change ratio to flag as drift
        """
        self.significance_level = significance_level
        self.min_samples = min_samples_per_period
        self.drift_threshold = drift_threshold
        self._baseline_distribution: Optional[Dict[str, float]] = None
        self._baseline_counts: Optional[Dict[str, int]] = None
        self._history: List[Dict] = []
    
    def set_baseline(self, df: pd.DataFrame, category_column: str) -> 'CategoryDriftDetector':
        """
        Set the baseline distribution to compare against.
        
        Args:
            df: Baseline data
            category_column: Column containing categories
            
        Returns:
            Self for method chaining
        """
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found")
        
        counts = df[category_column].value_counts()
        total = counts.sum()
        
        self._baseline_counts = counts.to_dict()
        self._baseline_distribution = {cat: count / total for cat, count in counts.items()}
        
        return self
    
    def detect_drift(self, df: pd.DataFrame, category_column: str,
                     datetime_column: Optional[str] = None) -> List[DriftResult]:
        """
        Detect drift in categories compared to baseline.
        
        Args:
            df: Current data to analyze
            category_column: Column with categories
            datetime_column: Optional datetime for trend analysis
            
        Returns:
            List of DriftResults for each category
        """
        if self._baseline_distribution is None:
            raise ValueError("No baseline set. Call set_baseline() first.")
        
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found")
        
        # Calculate current distribution
        current_counts = df[category_column].value_counts()
        current_total = current_counts.sum()
        current_dist = {cat: count / current_total for cat, count in current_counts.items()}
        
        # Get all categories from both distributions
        all_categories = set(self._baseline_distribution.keys()) | set(current_dist.keys())
        
        results = []
        for category in all_categories:
            baseline_rate = self._baseline_distribution.get(category, 0.0)
            current_rate = current_dist.get(category, 0.0)
            
            # Calculate change
            if baseline_rate > 0:
                change_pct = ((current_rate - baseline_rate) / baseline_rate) * 100
            elif current_rate > 0:
                change_pct = 100.0  # New category
            else:
                change_pct = 0.0
            
            # Determine drift type and severity
            drift_type, severity = self._classify_drift(baseline_rate, current_rate, change_pct)
            
            # Statistical test for this category
            p_value = self._test_proportion_change(
                self._baseline_counts.get(category, 0),
                sum(self._baseline_counts.values()),
                current_counts.get(category, 0),
                current_total
            )
            
            # Determine trend if datetime available
            trend = "stable"
            if datetime_column and datetime_column in df.columns:
                trend = self._detect_category_trend(df, category_column, datetime_column, category)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(category, drift_type, change_pct, severity)
            
            # Calculate confidence
            confidence = self._calculate_confidence(p_value, current_total, severity)
            
            results.append(DriftResult(
                category=category,
                drift_type=drift_type,
                severity=severity,
                current_rate=current_rate,
                baseline_rate=baseline_rate,
                change_pct=change_pct,
                trend_direction=trend,
                p_value=p_value,
                confidence=confidence,
                recommendation=recommendation
            ))
        
        # Sort by severity
        results.sort(key=lambda x: x.severity, reverse=True)
        return results
    
    def _classify_drift(self, baseline_rate: float, current_rate: float,
                        change_pct: float) -> Tuple[DriftType, float]:
        """Classify the type and severity of drift."""
        abs_change = abs(change_pct)
        
        # Calculate severity (0-1 scale)
        severity = min(1.0, abs_change / 100.0)
        
        if abs_change < self.drift_threshold * 100:
            return DriftType.NONE, severity
        
        if current_rate > baseline_rate:
            return DriftType.EMERGING, severity
        else:
            return DriftType.DECLINING, severity
    
    def _test_proportion_change(self, count1: int, n1: int, 
                                 count2: int, n2: int) -> Optional[float]:
        """Test if proportions are significantly different."""
        if not SCIPY_AVAILABLE:
            return None
        
        if n1 == 0 or n2 == 0:
            return 1.0
        
        # Two-proportion z-test
        p1 = count1 / n1
        p2 = count2 / n2
        p_pooled = (count1 + count2) / (n1 + n2)
        
        if p_pooled == 0 or p_pooled == 1:
            return 1.0
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        if se == 0:
            return 1.0
        
        z = (p1 - p2) / se
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        
        return p_value
    
    def _detect_category_trend(self, df: pd.DataFrame, category_column: str,
                                datetime_column: str, category: str) -> str:
        """Detect trend direction for a specific category."""
        df = df.copy()
        df['_dt'] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=['_dt'])
        
        if len(df) < self.min_samples:
            return "insufficient_data"
        
        # Group by week and count category occurrences
        df['_week'] = df['_dt'].dt.isocalendar().week
        df['_year'] = df['_dt'].dt.year
        df['_period'] = df['_year'].astype(str) + '_' + df['_week'].astype(str)
        
        df['_is_category'] = (df[category_column] == category).astype(int)
        weekly = df.groupby('_period')['_is_category'].sum().values
        
        if len(weekly) < 3:
            return "insufficient_data"
        
        # Simple trend detection using linear regression slope
        x = np.arange(len(weekly))
        slope, _, r_value, _, _ = scipy_stats.linregress(x, weekly)
        
        if abs(r_value) < 0.3:  # Weak correlation
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _generate_recommendation(self, category: str, drift_type: DriftType,
                                  change_pct: float, severity: float) -> str:
        """Generate actionable recommendation."""
        if drift_type == DriftType.NONE:
            return f"'{category}' is stable. No action required."
        
        if drift_type == DriftType.EMERGING:
            if severity > 0.5:
                return (f"ALERT: '{category}' has increased by {change_pct:.0f}%. "
                        "Investigate root cause and allocate additional resources.")
            else:
                return (f"'{category}' is trending upward (+{change_pct:.0f}%). "
                        "Monitor for continued growth.")
        
        if drift_type == DriftType.DECLINING:
            if severity > 0.5:
                return (f"'{category}' has decreased by {abs(change_pct):.0f}%. "
                        "Verify if this reflects improved processes or data issues.")
            else:
                return (f"'{category}' is trending downward ({change_pct:.0f}%). "
                        "Positive trend if intentional, otherwise investigate.")
        
        return f"'{category}' shows unusual patterns. Review data quality."
    
    def _calculate_confidence(self, p_value: Optional[float], 
                               sample_size: int, severity: float) -> float:
        """Calculate confidence in the drift detection."""
        confidence = 0.5  # Base confidence
        
        # Adjust for statistical significance
        if p_value is not None:
            if p_value < 0.01:
                confidence += 0.3
            elif p_value < 0.05:
                confidence += 0.2
            elif p_value < 0.1:
                confidence += 0.1
        
        # Adjust for sample size
        if sample_size >= 100:
            confidence += 0.1
        elif sample_size >= 50:
            confidence += 0.05
        
        # Adjust for severity
        confidence += severity * 0.1
        
        return min(1.0, confidence)
    
    def compare_distributions(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               category_column: str,
                               period1_name: str = "Period 1",
                               period2_name: str = "Period 2") -> DistributionComparison:
        """
        Compare category distributions between two datasets.
        
        Args:
            df1: First dataset
            df2: Second dataset
            category_column: Column with categories
            period1_name: Label for first period
            period2_name: Label for second period
            
        Returns:
            DistributionComparison with statistical results
        """
        # Get counts
        counts1 = df1[category_column].value_counts()
        counts2 = df2[category_column].value_counts()
        
        # Align categories
        all_cats = sorted(set(counts1.index) | set(counts2.index))
        freq1 = np.array([counts1.get(c, 0) for c in all_cats])
        freq2 = np.array([counts2.get(c, 0) for c in all_cats])
        
        # Chi-square test
        if SCIPY_AVAILABLE and freq1.sum() > 0 and freq2.sum() > 0:
            # Normalize to expected frequencies
            total1, total2 = freq1.sum(), freq2.sum()
            expected = (freq1 + freq2) / 2 * (total1 + total2) / 2
            expected = np.maximum(expected, 1)  # Avoid zeros
            
            chi2, p_value = scipy_stats.chisquare(freq1, f_exp=expected * total1 / expected.sum())
            
            # Jensen-Shannon divergence
            p1 = freq1 / freq1.sum() if freq1.sum() > 0 else freq1
            p2 = freq2 / freq2.sum() if freq2.sum() > 0 else freq2
            js_div = jensenshannon(p1, p2)
        else:
            chi2, p_value, js_div = 0.0, 1.0, 0.0
        
        # Find drifted categories
        drifted = []
        dist1 = {c: counts1.get(c, 0) / counts1.sum() for c in all_cats}
        dist2 = {c: counts2.get(c, 0) / counts2.sum() for c in all_cats}
        
        for cat in all_cats:
            r1, r2 = dist1.get(cat, 0), dist2.get(cat, 0)
            if r1 > 0:
                change = abs((r2 - r1) / r1)
                if change > self.drift_threshold:
                    drifted.append(cat)
            elif r2 > 0:
                drifted.append(cat)  # New category
        
        significant = p_value < self.significance_level
        
        summary = self._generate_comparison_summary(
            period1_name, period2_name, significant, js_div, drifted
        )
        
        return DistributionComparison(
            period1_name=period1_name,
            period2_name=period2_name,
            chi_square_stat=chi2,
            chi_square_pvalue=p_value,
            js_divergence=js_div,
            significant_drift=significant,
            drifted_categories=drifted,
            summary=summary
        )
    
    def _generate_comparison_summary(self, p1: str, p2: str, significant: bool,
                                      js_div: float, drifted: List[str]) -> str:
        """Generate human-readable comparison summary."""
        if not significant:
            return f"No statistically significant difference between {p1} and {p2}."
        
        if js_div > 0.5:
            severity = "dramatically"
        elif js_div > 0.3:
            severity = "significantly"
        else:
            severity = "moderately"
        
        drift_str = ", ".join(drifted[:5])
        if len(drifted) > 5:
            drift_str += f" and {len(drifted) - 5} more"
        
        return (f"Category distribution has {severity} shifted between {p1} and {p2}. "
                f"Categories with notable changes: {drift_str}.")
    
    def get_rolling_drift(self, df: pd.DataFrame, category_column: str,
                          datetime_column: str, window_days: int = 30,
                          step_days: int = 7) -> pd.DataFrame:
        """
        Calculate rolling drift over time.
        
        Args:
            df: Data with datetime
            category_column: Category column
            datetime_column: Datetime column
            window_days: Size of rolling window
            step_days: Step size between windows
            
        Returns:
            DataFrame with drift metrics over time
        """
        df = df.copy()
        df['_dt'] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=['_dt']).sort_values('_dt')
        
        min_date = df['_dt'].min()
        max_date = df['_dt'].max()
        
        results = []
        current_start = min_date + timedelta(days=window_days)
        
        # Set first window as baseline
        baseline_df = df[df['_dt'] < min_date + timedelta(days=window_days)]
        if len(baseline_df) < self.min_samples:
            warnings.warn("Insufficient data for rolling drift analysis")
            return pd.DataFrame()
        
        self.set_baseline(baseline_df, category_column)
        
        while current_start <= max_date:
            window_end = current_start
            window_start = current_start - timedelta(days=window_days)
            
            window_df = df[(df['_dt'] >= window_start) & (df['_dt'] < window_end)]
            
            if len(window_df) >= self.min_samples:
                drift_results = self.detect_drift(window_df, category_column)
                
                # Aggregate drift metrics
                max_severity = max([r.severity for r in drift_results], default=0)
                emerging_count = sum(1 for r in drift_results if r.drift_type == DriftType.EMERGING)
                declining_count = sum(1 for r in drift_results if r.drift_type == DriftType.DECLINING)
                
                results.append({
                    'period_end': window_end,
                    'sample_count': len(window_df),
                    'max_severity': max_severity,
                    'emerging_categories': emerging_count,
                    'declining_categories': declining_count,
                    'total_drift_categories': emerging_count + declining_count
                })
            
            current_start += timedelta(days=step_days)
        
        return pd.DataFrame(results)


def detect_category_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                          category_column: str) -> List[DriftResult]:
    """
    Convenience function to detect drift between two datasets.
    
    Args:
        baseline_df: Historical/baseline data
        current_df: Current data to compare
        category_column: Column with categories
        
    Returns:
        List of DriftResults
    """
    detector = CategoryDriftDetector()
    detector.set_baseline(baseline_df, category_column)
    return detector.detect_drift(current_df, category_column)


def compare_periods(df: pd.DataFrame, category_column: str,
                    datetime_column: str, split_date: datetime) -> DistributionComparison:
    """
    Compare category distributions before and after a date.
    
    Args:
        df: Full dataset
        category_column: Category column
        datetime_column: Datetime column
        split_date: Date to split on
        
    Returns:
        DistributionComparison
    """
    df = df.copy()
    df['_dt'] = pd.to_datetime(df[datetime_column], errors='coerce')
    
    before = df[df['_dt'] < split_date]
    after = df[df['_dt'] >= split_date]
    
    detector = CategoryDriftDetector()
    return detector.compare_distributions(
        before, after, category_column,
        f"Before {split_date.strftime('%Y-%m-%d')}",
        f"After {split_date.strftime('%Y-%m-%d')}"
    )


def get_emerging_categories(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                            category_column: str, 
                            min_severity: float = 0.2) -> List[DriftResult]:
    """
    Get categories that are increasing in frequency.
    
    Args:
        baseline_df: Baseline data
        current_df: Current data
        category_column: Category column
        min_severity: Minimum severity to include
        
    Returns:
        List of emerging categories sorted by severity
    """
    results = detect_category_drift(baseline_df, current_df, category_column)
    emerging = [r for r in results 
                if r.drift_type == DriftType.EMERGING and r.severity >= min_severity]
    return sorted(emerging, key=lambda x: x.severity, reverse=True)


def get_declining_categories(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                             category_column: str,
                             min_severity: float = 0.2) -> List[DriftResult]:
    """
    Get categories that are decreasing in frequency.
    
    Args:
        baseline_df: Baseline data
        current_df: Current data
        category_column: Category column
        min_severity: Minimum severity to include
        
    Returns:
        List of declining categories sorted by severity
    """
    results = detect_category_drift(baseline_df, current_df, category_column)
    declining = [r for r in results 
                 if r.drift_type == DriftType.DECLINING and r.severity >= min_severity]
    return sorted(declining, key=lambda x: x.severity, reverse=True)
