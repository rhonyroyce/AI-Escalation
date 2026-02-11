"""
Category Drift Detection.

Detects shifts in escalation categories over time using:
- Chi-square tests for distribution changes
- Jensen-Shannon divergence for similarity
- Trend analysis with Mann-Kendall test
- Rolling window comparison
- Emerging/declining category identification

Architecture Overview:
    This module provides statistical tools for detecting when the distribution
    of escalation ticket categories has changed compared to a historical
    baseline.  Such "drift" can indicate:
    - New failure modes emerging in the network (emerging categories)
    - Successful remediation efforts (declining categories)
    - Seasonal or cyclical variations in issue types
    - Data quality issues or classification model degradation

Statistical Methods:

    1. **Two-Proportion Z-Test** (per-category):
       For each category, tests whether its proportion in the current period
       is significantly different from the baseline period.  Uses a pooled
       proportion estimator and the standard normal approximation.
       Formula:
           z = (p1 - p2) / sqrt( p_pooled * (1 - p_pooled) * (1/n1 + 1/n2) )
       where p_pooled = (count1 + count2) / (n1 + n2).

    2. **Chi-Square Goodness-of-Fit** (whole distribution):
       Tests whether the entire category frequency distribution in period 2
       is consistent with the expected frequencies derived from both periods.
       Implemented via ``scipy.stats.chisquare``.

    3. **Jensen-Shannon Divergence** (whole distribution):
       A symmetric, bounded (0-1) measure of divergence between two
       probability distributions.  Computed via ``scipy.spatial.distance.jensenshannon``.
       - 0 = identical distributions
       - 1 = completely non-overlapping distributions
       Useful for quantifying *how much* the distribution has shifted.

    4. **Linear Regression Trend** (per-category, over time):
       Weekly counts of each category are regressed against time using
       ``scipy.stats.linregress``.  The slope and correlation coefficient (r)
       determine whether the category is trending up, down, or stable.

Drift Classification:
    Each category is classified as one of:
    - NONE: Change below the drift threshold (default 20%)
    - EMERGING: Category frequency is increasing
    - DECLINING: Category frequency is decreasing
    - SHIFT: Overall distribution change (used in whole-distribution tests)
    - VOLATILE: High variance, unstable (reserved for future use)

Convenience Functions:
    - ``detect_category_drift(baseline_df, current_df, category_column)``
    - ``compare_periods(df, category_column, datetime_column, split_date)``
    - ``get_emerging_categories(...)``
    - ``get_declining_categories(...)``
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
    """
    Types of category drift detected.

    Values:
        NONE: No significant change.
        EMERGING: Category is becoming more frequent.
        DECLINING: Category is becoming less frequent.
        SHIFT: Overall distribution change (multiple categories shifting).
        VOLATILE: High variance, unstable pattern (reserved for future use).
    """
    NONE = "none"
    EMERGING = "emerging"  # Category becoming more frequent
    DECLINING = "declining"  # Category becoming less frequent
    SHIFT = "shift"  # Overall distribution change
    VOLATILE = "volatile"  # High variance, unstable


@dataclass
class DriftResult:
    """
    Result of drift detection for a single category.

    Attributes:
        category (str): The category name.
        drift_type (DriftType): Classification of the detected drift.
        severity (float): Magnitude of drift on a 0-1 scale, computed as
            min(1.0, |change_pct| / 100).
        current_rate (float): Proportion of this category in the current data.
        baseline_rate (float): Proportion in the baseline data.
        change_pct (float): Percentage change from baseline to current.
            Positive = increasing, negative = decreasing.
        trend_direction (str): 'increasing', 'decreasing', 'stable', or
            'insufficient_data'.
        p_value (float, optional): P-value from the two-proportion z-test.
            None if scipy is unavailable.
        confidence (float): Confidence in the detection (0-1), derived from
            statistical significance, sample size, and severity.
        recommendation (str): Human-readable action recommendation.
    """
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
    """
    Comparison of two category distributions.

    Produced by ``CategoryDriftDetector.compare_distributions``.

    Attributes:
        period1_name (str): Label for the first period.
        period2_name (str): Label for the second period.
        chi_square_stat (float): Chi-square test statistic.
        chi_square_pvalue (float): P-value from the chi-square test.
        js_divergence (float): Jensen-Shannon divergence between the two
            distributions (0 = identical, 1 = completely different).
        significant_drift (bool): True if p-value < significance level.
        drifted_categories (list[str]): Categories with notable changes.
        summary (str): Human-readable comparison summary.
    """
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

    Usage:
        >>> detector = CategoryDriftDetector()
        >>> detector.set_baseline(historical_df, 'AI_Category')
        >>> results = detector.detect_drift(current_df, 'AI_Category')
        >>> for r in results:
        ...     print(f"{r.category}: {r.drift_type.value} ({r.change_pct:+.0f}%)")

    Attributes:
        significance_level (float): P-value threshold for statistical tests.
        min_samples (int): Minimum samples per period for valid analysis.
        drift_threshold (float): Minimum relative change ratio to flag as drift.
        _baseline_distribution (dict): Proportions from the baseline period.
        _baseline_counts (dict): Raw counts from the baseline period.
        _history (list): Record of past drift analyses (for audit trail).
    """

    def __init__(self,
                 significance_level: float = 0.05,
                 min_samples_per_period: int = 30,
                 drift_threshold: float = 0.2):
        """
        Initialize the detector.

        Args:
            significance_level (float): P-value threshold for statistical tests.
                Default 0.05 (95% confidence).
            min_samples_per_period (int): Minimum samples needed for a valid
                comparison.  Periods with fewer samples are skipped.
            drift_threshold (float): Minimum relative change ratio to flag as
                drift.  Default 0.2 means a category must change by at least
                20% relative to its baseline rate.
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

        Computes category counts and proportions from the provided DataFrame,
        which represent the "expected" or "normal" state.  Subsequent calls
        to ``detect_drift`` will compare new data against this baseline.

        Args:
            df (pd.DataFrame): Baseline data.
            category_column (str): Column containing categories.

        Returns:
            CategoryDriftDetector: Self, for method chaining.

        Raises:
            ValueError: If ``category_column`` is not found in ``df``.
        """
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found")

        counts = df[category_column].value_counts()
        total = counts.sum()

        self._baseline_counts = counts.to_dict()
        # Convert counts to proportions (each sums to 1.0)
        self._baseline_distribution = {cat: count / total for cat, count in counts.items()}

        return self

    def detect_drift(self, df: pd.DataFrame, category_column: str,
                     datetime_column: Optional[str] = None) -> List[DriftResult]:
        """
        Detect drift in categories compared to baseline.

        For each category (union of baseline and current), computes:
        1. The percentage change in proportion.
        2. A drift classification (NONE, EMERGING, DECLINING).
        3. A two-proportion z-test p-value.
        4. A trend direction (if datetime column is provided).
        5. A confidence score combining significance, sample size, and severity.
        6. A human-readable recommendation.

        Results are sorted by severity (most severe first).

        Args:
            df (pd.DataFrame): Current data to analyze.
            category_column (str): Column with categories.
            datetime_column (str, optional): Datetime column for trend analysis.

        Returns:
            list[DriftResult]: One result per category, sorted by severity.

        Raises:
            ValueError: If no baseline has been set or column not found.
        """
        if self._baseline_distribution is None:
            raise ValueError("No baseline set. Call set_baseline() first.")

        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found")

        # Calculate current distribution (proportions)
        current_counts = df[category_column].value_counts()
        current_total = current_counts.sum()
        current_dist = {cat: count / current_total for cat, count in current_counts.items()}

        # Get all categories from both distributions (union of keys)
        all_categories = set(self._baseline_distribution.keys()) | set(current_dist.keys())

        results = []
        for category in all_categories:
            baseline_rate = self._baseline_distribution.get(category, 0.0)
            current_rate = current_dist.get(category, 0.0)

            # Calculate percentage change relative to baseline
            if baseline_rate > 0:
                change_pct = ((current_rate - baseline_rate) / baseline_rate) * 100
            elif current_rate > 0:
                change_pct = 100.0  # Entirely new category
            else:
                change_pct = 0.0

            # Determine drift type and severity from the change percentage
            drift_type, severity = self._classify_drift(baseline_rate, current_rate, change_pct)

            # Run two-proportion z-test for this specific category
            p_value = self._test_proportion_change(
                self._baseline_counts.get(category, 0),
                sum(self._baseline_counts.values()),
                current_counts.get(category, 0),
                current_total
            )

            # Determine temporal trend if datetime column is available
            trend = "stable"
            if datetime_column and datetime_column in df.columns:
                trend = self._detect_category_trend(df, category_column, datetime_column, category)

            # Generate human-readable recommendation
            recommendation = self._generate_recommendation(category, drift_type, change_pct, severity)

            # Calculate composite confidence score
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

        # Sort by severity (most severe first)
        results.sort(key=lambda x: x.severity, reverse=True)
        return results

    def _classify_drift(self, baseline_rate: float, current_rate: float,
                        change_pct: float) -> Tuple[DriftType, float]:
        """
        Classify the type and severity of drift.

        Severity is computed as min(1.0, |change_pct| / 100), mapping a 100%
        change to severity 1.0.

        Classification Rules:
            - |change_pct| < drift_threshold * 100 => NONE
            - current_rate > baseline_rate         => EMERGING
            - current_rate < baseline_rate         => DECLINING

        Args:
            baseline_rate (float): Baseline proportion.
            current_rate (float): Current proportion.
            change_pct (float): Percentage change.

        Returns:
            tuple: (DriftType, severity_float)
        """
        abs_change = abs(change_pct)

        # Calculate severity (0-1 scale): a 100% change maps to 1.0
        severity = min(1.0, abs_change / 100.0)

        # Apply drift threshold (default 20% relative change)
        if abs_change < self.drift_threshold * 100:
            return DriftType.NONE, severity

        if current_rate > baseline_rate:
            return DriftType.EMERGING, severity
        else:
            return DriftType.DECLINING, severity

    def _test_proportion_change(self, count1: int, n1: int,
                                 count2: int, n2: int) -> Optional[float]:
        """
        Test if proportions are significantly different.

        Implements a two-proportion z-test (two-tailed):
            H0: p1 = p2  (proportions are equal)
            H1: p1 != p2 (proportions differ)

        The pooled proportion is used as the common estimate under H0:
            p_pooled = (count1 + count2) / (n1 + n2)

        The standard error of the difference is:
            SE = sqrt( p_pooled * (1 - p_pooled) * (1/n1 + 1/n2) )

        The test statistic is:
            z = (p1 - p2) / SE

        The two-tailed p-value is:
            p = 2 * (1 - Phi(|z|))

        Args:
            count1 (int): Count of the category in period 1.
            n1 (int): Total count in period 1.
            count2 (int): Count of the category in period 2.
            n2 (int): Total count in period 2.

        Returns:
            float or None: Two-tailed p-value, or None if scipy is unavailable,
                or 1.0 if the test is degenerate (zero denominator).
        """
        if not SCIPY_AVAILABLE:
            return None

        if n1 == 0 or n2 == 0:
            return 1.0  # Cannot compute with zero-size samples

        # Compute sample proportions
        p1 = count1 / n1
        p2 = count2 / n2
        # Pooled proportion under H0
        p_pooled = (count1 + count2) / (n1 + n2)

        # Degenerate case: all or none belong to this category
        if p_pooled == 0 or p_pooled == 1:
            return 1.0

        # Standard error of the difference in proportions
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        if se == 0:
            return 1.0

        # Test statistic
        z = (p1 - p2) / se
        # Two-tailed p-value via the standard normal CDF
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

        return p_value

    def _detect_category_trend(self, df: pd.DataFrame, category_column: str,
                                datetime_column: str, category: str) -> str:
        """
        Detect trend direction for a specific category.

        Approach:
            1. Parse the datetime column and group by ISO week.
            2. For each week, count how many tickets belong to ``category``.
            3. Fit a simple linear regression (OLS) of weekly counts vs. time index.
            4. Use the slope and correlation coefficient to determine trend:
               - |r| < 0.3: 'stable' (weak correlation)
               - slope > 0: 'increasing'
               - slope < 0: 'decreasing'

        Requires at least 3 weeks of data and ``min_samples`` total rows.

        Args:
            df (pd.DataFrame): Data with datetime and category columns.
            category_column (str): Category column name.
            datetime_column (str): Datetime column name.
            category (str): The specific category to analyse.

        Returns:
            str: 'increasing', 'decreasing', 'stable', or 'insufficient_data'.
        """
        df = df.copy()
        df['_dt'] = pd.to_datetime(df[datetime_column], errors='coerce')
        df = df.dropna(subset=['_dt'])

        if len(df) < self.min_samples:
            return "insufficient_data"

        # Group by ISO year-week and count occurrences of the category
        df['_week'] = df['_dt'].dt.isocalendar().week
        df['_year'] = df['_dt'].dt.year
        df['_period'] = df['_year'].astype(str) + '_' + df['_week'].astype(str)

        # Binary indicator: 1 if this row belongs to the target category, else 0
        df['_is_category'] = (df[category_column] == category).astype(int)
        weekly = df.groupby('_period')['_is_category'].sum().values

        if len(weekly) < 3:
            return "insufficient_data"

        # Simple linear regression: slope, intercept, r_value, p_value, std_err
        x = np.arange(len(weekly))
        slope, _, r_value, _, _ = scipy_stats.linregress(x, weekly)

        # Classify based on correlation strength and slope direction
        if abs(r_value) < 0.3:  # Weak correlation -> no clear trend
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _generate_recommendation(self, category: str, drift_type: DriftType,
                                  change_pct: float, severity: float) -> str:
        """
        Generate actionable recommendation.

        Produces context-dependent guidance based on drift type and severity:
        - NONE: "Stable, no action required."
        - EMERGING + high severity: "Investigate root cause and allocate resources."
        - EMERGING + low severity: "Monitor for continued growth."
        - DECLINING + high severity: "Verify if process improvement or data issue."
        - DECLINING + low severity: "Positive trend if intentional."

        Args:
            category (str): Category name.
            drift_type (DriftType): The classified drift type.
            change_pct (float): Percentage change.
            severity (float): Severity score (0-1).

        Returns:
            str: Recommendation text.
        """
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
        """
        Calculate confidence in the drift detection.

        Combines three signals into a composite confidence score:
        1. **Statistical significance** (p-value):
           - p < 0.01: +0.30
           - p < 0.05: +0.20
           - p < 0.10: +0.10
        2. **Sample size**:
           - >= 100: +0.10
           - >= 50:  +0.05
        3. **Severity**: +severity * 0.10

        Base confidence is 0.50.  Result is clamped to [0, 1].

        Args:
            p_value (float or None): P-value from the proportion test.
            sample_size (int): Number of samples in the current period.
            severity (float): Drift severity (0-1).

        Returns:
            float: Confidence score in [0, 1].
        """
        confidence = 0.5  # Base confidence

        # Adjust for statistical significance
        if p_value is not None:
            if p_value < 0.01:
                confidence += 0.3   # highly significant
            elif p_value < 0.05:
                confidence += 0.2   # significant at 5%
            elif p_value < 0.1:
                confidence += 0.1   # marginally significant

        # Adjust for sample size (larger samples -> more reliable)
        if sample_size >= 100:
            confidence += 0.1
        elif sample_size >= 50:
            confidence += 0.05

        # Adjust for severity (larger changes are easier to detect reliably)
        confidence += severity * 0.1

        return min(1.0, confidence)

    def compare_distributions(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               category_column: str,
                               period1_name: str = "Period 1",
                               period2_name: str = "Period 2") -> DistributionComparison:
        """
        Compare category distributions between two datasets.

        Performs two whole-distribution statistical tests:
        1. **Chi-square goodness-of-fit**: Tests whether df2's category
           frequencies are consistent with expected frequencies derived
           from the combined data of both periods.
        2. **Jensen-Shannon divergence**: Measures how different the two
           normalised distributions are on a 0-1 scale.

        Also identifies individual categories that have shifted beyond the
        drift threshold.

        Args:
            df1 (pd.DataFrame): First dataset.
            df2 (pd.DataFrame): Second dataset.
            category_column (str): Column with categories.
            period1_name (str): Label for first period.
            period2_name (str): Label for second period.

        Returns:
            DistributionComparison: Statistical comparison results.
        """
        # Get category counts from both datasets
        counts1 = df1[category_column].value_counts()
        counts2 = df2[category_column].value_counts()

        # Align categories: ensure both frequency arrays have the same categories
        all_cats = sorted(set(counts1.index) | set(counts2.index))
        freq1 = np.array([counts1.get(c, 0) for c in all_cats])
        freq2 = np.array([counts2.get(c, 0) for c in all_cats])

        # Chi-square test and Jensen-Shannon divergence
        if SCIPY_AVAILABLE and freq1.sum() > 0 and freq2.sum() > 0:
            # Compute expected frequencies from the combined distribution
            total1, total2 = freq1.sum(), freq2.sum()
            expected = (freq1 + freq2) / 2 * (total1 + total2) / 2
            expected = np.maximum(expected, 1)  # Avoid zeros in expected

            # Chi-square goodness-of-fit test
            chi2, p_value = scipy_stats.chisquare(freq1, f_exp=expected * total1 / expected.sum())

            # Jensen-Shannon divergence between the two normalised distributions
            # JSD is the square root of the Jensen-Shannon information divergence
            p1 = freq1 / freq1.sum() if freq1.sum() > 0 else freq1
            p2 = freq2 / freq2.sum() if freq2.sum() > 0 else freq2
            js_div = jensenshannon(p1, p2)
        else:
            chi2, p_value, js_div = 0.0, 1.0, 0.0

        # Identify individual categories that drifted beyond the threshold
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
                drifted.append(cat)  # New category in period 2

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
        """
        Generate human-readable comparison summary.

        Describes the magnitude of distribution shift using the Jensen-Shannon
        divergence:
            - > 0.5: "dramatically"
            - > 0.3: "significantly"
            - <= 0.3: "moderately"

        Args:
            p1 (str): Name of period 1.
            p2 (str): Name of period 2.
            significant (bool): Whether the chi-square test was significant.
            js_div (float): Jensen-Shannon divergence value.
            drifted (list[str]): Categories that shifted beyond the threshold.

        Returns:
            str: Summary text.
        """
        if not significant:
            return f"No statistically significant difference between {p1} and {p2}."

        # Describe magnitude using Jensen-Shannon divergence
        if js_div > 0.5:
            severity = "dramatically"
        elif js_div > 0.3:
            severity = "significantly"
        else:
            severity = "moderately"

        # List up to 5 drifted categories, with overflow count
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

        Slides a window across the time axis and computes drift metrics for
        each window position relative to the initial baseline window.

        Procedure:
            1. Set the first ``window_days`` of data as the baseline.
            2. Slide the window forward by ``step_days`` at each step.
            3. For each position, compute drift results and aggregate:
               - Maximum severity across all categories.
               - Count of emerging and declining categories.

        Args:
            df (pd.DataFrame): Data with datetime.
            category_column (str): Category column.
            datetime_column (str): Datetime column.
            window_days (int): Size of rolling window in days.
            step_days (int): Step size between windows in days.

        Returns:
            pd.DataFrame: One row per window position with columns:
                period_end, sample_count, max_severity,
                emerging_categories, declining_categories,
                total_drift_categories.
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

        # Slide the window forward
        while current_start <= max_date:
            window_end = current_start
            window_start = current_start - timedelta(days=window_days)

            window_df = df[(df['_dt'] >= window_start) & (df['_dt'] < window_end)]

            if len(window_df) >= self.min_samples:
                drift_results = self.detect_drift(window_df, category_column)

                # Aggregate drift metrics for this window position
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


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def detect_category_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                          category_column: str) -> List[DriftResult]:
    """
    Convenience function to detect drift between two datasets.

    Creates a temporary ``CategoryDriftDetector``, sets the baseline, and
    detects drift in one call.

    Args:
        baseline_df (pd.DataFrame): Historical/baseline data.
        current_df (pd.DataFrame): Current data to compare.
        category_column (str): Column with categories.

    Returns:
        list[DriftResult]: Per-category drift results sorted by severity.
    """
    detector = CategoryDriftDetector()
    detector.set_baseline(baseline_df, category_column)
    return detector.detect_drift(current_df, category_column)


def compare_periods(df: pd.DataFrame, category_column: str,
                    datetime_column: str, split_date: datetime) -> DistributionComparison:
    """
    Compare category distributions before and after a date.

    Splits the DataFrame into two subsets at ``split_date`` and runs a full
    distribution comparison (chi-square + Jensen-Shannon divergence).

    Args:
        df (pd.DataFrame): Full dataset.
        category_column (str): Category column.
        datetime_column (str): Datetime column.
        split_date (datetime): Date to split on.

    Returns:
        DistributionComparison: Statistical comparison results.
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

    Filters the full drift detection results to only return categories
    classified as EMERGING with severity above the threshold.

    Args:
        baseline_df (pd.DataFrame): Baseline data.
        current_df (pd.DataFrame): Current data.
        category_column (str): Category column.
        min_severity (float): Minimum severity to include.

    Returns:
        list[DriftResult]: Emerging categories sorted by severity descending.
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

    Filters the full drift detection results to only return categories
    classified as DECLINING with severity above the threshold.

    Args:
        baseline_df (pd.DataFrame): Baseline data.
        current_df (pd.DataFrame): Current data.
        category_column (str): Category column.
        min_severity (float): Minimum severity to include.

    Returns:
        list[DriftResult]: Declining categories sorted by severity descending.
    """
    results = detect_category_drift(baseline_df, current_df, category_column)
    declining = [r for r in results
                 if r.drift_type == DriftType.DECLINING and r.severity >= min_severity]
    return sorted(declining, key=lambda x: x.severity, reverse=True)
