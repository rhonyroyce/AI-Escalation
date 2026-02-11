"""
Resolution Time Predictor - ML-based resolution time prediction.

Predicts resolution time for escalation tickets based on historical patterns,
issue characteristics, and human-provided expectations.

Architecture Overview:
    This module implements a supervised regression pipeline that estimates how many
    calendar days a given escalation ticket will take to resolve. The primary model
    is a Random Forest regressor (GPU-accelerated via RAPIDS cuML when a compatible
    GPU is detected, otherwise falling back to scikit-learn on CPU).

Prediction Strategy (priority cascade):
    1. **ML model** -- If training succeeded with >= 10 labelled samples, the Random
       Forest is queried first.  A sanity-check window (0.1 -- 100 days) guards
       against degenerate predictions.
    2. **Category statistics** -- Median resolution time learned from historical
       data for the ticket's AI-assigned category.
    3. **Global statistics** -- Median of all per-category medians, used when the
       ticket's category was unseen during training.
    4. **Calibrated heuristics** -- Domain-expert lookup table of expected resolution
       days per telecom escalation category, adjusted by severity, text complexity
       indicators, recurrence risk, and strategic friction score.

Feature Engineering:
    Features fed to the Random Forest include:
    * ``category_hash``       -- Deterministic hash of the category string (mod 1000)
                                 to create a categorical integer representation.
    * ``category_avg_days``   -- Historical mean resolution time for the category.
    * ``category_median_days``-- Historical median resolution time for the category.
    * ``severity_level``      -- Ordinal encoding: Critical=4, High=3, Medium=2, Low=1.
    * ``text_length``         -- Character count of the issue summary.
    * ``word_count``          -- Token count (whitespace split) of the summary.
    * ``complexity_score``    -- Count of domain complexity keywords found in the
                                 summary (e.g., "migration", "integration").
    * ``similar_resolution_days`` -- Expected resolution days from similar-ticket
                                    analysis.
    * ``similar_count``       -- Number of similar historical tickets found.
    * ``ai_confidence``       -- Confidence score from the AI classifier.
    * ``recurrence_risk``     -- Ordinal encoding of the AI-predicted recurrence risk.

Accuracy Metrics:
    After training, the module reports Mean Absolute Error (MAE) and Root Mean
    Squared Error (RMSE) on the training set.  The ``get_accuracy_metrics`` method
    computes MAE, RMSE, MAPE, and Pearson correlation on any DataFrame that has
    both actual and predicted columns populated.

Human-in-the-Loop:
    Subject-matter experts can supply per-category expected resolution times via
    ``set_human_expectations``.  These are stored alongside ML predictions in a
    dedicated ``Human_Expected_Days`` column, enabling three-way comparison
    (actual vs. predicted vs. expected) on the dashboard.

GPU-accelerated with RAPIDS cuML when available.
"""

import logging
import numpy as np
import pandas as pd

from ..core.config import COL_SUMMARY, COL_SEVERITY, COL_DATETIME, COL_RESOLUTION_DATE, USE_GPU
from ..core.gpu_utils import GPURandomForestRegressor, is_gpu_available

logger = logging.getLogger(__name__)


class ResolutionTimePredictor:
    """
    ML-based Resolution Time Predictor.

    Predicts resolution time for tickets based on:
    - Historical resolution times by category
    - Issue severity and complexity indicators
    - Similar ticket resolution patterns
    - Human-provided expected times (for calibration)

    Provides three metrics:
    - Actual: Real resolution time from data
    - Predicted: ML-predicted resolution time
    - Expected: Human-provided expectation (from feedback)

    Attributes:
        model: The trained Random Forest regressor (GPU or CPU variant).
        feature_columns (list[str]): Ordered list of feature names corresponding
            to the columns of the training matrix X.  Used at inference time to
            ensure consistent feature ordering.
        category_stats (dict): Per-category descriptive statistics (mean, median,
            std, count) computed from the training set.  Used as a fallback when
            the ML model cannot be applied.
        severity_stats (dict): Reserved for future per-severity statistics.
        human_expectations (dict): Mapping of category name to human-supplied
            expected resolution days, loaded via ``set_human_expectations``.
        is_trained (bool): Flag indicating whether the Random Forest has been
            successfully fitted on historical data.
    """

    def __init__(self):
        """Initialize the predictor with empty state; no model is trained yet."""
        self.model = None
        self.feature_columns = []
        self.category_stats = {}
        self.severity_stats = {}
        self.human_expectations = {}
        self.is_trained = False

        logger.info("[Resolution Predictor] Initialized")

    def _extract_features(self, row):
        """
        Extract a flat feature dictionary from a single ticket row.

        The feature set is designed to be lightweight and computable from
        the enriched DataFrame columns produced by earlier pipeline stages
        (classification, similarity analysis, recurrence prediction).

        Args:
            row: A pandas Series or dict-like object representing one ticket.

        Returns:
            dict: Feature name -> numeric value mapping.  All values are
                guaranteed to be numeric (int or float) so the resulting
                dict can be directly converted to a numpy array.

        Feature Details:
            - category_hash: Hash of AI_Category string mod 1000.  This converts
              the categorical variable into a pseudo-numeric feature.  While not
              ideal for tree-based models (which prefer ordinal or one-hot
              encoding), it provides a deterministic integer identifier that the
              Random Forest can split on.
            - category_avg_days / category_median_days: Populated from
              ``self.category_stats`` if the category was seen during training;
              otherwise defaults to 5.0 / 3.0 days respectively.
            - severity_level: Ordinal mapping Critical=4 > High=3 > Medium=2 > Low=1.
            - text_length / word_count: Simple proxies for issue description
              verbosity, which may correlate with complexity.
            - complexity_score: Count of domain-specific complexity keywords found
              in the summary text.  More keywords suggest a harder problem.
            - similar_resolution_days: Expected resolution days propagated from the
              similar-ticket analysis module.
            - similar_count: How many similar historical tickets were found.
            - ai_confidence: Classification confidence from the AI engine.
            - recurrence_risk: Ordinal encoding of the predicted recurrence risk
              level (High=3, Medium=2, Low=1).
        """
        features = {}

        # -- Category identifier (hashed to integer for the tree model) --
        category = str(row.get('AI_Category', 'Unknown'))
        features['category_hash'] = hash(category) % 1000

        # -- Historical resolution statistics for this category --
        # If we have seen this category in training data, use learned stats;
        # otherwise fall back to conservative defaults.
        if category in self.category_stats:
            features['category_avg_days'] = self.category_stats[category]['mean']
            features['category_median_days'] = self.category_stats[category]['median']
        else:
            features['category_avg_days'] = 5.0       # default mean when unseen
            features['category_median_days'] = 3.0     # default median when unseen

        # -- Severity as ordinal integer --
        severity = str(row.get('AI_Severity', row.get(COL_SEVERITY, 'Medium')))
        severity_map = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        features['severity_level'] = severity_map.get(severity, 2)

        # -- Text complexity proxies --
        summary = str(row.get(COL_SUMMARY, ''))
        features['text_length'] = len(summary)          # character count
        features['word_count'] = len(summary.split())    # whitespace-tokenized word count

        # -- Domain complexity keyword scan --
        # Each keyword hit increments the score by 1; the total acts as a rough
        # measure of how operationally complex the issue is.
        complexity_keywords = ['complex', 'multiple', 'integration', 'migration', 'upgrade', 'critical']
        features['complexity_score'] = sum(1 for kw in complexity_keywords if kw in summary.lower())

        # -- Similar-ticket derived features --
        features['similar_resolution_days'] = row.get('Expected_Resolution_Days', 0) or 0
        features['similar_count'] = row.get('Similar_Ticket_Count', 0) or 0

        # -- AI classification confidence --
        features['ai_confidence'] = row.get('AI_Confidence', 0.5) or 0.5

        # -- Recurrence risk as ordinal integer --
        recurrence = str(row.get('AI_Recurrence_Risk', 'Medium'))
        recurrence_map = {'High': 3, 'Medium': 2, 'Low': 1, 'Unknown': 2}
        features['recurrence_risk'] = recurrence_map.get(recurrence, 2)

        return features

    def train(self, df):
        """
        Train the resolution time predictor on historical data.

        Training Procedure:
            1. Iterate through every row in ``df`` and compute the actual
               resolution days from issue-date / resolution-date pairs.
            2. Discard rows where the actual resolution is missing, non-positive,
               or exceeds 365 days (likely data errors or long-running outliers).
            3. Extract the feature vector for each valid row.
            4. If fewer than 10 valid samples exist, abort (the model would
               overfit on so few points).
            5. Build per-category descriptive statistics (mean, median, std,
               count) from the valid training data.
            6. Fit a Random Forest regressor on the feature matrix.

        Hyperparameter Scaling:
            ``n_estimators`` and ``max_depth`` are dynamically scaled relative to
            the number of training samples to prevent overfitting on small datasets
            and to avoid the RAPIDS cuML "bins" warning that occurs when the
            number of bins exceeds the number of samples.

        Args:
            df (pd.DataFrame): Full ticket DataFrame including date columns and
                all enrichment columns produced by earlier pipeline stages.

        Returns:
            bool: True if training succeeded, False otherwise.
        """
        logger.info("[Resolution Predictor] Training model...")

        # --- Step 1: Build labelled training set ---
        training_data = []

        for idx, row in df.iterrows():
            # Calculate the ground-truth resolution time (calendar days)
            actual_days = self._calculate_resolution_days(row)
            # Skip rows without valid resolution data or outlier durations
            if actual_days is None or actual_days <= 0 or actual_days > 365:
                continue

            features = self._extract_features(row)
            features['actual_days'] = actual_days  # target variable
            training_data.append(features)

        # --- Step 2: Guard against insufficient data ---
        if len(training_data) < 10:
            logger.warning("[Resolution Predictor] Insufficient training data (< 10 samples)")
            self.is_trained = False
            return False

        train_df = pd.DataFrame(training_data)

        # --- Step 3: Build per-category descriptive statistics ---
        # These statistics serve as fallback predictions when the ML model
        # cannot be used (e.g., for unseen categories or prediction failures).
        for category in df['AI_Category'].dropna().unique():
            cat_data = [t['actual_days'] for t in training_data
                       if t.get('category_hash') == hash(category) % 1000]
            if cat_data:
                self.category_stats[category] = {
                    'mean': np.mean(cat_data),
                    'median': np.median(cat_data),
                    'std': np.std(cat_data) if len(cat_data) > 1 else 0,
                    'count': len(cat_data)
                }

        # --- Step 4: Prepare feature matrix and target vector ---
        self.feature_columns = [c for c in train_df.columns if c != 'actual_days']
        X = train_df[self.feature_columns].values   # shape (n_samples, n_features)
        y = train_df['actual_days'].values           # shape (n_samples,)

        try:
            # --- Step 5: Select compute backend (GPU vs CPU) ---
            use_gpu = USE_GPU and is_gpu_available()
            if use_gpu:
                logger.info("[Resolution Predictor] Training with GPU acceleration")

            # Scale hyperparameters to dataset size to prevent overfitting and
            # avoid RAPIDS cuML warnings about excessive bins.
            n_samples = len(training_data)
            n_estimators = min(50, max(10, n_samples // 2))

            # --- Step 6: Instantiate and fit the Random Forest regressor ---
            self.model = GPURandomForestRegressor(
                use_gpu=use_gpu,
                n_estimators=n_estimators,
                max_depth=min(8, max(3, n_samples // 5)),  # Scale depth with samples
                random_state=42  # Reproducibility seed
            )
            self.model.fit(X, y)
            self.is_trained = True

            # --- Step 7: Compute in-sample error metrics (diagnostic only) ---
            predictions = self.model.predict(X)
            mae = np.mean(np.abs(predictions - y))
            rmse = np.sqrt(np.mean((predictions - y) ** 2))

            logger.info(f"[Resolution Predictor] Model trained on {len(training_data)} samples")
            logger.info(f"[Resolution Predictor] Training MAE: {mae:.2f} days, RMSE: {rmse:.2f} days")

            return True

        except Exception as e:
            logger.error(f"[Resolution Predictor] Training failed: {e}")
            self.is_trained = False
            return False

    def _calculate_resolution_days(self, row):
        """
        Calculate actual resolution days from ticket data.

        Computes the calendar-day difference between the issue creation date
        and the resolution date.  Both dates may be stored as strings or
        pandas Timestamps, so this method normalises them via ``pd.to_datetime``
        if necessary.

        Args:
            row: A pandas Series (or dict-like) containing at minimum the
                configured ``COL_DATETIME`` and ``COL_RESOLUTION_DATE`` fields.

        Returns:
            int or None: Non-negative integer day count, or None if either date
                is missing, unparseable, or the difference is negative (which
                would indicate a data-entry error).
        """
        try:
            issue_date = row.get(COL_DATETIME)
            resolution_date = row.get(COL_RESOLUTION_DATE)

            # Bail out if either date is missing
            if pd.isna(issue_date) or pd.isna(resolution_date):
                return None

            # Normalise string dates to Timestamps
            if isinstance(issue_date, str):
                issue_date = pd.to_datetime(issue_date)
            if isinstance(resolution_date, str):
                resolution_date = pd.to_datetime(resolution_date)

            days = (resolution_date - issue_date).days
            # Negative days imply data error; treat as missing
            return days if days >= 0 else None

        except Exception:
            return None

    def predict(self, row):
        """
        Predict resolution time for a single ticket.

        Implements a four-level priority cascade to maximise prediction coverage
        and gracefully degrade when the ML model is unavailable.

        Priority order:
            1. ML model prediction (if trained and successful)
            2. Category statistics from training data (median)
            3. Global statistics from training data
            4. Calibrated heuristics (scaled to match learned data)

        Confidence Calculation (ML path):
            Confidence is derived from the ratio of the category's standard
            deviation to the predicted value.  High variance within a category
            lowers confidence, while low variance raises it.  Clamped to
            [0.30, 0.95].

        Confidence Calculation (category_stats path):
            Starts at 0.40 and increases linearly with sample count, capping
            at 0.70 when 50+ samples are available.

        Args:
            row: A pandas Series or dict-like for one ticket.

        Returns:
            dict with keys:
                - predicted_days (float): Estimated resolution days (>= 0.5).
                - confidence (float): Confidence in [0.30, 0.95].
                - method (str): One of 'ml', 'category_stats', 'global_stats',
                  'heuristic'.
        """
        category = str(row.get('AI_Category', 'Unknown'))

        # ---- Priority 1: Try ML model ----
        if self.is_trained and self.model is not None:
            try:
                features = self._extract_features(row)
                # Build a single-row feature matrix in the same column order used during training
                X = np.array([[features[c] for c in self.feature_columns]])
                predicted = self.model.predict(X)[0]

                # Sanity check: discard predictions outside a reasonable range
                if 0.1 <= predicted <= 100:
                    # Confidence decreases as within-category variance grows
                    cat_std = self.category_stats.get(category, {}).get('std', 2.0)
                    confidence = max(0.3, min(0.95, 1 - (cat_std / (predicted + 1))))

                    return {
                        'predicted_days': max(0.5, round(predicted, 1)),
                        'confidence': confidence,
                        'method': 'ml'
                    }
            except Exception as e:
                logger.debug(f"[Resolution Predictor] ML prediction failed: {e}")

        # ---- Priority 2: Use learned category statistics ----
        if category in self.category_stats:
            cat_stat = self.category_stats[category]
            # Prefer median (robust to outliers) over mean
            predicted_days = cat_stat.get('median', cat_stat.get('mean', 3.0))
            sample_count = cat_stat.get('count', 1)
            # Confidence grows logarithmically with sample count, capped at 0.70
            confidence = min(0.7, 0.4 + (sample_count / 50))

            return {
                'predicted_days': max(0.5, round(predicted_days, 1)),
                'confidence': confidence,
                'method': 'category_stats'
            }

        # ---- Priority 3: Use global stats from training data for unknown category ----
        if self.category_stats:
            # Compute the global median of all per-category medians
            all_medians = [s.get('median', s.get('mean')) for s in self.category_stats.values()
                          if s.get('median') or s.get('mean')]
            if all_medians:
                global_median = np.median(all_medians)
                return {
                    'predicted_days': max(0.5, round(global_median, 1)),
                    'confidence': 0.35,  # low confidence for global fallback
                    'method': 'global_stats'
                }

        # ---- Priority 4: Calibrated heuristic prediction ----
        return self._predict_heuristic(row, category)

    def _predict_heuristic(self, row, category):
        """
        Heuristic-based resolution time prediction when ML model isn't trained.

        Uses category complexity, severity, and other indicators.
        Now heavily calibrated to actual data when available.

        Heuristic Adjustment Pipeline:
            1. **Base days**: Either the global median of all learned categories
               (when any training data exists) or a domain-expert lookup table
               of per-category resolution times specific to telecom escalations.
            2. **Severity multiplier**: Critical issues are resolved ~40% faster
               (urgency-driven), while low-severity issues take ~30% longer.
            3. **Complexity multiplier**: Presence of complexity keywords (e.g.,
               "migration", "integration") increases the estimate by 40%;
               simplicity keywords reduce it by 30%.
            4. **Recurrence multiplier**: High recurrence risk adds 20% (systemic
               problems are harder to resolve permanently).
            5. **Friction multiplier**: The Strategic Friction Score provides a
               fine-grained continuous adjustment in the range [0.9, 1.1].

        Args:
            row: Single ticket data (Series or dict-like).
            category (str): The AI-assigned category name.

        Returns:
            dict: Same schema as ``predict`` -- predicted_days, confidence, method.
        """
        # First, try to get a reasonable baseline from learned data
        if self.category_stats:
            # Calculate global median from all learned data
            all_medians = [s['median'] for s in self.category_stats.values() if s.get('median')]
            if all_medians:
                global_median = np.median(all_medians)
                # Use this as the base instead of high industry heuristics
                base_days = global_median
            else:
                base_days = 3.0  # Conservative default if no learned data
        else:
            # 8-category system resolution time estimates
            # Based on telecom escalation analysis
            category_base_days = {
                'Scheduling & Planning': 1.0,
                'Documentation & Reporting': 0.5,
                'Validation & QA': 1.5,
                'Process Compliance': 1.0,
                'Configuration & Data Mismatch': 2.5,
                'Site Readiness': 3.0,
                'Communication & Response': 0.5,
                'Nesting & Tool Errors': 2.0,
            }
            base_days = category_base_days.get(category, 2.0)

        # --- Severity adjustment ---
        # Critical / high-severity issues receive more attention and are resolved
        # faster; low-severity issues may sit in queue longer.
        severity = str(row.get('Severity_Norm', row.get(COL_SEVERITY, 'Medium'))).lower()
        if 'critical' in severity or 's1' in severity:
            base_days *= 0.6  # Faster resolution for critical issues
        elif 'high' in severity or 's2' in severity:
            base_days *= 0.8
        elif 'low' in severity:
            base_days *= 1.3  # Lower priority takes longer

        # --- Text complexity adjustment ---
        # Scan the issue summary for keywords indicating complex or simple work.
        summary = str(row.get(COL_SUMMARY, ''))
        complexity_indicators = ['multiple', 'complex', 'integration', 'migration', 'several', 'widespread']
        simple_indicators = ['simple', 'quick', 'minor', 'single', 'easy']

        if any(ind in summary.lower() for ind in complexity_indicators):
            base_days *= 1.4  # complex issues take longer
        elif any(ind in summary.lower() for ind in simple_indicators):
            base_days *= 0.7  # simple issues resolve faster

        # --- Recurrence risk adjustment ---
        # Repeat issues may be systemic and harder to fix permanently.
        recurrence_risk = str(row.get('AI_Recurrence_Risk', '')).lower()
        if 'high' in recurrence_risk:
            base_days *= 1.2
        elif 'low' in recurrence_risk:
            base_days *= 0.9

        # --- Strategic Friction Score adjustment ---
        # A continuous fine-grained multiplier derived from the friction score.
        # Score of 0 => multiplier 0.9, score of 100 => multiplier 1.1.
        friction = row.get('Strategic_Friction_Score', 50)
        if pd.notna(friction):
            base_days *= (0.9 + (float(friction) / 500))  # 0.9 to 1.1 multiplier

        # Round to 1 decimal place, minimum 0.5 days
        predicted_days = max(0.5, round(base_days, 1))

        # Confidence based on how specific our estimate is
        confidence = 0.4  # Heuristic confidence is moderate
        if category in category_base_days:
            confidence = 0.55  # slightly higher when the category was in the lookup table

        return {
            'predicted_days': predicted_days,
            'confidence': confidence,
            'method': 'heuristic'
        }

    def set_human_expectations(self, expectations_dict):
        """
        Set human-provided expected resolution times by category.

        These expectations are merged into the output DataFrame alongside ML
        predictions, enabling a three-way comparison on the dashboard:
        Actual vs. Predicted vs. Human Expected.

        Args:
            expectations_dict (dict): Mapping of category name (str) to
                expected resolution days (float).
        """
        self.human_expectations = expectations_dict
        logger.info(f"[Resolution Predictor] Loaded {len(expectations_dict)} human expectations")

    def process_all_tickets(self, df):
        """
        Process all tickets and add resolution time columns.

        End-to-end workflow:
            1. Train the ML model on the provided DataFrame (using rows that
               have both issue and resolution dates).
            2. For every ticket, compute the actual resolution days (if dates
               are available), the ML/heuristic predicted days, the prediction
               confidence and method, and the human expected days (if supplied).
            3. Append five new columns to the DataFrame:
               - ``Actual_Resolution_Days``
               - ``Predicted_Resolution_Days``
               - ``Human_Expected_Days``
               - ``Resolution_Prediction_Confidence``
               - ``Resolution_Prediction_Method``

        Args:
            df (pd.DataFrame): The enriched ticket DataFrame.

        Returns:
            pd.DataFrame: Same DataFrame with the five new columns populated.
        """
        logger.info(f"[Resolution Predictor] Processing {len(df)} tickets...")

        # First, train on the data (uses rows with known resolution dates)
        self.train(df)

        # Initialize output columns with None / null
        df['Actual_Resolution_Days'] = None
        df['Predicted_Resolution_Days'] = None
        df['Human_Expected_Days'] = None
        df['Resolution_Prediction_Confidence'] = None
        df['Resolution_Prediction_Method'] = None

        for idx, row in df.iterrows():
            # Compute ground-truth resolution time (may be None)
            actual = self._calculate_resolution_days(row)
            df.at[idx, 'Actual_Resolution_Days'] = actual

            # Generate prediction via the priority cascade
            prediction = self.predict(row)
            df.at[idx, 'Predicted_Resolution_Days'] = prediction['predicted_days']
            df.at[idx, 'Resolution_Prediction_Confidence'] = prediction['confidence']
            df.at[idx, 'Resolution_Prediction_Method'] = prediction['method']

            # Attach human expectation if available for this category
            category = str(row.get('AI_Category', 'Unknown'))
            if category in self.human_expectations:
                df.at[idx, 'Human_Expected_Days'] = self.human_expectations[category]

        # Log summary statistics
        has_actual = df['Actual_Resolution_Days'].notna().sum()
        has_predicted = df['Predicted_Resolution_Days'].notna().sum()
        has_expected = df['Human_Expected_Days'].notna().sum()

        logger.info(f"[Resolution Predictor] Complete:")
        logger.info(f"  → {has_actual} tickets with actual resolution times")
        logger.info(f"  → {has_predicted} tickets with ML predictions")
        logger.info(f"  → {has_expected} tickets with human expectations")

        return df

    def get_accuracy_metrics(self, df):
        """
        Calculate accuracy metrics comparing actual vs predicted resolution times.

        Requires at least 5 tickets that have both ``Actual_Resolution_Days``
        and ``Predicted_Resolution_Days`` populated.

        Metrics computed:
            - **MAE** (Mean Absolute Error): Average absolute deviation in days.
            - **RMSE** (Root Mean Squared Error): Penalises large errors more
              than MAE; useful for identifying outlier predictions.
            - **MAPE** (Mean Absolute Percentage Error): Scale-independent error
              metric expressed as a percentage.  A small epsilon (0.1) is added
              to the denominator to avoid division by zero.
            - **Correlation** (Pearson r): Linear correlation between actual and
              predicted.  Values near +1 indicate the model captures the ranking
              of resolution difficulty even if the absolute scale is off.
            - **Per-category MAE**: Breakdown of MAE for categories with >= 3
              samples.

        Args:
            df (pd.DataFrame): DataFrame with both actual and predicted columns.

        Returns:
            dict or None: Metrics dictionary, or None if insufficient data.
        """
        # Only consider rows where both actual and predicted are available
        valid = df.dropna(subset=['Actual_Resolution_Days', 'Predicted_Resolution_Days'])

        if len(valid) < 5:
            return None

        actual = np.array(valid['Actual_Resolution_Days'].values, dtype=float)
        predicted = np.array(valid['Predicted_Resolution_Days'].values, dtype=float)

        # Calculate Pearson correlation safely (requires at least 3 points
        # and non-zero variance in both arrays)
        try:
            if len(actual) > 2 and np.std(actual) > 0 and np.std(predicted) > 0:
                correlation = np.corrcoef(actual, predicted)[0, 1]
            else:
                correlation = 0.0
        except Exception:
            correlation = 0.0

        metrics = {
            'mae': np.mean(np.abs(predicted - actual)),
            'rmse': np.sqrt(np.mean((predicted - actual) ** 2)),
            # MAPE with epsilon in denominator to avoid division by zero
            'mape': np.mean(np.abs((actual - predicted) / (actual + 0.1))) * 100,
            'sample_count': len(valid),
            'correlation': correlation
        }

        # Per-category breakdown (only for categories with enough samples)
        metrics['by_category'] = {}
        for category in valid['AI_Category'].dropna().unique():
            cat_data = valid[valid['AI_Category'] == category]
            if len(cat_data) >= 3:
                cat_actual = np.array(cat_data['Actual_Resolution_Days'].values, dtype=float)
                cat_pred = np.array(cat_data['Predicted_Resolution_Days'].values, dtype=float)
                metrics['by_category'][category] = {
                    'mae': np.mean(np.abs(cat_pred - cat_actual)),
                    'count': len(cat_data)
                }

        return metrics


# ---------------------------------------------------------------------------
# Module-level singleton and convenience function
# ---------------------------------------------------------------------------

# Global resolution time predictor instance (lazily initialised by
# ``apply_resolution_time_prediction``).
resolution_time_predictor = None


def apply_resolution_time_prediction(df, human_expectations=None):
    """
    Apply ML-based resolution time prediction to the dataframe.

    This is the top-level entry point called by the main analysis pipeline.
    It instantiates a fresh ``ResolutionTimePredictor``, optionally loads
    human expectations, runs the full train-and-predict workflow, and logs
    accuracy metrics.

    Args:
        df (pd.DataFrame): Enriched ticket DataFrame.
        human_expectations (dict, optional): Mapping of category name to
            expected resolution days provided by subject-matter experts.

    Returns:
        pd.DataFrame: The input DataFrame augmented with resolution-time
            prediction columns.
    """
    global resolution_time_predictor

    logger.info("[Resolution Predictor] Initializing resolution time prediction...")

    resolution_time_predictor = ResolutionTimePredictor()

    if human_expectations:
        resolution_time_predictor.set_human_expectations(human_expectations)

    df = resolution_time_predictor.process_all_tickets(df)

    # Log accuracy metrics if enough labelled data was available
    metrics = resolution_time_predictor.get_accuracy_metrics(df)
    if metrics:
        logger.info(f"[Resolution Predictor] Accuracy Metrics:")
        logger.info(f"  → MAE: {metrics['mae']:.2f} days")
        logger.info(f"  → RMSE: {metrics['rmse']:.2f} days")
        logger.info(f"  → Correlation: {metrics['correlation']:.2f}")

    return df
