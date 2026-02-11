"""
Recurrence Predictor - ML-based ticket recurrence prediction.

Uses Gradient Boosting to predict which tickets are likely to recur
within 30 days, enabling proactive intervention.

Architecture Overview:
    This module implements a binary classification pipeline that estimates the
    probability that a given escalation ticket will recur (i.e., a substantially
    similar issue will be raised again within ~30 days).  Proactive identification
    of likely-recurrent tickets enables the operations team to invest in root-cause
    remediation rather than repeatedly treating symptoms.

Model Selection:
    The primary model is a Random Forest classifier, dispatched through the
    ``GPURandomForestClassifier`` abstraction layer.  When a compatible NVIDIA
    GPU is available and ``USE_GPU`` is enabled, RAPIDS cuML is used for
    hardware-accelerated training and inference; otherwise scikit-learn provides
    a transparent CPU fallback.

Feature Engineering:
    Features are drawn from multiple pipeline stages:
    * **Numeric features** (direct pass-through):
        - Strategic_Friction_Score, AI_Confidence, Engineer_Issue_Count,
          Days_Since_Issue, Recidivism_Score
    * **Categorical features** (label-encoded):
        - Severity_Norm, Type_Norm, Origin_Norm, Root_Cause_Category,
          AI_Category, LOB_Risk_Tier, Is_Human_Error
        - LabelEncoders are fitted during training and reused at prediction
          time.  Unseen category values are mapped to 'Unknown'.
    * **Binary flags**:
        - Is_Repeat_Offender (derived from Engineer_Flag)
        - Is_Aged (derived from Aging_Status; 3-level: 0, 0.5, 1)
    * **Embedding dimensions** (optional):
        - First 10 dimensions of the ticket text embedding vector, providing
          a compressed semantic representation.

Target Variable:
    ``Recurrence_Actual`` is a binary label ('Yes' / 'No') derived from the
    Phase 3 recidivism analysis module.  Tickets with ``Learning_Status``
    containing 'REPEAT' or 'POSSIBLE', or with a ``Recidivism_Score`` >= 0.7,
    are labelled as recurrences.

Training Requirements:
    - Minimum 50 labelled samples (configurable via ``min_samples``).
    - Both 'Yes' and 'No' classes must have >= 10 samples each.
    - An 80/20 stratified train/test split is used for evaluation.

Evaluation Metrics:
    - Accuracy, AUC-ROC, and top-5 feature importances are logged after
      training.

Persistence:
    Trained models and label encoders can be serialised to disk via pickle
    (``save`` / ``load`` methods), enabling warm-start on subsequent runs.

Heuristic Fallback:
    When the ML model is not available (insufficient data or loading failure),
    a rule-based heuristic computes a risk score from six weighted factors:
    Learning_Status, PM-reported risk, similar-ticket count, friction score,
    severity, and category risk tier.

GPU-accelerated with RAPIDS cuML when available.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from ..core.config import (
    ANCHORS,
    RECURRENCE_MODEL_PATH,
    RECURRENCE_ENCODERS_PATH,
    USE_GPU
)
from ..core.gpu_utils import (
    GPURandomForestClassifier,
    is_gpu_available,
    clear_gpu_memory
)

logger = logging.getLogger(__name__)


class RecurrencePredictor:
    """
    Machine Learning model to predict ticket recurrence probability.

    Uses historical data to learn patterns and predict which new tickets
    are likely to recur within 30 days, allowing proactive intervention.

    Features used:
    - Severity level
    - Issue type (Escalation/Concern/etc.)
    - Origin (Internal/External)
    - Root cause category
    - Engineer history (repeat offender status)
    - LOB risk tier
    - AI category
    - Text embedding similarity to past recurring issues

    Attributes:
        model: The trained Random Forest classifier (GPU or CPU variant).
        encoders (dict): Mapping of categorical feature name to its fitted
            ``sklearn.preprocessing.LabelEncoder``.  Persisted alongside the
            model so that inference can decode the same categorical space.
        feature_columns (list[str]): Ordered list of feature column names
            matching the training matrix layout.
        is_trained (bool): Whether the model has been successfully fitted.
        model_metrics (dict): Training evaluation metrics (accuracy, AUC-ROC,
            train/test sample counts, recurrence rate, top features).
    """

    def __init__(self):
        """Initialise the predictor with empty state; no model is trained yet."""
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.model_metrics = {}

    def _prepare_features(self, df, fit_encoders=False):
        """
        Prepare feature matrix for training/prediction.

        Transforms raw DataFrame columns into a numeric feature matrix suitable
        for the Random Forest model.  Three feature families are constructed:

        1. **Numeric features** -- Directly copied from the DataFrame with
           coercion to float and NaN-filling with 0.
        2. **Categorical features** -- Each is label-encoded into consecutive
           integers.  During training (``fit_encoders=True``), new
           ``LabelEncoder`` instances are fitted on pre-defined category
           vocabularies plus an 'Unknown' sentinel.  During prediction,
           values not seen during training are mapped to 'Unknown' before
           encoding.
        3. **Binary / derived flags** -- Engineered from text patterns in
           existing columns (e.g., 'Repeat' in ``Engineer_Flag``).

        Optionally, the first 10 dimensions of the pre-computed text embedding
        vector are appended as features, providing a compressed semantic signal.

        Args:
            df (pd.DataFrame): Ticket data.
            fit_encoders (bool): If True, fit new LabelEncoders (training mode).
                If False, reuse previously fitted encoders (prediction mode).

        Returns:
            tuple:
                - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
                - feature_names (list[str]): Ordered feature column names.
        """
        feature_df = pd.DataFrame()

        # ---- Numeric features (direct pass-through) ----
        # These columns are already numeric in the enriched DataFrame.
        # pd.to_numeric with errors='coerce' handles any stray non-numeric
        # values, and fillna(0) ensures no NaNs reach the model.
        numeric_features = [
            'Strategic_Friction_Score',
            'AI_Confidence',
            'Engineer_Issue_Count',
            'Days_Since_Issue',
            'Recidivism_Score'
        ]

        for feat in numeric_features:
            if feat in df.columns:
                feature_df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)

        # ---- Categorical features (label-encoded) ----
        # Each feature has a pre-defined vocabulary (known possible values).
        # 'Unknown' is always included as a catch-all for unseen values.
        categorical_features = {
            'Severity_Norm': ['Critical', 'High', 'Medium', 'Low', 'Unknown'],
            'Type_Norm': ['Escalations', 'Concerns', 'Lessons Learned', 'Unknown'],
            'Origin_Norm': ['Internal', 'External', 'Unknown'],
            'Root_Cause_Category': ['Human Error', 'External Party', 'Process Gap',
                                     'System/Technical', 'Training Gap', 'Communication',
                                     'Resource', 'Other', 'Unclassified'],
            'AI_Category': list(ANCHORS.keys()) + ['Unclassified'],
            'LOB_Risk_Tier': ['Critical', 'High', 'Medium', 'Low', 'Unknown'],
            'Is_Human_Error': ['Yes', 'No', 'External']
        }

        for feat, categories in categorical_features.items():
            if feat in df.columns:
                if fit_encoders:
                    # Create and fit a new encoder on the full vocabulary
                    # (including 'Unknown') so any future value can be mapped.
                    le = LabelEncoder()
                    le.fit(categories + ['Unknown'])
                    self.encoders[feat] = le

                if feat in self.encoders:
                    # Transform column values, replacing unseen values with 'Unknown'
                    values = df[feat].fillna('Unknown').astype(str)
                    values = values.apply(lambda x: x if x in self.encoders[feat].classes_ else 'Unknown')
                    feature_df[f'{feat}_encoded'] = self.encoders[feat].transform(values)

        # ---- Binary flags derived from text patterns ----
        # Is_Repeat_Offender: 1 if the engineer has been flagged as a repeat
        # offender in previous analyses, 0 otherwise.
        if 'Engineer_Flag' in df.columns:
            feature_df['Is_Repeat_Offender'] = df['Engineer_Flag'].apply(
                lambda x: 1 if 'Repeat' in str(x) else 0
            )

        # Is_Aged: Tri-level encoding of ticket aging status.
        #   >30 days => 1.0 (severely aged)
        #   >14 days => 0.5 (moderately aged)
        #   otherwise => 0.0 (within SLA)
        if 'Aging_Status' in df.columns:
            feature_df['Is_Aged'] = df['Aging_Status'].apply(
                lambda x: 1 if '>30' in str(x) else (0.5 if '>14' in str(x) else 0)
            )

        # ---- Embedding-based features (first 10 principal dimensions) ----
        # If an 'embedding' column exists (pre-computed text embeddings from the
        # AI engine), extract the first 10 dimensions as numeric features.
        # This provides a low-dimensional semantic fingerprint of the ticket text.
        if 'embedding' in df.columns:
            try:
                embeddings = np.vstack(df['embedding'].values)
                for i in range(min(10, embeddings.shape[1])):
                    feature_df[f'emb_dim_{i}'] = embeddings[:, i]
            except Exception:
                pass  # Skip if embeddings are not available or malformed

        # Store the final feature column ordering for consistent inference
        self.feature_columns = list(feature_df.columns)
        return feature_df.values, self.feature_columns

    def train(self, df, min_samples=50):
        """
        Train the recurrence prediction model on historical data.

        Training Pipeline:
            1. Verify that a ``Recurrence_Actual`` column exists with known
               outcomes ('Yes' / 'No').
            2. Filter to rows with known outcomes and enforce the minimum
               sample requirement.
            3. Prepare the feature matrix (fitting new LabelEncoders).
            4. Encode the target: 'Yes' -> 1, 'No' -> 0.
            5. Perform an 80/20 stratified train/test split (stratification
               preserves the class imbalance ratio in both splits).
            6. Dynamically scale ``n_estimators`` and ``max_depth`` based on
               sample count to avoid overfitting and RAPIDS cuML bin warnings.
            7. Fit the Random Forest classifier.
            8. Evaluate on the held-out test set: accuracy and AUC-ROC.
            9. Extract feature importances and log the top 5.

        Class Imbalance:
            The historical recurrence rate is logged but no explicit resampling
            or class-weight adjustment is applied.  The model relies on the
            Random Forest's inherent robustness to moderate imbalance.

        Args:
            df (pd.DataFrame): Historical ticket data including recurrence outcomes.
            min_samples (int): Minimum total samples required to train.

        Returns:
            dict: Training metrics (accuracy, auc_roc, sample counts, etc.)
                or an error dict if training could not proceed.
        """
        logger.info("[Recurrence Predictor] Training predictive model...")

        # --- Step 1: Verify target column exists ---
        if 'Recurrence_Actual' not in df.columns:
            logger.warning("No Recurrence_Actual column found. Cannot train.")
            return {'error': 'No recurrence data available'}

        # --- Step 2: Filter to rows with known binary outcomes ---
        train_df = df[df['Recurrence_Actual'].isin(['Yes', 'No'])].copy()

        if len(train_df) < min_samples:
            logger.warning(f"Insufficient training data ({len(train_df)} samples). Need {min_samples}+")
            return {'error': f'Need at least {min_samples} samples, have {len(train_df)}'}

        # --- Step 3: Prepare feature matrix ---
        X, feature_names = self._prepare_features(train_df, fit_encoders=True)
        # Binary target: 1 = recurrence, 0 = no recurrence
        y = (train_df['Recurrence_Actual'] == 'Yes').astype(int)

        # --- Step 4: Log class distribution ---
        recurrence_rate = y.mean()
        logger.info(f"  Historical recurrence rate: {recurrence_rate:.1%}")

        # --- Step 5: Stratified train/test split ---
        # Stratification ensures both train and test sets preserve the original
        # class ratio, critical for meaningful AUC-ROC evaluation.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        # --- Step 6: Select compute backend and scale hyperparameters ---
        use_gpu = USE_GPU and is_gpu_available()
        if use_gpu:
            logger.info("  [GPU] Training with cuML RandomForestClassifier")
        else:
            logger.info("  [CPU] Training with sklearn GradientBoostingClassifier")

        # Scale tree count and depth relative to sample size to prevent
        # overfitting on small datasets and avoid RAPIDS cuML bin warnings.
        n_samples = len(X_train)
        n_estimators = min(100, max(10, n_samples // 2))
        max_depth = min(8, max(3, n_samples // 5))

        # --- Step 7: Instantiate and fit the classifier ---
        self.model = GPURandomForestClassifier(
            use_gpu=use_gpu,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # --- Step 8: Evaluate on the held-out test set ---
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]  # probability of class 1 (recurrence)

        # Convert GPU tensors (cupy arrays) to numpy for metrics calculation
        if hasattr(y_pred, 'get'):
            y_pred = y_pred.get()  # cupy to numpy
        if hasattr(y_proba, 'get'):
            y_proba = y_proba.get()
        y_test_np = np.array(y_test.values)

        # AUC-ROC requires both classes present in the test set
        try:
            auc_score = roc_auc_score(y_test_np, y_proba) if len(np.unique(y_test_np)) > 1 else 0.5
        except Exception:
            auc_score = 0.5

        accuracy = (y_pred == y_test_np).mean()

        # --- Step 9: Extract feature importances ---
        # Feature importances may not be available for all model backends
        # (e.g., some cuML versions).
        try:
            feature_imp = self.model.feature_importances_
            if hasattr(feature_imp, 'get'):
                feature_imp = feature_imp.get()  # cupy to numpy
            feature_importance = dict(zip(feature_names, feature_imp))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        except (AttributeError, Exception) as e:
            logger.debug(f"Feature importances not available: {e}")
            top_features = [(f, 0.0) for f in feature_names[:5]]

        # Store metrics for later retrieval
        self.model_metrics = {
            'accuracy': accuracy,
            'auc_roc': auc_score,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'recurrence_rate': recurrence_rate,
            'top_features': top_features
        }

        self.is_trained = True

        # Log training results
        logger.info(f"  ✓ Model trained on {len(X_train)} samples")
        logger.info(f"  → Accuracy: {accuracy:.1%}")
        logger.info(f"  → AUC-ROC: {auc_score:.3f}")
        logger.info(f"  → Top predictive features:")
        for feat, importance in top_features:
            logger.info(f"      • {feat}: {importance:.3f}")

        return self.model_metrics

    def predict(self, df):
        """
        Predict recurrence probability for tickets.

        If the ML model is trained, class-1 probabilities from the Random
        Forest are used directly.  Otherwise, the heuristic fallback is
        invoked.

        Risk Categorisation:
            Probabilities are mapped to four risk tiers:
            - >= 0.7  : High Risk
            - >= 0.5  : Elevated
            - >= 0.3  : Moderate
            - <  0.3  : Low

        Confidence Calculation:
            Confidence reflects how far the predicted probability is from the
            decision boundary (0.5).  Predictions near 0 or 1 are high
            confidence; predictions near 0.5 are low confidence.
            ``certainty = |prob - 0.5| * 2`` maps to:
            - >= 0.6 : High confidence
            - >= 0.3 : Medium confidence
            - <  0.3 : Low confidence

        Args:
            df (pd.DataFrame): Ticket data (may be the full dataset or a
                single-row DataFrame).

        Returns:
            pd.DataFrame: Copy of ``df`` with three new columns:
                - AI_Recurrence_Probability (float, 0-1)
                - AI_Recurrence_Risk (str, human-readable risk tier)
                - AI_Recurrence_Confidence (str, 'High'/'Medium'/'Low')
        """
        df = df.copy()

        # Initialise prediction columns with safe defaults
        df['AI_Recurrence_Probability'] = 0.0
        df['AI_Recurrence_Risk'] = 'Unknown'
        df['AI_Recurrence_Confidence'] = 'Low'

        if not self.is_trained:
            # Use heuristic-based prediction when model isn't trained
            logger.info("[Recurrence Predictor] Model not trained, using heuristic prediction")
            df = self._predict_heuristic(df)
            return df

        try:
            # Prepare feature matrix using the encoders fitted during training
            X, _ = self._prepare_features(df, fit_encoders=False)

            # Get class-1 (recurrence) probabilities from the model
            probabilities = self.model.predict_proba(X)[:, 1]

            df['AI_Recurrence_Probability'] = probabilities

            # Map probabilities to human-readable risk tiers
            def categorize_risk(prob):
                """Assign a colour-coded risk tier based on recurrence probability."""
                if prob >= 0.7:
                    return '🔴 High Risk (>70%)'
                elif prob >= 0.5:
                    return '🟠 Elevated (50-70%)'
                elif prob >= 0.3:
                    return '🟡 Moderate (30-50%)'
                else:
                    return '🟢 Low (<30%)'

            df['AI_Recurrence_Risk'] = df['AI_Recurrence_Probability'].apply(categorize_risk)

            # Map probabilities to confidence levels based on distance from
            # the 0.5 decision boundary.
            def get_confidence(prob):
                """Determine confidence from distance to decision boundary."""
                # certainty ranges from 0 (prob=0.5) to 1 (prob=0 or prob=1)
                certainty = abs(prob - 0.5) * 2
                if certainty >= 0.6:
                    return 'High'
                elif certainty >= 0.3:
                    return 'Medium'
                else:
                    return 'Low'

            df['AI_Recurrence_Confidence'] = df['AI_Recurrence_Probability'].apply(get_confidence)

            # Log summary statistics
            high_risk = (df['AI_Recurrence_Probability'] >= 0.5).sum()
            logger.info(f"[Recurrence Predictor] Predictions complete:")
            logger.info(f"  → {high_risk} tickets flagged as high recurrence risk (≥50%)")
            logger.info(f"  → Average recurrence probability: {probabilities.mean():.1%}")

        except Exception as e:
            logger.warning(f"[Recurrence Predictor] ML prediction failed: {e}")
            logger.info("[Recurrence Predictor] Falling back to heuristic prediction")
            df = self._predict_heuristic(df)

        return df

    def _predict_heuristic(self, df):
        """
        Heuristic-based recurrence prediction when ML model isn't trained.

        Uses available signals like Learning_Status, PM risk, friction score, etc.

        The heuristic computes a weighted additive risk score in [0, 1] from
        six independent factors:

        Factor 1 -- Learning Status (max +0.25):
            'REPEAT' in Learning_Status adds 0.25; 'POSSIBLE' adds 0.15.
            This is the strongest single signal, derived from the recidivism
            analysis module.

        Factor 2 -- PM-reported recurrence risk (max +0.30 / min -0.10):
            The project manager's subjective assessment of recurrence likelihood.
            'high' adds 0.30, 'medium' adds 0.15, 'low' subtracts 0.10.

        Factor 3 -- Similar ticket count (max +0.20):
            Each similar historical ticket contributes 0.04, capped at 0.20
            (i.e., 5 similar tickets saturate this factor).

        Factor 4 -- Strategic Friction Score (max +0.15):
            Linear scaling from the 0-100 friction score, representing
            operational difficulty at the organisational level.

        Factor 5 -- Severity (max +0.10):
            Critical/S1 severity adds 0.10; High/S2 adds 0.05.  Higher
            severity issues tend to indicate deeper systemic problems.

        Factor 6 -- Category risk (max +0.05):
            Certain AI categories (OSS, system, process, configuration) have
            historically higher recurrence rates.

        Base probability: 0.3 (prior).
        Final score: clamped to [0.0, 1.0].

        Args:
            df (pd.DataFrame): Ticket data.

        Returns:
            pd.DataFrame: Same DataFrame with prediction columns populated.
        """
        from ..core.config import COL_RECURRENCE_RISK

        def calculate_risk_score(row):
            """Calculate recurrence risk score (0-1) based on multiple factors."""
            score = 0.3  # Base probability (prior)

            # Factor 1: Learning Status (repeat offender indicator)
            learning = str(row.get('Learning_Status', '')).upper()
            if 'REPEAT' in learning:
                score += 0.25  # confirmed repeat offender
            elif 'POSSIBLE' in learning:
                score += 0.15  # possible repeat offender

            # Factor 2: PM-reported recurrence risk
            pm_risk = str(row.get(COL_RECURRENCE_RISK, row.get('tickets_data_risk_for_recurrence_pm', ''))).lower()
            if 'high' in pm_risk:
                score += 0.3
            elif 'medium' in pm_risk or 'moderate' in pm_risk:
                score += 0.15
            elif 'low' in pm_risk:
                score -= 0.1  # low PM risk reduces the score

            # Factor 3: Similar ticket count (more similar tickets = higher risk pattern)
            similar_count = row.get('Similar_Ticket_Count', 0)
            if pd.notna(similar_count) and similar_count > 0:
                # Each similar ticket contributes 0.04, capped at 0.20
                score += min(0.2, int(similar_count) * 0.04)

            # Factor 4: Strategic Friction Score (higher friction = systemic issues)
            friction = row.get('Strategic_Friction_Score', 0)
            if pd.notna(friction):
                # Linear scaling: 0 -> 0, 100 -> 0.15
                score += min(0.15, float(friction) / 100 * 0.15)

            # Factor 5: Severity (higher severity issues tend to recur more)
            severity = str(row.get('Severity_Norm', row.get('tickets_data_severity', ''))).lower()
            if 'critical' in severity or 'sev1' in severity or 's1' in severity:
                score += 0.1
            elif 'high' in severity or 'sev2' in severity or 's2' in severity:
                score += 0.05

            # Factor 6: Category risk (some categories have higher recurrence)
            category = str(row.get('AI_Category', '')).lower()
            high_risk_categories = ['oss', 'system', 'process', 'configuration']
            if any(cat in category for cat in high_risk_categories):
                score += 0.05

            return min(1.0, max(0.0, score))  # Clamp to 0-1

        def categorize_risk(prob):
            """Assign colour-coded risk tier from heuristic probability."""
            if prob >= 0.7:
                return '🔴 High (>70%)'
            elif prob >= 0.5:
                return '🟠 Elevated (50-70%)'
            elif prob >= 0.3:
                return '🟡 Moderate (30-50%)'
            else:
                return '🟢 Low (<30%)'

        def get_confidence(prob):
            """Heuristic predictions have inherently lower confidence."""
            # Extreme probabilities (near 0 or 1) get Medium confidence;
            # middle-range probabilities get Low confidence.
            if prob >= 0.7 or prob <= 0.2:
                return 'Medium'
            else:
                return 'Low'

        # Apply the heuristic score function to every row
        df['AI_Recurrence_Probability'] = df.apply(calculate_risk_score, axis=1)
        df['AI_Recurrence_Risk'] = df['AI_Recurrence_Probability'].apply(categorize_risk)
        df['AI_Recurrence_Confidence'] = df['AI_Recurrence_Probability'].apply(get_confidence)

        # Log summary
        high_risk = (df['AI_Recurrence_Probability'] >= 0.5).sum()
        logger.info(f"[Recurrence Predictor] Heuristic predictions complete:")
        logger.info(f"  → {high_risk}/{len(df)} tickets flagged as elevated/high risk (≥50%)")
        logger.info(f"  → Average risk probability: {df['AI_Recurrence_Probability'].mean():.1%}")

        return df

    def save(self, model_path=None, encoders_path=None):
        """
        Save trained model and encoders to disk.

        Persistence Format:
            Two pickle files are written:
            1. **Model file** -- Contains the fitted model object, the ordered
               feature column list, and training metrics.
            2. **Encoders file** -- Contains the dict of fitted LabelEncoders.

        Args:
            model_path (str, optional): Path for the model pickle file.
                Defaults to ``RECURRENCE_MODEL_PATH`` from config.
            encoders_path (str, optional): Path for the encoders pickle file.
                Defaults to ``RECURRENCE_ENCODERS_PATH`` from config.
        """
        model_path = model_path or RECURRENCE_MODEL_PATH
        encoders_path = encoders_path or RECURRENCE_ENCODERS_PATH

        if self.is_trained:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'metrics': self.model_metrics
                }, f)

            with open(encoders_path, 'wb') as f:
                pickle.dump(self.encoders, f)

            logger.info(f"[Recurrence Predictor] Model saved to {model_path}")

    def load(self, model_path=None, encoders_path=None):
        """
        Load trained model and encoders from disk.

        If both the model and encoders files exist, they are unpickled and
        the predictor is marked as trained.  This enables warm-start
        predictions without re-training.

        Args:
            model_path (str, optional): Path to the model pickle file.
            encoders_path (str, optional): Path to the encoders pickle file.

        Returns:
            bool: True if loading succeeded, False otherwise.
        """
        model_path = model_path or RECURRENCE_MODEL_PATH
        encoders_path = encoders_path or RECURRENCE_ENCODERS_PATH

        try:
            if os.path.exists(model_path) and os.path.exists(encoders_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.feature_columns = data['feature_columns']
                    self.model_metrics = data.get('metrics', {})

                with open(encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)

                self.is_trained = True
                logger.info(f"[Recurrence Predictor] Model loaded from {model_path}")
                return True
        except Exception as e:
            logger.warning(f"[Recurrence Predictor] Could not load model: {e}")

        return False

    def get_risk_factors(self, row):
        """
        Explain why a specific ticket is flagged as high risk.

        Inspects the enriched columns of a single ticket and returns a
        human-readable list of contributing risk factors.  This provides
        interpretability for tickets with AI_Recurrence_Probability >= 0.5.

        The explanations are rule-based (not SHAP or LIME), checking for:
        - Human error root cause
        - High engineer issue count (>= 3 prior issues)
        - Process gap root cause
        - High severity
        - High recidivism score (>= 0.5)
        - High-risk LOB tier

        Args:
            row: A pandas Series or dict-like for one ticket.

        Returns:
            list[str]: Contributing risk factors, or a single-element list
                with "No specific factors identified" if none apply.
        """
        factors = []

        if row.get('AI_Recurrence_Probability', 0) >= 0.5:
            if row.get('Is_Human_Error') == 'Yes':
                factors.append("Human error root cause (historically high recurrence)")

            if row.get('Engineer_Issue_Count', 0) >= 3:
                factors.append(f"Engineer has {row.get('Engineer_Issue_Count')} prior issues")

            if row.get('Root_Cause_Category') == 'Process Gap':
                factors.append("Process gap issues tend to recur until fixed")

            if row.get('Severity_Norm') in ['Critical', 'High']:
                factors.append("High severity issues often indicate systemic problems")

            if row.get('Recidivism_Score', 0) >= 0.5:
                factors.append(f"Similar issue occurred before (similarity: {row.get('Recidivism_Score', 0):.0%})")

            if row.get('LOB_Risk_Tier') in ['Critical', 'High']:
                factors.append(f"LOB has elevated risk profile")

        return factors if factors else ["No specific factors identified"]


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

# Global predictor instance (shared across the application lifecycle).
recurrence_predictor = RecurrencePredictor()


def apply_recurrence_predictions(df, train_if_possible=True):
    """
    Apply recurrence predictions to the dataframe.

    This is the top-level entry point called by the main analysis pipeline.
    It orchestrates the full workflow:

    1. Derives ``Recurrence_Actual`` labels from the recidivism analysis
       module's ``Learning_Status`` column (if not already present).
    2. Attempts to load a previously trained model from disk.
    3. If no model is found and ``train_if_possible`` is True, trains a new
       model provided there are at least 10 'Yes' and 10 'No' labels.
    4. Applies predictions (ML or heuristic fallback) to all tickets.

    Args:
        df (pd.DataFrame): Enriched ticket DataFrame.
        train_if_possible (bool): Whether to train a new model if none exists.

    Returns:
        pd.DataFrame: DataFrame with recurrence prediction columns added:
            - AI_Recurrence_Probability
            - AI_Recurrence_Risk
            - AI_Recurrence_Confidence
    """
    global recurrence_predictor

    logger.info("[Recurrence Predictor] Initializing AI-based recurrence prediction...")

    # Step 1: Derive Recurrence_Actual from recidivism analysis if not already present
    if 'Recurrence_Actual' not in df.columns:
        df = _derive_recurrence_labels(df)

    # Step 2: Try to load an existing pre-trained model from disk
    if recurrence_predictor.load():
        logger.info("  Using pre-trained model")
    elif train_if_possible:
        # Step 3: Train new model if we have sufficient recurrence data
        if 'Recurrence_Actual' in df.columns:
            # Verify we have both positive and negative labels in sufficient quantity
            label_counts = df['Recurrence_Actual'].value_counts()
            if 'Yes' in label_counts and 'No' in label_counts:
                if label_counts['Yes'] >= 10 and label_counts['No'] >= 10:
                    metrics = recurrence_predictor.train(df)
                    if 'error' not in metrics:
                        recurrence_predictor.save()  # Persist for future runs
                else:
                    logger.info(f"  Insufficient balanced training data (Yes: {label_counts.get('Yes', 0)}, No: {label_counts.get('No', 0)})")
            else:
                logger.info("  Training data lacks both Yes/No labels - using heuristic prediction")
        else:
            logger.info("  No historical recurrence data - will train on next run with outcomes")

    # Step 4: Apply predictions (ML model or heuristic fallback)
    df = recurrence_predictor.predict(df)

    return df


def _derive_recurrence_labels(df):
    """
    Derive Recurrence_Actual labels from recidivism analysis results.

    Uses ``Learning_Status`` (from Phase 3 recidivism analysis) to determine
    which tickets represent actual recurrences vs. new issues.

    Labelling Rules:
        1. Default all tickets to 'No' (not a recurrence).
        2. Mark as 'Yes' if ``Learning_Status`` contains 'REPEAT' (confirmed
           repeat offense) or 'POSSIBLE' (likely repeat offense).
        3. Additionally mark as 'Yes' if ``Recidivism_Score`` >= 0.7 (high
           semantic similarity to a prior ticket), even if Learning_Status
           did not flag it.

    Args:
        df (pd.DataFrame): Ticket DataFrame with recidivism analysis columns.

    Returns:
        pd.DataFrame: Copy of ``df`` with ``Recurrence_Actual`` column added.
    """
    df = df.copy()

    # Initialize as 'No' (not a recurrence)
    df['Recurrence_Actual'] = 'No'

    # Mark as 'Yes' based on recidivism analysis findings (Learning_Status)
    if 'Learning_Status' in df.columns:
        # REPEAT OFFENSE = confirmed recurrence
        repeat_mask = df['Learning_Status'].astype(str).str.contains('REPEAT', case=False, na=False)
        df.loc[repeat_mask, 'Recurrence_Actual'] = 'Yes'

        # POSSIBLE REPEAT = likely recurrence
        possible_mask = df['Learning_Status'].astype(str).str.contains('POSSIBLE', case=False, na=False)
        df.loc[possible_mask, 'Recurrence_Actual'] = 'Yes'

        yes_count = repeat_mask.sum() + possible_mask.sum()
        no_count = len(df) - yes_count
        logger.info(f"  Derived recurrence labels: {yes_count} recurrences, {no_count} new issues")

    # Also consider high recidivism scores (semantic similarity >= 0.7)
    # that may not have been caught by the Learning_Status text matching.
    if 'Recidivism_Score' in df.columns:
        high_score_mask = df['Recidivism_Score'] >= 0.7
        # Only upgrade tickets that were previously labelled 'No'
        additional = high_score_mask & (df['Recurrence_Actual'] == 'No')
        df.loc[additional, 'Recurrence_Actual'] = 'Yes'
        if additional.sum() > 0:
            logger.info(f"  Additional {additional.sum()} tickets marked as recurrence from high similarity scores")

    return df
