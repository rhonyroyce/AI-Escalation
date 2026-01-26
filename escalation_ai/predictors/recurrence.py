"""
Recurrence Predictor - ML-based ticket recurrence prediction.

Uses Gradient Boosting to predict which tickets are likely to recur
within 30 days, enabling proactive intervention.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from ..core.config import (
    ANCHORS, 
    RECURRENCE_MODEL_PATH,
    RECURRENCE_ENCODERS_PATH
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
    """
    
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.model_metrics = {}
        
    def _prepare_features(self, df, fit_encoders=False):
        """
        Prepare feature matrix for training/prediction.
        
        Args:
            df: DataFrame with ticket data
            fit_encoders: If True, fit new encoders (training mode)
            
        Returns:
            X: Feature matrix (numpy array)
            feature_names: List of feature column names
        """
        feature_df = pd.DataFrame()
        
        # Numeric features (direct use)
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
        
        # Categorical features (need encoding)
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
                    # Create and fit new encoder
                    le = LabelEncoder()
                    # Fit on all possible categories to handle unseen values
                    le.fit(categories + ['Unknown'])
                    self.encoders[feat] = le
                
                if feat in self.encoders:
                    # Transform, handling unseen values
                    values = df[feat].fillna('Unknown').astype(str)
                    values = values.apply(lambda x: x if x in self.encoders[feat].classes_ else 'Unknown')
                    feature_df[f'{feat}_encoded'] = self.encoders[feat].transform(values)
        
        # Binary flags
        if 'Engineer_Flag' in df.columns:
            feature_df['Is_Repeat_Offender'] = df['Engineer_Flag'].apply(
                lambda x: 1 if 'Repeat' in str(x) else 0
            )
        
        if 'Aging_Status' in df.columns:
            feature_df['Is_Aged'] = df['Aging_Status'].apply(
                lambda x: 1 if '>30' in str(x) else (0.5 if '>14' in str(x) else 0)
            )
        
        # Embedding-based features (if available)
        if 'embedding' in df.columns:
            try:
                embeddings = np.vstack(df['embedding'].values)
                for i in range(min(10, embeddings.shape[1])):
                    feature_df[f'emb_dim_{i}'] = embeddings[:, i]
            except Exception:
                pass  # Skip if embeddings not available
        
        self.feature_columns = list(feature_df.columns)
        return feature_df.values, self.feature_columns
    
    def train(self, df, min_samples=50):
        """
        Train the recurrence prediction model on historical data.
        
        Args:
            df: DataFrame with historical ticket data including recurrence outcomes
            min_samples: Minimum samples required to train
            
        Returns:
            dict: Training metrics
        """
        logger.info("[Recurrence Predictor] Training predictive model...")
        
        # Determine target variable (actual recurrence)
        if 'Recurrence_Actual' not in df.columns:
            logger.warning("No Recurrence_Actual column found. Cannot train.")
            return {'error': 'No recurrence data available'}
        
        # Filter to rows with known outcomes
        train_df = df[df['Recurrence_Actual'].isin(['Yes', 'No'])].copy()
        
        if len(train_df) < min_samples:
            logger.warning(f"Insufficient training data ({len(train_df)} samples). Need {min_samples}+")
            return {'error': f'Need at least {min_samples} samples, have {len(train_df)}'}
        
        # Prepare features
        X, feature_names = self._prepare_features(train_df, fit_encoders=True)
        y = (train_df['Recurrence_Actual'] == 'Yes').astype(int)
        
        # Handle class imbalance
        recurrence_rate = y.mean()
        logger.info(f"  Historical recurrence rate: {recurrence_rate:.1%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
        
        # Train model (Gradient Boosting for better probability calibration)
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        try:
            auc_score = roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.5
        except Exception:
            auc_score = 0.5
        
        accuracy = (y_pred == y_test).mean()
        
        # Feature importance
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        self.model_metrics = {
            'accuracy': accuracy,
            'auc_roc': auc_score,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'recurrence_rate': recurrence_rate,
            'top_features': top_features
        }
        
        self.is_trained = True
        
        # Log results
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
        
        Args:
            df: DataFrame with ticket data
            
        Returns:
            DataFrame with prediction columns added
        """
        df = df.copy()
        
        # Initialize prediction columns
        df['AI_Recurrence_Probability'] = 0.0
        df['AI_Recurrence_Risk'] = 'Unknown'
        df['AI_Recurrence_Confidence'] = 'Low'
        
        if not self.is_trained:
            logger.info("[Recurrence Predictor] Model not trained, skipping predictions")
            return df
        
        try:
            # Prepare features
            X, _ = self._prepare_features(df, fit_encoders=False)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(X)[:, 1]
            
            df['AI_Recurrence_Probability'] = probabilities
            
            # Categorize risk level
            def categorize_risk(prob):
                if prob >= 0.7:
                    return '🔴 High Risk (>70%)'
                elif prob >= 0.5:
                    return '🟠 Elevated (50-70%)'
                elif prob >= 0.3:
                    return '🟡 Moderate (30-50%)'
                else:
                    return '🟢 Low (<30%)'
            
            df['AI_Recurrence_Risk'] = df['AI_Recurrence_Probability'].apply(categorize_risk)
            
            # Confidence based on model certainty
            def get_confidence(prob):
                certainty = abs(prob - 0.5) * 2
                if certainty >= 0.6:
                    return 'High'
                elif certainty >= 0.3:
                    return 'Medium'
                else:
                    return 'Low'
            
            df['AI_Recurrence_Confidence'] = df['AI_Recurrence_Probability'].apply(get_confidence)
            
            # Log summary
            high_risk = (df['AI_Recurrence_Probability'] >= 0.5).sum()
            logger.info(f"[Recurrence Predictor] Predictions complete:")
            logger.info(f"  → {high_risk} tickets flagged as high recurrence risk (≥50%)")
            logger.info(f"  → Average recurrence probability: {probabilities.mean():.1%}")
            
        except Exception as e:
            logger.warning(f"[Recurrence Predictor] Prediction failed: {e}")
        
        return df
    
    def save(self, model_path=None, encoders_path=None):
        """Save trained model and encoders to disk."""
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
        """Load trained model and encoders from disk."""
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
        
        Returns: List of contributing risk factors
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


# Global predictor instance
recurrence_predictor = RecurrencePredictor()


def apply_recurrence_predictions(df, train_if_possible=True):
    """
    Apply recurrence predictions to the dataframe.
    
    This function:
    1. Tries to load a pre-trained model
    2. If not available and data permits, trains a new model
    3. Applies predictions to all tickets
    
    Args:
        df: DataFrame with ticket data
        train_if_possible: Whether to train if no model exists
        
    Returns:
        DataFrame with prediction columns added
    """
    global recurrence_predictor
    
    logger.info("[Recurrence Predictor] Initializing AI-based recurrence prediction...")
    
    # Try to load existing model
    if recurrence_predictor.load():
        logger.info("  Using pre-trained model")
    elif train_if_possible:
        # Train new model if we have recurrence data
        if 'Recurrence_Actual' in df.columns:
            metrics = recurrence_predictor.train(df)
            if 'error' not in metrics:
                recurrence_predictor.save()
        else:
            logger.info("  No historical recurrence data - will train on next run with outcomes")
    
    # Apply predictions
    df = recurrence_predictor.predict(df)
    
    return df
