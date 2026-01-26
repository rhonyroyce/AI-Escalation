"""
Resolution Time Predictor - ML-based resolution time prediction.

Predicts resolution time for tickets based on historical patterns,
issue characteristics, and human-provided expectations.

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
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.category_stats = {}
        self.severity_stats = {}
        self.human_expectations = {}
        self.is_trained = False
        
        logger.info("[Resolution Predictor] Initialized")
    
    def _extract_features(self, row):
        """Extract features for resolution time prediction."""
        features = {}
        
        category = str(row.get('AI_Category', 'Unknown'))
        features['category_hash'] = hash(category) % 1000
        
        if category in self.category_stats:
            features['category_avg_days'] = self.category_stats[category]['mean']
            features['category_median_days'] = self.category_stats[category]['median']
        else:
            features['category_avg_days'] = 5.0
            features['category_median_days'] = 3.0
        
        severity = str(row.get('AI_Severity', row.get(COL_SEVERITY, 'Medium')))
        severity_map = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        features['severity_level'] = severity_map.get(severity, 2)
        
        summary = str(row.get(COL_SUMMARY, ''))
        features['text_length'] = len(summary)
        features['word_count'] = len(summary.split())
        
        complexity_keywords = ['complex', 'multiple', 'integration', 'migration', 'upgrade', 'critical']
        features['complexity_score'] = sum(1 for kw in complexity_keywords if kw in summary.lower())
        
        features['similar_resolution_days'] = row.get('Expected_Resolution_Days', 0) or 0
        features['similar_count'] = row.get('Similar_Ticket_Count', 0) or 0
        
        features['ai_confidence'] = row.get('AI_Confidence', 0.5) or 0.5
        
        recurrence = str(row.get('AI_Recurrence_Risk', 'Medium'))
        recurrence_map = {'High': 3, 'Medium': 2, 'Low': 1, 'Unknown': 2}
        features['recurrence_risk'] = recurrence_map.get(recurrence, 2)
        
        return features
    
    def train(self, df):
        """Train the resolution time predictor on historical data."""
        logger.info("[Resolution Predictor] Training model...")
        
        training_data = []
        
        for idx, row in df.iterrows():
            actual_days = self._calculate_resolution_days(row)
            if actual_days is None or actual_days <= 0 or actual_days > 365:
                continue
            
            features = self._extract_features(row)
            features['actual_days'] = actual_days
            training_data.append(features)
        
        if len(training_data) < 10:
            logger.warning("[Resolution Predictor] Insufficient training data (< 10 samples)")
            self.is_trained = False
            return False
        
        train_df = pd.DataFrame(training_data)
        
        # Build category statistics
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
        
        self.feature_columns = [c for c in train_df.columns if c != 'actual_days']
        X = train_df[self.feature_columns].values
        y = train_df['actual_days'].values
        
        try:
            # Use GPU-accelerated model when available
            use_gpu = USE_GPU and is_gpu_available()
            if use_gpu:
                logger.info("[Resolution Predictor] Training with GPU acceleration")
            
            self.model = GPURandomForestRegressor(
                use_gpu=use_gpu,
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            self.model.fit(X, y)
            self.is_trained = True
            
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
        """Calculate actual resolution days from ticket data."""
        try:
            issue_date = row.get(COL_DATETIME)
            resolution_date = row.get(COL_RESOLUTION_DATE)
            
            if pd.isna(issue_date) or pd.isna(resolution_date):
                return None
            
            if isinstance(issue_date, str):
                issue_date = pd.to_datetime(issue_date)
            if isinstance(resolution_date, str):
                resolution_date = pd.to_datetime(resolution_date)
            
            days = (resolution_date - issue_date).days
            return days if days >= 0 else None
            
        except Exception:
            return None
    
    def predict(self, row):
        """Predict resolution time for a single ticket."""
        category = str(row.get('AI_Category', 'Unknown'))
        
        if self.is_trained and self.model is not None:
            try:
                features = self._extract_features(row)
                X = np.array([[features[c] for c in self.feature_columns]])
                predicted = self.model.predict(X)[0]
                
                cat_std = self.category_stats.get(category, {}).get('std', 2.0)
                confidence = max(0.3, min(0.95, 1 - (cat_std / (predicted + 1))))
                
                return {
                    'predicted_days': max(0.5, round(predicted, 1)),
                    'confidence': confidence,
                    'method': 'ml'
                }
            except Exception as e:
                logger.debug(f"[Resolution Predictor] ML prediction failed: {e}")
        
        if category in self.category_stats:
            return {
                'predicted_days': round(self.category_stats[category]['median'], 1),
                'confidence': 0.5,
                'method': 'category_stats'
            }
        
        return {
            'predicted_days': 5.0,
            'confidence': 0.2,
            'method': 'default'
        }
    
    def set_human_expectations(self, expectations_dict):
        """Set human-provided expected resolution times by category."""
        self.human_expectations = expectations_dict
        logger.info(f"[Resolution Predictor] Loaded {len(expectations_dict)} human expectations")
    
    def process_all_tickets(self, df):
        """Process all tickets and add resolution time columns."""
        logger.info(f"[Resolution Predictor] Processing {len(df)} tickets...")
        
        # First, train on the data
        self.train(df)
        
        # Initialize columns
        df['Actual_Resolution_Days'] = None
        df['Predicted_Resolution_Days'] = None
        df['Human_Expected_Days'] = None
        df['Resolution_Prediction_Confidence'] = None
        df['Resolution_Prediction_Method'] = None
        
        for idx, row in df.iterrows():
            actual = self._calculate_resolution_days(row)
            df.at[idx, 'Actual_Resolution_Days'] = actual
            
            prediction = self.predict(row)
            df.at[idx, 'Predicted_Resolution_Days'] = prediction['predicted_days']
            df.at[idx, 'Resolution_Prediction_Confidence'] = prediction['confidence']
            df.at[idx, 'Resolution_Prediction_Method'] = prediction['method']
            
            category = str(row.get('AI_Category', 'Unknown'))
            if category in self.human_expectations:
                df.at[idx, 'Human_Expected_Days'] = self.human_expectations[category]
        
        has_actual = df['Actual_Resolution_Days'].notna().sum()
        has_predicted = df['Predicted_Resolution_Days'].notna().sum()
        has_expected = df['Human_Expected_Days'].notna().sum()
        
        logger.info(f"[Resolution Predictor] Complete:")
        logger.info(f"  → {has_actual} tickets with actual resolution times")
        logger.info(f"  → {has_predicted} tickets with ML predictions")
        logger.info(f"  → {has_expected} tickets with human expectations")
        
        return df
    
    def get_accuracy_metrics(self, df):
        """Calculate accuracy metrics comparing actual vs predicted."""
        valid = df.dropna(subset=['Actual_Resolution_Days', 'Predicted_Resolution_Days'])
        
        if len(valid) < 5:
            return None
        
        actual = valid['Actual_Resolution_Days'].values
        predicted = valid['Predicted_Resolution_Days'].values
        
        metrics = {
            'mae': np.mean(np.abs(predicted - actual)),
            'rmse': np.sqrt(np.mean((predicted - actual) ** 2)),
            'mape': np.mean(np.abs((actual - predicted) / (actual + 0.1))) * 100,
            'sample_count': len(valid),
            'correlation': np.corrcoef(actual, predicted)[0, 1] if len(valid) > 2 else 0
        }
        
        metrics['by_category'] = {}
        for category in valid['AI_Category'].dropna().unique():
            cat_data = valid[valid['AI_Category'] == category]
            if len(cat_data) >= 3:
                cat_actual = cat_data['Actual_Resolution_Days'].values
                cat_pred = cat_data['Predicted_Resolution_Days'].values
                metrics['by_category'][category] = {
                    'mae': np.mean(np.abs(cat_pred - cat_actual)),
                    'count': len(cat_data)
                }
        
        return metrics


# Global resolution time predictor instance
resolution_time_predictor = None


def apply_resolution_time_prediction(df, human_expectations=None):
    """Apply ML-based resolution time prediction to the dataframe."""
    global resolution_time_predictor
    
    logger.info("[Resolution Predictor] Initializing resolution time prediction...")
    
    resolution_time_predictor = ResolutionTimePredictor()
    
    if human_expectations:
        resolution_time_predictor.set_human_expectations(human_expectations)
    
    df = resolution_time_predictor.process_all_tickets(df)
    
    metrics = resolution_time_predictor.get_accuracy_metrics(df)
    if metrics:
        logger.info(f"[Resolution Predictor] Accuracy Metrics:")
        logger.info(f"  → MAE: {metrics['mae']:.2f} days")
        logger.info(f"  → RMSE: {metrics['rmse']:.2f} days")
        logger.info(f"  → Correlation: {metrics['correlation']:.2f}")
    
    return df
