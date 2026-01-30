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
            
            # Adjust n_estimators based on sample size (avoid cuML bins warning)
            n_samples = len(training_data)
            n_estimators = min(50, max(10, n_samples // 2))
            
            self.model = GPURandomForestRegressor(
                use_gpu=use_gpu,
                n_estimators=n_estimators,
                max_depth=min(8, max(3, n_samples // 5)),  # Scale depth with samples
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
        """Predict resolution time for a single ticket.
        
        Priority order:
        1. ML model prediction (if trained and successful)
        2. Category statistics from training data (median)
        3. Global statistics from training data
        4. Calibrated heuristics (scaled to match learned data)
        """
        category = str(row.get('AI_Category', 'Unknown'))
        
        # Priority 1: Try ML model
        if self.is_trained and self.model is not None:
            try:
                features = self._extract_features(row)
                X = np.array([[features[c] for c in self.feature_columns]])
                predicted = self.model.predict(X)[0]
                
                # Sanity check: prediction should be reasonable
                if 0.1 <= predicted <= 100:
                    cat_std = self.category_stats.get(category, {}).get('std', 2.0)
                    confidence = max(0.3, min(0.95, 1 - (cat_std / (predicted + 1))))
                    
                    return {
                        'predicted_days': max(0.5, round(predicted, 1)),
                        'confidence': confidence,
                        'method': 'ml'
                    }
            except Exception as e:
                logger.debug(f"[Resolution Predictor] ML prediction failed: {e}")
        
        # Priority 2: Use learned category statistics
        if category in self.category_stats:
            cat_stat = self.category_stats[category]
            predicted_days = cat_stat.get('median', cat_stat.get('mean', 3.0))
            sample_count = cat_stat.get('count', 1)
            confidence = min(0.7, 0.4 + (sample_count / 50))
            
            return {
                'predicted_days': max(0.5, round(predicted_days, 1)),
                'confidence': confidence,
                'method': 'category_stats'
            }
        
        # Priority 3: Use global stats from training data for unknown category
        if self.category_stats:
            all_medians = [s.get('median', s.get('mean')) for s in self.category_stats.values() 
                          if s.get('median') or s.get('mean')]
            if all_medians:
                global_median = np.median(all_medians)
                return {
                    'predicted_days': max(0.5, round(global_median, 1)),
                    'confidence': 0.35,
                    'method': 'global_stats'
                }
        
        # Priority 4: Calibrated heuristic prediction
        return self._predict_heuristic(row, category)
    
    def _predict_heuristic(self, row, category):
        """
        Heuristic-based resolution time prediction when ML model isn't trained.
        Uses category complexity, severity, and other indicators.
        Now heavily calibrated to actual data when available.
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
        
        # Adjust for severity
        severity = str(row.get('Severity_Norm', row.get(COL_SEVERITY, 'Medium'))).lower()
        if 'critical' in severity or 's1' in severity:
            base_days *= 0.6  # Faster resolution for critical issues
        elif 'high' in severity or 's2' in severity:
            base_days *= 0.8
        elif 'low' in severity:
            base_days *= 1.3  # Lower priority takes longer
        
        # Adjust for complexity (based on summary text)
        summary = str(row.get(COL_SUMMARY, ''))
        complexity_indicators = ['multiple', 'complex', 'integration', 'migration', 'several', 'widespread']
        simple_indicators = ['simple', 'quick', 'minor', 'single', 'easy']
        
        if any(ind in summary.lower() for ind in complexity_indicators):
            base_days *= 1.4
        elif any(ind in summary.lower() for ind in simple_indicators):
            base_days *= 0.7
        
        # Adjust for recurrence risk (repeat issues may be systemic and harder to fix)
        recurrence_risk = str(row.get('AI_Recurrence_Risk', '')).lower()
        if 'high' in recurrence_risk:
            base_days *= 1.2
        elif 'low' in recurrence_risk:
            base_days *= 0.9
        
        # Add some variability based on friction score
        friction = row.get('Strategic_Friction_Score', 50)
        if pd.notna(friction):
            base_days *= (0.9 + (float(friction) / 500))  # 0.9 to 1.1 multiplier
        
        # Round to 1 decimal place, minimum 0.5 days
        predicted_days = max(0.5, round(base_days, 1))
        
        # Confidence based on how specific our estimate is
        confidence = 0.4  # Heuristic confidence is moderate
        if category in category_base_days:
            confidence = 0.55
        
        return {
            'predicted_days': predicted_days,
            'confidence': confidence,
            'method': 'heuristic'
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
        
        actual = np.array(valid['Actual_Resolution_Days'].values, dtype=float)
        predicted = np.array(valid['Predicted_Resolution_Days'].values, dtype=float)
        
        # Calculate correlation safely
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
            'mape': np.mean(np.abs((actual - predicted) / (actual + 0.1))) * 100,
            'sample_count': len(valid),
            'correlation': correlation
        }
        
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
