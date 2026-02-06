"""
Strategic Friction Scoring Engine.
Multi-variable risk scoring based on McKinsey framework.
"""

import logging
import pandas as pd
from typing import Dict

from ..core.config import (
    WEIGHTS, COL_SEVERITY, COL_TYPE, COL_ORIGIN, COL_IMPACT,
    COL_SUMMARY, COL_ENGINEER, COL_DATETIME, COL_ROOT_CAUSE,
    COL_RECURRENCE_RISK, ROOT_CAUSE_CATEGORIES, REQUIRED_COLUMNS
)
from ..feedback.price_catalog import get_price_catalog
from ..core.utils import validate_columns

logger = logging.getLogger(__name__)


def calculate_strategic_friction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multi-variable risk scoring to the dataframe.
    
    Scoring formula:
    Score = Base_Severity × Type_Multiplier × Origin_Multiplier × Impact_Multiplier
    
    Returns:
        DataFrame with Strategic_Friction_Score and additional metric columns
    """
    logger.info("[Strategic Engine] Applying Multi-Variable Risk Scoring...")
    df = df.copy()
    
    # Validate columns
    validate_columns(df, REQUIRED_COLUMNS)
   
    # Normalize with safe column access
    df['Severity_Norm'] = df[COL_SEVERITY].astype(str).str.title().str.strip() if COL_SEVERITY in df.columns else 'Default'
    df['Type_Norm'] = df[COL_TYPE].astype(str).str.title().str.strip() if COL_TYPE in df.columns else ''
    df['Origin_Norm'] = df[COL_ORIGIN].astype(str).str.title().str.strip() if COL_ORIGIN in df.columns else ''
    df['Impact_Norm'] = df[COL_IMPACT].fillna('None').astype(str).str.title().str.strip() if COL_IMPACT in df.columns else 'None'

    # Load price catalog for financial impact (always reload to pick up changes)
    price_catalog = get_price_catalog()
    price_catalog.load_catalog()  # Always reload to ensure latest values from price_catalog.xlsx

    def get_score(row):
        """Calculate strategic friction score for a row."""
        # 1. Base Score (Severity)
        base = WEIGHTS['BASE_SEVERITY'].get(row['Severity_Norm'], 5)
       
        # 2. Type Multiplier (Escalation > Concern)
        m_type = 1.0
        if 'Escalation' in row['Type_Norm']: 
            m_type = WEIGHTS['TYPE_MULTIPLIER']['Escalations']
        elif 'Lesson' in row['Type_Norm']: 
            m_type = 0.0  # Lessons are not risks
       
        # 3. Origin Multiplier (External > Internal)
        m_origin = WEIGHTS['ORIGIN_MULTIPLIER'].get(row['Origin_Norm'], 1.0)
       
        # 4. Impact Multiplier
        m_impact = 1.0
        if 'High' in row['Impact_Norm']: 
            m_impact = WEIGHTS['IMPACT_MULTIPLIER']['High']
       
        return base * m_type * m_origin * m_impact

    def get_financial_impact(row):
        """Calculate financial impact using price catalog."""
        category = row.get('AI_Category', 'Unclassified')
        severity = row['Severity_Norm']
        origin = row['Origin_Norm']
        description = str(row.get(COL_SUMMARY, ''))
        
        impact = price_catalog.calculate_financial_impact(
            category=category,
            severity=severity,
            origin=origin,
            description=description,
            delay_hours=4.0
        )
        return impact['total_impact']

    df['Strategic_Friction_Score'] = df.apply(get_score, axis=1)
    
    # Add financial impact column
    df['Financial_Impact'] = df.apply(get_financial_impact, axis=1)
    logger.info(f"  → Total estimated financial impact: ${df['Financial_Impact'].sum():,.2f}")
    
    # Additional metrics
    df = _add_risk_tier(df)
    df = _add_engineer_accountability(df)
    df = _add_aging_status(df)
    df = _add_human_error_flags(df)
    df = _add_root_cause_classification(df)
    df = _add_pm_recurrence_risk(df)
    df = _add_priority_and_actions(df)
    
    return df


def _add_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk tier based on friction score."""
    def get_risk_tier(score):
        if score >= 150:
            return "Critical"
        elif score >= 75:
            return "High"
        elif score >= 25:
            return "Medium"
        else:
            return "Low"
    
    df['Risk_Tier'] = df['Strategic_Friction_Score'].apply(get_risk_tier)
    return df


def _add_engineer_accountability(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineer tracking columns."""
    if COL_ENGINEER in df.columns:
        df['Engineer'] = df[COL_ENGINEER].fillna('Unknown').astype(str).str.strip()
        
        # Count issues per engineer
        engineer_counts = df[df['Type_Norm'].isin(['Escalations', 'Concerns'])].groupby('Engineer').size()
        df['Engineer_Issue_Count'] = df['Engineer'].map(engineer_counts).fillna(0).astype(int)
        
        # Calculate engineer's total friction
        engineer_friction = df[df['Type_Norm'].isin(['Escalations', 'Concerns'])].groupby('Engineer')['Strategic_Friction_Score'].sum()
        df['Engineer_Total_Friction'] = df['Engineer'].map(engineer_friction).fillna(0)
        
        # Flag repeat offenders
        df['Engineer_Flag'] = df['Engineer_Issue_Count'].apply(
            lambda x: '🔴 Repeat Offender' if x >= 5 else ('🟡 Multiple Issues' if x >= 3 else '')
        )
        
        logger.info(f"  → Engineer accountability tracked for {df['Engineer'].nunique()} unique engineers")
    
    return df


def _add_aging_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add issue aging tracking."""
    if COL_DATETIME in df.columns:
        df['Issue_Date'] = pd.to_datetime(df[COL_DATETIME], errors='coerce')
        df['Days_Since_Issue'] = (pd.Timestamp.now() - df['Issue_Date']).dt.days
        df['Days_Since_Issue'] = df['Days_Since_Issue'].fillna(-1).astype(int)
        
        df['Aging_Status'] = df['Days_Since_Issue'].apply(
            lambda x: '🔴 >30 days' if x > 30 else ('🟡 >14 days' if x > 14 else ('🟢 Recent' if x >= 0 else 'Unknown'))
        )
    
    return df


def _add_human_error_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add human error flag based on origin and type."""
    df['Is_Human_Error'] = (
        (df['Origin_Norm'] == 'Internal') & 
        (df['Type_Norm'].isin(['Escalations', 'Concerns']))
    ).map({True: 'Yes', False: 'No'})
    
    return df


def _add_root_cause_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Add root cause classification."""
    df['Root_Cause_Category'] = 'Unclassified'
    df['Root_Cause_Original'] = ''
    
    if COL_ROOT_CAUSE in df.columns:
        df['Root_Cause_Original'] = df[COL_ROOT_CAUSE].fillna('').astype(str).str.strip()
        
        def classify_root_cause(root_cause_text):
            if pd.isna(root_cause_text) or not root_cause_text:
                return 'Unclassified'
            text_lower = str(root_cause_text).lower()
            
            for category, keywords in ROOT_CAUSE_CATEGORIES.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return category
            return 'Other'
        
        df['Root_Cause_Category'] = df['Root_Cause_Original'].apply(classify_root_cause)
        
        # Update human error flag based on root cause
        human_error_mask = df['Root_Cause_Category'] == 'Human Error'
        df.loc[human_error_mask, 'Is_Human_Error'] = 'Yes'
        
        external_mask = df['Root_Cause_Category'] == 'External Party'
        df.loc[external_mask, 'Is_Human_Error'] = 'External'
        
        logger.info(f"  → Root cause classified: {df['Root_Cause_Category'].value_counts().to_dict()}")
    
    return df


def _add_pm_recurrence_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add PM recurrence risk normalization."""
    df['PM_Recurrence_Risk'] = 'Unknown'
    df['PM_Recurrence_Risk_Norm'] = 'Unknown'
    
    if COL_RECURRENCE_RISK in df.columns:
        df['PM_Recurrence_Risk'] = df[COL_RECURRENCE_RISK].fillna('Unknown').astype(str).str.strip()
        
        def normalize_recurrence_risk(risk_text):
            if pd.isna(risk_text) or not risk_text:
                return 'Unknown'
            text_lower = str(risk_text).lower().strip()
            
            if text_lower in ['high', 'yes', 'likely', 'probable', 'very high']:
                return 'High'
            elif text_lower in ['medium', 'moderate', 'possible', 'maybe']:
                return 'Medium'
            elif text_lower in ['low', 'no', 'unlikely', 'none', 'very low', 'minimal']:
                return 'Low'
            else:
                return 'Unknown'
        
        df['PM_Recurrence_Risk_Norm'] = df['PM_Recurrence_Risk'].apply(normalize_recurrence_risk)
        logger.info(f"  → PM Recurrence risk: {df['PM_Recurrence_Risk_Norm'].value_counts().to_dict()}")
    
    return df


def _add_priority_and_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Add priority score and action required columns."""
    # Priority Score
    df['Priority_Score'] = df['Strategic_Friction_Score']
    if 'Days_Since_Issue' in df.columns:
        df['Priority_Score'] = df['Priority_Score'] * (1 + (30 - df['Days_Since_Issue'].clip(0, 30)) / 100)
    
    # Action Required
    def get_action_required(row):
        actions = []
        if row['Risk_Tier'] in ['Critical', 'High']:
            actions.append('Immediate Review')
        if row.get('Is_Human_Error') == 'Yes':
            actions.append('Training Review')
        if row.get('Engineer_Flag') and 'Repeat' in str(row.get('Engineer_Flag', '')):
            actions.append('Performance Discussion')
        if row.get('Learning_Status') == 'Confirmed Repeat':
            actions.append('Process Fix Required')
        return ' | '.join(actions) if actions else 'Monitor'
    
    df['Action_Required'] = df.apply(get_action_required, axis=1)
    
    return df
