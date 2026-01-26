"""
Chart Generator for McKinsey-style executive visualizations.

This module provides the ChartGenerator class that creates 15 executive-quality
charts for escalation analysis reporting.
"""

import os
import logging
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# McKinsey Blue - primary brand color
MC_BLUE = '#004C97'

# McKinsey color palette for charts
MCKINSEY_COLORS = [
    '#004C97',  # Primary Blue
    '#0078D4',  # Light Blue
    '#00A6ED',  # Cyan
    '#6CC24A',  # Green
    '#F0AD4E',  # Orange/Warning
    '#D9534F',  # Red/Danger
    '#9B59B6',  # Purple
    '#3498DB',  # Sky Blue
]

# Risk tier color mapping
RISK_TIER_COLORS = {
    'Critical': '#D9534F',
    'High': '#F0AD4E',
    'Medium': '#5BC0DE',
    'Low': '#5CB85C'
}

# Root cause color mapping
ROOT_CAUSE_COLORS = {
    'Human Error': '#D9534F',
    'External Party': '#5BC0DE',
    'Process Gap': '#F0AD4E',
    'System/Technical': '#5CB85C',
    'Training Gap': '#9B59B6',
    'Communication': '#3498DB',
    'Resource': '#E74C3C',
    'Other': '#95A5A6',
    'Unclassified': '#BDC3C7'
}


class ChartGenerator:
    """
    Generates McKinsey-style executive charts for escalation analysis.
    
    Charts include:
    1. Strategic Friction Pareto
    2. Risk Origin Distribution
    3. Learning Integrity (Recidivism)
    4. 7-Day Rolling Trend
    5. Category × Severity Heatmap
    6. Top Engineers by Friction
    7. Engineer Learning Behavior
    8. LOB Risk Exposure
    9. LOB Strategic Matrix
    10. LOB Issue Category Breakdown
    11. Root Cause Analysis
    12. PM Prediction Accuracy
    13. AI Recurrence Prediction
    14. Financial Impact
    15. Resolution Time Comparison
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.paths: List[Optional[str]] = [None] * 15
    
    def generate_all(self, df: pd.DataFrame) -> List[Optional[str]]:
        """
        Generate all 15 executive charts.
        
        Args:
            df: DataFrame with processed escalation data
            
        Returns:
            List of file paths to generated charts (None for skipped charts)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib and seaborn required for chart generation")
            return self.paths
        
        logger.info("[Vis Engine] Generating 15 Executive Charts...")
        
        # Set global style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})
        
        # Generate each chart (with error handling)
        self._chart_01_friction_pareto(df, plt, sns)
        self._chart_02_risk_origin(df, plt)
        self._chart_03_learning_integrity(df, plt)
        self._chart_04_risk_trend(df, plt, sns)
        self._chart_05_severity_heatmap(df, plt, sns)
        self._chart_06_engineer_friction(df, plt)
        self._chart_07_engineer_learning(df, plt)
        self._chart_08_lob_friction(df, plt)
        self._chart_09_lob_matrix(df, plt)
        self._chart_10_lob_categories(df, plt)
        self._chart_11_root_cause(df, plt)
        self._chart_12_pm_accuracy(df, plt)
        self._chart_13_ai_recurrence(df, plt)
        self._chart_14_financial_impact(df, plt, sns)
        self._chart_15_resolution_time(df, plt)
        
        return self.paths
    
    def _chart_01_friction_pareto(self, df, plt, sns):
        """Chart 1: Top Strategic Friction Sources."""
        try:
            plt.figure(figsize=(8, 5))
            risk_df = df[df['Strategic_Friction_Score'] > 0]
            if not risk_df.empty and 'AI_Category' in df.columns:
                cat_scores = risk_df.groupby('AI_Category')['Strategic_Friction_Score'].sum().nlargest(10)
                sns.barplot(x=cat_scores.values, y=cat_scores.index, palette="Reds_r")
                plt.title('Top Strategic Friction Sources (AI Classified)', fontweight='bold')
                plt.xlabel('Weighted Risk Score')
                plt.tight_layout()
                path = os.path.join(self.plot_dir, "friction.png")
                plt.savefig(path)
                self.paths[0] = path
        except Exception as e:
            logger.warning(f"Chart 1 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_02_risk_origin(self, df, plt):
        """Chart 2: Risk Origin Distribution."""
        try:
            from ..core.config import COL_ORIGIN
            plt.figure(figsize=(6, 4))
            if COL_ORIGIN in df.columns:
                ext = df[df[COL_ORIGIN]=='External']['Strategic_Friction_Score'].sum()
                int_ = df[df[COL_ORIGIN]=='Internal']['Strategic_Friction_Score'].sum()
                if ext + int_ > 0:
                    plt.pie([ext, int_], labels=['External', 'Internal'], 
                           colors=['#D9534F', MC_BLUE], autopct='%1.1f%%', startangle=90)
                    plt.title('Risk Origin (Weighted)', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "origin.png")
                    plt.savefig(path)
                    self.paths[1] = path
        except Exception as e:
            logger.warning(f"Chart 2 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_03_learning_integrity(self, df, plt):
        """Chart 3: Learning Integrity (Recidivism)."""
        try:
            plt.figure(figsize=(7, 4))
            high_conf = df['Learning_Status'].astype(str).str.contains('REPEAT OFFENSE').sum()
            medium_conf = df['Learning_Status'].astype(str).str.contains('POSSIBLE REPEAT').sum()
            new_issues = len(df) - high_conf - medium_conf
            
            labels, sizes, colors = [], [], []
            if new_issues > 0:
                labels.append(f'New Issues ({new_issues})')
                sizes.append(new_issues)
                colors.append('#5CB85C')
            if high_conf > 0:
                labels.append(f'Confirmed Repeats ({high_conf})')
                sizes.append(high_conf)
                colors.append('#D9534F')
            if medium_conf > 0:
                labels.append(f'Possible Repeats ({medium_conf})')
                sizes.append(medium_conf)
                colors.append('#F0AD4E')
            
            if sizes:
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=60)
                plt.title('Institutional Learning Integrity', fontweight='bold')
                plt.tight_layout()
                path = os.path.join(self.plot_dir, "learning.png")
                plt.savefig(path)
                self.paths[2] = path
        except Exception as e:
            logger.warning(f"Chart 3 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_04_risk_trend(self, df, plt, sns):
        """Chart 4: 7-Day Rolling Risk Trend."""
        try:
            from ..core.config import COL_DATETIME
            plt.figure(figsize=(10, 4))
            if COL_DATETIME in df.columns:
                df_temp = df.copy()
                df_temp['Date'] = pd.to_datetime(df_temp[COL_DATETIME], errors='coerce')
                daily = df_temp.groupby('Date')['Strategic_Friction_Score'].sum().rolling(7, min_periods=1).mean()
                if not daily.empty:
                    sns.lineplot(data=daily, color=MC_BLUE, linewidth=2.5)
                    plt.fill_between(daily.index, daily.values, alpha=0.1, color=MC_BLUE)
                    plt.title('7-Day Rolling Risk Trend', fontweight='bold')
                    plt.ylabel('Avg Daily Friction')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "trend.png")
                    plt.savefig(path)
                    self.paths[3] = path
        except Exception as e:
            logger.warning(f"Chart 4 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_05_severity_heatmap(self, df, plt, sns):
        """Chart 5: Category × Severity Heatmap."""
        try:
            plt.figure(figsize=(10, 6))
            if 'AI_Category' in df.columns and 'Severity_Norm' in df.columns:
                heatmap_data = pd.crosstab(
                    df['AI_Category'], df['Severity_Norm'],
                    values=df['Strategic_Friction_Score'], aggfunc='sum'
                ).fillna(0)
                
                severity_order = ['Critical', 'Major', 'Minor', 'Default']
                existing_cols = [c for c in severity_order if c in heatmap_data.columns]
                heatmap_data = heatmap_data[existing_cols]
                
                if not heatmap_data.empty:
                    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                               linewidths=0.5, cbar_kws={'label': 'Friction Score'})
                    plt.title('Risk Heatmap: Category × Severity', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "heatmap.png")
                    plt.savefig(path)
                    self.paths[4] = path
        except Exception as e:
            logger.warning(f"Chart 5 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_06_engineer_friction(self, df, plt):
        """Chart 6: Top Engineers by Friction."""
        try:
            plt.figure(figsize=(9, 5))
            if 'Engineer' in df.columns and 'Strategic_Friction_Score' in df.columns:
                df_issues = df[df['Type_Norm'].isin(['Escalations', 'Concerns'])] if 'Type_Norm' in df.columns else df
                if not df_issues.empty:
                    engineer_friction = df_issues.groupby('Engineer').agg({
                        'Strategic_Friction_Score': 'sum',
                        'Engineer_Is_Repeat_Offender': 'first'
                    }).reset_index()
                    engineer_friction = engineer_friction.nlargest(10, 'Strategic_Friction_Score')
                    
                    colors = ['#D9534F' if row.get('Engineer_Is_Repeat_Offender', False) else MC_BLUE 
                             for _, row in engineer_friction.iterrows()]
                    
                    plt.barh(engineer_friction['Engineer'], 
                            engineer_friction['Strategic_Friction_Score'], color=colors)
                    plt.xlabel('Total Friction Score')
                    plt.title('Top 10 Engineers by Friction Contribution', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "engineer_friction.png")
                    plt.savefig(path)
                    self.paths[5] = path
        except Exception as e:
            logger.warning(f"Chart 6 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_07_engineer_learning(self, df, plt):
        """Chart 7: Engineer Learning Behavior."""
        try:
            plt.figure(figsize=(8, 5))
            if 'Engineer_Learning_Score' in df.columns:
                df_issues = df[df['Type_Norm'].isin(['Escalations', 'Concerns'])] if 'Type_Norm' in df.columns else df
                if not df_issues.empty:
                    engineer_learning = df_issues.groupby('Engineer')['Engineer_Learning_Score'].first().dropna()
                    if len(engineer_learning) > 0:
                        categories = {
                            'Active Learners (≥0.8)': (engineer_learning >= 0.8).sum(),
                            'Moderate (0.4-0.8)': ((engineer_learning >= 0.4) & (engineer_learning < 0.8)).sum(),
                            'Low (0.1-0.4)': ((engineer_learning >= 0.1) & (engineer_learning < 0.4)).sum(),
                            'No Lessons (0)': (engineer_learning == 0).sum()
                        }
                        categories = {k: v for k, v in categories.items() if v > 0}
                        
                        if categories:
                            colors_learning = ['#5CB85C', '#5BC0DE', '#F0AD4E', '#D9534F'][:len(categories)]
                            plt.pie(list(categories.values()), labels=list(categories.keys()),
                                   colors=colors_learning, autopct='%1.1f%%', startangle=90)
                            plt.title('Engineer Learning Behavior Distribution', fontweight='bold')
                            plt.tight_layout()
                            path = os.path.join(self.plot_dir, "engineer_learning.png")
                            plt.savefig(path)
                            self.paths[6] = path
        except Exception as e:
            logger.warning(f"Chart 7 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_08_lob_friction(self, df, plt):
        """Chart 8: LOB Risk Exposure."""
        try:
            plt.figure(figsize=(10, 6))
            if 'LOB' in df.columns and 'LOB_Risk_Tier' in df.columns:
                df_issues = df[df['Type_Norm'].isin(['Escalations', 'Concerns'])] if 'Type_Norm' in df.columns else df
                if not df_issues.empty:
                    lob_friction = df_issues.groupby('LOB').agg({
                        'Strategic_Friction_Score': 'sum',
                        'LOB_Risk_Tier': 'first'
                    }).reset_index()
                    lob_friction = lob_friction.nlargest(10, 'Strategic_Friction_Score')
                    
                    colors = [RISK_TIER_COLORS.get(row['LOB_Risk_Tier'], MC_BLUE) 
                             for _, row in lob_friction.iterrows()]
                    
                    plt.barh(lob_friction['LOB'], lob_friction['Strategic_Friction_Score'], color=colors)
                    plt.xlabel('Total Strategic Friction')
                    plt.title('LOB Risk Exposure (by Friction)', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "lob_friction.png")
                    plt.savefig(path)
                    self.paths[7] = path
        except Exception as e:
            logger.warning(f"Chart 8 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_09_lob_matrix(self, df, plt):
        """Chart 9: LOB Strategic Matrix (Efficiency vs Learning)."""
        try:
            plt.figure(figsize=(10, 7))
            if 'LOB' in df.columns and 'LOB_Efficiency_Score' in df.columns:
                df_unique = df.drop_duplicates(subset=['LOB'])
                df_unique = df_unique[df_unique['LOB'] != 'Unknown']
                
                if len(df_unique) > 0:
                    colors = [RISK_TIER_COLORS.get(tier, MC_BLUE) for tier in df_unique['LOB_Risk_Tier']]
                    sizes = (df_unique['LOB_Total_Friction'] / df_unique['LOB_Total_Friction'].max() * 1000) + 100
                    
                    plt.scatter(
                        df_unique['LOB_Efficiency_Score'],
                        df_unique['LOB_Learning_Rate'] * 100,
                        s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=1
                    )
                    
                    for _, row in df_unique.iterrows():
                        plt.annotate(row['LOB'][:15], 
                                    (row['LOB_Efficiency_Score'], row['LOB_Learning_Rate'] * 100),
                                    xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    plt.xlabel('Operational Efficiency Score (0-100)')
                    plt.ylabel('Learning Rate (%)')
                    plt.title('LOB Strategic Matrix', fontweight='bold')
                    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
                    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
                    plt.xlim(0, 100)
                    plt.ylim(0, 100)
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "lob_matrix.png")
                    plt.savefig(path)
                    self.paths[8] = path
        except Exception as e:
            logger.warning(f"Chart 9 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_10_lob_categories(self, df, plt):
        """Chart 10: LOB Issue Category Breakdown."""
        try:
            plt.figure(figsize=(12, 6))
            if 'LOB' in df.columns and 'AI_Category' in df.columns:
                lob_category = pd.crosstab(df['LOB'], df['AI_Category'])
                lob_totals = lob_category.sum(axis=1).nlargest(8)
                lob_category = lob_category.loc[lob_totals.index]
                cat_totals = lob_category.sum(axis=0).nlargest(6)
                lob_category = lob_category[cat_totals.index]
                
                if not lob_category.empty:
                    lob_category.plot(kind='barh', stacked=True, 
                                     color=MCKINSEY_COLORS[:len(lob_category.columns)],
                                     figsize=(12, 6), width=0.7)
                    plt.xlabel('Number of Issues')
                    plt.ylabel('Line of Business')
                    plt.title('LOB Issue Breakdown by Category', fontweight='bold')
                    plt.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "lob_categories.png")
                    plt.savefig(path)
                    self.paths[9] = path
        except Exception as e:
            logger.warning(f"Chart 10 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_11_root_cause(self, df, plt):
        """Chart 11: Root Cause Analysis (Donut)."""
        try:
            plt.figure(figsize=(9, 6))
            if 'Root_Cause_Category' in df.columns:
                counts = df['Root_Cause_Category'].value_counts()
                counts = counts[counts >= 1]
                
                if not counts.empty:
                    colors = [ROOT_CAUSE_COLORS.get(cat, '#95A5A6') for cat in counts.index]
                    plt.pie(counts.values, labels=counts.index, colors=colors,
                           autopct='%1.1f%%', startangle=90, pctdistance=0.75)
                    
                    from matplotlib.patches import Circle
                    centre_circle = Circle((0, 0), 0.50, fc='white')
                    plt.gca().add_patch(centre_circle)
                    
                    plt.title('Root Cause Distribution', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "root_cause.png")
                    plt.savefig(path)
                    self.paths[10] = path
        except Exception as e:
            logger.warning(f"Chart 11 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_12_pm_accuracy(self, df, plt):
        """Chart 12: PM Prediction Accuracy."""
        try:
            plt.figure(figsize=(9, 5))
            if 'PM_Prediction_Accuracy' in df.columns:
                accuracy = df['PM_Prediction_Accuracy'].value_counts()
                accuracy = accuracy[~accuracy.index.isin(['N/A', 'No Prediction'])]
                
                if not accuracy.empty:
                    colors = []
                    for cat in accuracy.index:
                        if 'Correct' in cat:
                            colors.append('#5CB85C')
                        elif 'MISSED' in cat:
                            colors.append('#D9534F')
                        elif 'Overestimate' in cat:
                            colors.append('#F0AD4E')
                        else:
                            colors.append('#5BC0DE')
                    
                    plt.barh(accuracy.index, accuracy.values, color=colors)
                    plt.xlabel('Number of Tickets')
                    plt.title('PM Recurrence Prediction Accuracy', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "pm_accuracy.png")
                    plt.savefig(path)
                    self.paths[11] = path
        except Exception as e:
            logger.warning(f"Chart 12 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_13_ai_recurrence(self, df, plt):
        """Chart 13: AI Recurrence Prediction Distribution."""
        try:
            plt.figure(figsize=(8, 5))
            if 'AI_Recurrence_Risk' in df.columns:
                risk_counts = df['AI_Recurrence_Risk'].value_counts()
                if not risk_counts.empty:
                    colors = []
                    for cat in risk_counts.index:
                        if 'High' in str(cat):
                            colors.append('#D9534F')
                        elif 'Elevated' in str(cat):
                            colors.append('#F0AD4E')
                        elif 'Moderate' in str(cat):
                            colors.append('#5BC0DE')
                        else:
                            colors.append('#5CB85C')
                    
                    plt.bar(risk_counts.index, risk_counts.values, color=colors)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Number of Tickets')
                    plt.title('AI Recurrence Risk Prediction', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "ai_recurrence.png")
                    plt.savefig(path)
                    self.paths[12] = path
        except Exception as e:
            logger.warning(f"Chart 13 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_14_financial_impact(self, df, plt, sns):
        """Chart 14: Financial Impact by Category."""
        try:
            plt.figure(figsize=(10, 5))
            if 'Financial_Impact' in df.columns and 'AI_Category' in df.columns:
                impact = df.groupby('AI_Category')['Financial_Impact'].sum().nlargest(10)
                if not impact.empty:
                    sns.barplot(x=impact.values, y=impact.index, palette="Blues_r")
                    plt.xlabel('Financial Impact ($)')
                    plt.title('Financial Impact by Category', fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(self.plot_dir, "financial.png")
                    plt.savefig(path)
                    self.paths[13] = path
        except Exception as e:
            logger.warning(f"Chart 14 Skipped: {e}")
        finally:
            plt.close()
    
    def _chart_15_resolution_time(self, df, plt):
        """Chart 15: Resolution Time Comparison (Actual vs Predicted vs Expected)."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            has_data = False
            
            # Left: Bar chart by category
            if 'AI_Category' in df.columns:
                comparison_data = []
                for cat in df['AI_Category'].unique():
                    cat_df = df[df['AI_Category'] == cat]
                    if len(cat_df) < 2:
                        continue
                    
                    actual = cat_df.get('Actual_Resolution_Days', pd.Series()).mean()
                    predicted = cat_df.get('Predicted_Resolution_Days', pd.Series()).mean()
                    expected = cat_df.get('Human_Expected_Days', pd.Series()).mean()
                    
                    if pd.notna(actual) or pd.notna(predicted):
                        comparison_data.append({
                            'Category': cat[:20],
                            'Actual': actual if pd.notna(actual) else 0,
                            'Predicted': predicted if pd.notna(predicted) else 0,
                            'Expected': expected if pd.notna(expected) else 0
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df = comp_df.nlargest(8, 'Actual')
                    
                    x = np.arange(len(comp_df))
                    width = 0.25
                    
                    axes[0].bar(x - width, comp_df['Actual'], width, label='Actual', color='#004C97')
                    axes[0].bar(x, comp_df['Predicted'], width, label='Predicted', color='#F0AD4E')
                    axes[0].bar(x + width, comp_df['Expected'], width, label='Expected', color='#5CB85C')
                    
                    axes[0].set_xticks(x)
                    axes[0].set_xticklabels(comp_df['Category'], rotation=45, ha='right')
                    axes[0].set_ylabel('Days')
                    axes[0].set_title('Resolution Time by Category', fontweight='bold')
                    axes[0].legend()
                    has_data = True
            
            # Right: Scatter plot (Predicted vs Actual)
            if 'Predicted_Resolution_Days' in df.columns and 'Actual_Resolution_Days' in df.columns:
                valid = df[df['Predicted_Resolution_Days'].notna() & df['Actual_Resolution_Days'].notna()]
                if len(valid) > 5:
                    axes[1].scatter(valid['Actual_Resolution_Days'], valid['Predicted_Resolution_Days'],
                                   alpha=0.5, c='#004C97')
                    
                    # Add perfect prediction line
                    max_val = max(valid['Actual_Resolution_Days'].max(), valid['Predicted_Resolution_Days'].max())
                    axes[1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
                    
                    # Calculate R²
                    from scipy import stats
                    corr, _ = stats.pearsonr(valid['Actual_Resolution_Days'], valid['Predicted_Resolution_Days'])
                    axes[1].text(0.05, 0.95, f'R² = {corr**2:.2f}', transform=axes[1].transAxes,
                                fontsize=12, verticalalignment='top')
                    
                    axes[1].set_xlabel('Actual Days')
                    axes[1].set_ylabel('Predicted Days')
                    axes[1].set_title('Prediction Accuracy', fontweight='bold')
                    axes[1].legend()
                    has_data = True
            
            if has_data:
                plt.tight_layout()
                path = os.path.join(self.plot_dir, "resolution_time.png")
                plt.savefig(path)
                self.paths[14] = path
        except Exception as e:
            logger.warning(f"Chart 15 Skipped: {e}")
        finally:
            plt.close()
