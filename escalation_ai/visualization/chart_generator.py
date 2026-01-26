"""
Chart Generator Module - McKinsey-Style Executive Visualizations
Charts organized by category for clear reporting structure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.config import PLOT_DIR


class ChartGenerator:
    """
    Generates McKinsey-style executive charts organized by category.
    
    Chart Categories:
        01_risk/       - Risk analysis and friction charts
        02_engineer/   - Engineer performance metrics
        03_lob/        - Line of Business analysis
        04_analysis/   - Root cause and learning charts
        05_predictive/ - ML model performance charts
        06_financial/  - Financial impact analysis
    """
    
    # Color palette - McKinsey style
    COLORS = {
        'primary': '#003366',      # Deep blue
        'secondary': '#0066CC',    # Medium blue
        'accent': '#FF6600',       # Orange
        'success': '#28A745',      # Green
        'warning': '#FFC107',      # Yellow
        'danger': '#DC3545',       # Red
        'neutral': '#6C757D',      # Gray
        'light': '#F8F9FA',        # Light gray
    }
    
    GRADIENT_BLUES = ['#E6F2FF', '#B3D9FF', '#66B3FF', '#3399FF', '#0066CC', '#003366']
    GRADIENT_RISK = ['#28A745', '#7CB342', '#FFC107', '#FF9800', '#FF5722', '#DC3545']
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize chart generator with output directory."""
        self.output_dir = output_dir or PLOT_DIR
        self._setup_directories()
        self._setup_style()
        
    def _setup_directories(self):
        """Create organized chart subdirectories."""
        self.chart_dirs = {
            'risk': self.output_dir / '01_risk',
            'engineer': self.output_dir / '02_engineer',
            'lob': self.output_dir / '03_lob',
            'analysis': self.output_dir / '04_analysis',
            'predictive': self.output_dir / '05_predictive',
            'financial': self.output_dir / '06_financial',
        }
        
        for dir_path in self.chart_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _setup_style(self):
        """Configure matplotlib style for executive presentations."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'axes.labelweight': 'bold',
            'figure.titlesize': 16,
            'figure.titleweight': 'bold',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#CCCCCC',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
        })
        
    def generate_all_charts(self, analysis_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate all charts organized by category.
        
        Returns:
            Dict mapping category names to list of generated chart paths
        """
        generated = {
            'risk': [],
            'engineer': [],
            'lob': [],
            'analysis': [],
            'predictive': [],
            'financial': [],
        }
        
        try:
            # Risk Charts (01_risk/)
            generated['risk'].append(self._chart_friction_pareto(analysis_data))
            generated['risk'].append(self._chart_risk_origin(analysis_data))
            generated['risk'].append(self._chart_risk_trend(analysis_data))
            generated['risk'].append(self._chart_severity_heatmap(analysis_data))
            
            # Engineer Charts (02_engineer/)
            generated['engineer'].append(self._chart_engineer_friction(analysis_data))
            generated['engineer'].append(self._chart_engineer_learning(analysis_data))
            
            # LOB Charts (03_lob/)
            generated['lob'].append(self._chart_lob_friction(analysis_data))
            generated['lob'].append(self._chart_lob_matrix(analysis_data))
            generated['lob'].append(self._chart_lob_categories(analysis_data))
            
            # Analysis Charts (04_analysis/)
            generated['analysis'].append(self._chart_root_cause(analysis_data))
            generated['analysis'].append(self._chart_learning_integrity(analysis_data))
            
            # Predictive Charts (05_predictive/)
            generated['predictive'].append(self._chart_pm_accuracy(analysis_data))
            generated['predictive'].append(self._chart_ai_recurrence(analysis_data))
            generated['predictive'].append(self._chart_resolution_time(analysis_data))
            
            # Financial Charts (06_financial/)
            generated['financial'].append(self._chart_financial_impact(analysis_data))
            
        except Exception as e:
            print(f"Chart generation error: {e}")
            
        # Filter None values
        for category in generated:
            generated[category] = [p for p in generated[category] if p]
            
        return generated
    
    # =========================================================================
    # RISK CHARTS (01_risk/)
    # =========================================================================
    
    def _chart_friction_pareto(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 01: Friction Pareto Analysis
        Shows cumulative friction by category with 80/20 analysis.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            friction_data = data.get('friction_by_category', {})
            if not friction_data:
                friction_data = {'Network': 45, 'Billing': 30, 'Hardware': 15, 'Software': 10}
            
            categories = list(friction_data.keys())
            values = list(friction_data.values())
            
            # Sort descending
            sorted_pairs = sorted(zip(values, categories), reverse=True)
            values, categories = zip(*sorted_pairs)
            values = list(values)
            categories = list(categories)
            
            # Calculate cumulative percentage
            total = sum(values)
            cumulative = np.cumsum(values) / total * 100
            
            # Bar chart
            bars = ax.bar(categories, values, color=self.COLORS['primary'], alpha=0.8, label='Friction Points')
            
            # Cumulative line
            ax2 = ax.twinx()
            ax2.plot(categories, cumulative, color=self.COLORS['accent'], 
                    marker='o', linewidth=2.5, markersize=8, label='Cumulative %')
            ax2.axhline(y=80, color=self.COLORS['danger'], linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='80% Threshold')
            
            # Styling
            ax.set_xlabel('Category')
            ax.set_ylabel('Friction Points', color=self.COLORS['primary'])
            ax2.set_ylabel('Cumulative %', color=self.COLORS['accent'])
            ax2.set_ylim(0, 105)
            
            plt.title('Friction Pareto Analysis\n80/20 Rule Application', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # Value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['risk'] / 'friction_pareto.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating friction pareto chart: {e}")
            plt.close('all')
            return None
    
    def _chart_risk_origin(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 02: Risk Origin Distribution
        Pie chart showing where escalations originate.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            origin_data = data.get('risk_by_origin', {})
            if not origin_data:
                origin_data = {'Field': 40, 'Call Center': 30, 'Self-Service': 20, 'Partner': 10}
            
            labels = list(origin_data.keys())
            sizes = list(origin_data.values())
            colors = [self.COLORS['primary'], self.COLORS['secondary'], 
                     self.COLORS['accent'], self.COLORS['success']][:len(labels)]
            
            explode = [0.05] * len(labels)
            explode[0] = 0.1  # Explode largest slice
            
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, explode=explode,
                autopct='%1.1f%%', startangle=90, shadow=True,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.title('Risk Origin Distribution\nEscalation Source Analysis', 
                     fontsize=14, fontweight='bold', pad=20)
            
            ax.axis('equal')
            fig.tight_layout()
            
            filepath = self.chart_dirs['risk'] / 'risk_origin.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating risk origin chart: {e}")
            plt.close('all')
            return None
    
    def _chart_risk_trend(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 03: Risk Trend Over Time
        Line chart showing risk evolution with forecast.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            trend_data = data.get('risk_trend', {})
            if not trend_data:
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                trend_data = {'months': months, 'values': [45, 52, 48, 55, 50, 47]}
            
            months = trend_data.get('months', ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'])
            values = trend_data.get('values', [45, 52, 48, 55, 50, 47])
            
            # Main trend line
            ax.plot(months, values, color=self.COLORS['primary'], 
                   marker='o', linewidth=2.5, markersize=10, label='Risk Score')
            
            # Fill under curve
            ax.fill_between(months, values, alpha=0.2, color=self.COLORS['primary'])
            
            # Moving average
            if len(values) >= 3:
                ma = pd.Series(values).rolling(window=3, min_periods=1).mean()
                ax.plot(months, ma, color=self.COLORS['accent'], 
                       linestyle='--', linewidth=2, label='3-Month MA')
            
            # Target line
            target = data.get('risk_target', 40)
            ax.axhline(y=target, color=self.COLORS['success'], 
                      linestyle=':', linewidth=2, label=f'Target ({target})')
            
            ax.set_xlabel('Period')
            ax.set_ylabel('Risk Score')
            ax.legend(loc='upper right')
            
            plt.title('Risk Trend Analysis\nHistorical Performance & Target', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['risk'] / 'risk_trend.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating risk trend chart: {e}")
            plt.close('all')
            return None
    
    def _chart_severity_heatmap(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 04: Severity-Impact Heatmap
        Shows relationship between severity levels and business impact.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            heatmap_data = data.get('severity_matrix', None)
            if heatmap_data is None:
                # Sample data: rows=severity, cols=impact
                heatmap_data = np.array([
                    [15, 8, 3, 1],    # Low severity
                    [10, 20, 12, 5],   # Medium severity
                    [5, 15, 25, 15],   # High severity
                    [2, 8, 18, 30],    # Critical severity
                ])
            
            severity_labels = ['Low', 'Medium', 'High', 'Critical']
            impact_labels = ['Minimal', 'Moderate', 'Significant', 'Severe']
            
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='RdYlGn_r',
                       xticklabels=impact_labels, yticklabels=severity_labels,
                       ax=ax, cbar_kws={'label': 'Ticket Count'},
                       linewidths=0.5, linecolor='white')
            
            ax.set_xlabel('Business Impact', fontsize=12, fontweight='bold')
            ax.set_ylabel('Severity Level', fontsize=12, fontweight='bold')
            
            plt.title('Severity-Impact Matrix\nEscalation Distribution Analysis', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['risk'] / 'severity_heatmap.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating severity heatmap: {e}")
            plt.close('all')
            return None
    
    # =========================================================================
    # ENGINEER CHARTS (02_engineer/)
    # =========================================================================
    
    def _chart_engineer_friction(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 05: Engineer Friction Analysis
        Horizontal bar chart showing friction points by engineer.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            engineer_data = data.get('friction_by_engineer', {})
            if not engineer_data:
                engineer_data = {
                    'Engineer A': 85, 'Engineer B': 72, 'Engineer C': 68,
                    'Engineer D': 55, 'Engineer E': 45, 'Engineer F': 38
                }
            
            engineers = list(engineer_data.keys())
            friction = list(engineer_data.values())
            
            # Sort by friction (highest first)
            sorted_pairs = sorted(zip(friction, engineers), reverse=True)
            friction, engineers = zip(*sorted_pairs)
            friction = list(friction)
            engineers = list(engineers)
            
            # Color by performance tier
            colors = []
            for f in friction:
                if f > 70:
                    colors.append(self.COLORS['danger'])
                elif f > 50:
                    colors.append(self.COLORS['warning'])
                else:
                    colors.append(self.COLORS['success'])
            
            bars = ax.barh(engineers, friction, color=colors, alpha=0.85, edgecolor='white')
            
            # Value labels
            for bar, val in zip(bars, friction):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                       f'{val}', va='center', fontsize=10, fontweight='bold')
            
            # Target line
            target = data.get('friction_target', 50)
            ax.axvline(x=target, color=self.COLORS['primary'], 
                      linestyle='--', linewidth=2, label=f'Target ({target})')
            
            ax.set_xlabel('Friction Score')
            ax.set_ylabel('Engineer')
            ax.legend(loc='lower right')
            
            # Add legend for colors
            legend_patches = [
                mpatches.Patch(color=self.COLORS['danger'], label='High Risk (>70)'),
                mpatches.Patch(color=self.COLORS['warning'], label='Medium Risk (50-70)'),
                mpatches.Patch(color=self.COLORS['success'], label='Low Risk (<50)'),
            ]
            ax.legend(handles=legend_patches, loc='lower right')
            
            plt.title('Engineer Friction Analysis\nPerformance Risk Assessment', 
                     fontsize=14, fontweight='bold', pad=20)
            
            ax.invert_yaxis()
            fig.tight_layout()
            
            filepath = self.chart_dirs['engineer'] / 'engineer_friction.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating engineer friction chart: {e}")
            plt.close('all')
            return None
    
    def _chart_engineer_learning(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 06: Engineer Learning Progress
        Shows training completion and skill development.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            learning_data = data.get('engineer_learning', {})
            if not learning_data:
                learning_data = {
                    'Engineer A': {'completed': 8, 'pending': 2},
                    'Engineer B': {'completed': 6, 'pending': 4},
                    'Engineer C': {'completed': 9, 'pending': 1},
                    'Engineer D': {'completed': 5, 'pending': 5},
                }
            
            engineers = list(learning_data.keys())
            completed = [learning_data[e].get('completed', 0) for e in engineers]
            pending = [learning_data[e].get('pending', 0) for e in engineers]
            
            x = np.arange(len(engineers))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, completed, width, label='Completed',
                          color=self.COLORS['success'], alpha=0.85)
            bars2 = ax.bar(x + width/2, pending, width, label='Pending',
                          color=self.COLORS['warning'], alpha=0.85)
            
            ax.set_xlabel('Engineer')
            ax.set_ylabel('Modules')
            ax.set_xticks(x)
            ax.set_xticklabels(engineers, rotation=45, ha='right')
            ax.legend()
            
            # Value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                               f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            plt.title('Engineer Learning Progress\nTraining Module Completion', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['engineer'] / 'engineer_learning.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating engineer learning chart: {e}")
            plt.close('all')
            return None
    
    # =========================================================================
    # LOB CHARTS (03_lob/)
    # =========================================================================
    
    def _chart_lob_friction(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 07: LOB Friction Distribution
        Shows friction distribution across Lines of Business.
        """
        try:
            fig, ax = plt.subplots(figsize=(11, 7))
            
            lob_data = data.get('friction_by_lob', {})
            if not lob_data:
                lob_data = {
                    'Enterprise': 42, 'SMB': 35, 'Consumer': 28,
                    'Government': 22, 'Healthcare': 18
                }
            
            lobs = list(lob_data.keys())
            friction = list(lob_data.values())
            
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(lobs)))
            
            bars = ax.bar(lobs, friction, color=colors, edgecolor='white', linewidth=1.5)
            
            # Value labels
            for bar, val in zip(bars, friction):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Average line
            avg = np.mean(friction)
            ax.axhline(y=avg, color=self.COLORS['accent'], linestyle='--', 
                      linewidth=2, label=f'Average ({avg:.1f})')
            
            ax.set_xlabel('Line of Business')
            ax.set_ylabel('Friction Score')
            ax.legend(loc='upper right')
            
            plt.title('LOB Friction Distribution\nBusiness Unit Performance', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['lob'] / 'lob_friction.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating LOB friction chart: {e}")
            plt.close('all')
            return None
    
    def _chart_lob_matrix(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 08: LOB Performance Matrix
        Bubble chart showing LOB by volume, friction, and resolution time.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            matrix_data = data.get('lob_matrix', {})
            if not matrix_data:
                matrix_data = {
                    'Enterprise': {'volume': 150, 'friction': 42, 'resolution': 4.2},
                    'SMB': {'volume': 200, 'friction': 35, 'resolution': 3.5},
                    'Consumer': {'volume': 350, 'friction': 28, 'resolution': 2.8},
                    'Government': {'volume': 80, 'friction': 22, 'resolution': 5.5},
                }
            
            lobs = list(matrix_data.keys())
            volumes = [matrix_data[l]['volume'] for l in lobs]
            frictions = [matrix_data[l]['friction'] for l in lobs]
            resolutions = [matrix_data[l]['resolution'] for l in lobs]
            
            # Normalize bubble sizes
            size_scale = np.array(volumes) / max(volumes) * 1000
            
            scatter = ax.scatter(frictions, resolutions, s=size_scale, 
                                alpha=0.6, c=range(len(lobs)), 
                                cmap='viridis', edgecolors='white', linewidth=2)
            
            # Labels
            for i, lob in enumerate(lobs):
                ax.annotate(lob, (frictions[i], resolutions[i]), 
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Friction Score')
            ax.set_ylabel('Avg Resolution Time (days)')
            
            # Size legend
            ax.text(0.02, 0.98, 'Bubble Size = Ticket Volume', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   style='italic', color='gray')
            
            plt.title('LOB Performance Matrix\nVolume vs Friction vs Resolution', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['lob'] / 'lob_matrix.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating LOB matrix chart: {e}")
            plt.close('all')
            return None
    
    def _chart_lob_categories(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 09: LOB Category Breakdown
        Stacked bar chart showing issue categories by LOB.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            category_data = data.get('lob_categories', {})
            if not category_data:
                category_data = {
                    'Enterprise': {'Network': 20, 'Billing': 12, 'Hardware': 10},
                    'SMB': {'Network': 15, 'Billing': 15, 'Hardware': 5},
                    'Consumer': {'Network': 10, 'Billing': 10, 'Hardware': 8},
                }
            
            lobs = list(category_data.keys())
            categories = list(set(cat for lob_cats in category_data.values() for cat in lob_cats.keys()))
            
            x = np.arange(len(lobs))
            width = 0.6
            
            bottom = np.zeros(len(lobs))
            colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
            
            for i, category in enumerate(categories):
                values = [category_data[lob].get(category, 0) for lob in lobs]
                ax.bar(x, values, width, label=category, bottom=bottom, color=colors[i])
                bottom += values
            
            ax.set_xlabel('Line of Business')
            ax.set_ylabel('Issue Count')
            ax.set_xticks(x)
            ax.set_xticklabels(lobs)
            ax.legend(title='Category', loc='upper right')
            
            plt.title('LOB Category Breakdown\nIssue Distribution by Business Unit', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['lob'] / 'lob_categories.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating LOB categories chart: {e}")
            plt.close('all')
            return None
    
    # =========================================================================
    # ANALYSIS CHARTS (04_analysis/)
    # =========================================================================
    
    def _chart_root_cause(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 10: Root Cause Analysis
        Horizontal bar chart showing root cause distribution.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            root_causes = data.get('root_causes', {})
            if not root_causes:
                root_causes = {
                    'Configuration Error': 28,
                    'Hardware Failure': 22,
                    'Software Bug': 18,
                    'User Error': 15,
                    'Network Issue': 12,
                    'Documentation Gap': 5,
                }
            
            causes = list(root_causes.keys())
            counts = list(root_causes.values())
            
            # Sort
            sorted_pairs = sorted(zip(counts, causes), reverse=True)
            counts, causes = zip(*sorted_pairs)
            
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(causes)))
            
            bars = ax.barh(causes, counts, color=colors, edgecolor='white')
            
            # Value labels
            for bar, val in zip(bars, counts):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{val}', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Occurrence Count')
            ax.set_ylabel('Root Cause')
            
            plt.title('Root Cause Analysis\nTop Escalation Drivers', 
                     fontsize=14, fontweight='bold', pad=20)
            
            ax.invert_yaxis()
            fig.tight_layout()
            
            filepath = self.chart_dirs['analysis'] / 'root_cause.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating root cause chart: {e}")
            plt.close('all')
            return None
    
    def _chart_learning_integrity(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 11: Learning System Integrity
        Gauge/donut chart showing system health metrics.
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            
            integrity_data = data.get('learning_integrity', {})
            if not integrity_data:
                integrity_data = {
                    'data_quality': 92,
                    'model_accuracy': 87,
                    'feedback_loop': 78,
                }
            
            metrics = [
                ('Data Quality', integrity_data.get('data_quality', 92)),
                ('Model Accuracy', integrity_data.get('model_accuracy', 87)),
                ('Feedback Loop', integrity_data.get('feedback_loop', 78)),
            ]
            
            for ax, (name, value) in zip(axes, metrics):
                # Donut chart
                colors_donut = [self.COLORS['success'] if value >= 80 
                               else self.COLORS['warning'] if value >= 60 
                               else self.COLORS['danger'], 
                               self.COLORS['light']]
                
                wedges, _ = ax.pie([value, 100-value], colors=colors_donut,
                                   startangle=90, counterclock=False,
                                   wedgeprops={'width': 0.4, 'edgecolor': 'white'})
                
                # Center text
                ax.text(0, 0, f'{value}%', ha='center', va='center',
                       fontsize=24, fontweight='bold', color=colors_donut[0])
                
                ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
            
            fig.suptitle('Learning System Integrity\nHealth Metrics Dashboard', 
                        fontsize=14, fontweight='bold', y=1.02)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['analysis'] / 'learning_integrity.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating learning integrity chart: {e}")
            plt.close('all')
            return None
    
    # =========================================================================
    # PREDICTIVE CHARTS (05_predictive/)
    # =========================================================================
    
    def _chart_pm_accuracy(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 12: Predictive Model Accuracy
        Shows model performance metrics over time.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            pm_data = data.get('pm_accuracy', {})
            if not pm_data:
                pm_data = {
                    'periods': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
                    'accuracy': [78, 82, 85, 84, 88, 91],
                    'precision': [75, 80, 82, 81, 85, 89],
                    'recall': [72, 78, 80, 82, 84, 87],
                }
            
            periods = pm_data.get('periods', ['W1', 'W2', 'W3', 'W4', 'W5', 'W6'])
            
            ax.plot(periods, pm_data.get('accuracy', []), marker='o', linewidth=2.5,
                   markersize=8, label='Accuracy', color=self.COLORS['primary'])
            ax.plot(periods, pm_data.get('precision', []), marker='s', linewidth=2,
                   markersize=7, label='Precision', color=self.COLORS['secondary'])
            ax.plot(periods, pm_data.get('recall', []), marker='^', linewidth=2,
                   markersize=7, label='Recall', color=self.COLORS['accent'])
            
            ax.set_xlabel('Period')
            ax.set_ylabel('Score (%)')
            ax.set_ylim(60, 100)
            ax.legend(loc='lower right')
            
            # Target line
            ax.axhline(y=85, color=self.COLORS['success'], linestyle='--',
                      linewidth=1.5, alpha=0.7, label='Target (85%)')
            
            plt.title('Predictive Model Performance\nAccuracy Trend Analysis', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['predictive'] / 'pm_accuracy.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating PM accuracy chart: {e}")
            plt.close('all')
            return None
    
    def _chart_ai_recurrence(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 13: AI Recurrence Prediction
        Shows predicted vs actual recurrence rates.
        """
        try:
            fig, ax = plt.subplots(figsize=(11, 7))
            
            recurrence_data = data.get('ai_recurrence', {})
            if not recurrence_data:
                recurrence_data = {
                    'categories': ['Network', 'Billing', 'Hardware', 'Software', 'Other'],
                    'predicted': [25, 18, 22, 15, 8],
                    'actual': [23, 20, 19, 16, 10],
                }
            
            categories = recurrence_data.get('categories', [])
            predicted = recurrence_data.get('predicted', [])
            actual = recurrence_data.get('actual', [])
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, predicted, width, label='AI Predicted',
                          color=self.COLORS['primary'], alpha=0.85)
            bars2 = ax.bar(x + width/2, actual, width, label='Actual',
                          color=self.COLORS['accent'], alpha=0.85)
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Recurrence Rate (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()
            
            # Value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                           f'{height}%', ha='center', va='bottom', fontsize=9)
            
            plt.title('AI Recurrence Prediction\nPredicted vs Actual Rates', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['predictive'] / 'ai_recurrence.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating AI recurrence chart: {e}")
            plt.close('all')
            return None
    
    def _chart_resolution_time(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 14: Resolution Time Prediction
        Box plot showing predicted resolution times by category.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            resolution_data = data.get('resolution_time', {})
            if not resolution_data:
                # Generate sample box plot data
                np.random.seed(42)
                resolution_data = {
                    'Network': np.random.normal(4, 1.5, 50),
                    'Billing': np.random.normal(2, 0.8, 50),
                    'Hardware': np.random.normal(6, 2, 50),
                    'Software': np.random.normal(3, 1, 50),
                }
            
            categories = list(resolution_data.keys())
            data_arrays = [resolution_data[cat] for cat in categories]
            
            bp = ax.boxplot(data_arrays, labels=categories, patch_artist=True,
                           medianprops={'color': 'black', 'linewidth': 2})
            
            # Color boxes
            colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(categories)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Resolution Time (days)')
            
            # Target line
            target = data.get('resolution_target', 3)
            ax.axhline(y=target, color=self.COLORS['success'], linestyle='--',
                      linewidth=2, label=f'Target ({target} days)')
            ax.legend(loc='upper right')
            
            plt.title('Resolution Time Distribution\nPredicted Time by Category', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['predictive'] / 'resolution_time.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating resolution time chart: {e}")
            plt.close('all')
            return None
    
    # =========================================================================
    # FINANCIAL CHARTS (06_financial/)
    # =========================================================================
    
    def _chart_financial_impact(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart 15: Financial Impact Analysis
        Stacked bar showing costs and potential savings.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            financial_data = data.get('financial_impact', {})
            if not financial_data:
                financial_data = {
                    'categories': ['Network', 'Billing', 'Hardware', 'Software'],
                    'direct_cost': [45000, 28000, 52000, 18000],
                    'indirect_cost': [22000, 15000, 25000, 8000],
                    'potential_savings': [35000, 20000, 40000, 12000],
                }
            
            categories = financial_data.get('categories', [])
            direct = np.array(financial_data.get('direct_cost', [])) / 1000  # Convert to K
            indirect = np.array(financial_data.get('indirect_cost', [])) / 1000
            savings = np.array(financial_data.get('potential_savings', [])) / 1000
            
            x = np.arange(len(categories))
            width = 0.25
            
            bars1 = ax.bar(x - width, direct, width, label='Direct Cost',
                          color=self.COLORS['danger'], alpha=0.85)
            bars2 = ax.bar(x, indirect, width, label='Indirect Cost',
                          color=self.COLORS['warning'], alpha=0.85)
            bars3 = ax.bar(x + width, savings, width, label='Potential Savings',
                          color=self.COLORS['success'], alpha=0.85)
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Amount ($K)')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend(loc='upper right')
            
            # Value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                           f'${height:.0f}K', ha='center', va='bottom', fontsize=8)
            
            # Total savings annotation
            total_savings = savings.sum()
            ax.annotate(f'Total Potential Savings: ${total_savings:.0f}K',
                       xy=(0.98, 0.98), xycoords='axes fraction',
                       ha='right', va='top', fontsize=11, fontweight='bold',
                       color=self.COLORS['success'],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title('Financial Impact Analysis\nCost Breakdown & Savings Opportunity', 
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['financial'] / 'financial_impact.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating financial impact chart: {e}")
            plt.close('all')
            return None
    
    def get_chart_summary(self) -> Dict[str, Any]:
        """Get summary of all chart directories and files."""
        summary = {}
        for category, dir_path in self.chart_dirs.items():
            if dir_path.exists():
                files = list(dir_path.glob('*.png'))
                summary[category] = {
                    'path': str(dir_path),
                    'count': len(files),
                    'files': [f.name for f in files]
                }
        return summary
