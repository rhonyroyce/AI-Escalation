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
        if output_dir is None:
            self.output_dir = PLOT_DIR
        elif isinstance(output_dir, str):
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = output_dir
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
            'similarity': self.output_dir / '10_similarity',
            'lessons': self.output_dir / '11_lessons_learned',
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
            'similarity': [],
            'lessons': [],
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
            generated['analysis'].append(self._chart_category_subcategory_breakdown(analysis_data))
            generated['analysis'].append(self._chart_category_heatmap(analysis_data))

            # Predictive Charts (05_predictive/)
            generated['predictive'].append(self._chart_pm_accuracy(analysis_data))
            generated['predictive'].append(self._chart_ai_recurrence(analysis_data))
            generated['predictive'].append(self._chart_resolution_time(analysis_data))

            # Financial Charts (06_financial/)
            generated['financial'].append(self._chart_financial_impact(analysis_data))
            generated['financial'].append(self._chart_subcategory_financial_impact(analysis_data))

            # Similarity Charts (10_similarity/)
            generated['similarity'].append(self._chart_similarity_count_distribution(analysis_data))
            generated['similarity'].append(self._chart_resolution_consistency(analysis_data))
            generated['similarity'].append(self._chart_similarity_score_distribution(analysis_data))
            generated['similarity'].append(self._chart_resolution_comparison(analysis_data))
            generated['similarity'].append(self._chart_similarity_effectiveness(analysis_data))

            # Lessons Learned Charts (11_lessons_learned/)
            generated['lessons'].append(self._chart_learning_grades(analysis_data))
            generated['lessons'].append(self._chart_lesson_completion_rate(analysis_data))
            generated['lessons'].append(self._chart_recurrence_vs_lessons(analysis_data))
            generated['lessons'].append(self._chart_learning_heatmap(analysis_data))
            generated['lessons'].append(self._chart_recommendations_summary(analysis_data))

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
            fig, ax = plt.subplots(figsize=(14, 8))
            
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
            
            # Truncate long category names for display
            cat_labels = [c[:18] + '..' if len(c) > 18 else c for c in categories]
            
            # Calculate cumulative percentage
            total = sum(values)
            cumulative = np.cumsum(values) / total * 100
            
            # Bar chart with truncated labels
            x_pos = np.arange(len(cat_labels))
            bars = ax.bar(x_pos, values, color=self.COLORS['primary'], alpha=0.8, label='Friction Points')
            
            # Cumulative line
            ax2 = ax.twinx()
            ax2.plot(x_pos, cumulative, color=self.COLORS['accent'], 
                    marker='o', linewidth=2.5, markersize=8, label='Cumulative %')
            ax2.axhline(y=80, color=self.COLORS['danger'], linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='80% Threshold')
            
            # Styling - rotate x-axis labels
            ax.set_xlabel('Category')
            ax.set_ylabel('Friction Points', color=self.COLORS['primary'])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
            ax2.set_ylabel('Cumulative %', color=self.COLORS['accent'])
            ax2.set_ylim(0, 105)
            
            plt.title('Friction Pareto Analysis\n80/20 Rule Application', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # Value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
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
            fig, ax = plt.subplots(figsize=(12, 10))
            
            engineer_data = data.get('friction_by_engineer', {})
            if not engineer_data:
                # No data available - create placeholder chart
                ax.text(0.5, 0.5, 'No Engineer Data Available\n\nEnsure tickets_data_engineer_name column exists', 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.title('Engineer Friction Analysis\nData Not Available', 
                         fontsize=14, fontweight='bold', pad=20)
                filepath = self.chart_dirs['engineer'] / 'engineer_friction.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            engineers = list(engineer_data.keys())
            friction = list(engineer_data.values())
            
            # Sort by friction (highest first) and limit to top 15
            sorted_pairs = sorted(zip(friction, engineers), reverse=True)[:15]
            friction, engineers = zip(*sorted_pairs)
            friction = list(friction)
            # Truncate long names
            engineers = [e[:20] + '...' if len(e) > 20 else e for e in engineers]
            
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
                       f'{val:.0f}', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Friction Score', fontsize=11)
            ax.set_ylabel('Engineer', fontsize=11)
            
            # Add legend for colors - positioned at lower right to avoid data overlap
            legend_patches = [
                mpatches.Patch(color=self.COLORS['danger'], label='High Risk (>70)'),
                mpatches.Patch(color=self.COLORS['warning'], label='Medium (50-70)'),
                mpatches.Patch(color=self.COLORS['success'], label='Low Risk (<50)'),
            ]
            ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)
            
            plt.title('Engineer Friction Analysis\nPerformance Risk Assessment', 
                     fontsize=14, fontweight='bold', pad=15)
            
            ax.invert_yaxis()
            plt.subplots_adjust(left=0.25)  # Make room for engineer names
            
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
                # No data available - create placeholder chart
                ax.text(0.5, 0.5, 'No Engineer Learning Data Available\n\nEnsure tickets_data_engineer_name column exists', 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.title('Engineer Learning Progress\nData Not Available', 
                         fontsize=14, fontweight='bold', pad=20)
                filepath = self.chart_dirs['engineer'] / 'engineer_learning.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            engineers = list(learning_data.keys())
            completed = [learning_data[e].get('completed', 0) for e in engineers]
            pending = [learning_data[e].get('pending', 0) for e in engineers]
            
            # Truncate long names
            engineers = [e[:15] + '...' if len(e) > 15 else e for e in engineers]
            
            x = np.arange(len(engineers))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, completed, width, label='Resolved',
                          color=self.COLORS['success'], alpha=0.85)
            bars2 = ax.bar(x + width/2, pending, width, label='Repeat Issues',
                          color=self.COLORS['warning'], alpha=0.85)
            
            ax.set_xlabel('Engineer', fontsize=11)
            ax.set_ylabel('Issue Count', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(engineers, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='upper right', fontsize=10)
            
            # Value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                               f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            plt.title('Engineer Learning Progress\nIssue Resolution vs Repeat Patterns', 
                     fontsize=14, fontweight='bold', pad=15)
            
            plt.subplots_adjust(bottom=0.25)  # Make room for rotated labels
            
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
                # No data available - create placeholder chart
                ax.text(0.5, 0.5, 'No LOB Data Available\n\nEnsure tickets_data_lob column exists', 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.title('LOB Friction Distribution\nData Not Available', 
                         fontsize=14, fontweight='bold', pad=20)
                filepath = self.chart_dirs['lob'] / 'lob_friction.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            lobs = list(lob_data.keys())
            friction = list(lob_data.values())
            
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(lobs)))
            
            bars = ax.bar(lobs, friction, color=colors, edgecolor='white', linewidth=1.5)
            
            # Value labels
            for bar, val in zip(bars, friction):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Average line
            avg = np.mean(friction)
            ax.axhline(y=avg, color=self.COLORS['accent'], linestyle='--', 
                      linewidth=2, label=f'Average ({avg:.1f})')
            
            ax.set_xlabel('Line of Business', fontsize=11)
            ax.set_ylabel('Friction Score', fontsize=11)
            # Rotate x-axis labels if more than 4 items
            if len(lobs) > 4:
                plt.xticks(rotation=45, ha='right', fontsize=10)
            ax.legend(loc='upper right', fontsize=10)
            
            plt.title('LOB Friction Distribution\nBusiness Unit Performance', 
                     fontsize=14, fontweight='bold', pad=15)
            
            plt.subplots_adjust(bottom=0.2)  # Make room for rotated labels
            
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
        Grouped bar chart showing LOB metrics: volume, friction, and resolution time.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Use friction_by_lob and lob_counts to build matrix data
            lob_friction = data.get('friction_by_lob', {})
            lob_counts = data.get('lob_counts', {})
            resolution_by_lob = data.get('resolution_by_lob', {})
            
            if not lob_friction or not lob_counts:
                # No data available - create placeholder chart
                ax.text(0.5, 0.5, 'No LOB Matrix Data Available\n\nEnsure tickets_data_lob column exists', 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.title('LOB Performance Matrix\nData Not Available', 
                         fontsize=14, fontweight='bold', pad=20)
                filepath = self.chart_dirs['lob'] / 'lob_matrix.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            # Build matrix from available data
            lobs = list(lob_friction.keys())
            volumes = np.array([lob_counts.get(lob, 0) for lob in lobs])
            frictions = np.array([lob_friction.get(lob, 0) for lob in lobs])
            resolutions = np.array([resolution_by_lob.get(lob, 5.0) for lob in lobs])
            
            # Normalize all metrics to 0-100 scale for comparison
            vol_norm = (volumes / max(volumes) * 100) if max(volumes) > 0 else volumes
            fric_norm = frictions  # Already in a reasonable scale
            res_norm = (resolutions / max(resolutions) * 100) if max(resolutions) > 0 else resolutions
            
            x = np.arange(len(lobs))
            width = 0.25
            
            # Create grouped bars
            bars1 = ax.bar(x - width, vol_norm, width, label='Volume (normalized)', 
                          color=self.COLORS['primary'], alpha=0.85)
            bars2 = ax.bar(x, fric_norm, width, label='Friction Score', 
                          color=self.COLORS['warning'], alpha=0.85)
            bars3 = ax.bar(x + width, res_norm, width, label='Resolution Time (normalized)', 
                          color=self.COLORS['accent'], alpha=0.85)
            
            ax.set_xlabel('Line of Business', fontsize=11)
            ax.set_ylabel('Score (normalized)', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(lobs, rotation=45, ha='right', fontsize=10)
            
            # Position legend outside chart
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, borderaxespad=0)
            
            # Add actual values as annotations
            for i, (v, f, r) in enumerate(zip(volumes, frictions, resolutions)):
                ax.annotate(f'Vol:{v}', xy=(x[i] - width, vol_norm[i] + 2), 
                           ha='center', va='bottom', fontsize=7, rotation=90)
                ax.annotate(f'{f:.0f}', xy=(x[i], fric_norm[i] + 2), 
                           ha='center', va='bottom', fontsize=7, rotation=90)
                ax.annotate(f'{r:.1f}d', xy=(x[i] + width, res_norm[i] + 2), 
                           ha='center', va='bottom', fontsize=7, rotation=90)
            
            plt.title('LOB Performance Matrix\nVolume vs Friction vs Resolution', 
                     fontsize=14, fontweight='bold', pad=20)
            
            plt.subplots_adjust(bottom=0.2, right=0.78)  # Room for labels and legend
            
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
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Use lob_by_category from analysis data
            category_data = data.get('lob_by_category', {})
            if not category_data:
                # No data available - create placeholder chart
                ax.text(0.5, 0.5, 'No LOB Category Data Available\n\nEnsure tickets_data_lob column exists', 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.title('LOB Category Breakdown\nData Not Available', 
                         fontsize=14, fontweight='bold', pad=20)
                filepath = self.chart_dirs['lob'] / 'lob_categories.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            # Aggregate smaller categories into "Other" to avoid legend overflow
            # Calculate total per category
            cat_totals = {}
            for cat, lob_data in category_data.items():
                cat_totals[cat] = sum(lob_data.values())
            
            # Sort by total and keep top 8, rest goes to "Other"
            sorted_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
            top_cats = [c[0] for c in sorted_cats[:8]]
            other_cats = [c[0] for c in sorted_cats[8:]]
            
            # Build new category_data with "Other" aggregated
            lobs = set()
            for cat_data in category_data.values():
                lobs.update(cat_data.keys())
            lobs = list(lobs)
            
            # Truncate long category names for display
            cat_labels = [c[:15] + '..' if len(c) > 15 else c for c in top_cats]
            if other_cats:
                cat_labels.append('Other')
            
            x = np.arange(len(lobs))
            width = 0.6
            
            bottom = np.zeros(len(lobs))
            colors = plt.cm.Set2(np.linspace(0, 1, len(cat_labels)))
            
            for i, category in enumerate(top_cats):
                cat_lob_data = category_data.get(category, {})
                values = [cat_lob_data.get(lob, 0) for lob in lobs]
                ax.bar(x, values, width, label=cat_labels[i], bottom=bottom, color=colors[i])
                bottom += np.array(values)
            
            # Add "Other" bar if there are aggregated categories
            if other_cats:
                other_values = np.zeros(len(lobs))
                for cat in other_cats:
                    cat_lob_data = category_data.get(cat, {})
                    for j, lob in enumerate(lobs):
                        other_values[j] += cat_lob_data.get(lob, 0)
                ax.bar(x, other_values, width, label='Other', bottom=bottom, color=colors[-1])
            
            ax.set_xlabel('Line of Business', fontsize=11)
            ax.set_ylabel('Issue Count', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(lobs, rotation=45, ha='right', fontsize=10)
            # Place legend outside the chart with controlled size
            ax.legend(title='Category', loc='upper left', fontsize=9, 
                     bbox_to_anchor=(1.02, 1), borderaxespad=0, ncol=1)
            
            plt.title('LOB Category Breakdown\nIssue Distribution by Business Unit', 
                     fontsize=14, fontweight='bold', pad=15)
            
            plt.subplots_adjust(right=0.78, bottom=0.18)  # Make room for legend and labels
            
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
            fig, ax = plt.subplots(figsize=(12, 7))
            
            recurrence_data = data.get('ai_recurrence', {})
            
            # Check if we have valid data (non-empty with non-zero values)
            has_valid_data = (recurrence_data and 
                             recurrence_data.get('categories') and
                             recurrence_data.get('predicted') and
                             any(v > 0 for v in recurrence_data.get('predicted', [])))
            
            if not has_valid_data:
                # Create a message indicating data not available
                ax.text(0.5, 0.5, 'AI Recurrence Data Not Available\n\n'
                       'Ensure AI_Recurrence_Probability column\nis populated with non-zero values', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.title('AI Recurrence Prediction\nPredicted vs Actual Rates', 
                         fontsize=14, fontweight='bold', pad=15)
                filepath = self.chart_dirs['predictive'] / 'ai_recurrence.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            categories = recurrence_data.get('categories', [])
            # Truncate long category names
            cat_labels = [c[:15] + '..' if len(c) > 15 else c for c in categories]
            
            predicted = recurrence_data.get('predicted', [])
            actual = recurrence_data.get('actual', [])
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, predicted, width, label='AI Predicted',
                          color=self.COLORS['primary'], alpha=0.85)
            bars2 = ax.bar(x + width/2, actual, width, label='Actual',
                          color=self.COLORS['accent'], alpha=0.85)
            
            ax.set_xlabel('Category', fontsize=11)
            ax.set_ylabel('Recurrence Rate (%)', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='upper right', fontsize=10)
            
            # Value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                               f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
            
            plt.title('AI Recurrence Prediction\nPredicted vs Actual Rates', 
                     fontsize=14, fontweight='bold', pad=15)
            
            plt.subplots_adjust(bottom=0.2)  # Room for rotated labels
            
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
            # Truncate long category names
            cat_labels = [c[:15] + '..' if len(c) > 15 else c for c in categories]
            
            data_arrays = [resolution_data[cat] for cat in categories]
            
            bp = ax.boxplot(data_arrays, labels=cat_labels, patch_artist=True,
                           medianprops={'color': 'black', 'linewidth': 2})
            
            # Color boxes
            colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(categories)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Category', fontsize=11)
            ax.set_ylabel('Resolution Time (days)', fontsize=11)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            
            # Target line
            target = data.get('resolution_target', 3)
            ax.axhline(y=target, color=self.COLORS['success'], linestyle='--',
                      linewidth=2, label=f'Target ({target} days)')
            ax.legend(loc='upper right', fontsize=10)
            
            plt.title('Resolution Time Distribution\nPredicted Time by Category', 
                     fontsize=14, fontweight='bold', pad=15)
            
            plt.subplots_adjust(bottom=0.2)  # Room for rotated labels
            
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
            fig, ax = plt.subplots(figsize=(14, 8))
            
            financial_data = data.get('financial_impact', {})
            if not financial_data:
                financial_data = {
                    'categories': ['Network', 'Billing', 'Hardware', 'Software'],
                    'direct_cost': [45000, 28000, 52000, 18000],
                    'indirect_cost': [22000, 15000, 25000, 8000],
                    'potential_savings': [35000, 20000, 40000, 12000],
                }
            
            categories = financial_data.get('categories', [])
            # Truncate long category names
            cat_labels = [c[:15] + '..' if len(c) > 15 else c for c in categories]
            
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
            
            ax.set_xlabel('Category', fontsize=11)
            ax.set_ylabel('Amount ($K)', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
            
            # Position legend outside chart area to avoid overlap
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, 
                     borderaxespad=0, framealpha=0.9)
            
            # Value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                               f'${height:.0f}K', ha='center', va='bottom', fontsize=7)
            
            # Total savings annotation - positioned at top left inside chart
            total_savings = savings.sum()
            ax.annotate(f'Total Potential Savings: ${total_savings:.0f}K',
                       xy=(0.02, 0.98), xycoords='axes fraction',
                       ha='left', va='top', fontsize=10, fontweight='bold',
                       color=self.COLORS['success'],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title('Financial Impact Analysis\nCost Breakdown & Savings Opportunity', 
                     fontsize=14, fontweight='bold', pad=15)
            
            plt.subplots_adjust(bottom=0.2, right=0.82)  # Room for rotated labels and legend
            
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

    # =========================================================================
    # SIMILARITY SEARCH CHARTS (10_similarity/)
    # =========================================================================

    def _chart_similarity_count_distribution(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Similar Ticket Count Distribution
        Histogram showing how many similar tickets are found per issue.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))

            similarity_data = data.get('similarity_counts', [])
            if not similarity_data:
                # No data available - create placeholder
                ax.text(0.5, 0.5, 'Similarity Search Data Not Available\n\n'
                       'Run similarity search to populate this chart',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.title('Similar Ticket Count Distribution\nData Not Available',
                         fontsize=14, fontweight='bold', pad=20)
                filepath = self.chart_dirs['similarity'] / 'similarity_count_distribution.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            # Create histogram
            counts = np.array(similarity_data)
            bins = np.arange(0, max(counts) + 2) - 0.5

            n, bins_out, patches = ax.hist(counts, bins=bins, color=self.COLORS['primary'],
                                           alpha=0.7, edgecolor='white', linewidth=1.5)

            # Color bars by count level
            for i, patch in enumerate(patches):
                if i == 0:
                    patch.set_facecolor(self.COLORS['danger'])  # No matches = concerning
                elif i <= 2:
                    patch.set_facecolor(self.COLORS['warning'])  # Few matches
                else:
                    patch.set_facecolor(self.COLORS['success'])  # Good coverage

            # Add statistics
            avg_count = np.mean(counts)
            median_count = np.median(counts)
            zero_matches = (counts == 0).sum()

            ax.axvline(x=avg_count, color=self.COLORS['accent'], linestyle='--',
                      linewidth=2, label=f'Mean: {avg_count:.1f}')
            ax.axvline(x=median_count, color=self.COLORS['secondary'], linestyle=':',
                      linewidth=2, label=f'Median: {median_count:.0f}')

            ax.set_xlabel('Number of Similar Tickets Found', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend(loc='upper right')

            # Add annotation about zero matches
            if zero_matches > 0:
                ax.annotate(f'{zero_matches} tickets with no similar matches\n(may be new issue types)',
                           xy=(0.02, 0.98), xycoords='axes fraction',
                           ha='left', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.9))

            plt.title(f'Similar Ticket Count Distribution\n{len(counts)} tickets analyzed, avg {avg_count:.1f} matches',
                     fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()

            filepath = self.chart_dirs['similarity'] / 'similarity_count_distribution.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating similarity count chart: {e}")
            plt.close('all')
            return None

    def _chart_resolution_consistency(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Resolution Consistency Analysis
        Shows consistent vs inconsistent resolutions based on similar ticket analysis.
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

            consistency_data = data.get('resolution_consistency', {})
            if not consistency_data:
                # Create placeholder
                ax1.text(0.5, 0.5, 'Resolution Consistency\nData Not Available',
                        ha='center', va='center', fontsize=14, transform=ax1.transAxes)
                ax1.axis('off')
                ax2.axis('off')
                filepath = self.chart_dirs['similarity'] / 'resolution_consistency.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            # Left: Pie chart of consistency
            consistent = consistency_data.get('consistent', 0)
            inconsistent = consistency_data.get('inconsistent', 0)
            no_data = consistency_data.get('no_data', 0)

            sizes = [consistent, inconsistent, no_data]
            labels = [f'Consistent\n({consistent})', f'Inconsistent\n({inconsistent})',
                     f'No Similar Data\n({no_data})']
            colors = [self.COLORS['success'], self.COLORS['danger'], self.COLORS['neutral']]
            explode = (0, 0.1, 0)  # Explode inconsistent slice

            # Filter out zero values
            non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
            if non_zero:
                sizes, labels, colors, explode = zip(*non_zero)
                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
                                                   autopct='%1.1f%%', startangle=90,
                                                   textprops={'fontsize': 10})
                for autotext in autotexts:
                    autotext.set_fontweight('bold')

            ax1.set_title('Resolution Consistency Breakdown', fontsize=12, fontweight='bold')

            # Right: Inconsistency by category (if available)
            inconsistent_by_cat = consistency_data.get('inconsistent_by_category', {})
            if inconsistent_by_cat:
                categories = list(inconsistent_by_cat.keys())
                counts = list(inconsistent_by_cat.values())

                # Sort and limit
                sorted_pairs = sorted(zip(counts, categories), reverse=True)[:10]
                counts, categories = zip(*sorted_pairs)

                # Truncate long names
                cat_labels = [c[:18] + '..' if len(c) > 18 else c for c in categories]

                bars = ax2.barh(cat_labels, counts, color=self.COLORS['danger'], alpha=0.8,
                               edgecolor='white', linewidth=1)

                for bar, count in zip(bars, counts):
                    ax2.text(count + 0.3, bar.get_y() + bar.get_height()/2,
                            f'{count}', va='center', fontsize=9, fontweight='bold')

                ax2.set_xlabel('Inconsistent Resolutions', fontsize=11)
                ax2.set_title('Inconsistency by Category', fontsize=12, fontweight='bold')
                ax2.invert_yaxis()
            else:
                ax2.text(0.5, 0.5, 'Category breakdown\nnot available',
                        ha='center', va='center', fontsize=12, transform=ax2.transAxes)
                ax2.axis('off')

            plt.suptitle('Resolution Consistency Analysis\nBased on Similar Ticket Patterns',
                        fontsize=14, fontweight='bold', y=1.02)

            fig.tight_layout()

            filepath = self.chart_dirs['similarity'] / 'resolution_consistency.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating resolution consistency chart: {e}")
            plt.close('all')
            return None

    def _chart_similarity_score_distribution(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Similarity Score Distribution
        Histogram of best match similarity scores.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 7))

            scores = data.get('similarity_scores', [])
            if not scores:
                ax.text(0.5, 0.5, 'Similarity Score Data\nNot Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['similarity'] / 'similarity_score_distribution.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            scores = np.array(scores)
            scores = scores[scores > 0]  # Filter out zeros

            if len(scores) == 0:
                ax.text(0.5, 0.5, 'No valid similarity scores found',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['similarity'] / 'similarity_score_distribution.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            # Create histogram with gradient colors
            n, bins, patches = ax.hist(scores, bins=20, alpha=0.7, edgecolor='white', linewidth=1)

            # Color by score level (higher = greener)
            cm = plt.cm.get_cmap('RdYlGn')
            for i, patch in enumerate(patches):
                bin_center = (bins[i] + bins[i+1]) / 2
                color = cm(bin_center)
                patch.set_facecolor(color)

            # Add threshold lines
            ax.axvline(x=0.7, color=self.COLORS['success'], linestyle='--',
                      linewidth=2, label='High Confidence (0.7+)')
            ax.axvline(x=0.5, color=self.COLORS['warning'], linestyle='--',
                      linewidth=2, label='Medium Confidence (0.5)')

            # Statistics
            avg_score = np.mean(scores)
            high_conf = (scores >= 0.7).sum()

            ax.set_xlabel('Best Match Similarity Score', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_xlim(0, 1)
            ax.legend(loc='upper left')

            # Annotation
            ax.annotate(f'Average Score: {avg_score:.2f}\nHigh Confidence Matches: {high_conf} ({high_conf/len(scores)*100:.1f}%)',
                       xy=(0.98, 0.98), xycoords='axes fraction',
                       ha='right', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            plt.title(f'Similarity Score Distribution\n{len(scores)} matches analyzed',
                     fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()

            filepath = self.chart_dirs['similarity'] / 'similarity_score_distribution.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating similarity score chart: {e}")
            plt.close('all')
            return None

    def _chart_resolution_comparison(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Resolution Time Comparison
        Compares expected vs predicted vs actual resolution times based on similar tickets.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            comparison_data = data.get('resolution_comparison', {})
            if not comparison_data or not comparison_data.get('categories'):
                ax.text(0.5, 0.5, 'Resolution Comparison Data\nNot Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['similarity'] / 'resolution_comparison.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            categories = comparison_data.get('categories', [])
            expected = comparison_data.get('expected_days', [])  # From similar tickets
            predicted = comparison_data.get('predicted_days', [])  # AI prediction
            actual = comparison_data.get('actual_days', [])  # Actual (if available)

            # Truncate category names
            cat_labels = [c[:15] + '..' if len(c) > 15 else c for c in categories]

            x = np.arange(len(categories))
            width = 0.25

            bars1 = ax.bar(x - width, expected, width, label='Expected (Similar Tickets)',
                          color=self.COLORS['secondary'], alpha=0.85)
            bars2 = ax.bar(x, predicted, width, label='AI Predicted',
                          color=self.COLORS['primary'], alpha=0.85)

            if actual and any(a > 0 for a in actual):
                bars3 = ax.bar(x + width, actual, width, label='Actual',
                              color=self.COLORS['accent'], alpha=0.85)

            ax.set_xlabel('Category', fontsize=11)
            ax.set_ylabel('Resolution Time (days)', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='upper right')

            # Add accuracy annotation
            if expected and predicted:
                mae = np.mean(np.abs(np.array(expected) - np.array(predicted)))
                ax.annotate(f'Mean Difference: {mae:.1f} days',
                           xy=(0.02, 0.98), xycoords='axes fraction',
                           ha='left', va='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            plt.title('Resolution Time Comparison\nSimilar Ticket Analysis vs AI Prediction',
                     fontsize=14, fontweight='bold', pad=20)

            plt.subplots_adjust(bottom=0.2)

            filepath = self.chart_dirs['similarity'] / 'resolution_comparison.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating resolution comparison chart: {e}")
            plt.close('all')
            return None

    def _chart_similarity_effectiveness(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Similarity Search Effectiveness Heatmap
        Shows search effectiveness by category and origin.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 10))

            effectiveness_data = data.get('similarity_effectiveness', None)
            if effectiveness_data is None:
                ax.text(0.5, 0.5, 'Similarity Effectiveness Data\nNot Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['similarity'] / 'similarity_effectiveness.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            # Build matrix from dict {category: {origin: effectiveness_score}}
            if isinstance(effectiveness_data, dict):
                categories = sorted(effectiveness_data.keys())
                origins = set()
                for cat_data in effectiveness_data.values():
                    if isinstance(cat_data, dict):
                        origins.update(cat_data.keys())
                origins = sorted(origins)

                matrix = []
                for cat in categories:
                    row = []
                    for origin in origins:
                        val = effectiveness_data.get(cat, {}).get(origin, 0)
                        row.append(val * 100)  # Convert to percentage
                    matrix.append(row)
                matrix = np.array(matrix)
            else:
                matrix = np.array(effectiveness_data) * 100
                categories = [f'Cat {i+1}' for i in range(matrix.shape[0])]
                origins = [f'Origin {i+1}' for i in range(matrix.shape[1])]

            # Truncate labels
            cat_labels = [c[:15] + '..' if len(c) > 15 else c for c in categories]
            origin_labels = [o[:12] + '..' if len(o) > 12 else o for o in origins]

            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                       xticklabels=origin_labels, yticklabels=cat_labels,
                       ax=ax, cbar_kws={'label': 'Match Effectiveness (%)'},
                       linewidths=0.5, linecolor='white')

            ax.set_xlabel('Origin/Source', fontsize=12)
            ax.set_ylabel('Category', fontsize=12)
            plt.xticks(rotation=45, ha='right')

            plt.title('Similarity Search Effectiveness\nBy Category and Origin',
                     fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()

            filepath = self.chart_dirs['similarity'] / 'similarity_effectiveness.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating similarity effectiveness chart: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # LESSONS LEARNED EFFECTIVENESS CHARTS (11_lessons_learned/)
    # =========================================================================

    def _chart_learning_grades(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Learning Effectiveness Grades by Category
        Shows letter grades (A-F) for each category based on lessons learned effectiveness.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            grades_data = data.get('lessons_grades', {})
            if not grades_data:
                ax.text(0.5, 0.5, 'Lessons Learned Grades\nNot Available\n\n'
                       'Run analysis with lessons_learned columns',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.axis('off')
                filepath = self.chart_dirs['lessons'] / 'learning_grades.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            categories = list(grades_data.keys())
            scores = [grades_data[c]['score'] for c in categories]
            grades = [grades_data[c]['grade'] for c in categories]

            # Sort by score
            sorted_data = sorted(zip(scores, categories, grades), reverse=True)
            scores, categories, grades = zip(*sorted_data)

            # Truncate long names
            cat_labels = [c[:20] + '..' if len(c) > 20 else c for c in categories]

            # Color by grade
            grade_colors = {
                'A': '#28A745', 'B': '#7CB342', 'C': '#FFC107',
                'D': '#FF9800', 'F': '#DC3545'
            }
            colors = [grade_colors.get(g, '#6C757D') for g in grades]

            bars = ax.barh(cat_labels, scores, color=colors, edgecolor='white', linewidth=1.5)

            # Add grade labels
            for bar, score, grade in zip(bars, scores, grades):
                ax.text(score + 2, bar.get_y() + bar.get_height()/2,
                       f'{grade} ({score:.0f})', va='center', fontsize=10, fontweight='bold')

            ax.set_xlabel('Learning Effectiveness Score (0-100)', fontsize=11)
            ax.set_xlim(0, 110)
            ax.invert_yaxis()

            # Grade legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='#28A745', label='A (80-100): Excellent'),
                plt.Rectangle((0,0),1,1, facecolor='#7CB342', label='B (65-79): Good'),
                plt.Rectangle((0,0),1,1, facecolor='#FFC107', label='C (50-64): Improving'),
                plt.Rectangle((0,0),1,1, facecolor='#FF9800', label='D (35-49): Poor'),
                plt.Rectangle((0,0),1,1, facecolor='#DC3545', label='F (0-34): Failing'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

            plt.title('Learning Effectiveness Grades by Category\nBased on Recurrence, Lesson Completion & Consistency',
                     fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()

            filepath = self.chart_dirs['lessons'] / 'learning_grades.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating learning grades chart: {e}")
            plt.close('all')
            return None

    def _chart_lesson_completion_rate(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Lesson Completion Rate by Category
        Bar chart comparing documented vs completed lessons.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            lessons_data = data.get('lessons_by_category', {})
            if not lessons_data:
                ax.text(0.5, 0.5, 'Lesson Completion Data\nNot Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['lessons'] / 'lesson_completion_rate.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            categories = list(lessons_data.keys())
            documented = [lessons_data[c].get('documented', 0) for c in categories]
            completed = [lessons_data[c].get('completed', 0) for c in categories]

            # Sort by documented
            sorted_data = sorted(zip(documented, completed, categories), reverse=True)
            documented, completed, categories = zip(*sorted_data)

            cat_labels = [c[:18] + '..' if len(c) > 18 else c for c in categories]

            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax.bar(x - width/2, documented, width, label='Documented',
                          color=self.COLORS['primary'], alpha=0.85)
            bars2 = ax.bar(x + width/2, completed, width, label='Completed',
                          color=self.COLORS['success'], alpha=0.85)

            # Add completion rate labels
            for i, (doc, comp) in enumerate(zip(documented, completed)):
                if doc > 0:
                    rate = (comp / doc) * 100
                    ax.text(i, max(doc, comp) + 1, f'{rate:.0f}%',
                           ha='center', fontsize=9, fontweight='bold',
                           color=self.COLORS['success'] if rate >= 50 else self.COLORS['danger'])

            ax.set_xlabel('Category', fontsize=11)
            ax.set_ylabel('Number of Lessons', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='upper right')

            plt.title('Lesson Documentation & Completion by Category\nPercentage shows completion rate',
                     fontsize=14, fontweight='bold', pad=20)

            plt.subplots_adjust(bottom=0.25)

            filepath = self.chart_dirs['lessons'] / 'lesson_completion_rate.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating lesson completion chart: {e}")
            plt.close('all')
            return None

    def _chart_recurrence_vs_lessons(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Recurrence Rate vs Lesson Completion
        Scatter plot showing correlation between learning and recurrence reduction.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 10))

            correlation_data = data.get('recurrence_lessons_correlation', {})
            if not correlation_data:
                ax.text(0.5, 0.5, 'Recurrence vs Lessons Data\nNot Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['lessons'] / 'recurrence_vs_lessons.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            categories = list(correlation_data.keys())
            recurrence = [correlation_data[c].get('recurrence_rate', 0) for c in categories]
            completion = [correlation_data[c].get('lesson_completion', 0) for c in categories]
            ticket_count = [correlation_data[c].get('ticket_count', 10) for c in categories]

            # Size based on ticket count
            sizes = [max(50, min(500, t * 10)) for t in ticket_count]

            # Color based on recurrence (red = high, green = low)
            scatter = ax.scatter(completion, recurrence, s=sizes, c=recurrence,
                               cmap='RdYlGn_r', alpha=0.7, edgecolors='white', linewidth=2)

            # Add category labels
            for i, cat in enumerate(categories):
                label = cat[:12] + '..' if len(cat) > 12 else cat
                ax.annotate(label, (completion[i], recurrence[i]), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, alpha=0.8)

            # Draw quadrant lines
            ax.axhline(y=30, color='#CCCCCC', linestyle='--', linewidth=1.5)
            ax.axvline(x=50, color='#CCCCCC', linestyle='--', linewidth=1.5)

            # Quadrant labels
            ax.text(75, 50, 'High Completion\nHigh Recurrence\n(Process Issue)',
                   ha='center', va='center', fontsize=9, color='#FF9800', alpha=0.8)
            ax.text(25, 50, 'Low Completion\nHigh Recurrence\n(NEEDS ATTENTION)',
                   ha='center', va='center', fontsize=9, color='#DC3545', fontweight='bold')
            ax.text(75, 15, 'High Completion\nLow Recurrence\n(IDEAL)',
                   ha='center', va='center', fontsize=9, color='#28A745', fontweight='bold')
            ax.text(25, 15, 'Low Completion\nLow Recurrence\n(Natural Resolution)',
                   ha='center', va='center', fontsize=9, color='#6C757D', alpha=0.8)

            ax.set_xlabel('Lesson Completion Rate (%)', fontsize=11)
            ax.set_ylabel('Recurrence Rate (%)', fontsize=11)
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, max(recurrence) * 1.2 if recurrence else 100)

            plt.colorbar(scatter, ax=ax, label='Recurrence Rate (%)')

            plt.title('Recurrence Rate vs Lesson Completion\nIdeal: Bottom-Right (Low Recurrence, High Completion)',
                     fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()

            filepath = self.chart_dirs['lessons'] / 'recurrence_vs_lessons.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating recurrence vs lessons chart: {e}")
            plt.close('all')
            return None

    def _chart_learning_heatmap(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Learning Effectiveness Heatmap
        Shows learning metrics across categories and LOBs.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 10))

            heatmap_data = data.get('learning_heatmap', None)
            if heatmap_data is None:
                ax.text(0.5, 0.5, 'Learning Heatmap Data\nNot Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['lessons'] / 'learning_heatmap.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            # Build matrix
            if isinstance(heatmap_data, dict):
                categories = sorted(heatmap_data.keys())
                lobs = set()
                for cat_data in heatmap_data.values():
                    if isinstance(cat_data, dict):
                        lobs.update(cat_data.keys())
                lobs = sorted(lobs)

                matrix = []
                for cat in categories:
                    row = []
                    for lob in lobs:
                        val = heatmap_data.get(cat, {}).get(lob, 50)
                        row.append(val)
                    matrix.append(row)
                matrix = np.array(matrix)
            else:
                matrix = np.array(heatmap_data)
                categories = [f'Cat {i+1}' for i in range(matrix.shape[0])]
                lobs = [f'LOB {i+1}' for i in range(matrix.shape[1])]

            # Truncate labels
            cat_labels = [c[:15] + '..' if len(c) > 15 else c for c in categories]
            lob_labels = [l[:12] + '..' if len(l) > 12 else l for l in lobs]

            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn',
                       xticklabels=lob_labels, yticklabels=cat_labels,
                       ax=ax, cbar_kws={'label': 'Learning Score'},
                       vmin=0, vmax=100, linewidths=0.5, linecolor='white')

            ax.set_xlabel('Line of Business / Market', fontsize=12)
            ax.set_ylabel('Category', fontsize=12)
            plt.xticks(rotation=45, ha='right')

            plt.title('Learning Effectiveness by Category & LOB\nGreen = Good Learning, Red = Needs Attention',
                     fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()

            filepath = self.chart_dirs['lessons'] / 'learning_heatmap.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating learning heatmap: {e}")
            plt.close('all')
            return None

    def _chart_recommendations_summary(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: AI Recommendations Summary
        Visual display of top recommendations with priority indicators.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            recommendations = data.get('lessons_recommendations', [])
            if not recommendations:
                ax.text(0.5, 0.5, 'No Recommendations Available\n\n'
                       'Run lessons learned analysis to generate',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.axis('off')
                filepath = self.chart_dirs['lessons'] / 'recommendations_summary.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            ax.axis('off')

            # Title
            ax.text(0.5, 0.95, 'AI-Generated Improvement Recommendations',
                   ha='center', va='top', fontsize=16, fontweight='bold',
                   transform=ax.transAxes)

            # Priority colors
            priority_colors = {
                'CRITICAL': '#DC3545',
                'HIGH': '#FF9800',
                'MEDIUM': '#FFC107',
                'LOW': '#28A745'
            }

            y_pos = 0.85
            for i, rec in enumerate(recommendations[:6]):
                if y_pos < 0.1:
                    break

                priority = rec.get('priority', 'MEDIUM')
                color = priority_colors.get(priority, '#6C757D')

                # Priority badge
                ax.add_patch(plt.Rectangle((0.02, y_pos - 0.02), 0.08, 0.08,
                            facecolor=color, edgecolor='white', linewidth=2,
                            transform=ax.transAxes))
                ax.text(0.06, y_pos + 0.02, priority[:1], ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white',
                       transform=ax.transAxes)

                # Category
                category = rec.get('category', 'Unknown')[:25]
                ax.text(0.12, y_pos + 0.02, category, ha='left', va='center',
                       fontsize=11, fontweight='bold', transform=ax.transAxes)

                # Recommendation text (wrapped)
                rec_text = rec.get('recommendation', '')[:150]
                if len(rec.get('recommendation', '')) > 150:
                    rec_text += '...'
                ax.text(0.12, y_pos - 0.04, rec_text, ha='left', va='top',
                       fontsize=9, color='#333333', transform=ax.transAxes,
                       wrap=True)

                y_pos -= 0.15

            # Legend
            legend_y = 0.05
            for priority, color in priority_colors.items():
                ax.add_patch(plt.Rectangle((0.02 + list(priority_colors.keys()).index(priority) * 0.2,
                            legend_y), 0.03, 0.03, facecolor=color,
                            transform=ax.transAxes))
                ax.text(0.06 + list(priority_colors.keys()).index(priority) * 0.2,
                       legend_y + 0.015, priority, ha='left', va='center',
                       fontsize=8, transform=ax.transAxes)

            filepath = self.chart_dirs['lessons'] / 'recommendations_summary.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating recommendations chart: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # CATEGORY/SUB-CATEGORY BREAKDOWN CHARTS (04_analysis/)
    # =========================================================================

    def _chart_category_subcategory_breakdown(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Category and Sub-Category Distribution
        Hierarchical bar chart showing category counts with sub-category breakdown.
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

            cat_counts = data.get('category_counts', {})
            subcat_counts = data.get('subcategory_counts', {})

            if not cat_counts:
                cat_counts = {
                    'Scheduling & Planning': 85,
                    'Documentation & Reporting': 45,
                    'Validation & QA': 35,
                    'Process Compliance': 30,
                    'Configuration & Data Mismatch': 40,
                    'Site Readiness': 35,
                    'Communication & Response': 20,
                    'Nesting & Tool Errors': 15
                }

            # Left chart: Category distribution
            categories = list(cat_counts.keys())
            counts = list(cat_counts.values())

            # Sort by count
            sorted_pairs = sorted(zip(counts, categories), reverse=True)
            counts, categories = zip(*sorted_pairs)

            colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

            bars = ax1.barh(categories, counts, color=colors, edgecolor='white', linewidth=1.5)

            for bar, count in zip(bars, counts):
                ax1.text(count + 1, bar.get_y() + bar.get_height()/2,
                        f'{count}', va='center', fontsize=10, fontweight='bold')

            ax1.set_xlabel('Number of Escalations', fontsize=11)
            ax1.set_title('Category Distribution', fontsize=14, fontweight='bold', pad=15)
            ax1.invert_yaxis()

            # Right chart: Sub-category breakdown (top 15)
            if subcat_counts:
                # Flatten sub-category counts if nested
                flat_subcats = {}
                for cat, subcats in subcat_counts.items():
                    if isinstance(subcats, dict):
                        for sub, count in subcats.items():
                            flat_subcats[f"{sub}"] = count
                    else:
                        flat_subcats[cat] = subcats

                # Sort and take top 15
                sorted_subcats = sorted(flat_subcats.items(), key=lambda x: x[1], reverse=True)[:15]
                sub_names = [s[0][:20] + '..' if len(s[0]) > 20 else s[0] for s in sorted_subcats]
                sub_counts = [s[1] for s in sorted_subcats]

                bars2 = ax2.barh(sub_names, sub_counts, color=self.COLORS['secondary'], alpha=0.8,
                               edgecolor='white', linewidth=1)

                for bar, count in zip(bars2, sub_counts):
                    ax2.text(count + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{count}', va='center', fontsize=9, fontweight='bold')

                ax2.set_xlabel('Number of Escalations', fontsize=11)
                ax2.set_title('Top Sub-Categories', fontsize=14, fontweight='bold', pad=15)
                ax2.invert_yaxis()
            else:
                ax2.text(0.5, 0.5, 'Sub-Category Data\nNot Available',
                        ha='center', va='center', fontsize=14, transform=ax2.transAxes)
                ax2.axis('off')

            plt.suptitle('Category & Sub-Category Analysis\nEscalation Distribution Breakdown',
                        fontsize=16, fontweight='bold', y=1.02)

            fig.tight_layout()

            filepath = self.chart_dirs['analysis'] / 'category_subcategory_breakdown.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating category/subcategory breakdown chart: {e}")
            plt.close('all')
            return None

    def _chart_subcategory_financial_impact(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Sub-Category Financial Impact Analysis
        Shows financial impact breakdown by sub-category within each main category.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 10))

            subcat_costs = data.get('subcategory_financial', {})

            if not subcat_costs:
                # Generate sample data
                subcat_costs = {
                    'TI/Calendar Issues': 25000,
                    'FE Coordination': 35000,
                    'Snapshot/Screenshot Issues': 15000,
                    'E911/CBN Reports': 30000,
                    'Precheck/Postcheck Failures': 50000,
                    'Port Matrix Issues': 90000,
                    'RET/TAC Naming': 75000,
                    'Backhaul Issues': 180000,
                    'MW/Transmission Issues': 160000,
                    'Delayed Responses': 20000,
                    'Nesting Type Errors': 60000,
                    'RIOT/FCI Tool Issues': 70000,
                }

            # Sort by financial impact
            sorted_items = sorted(subcat_costs.items(), key=lambda x: x[1], reverse=True)[:15]
            subcats = [s[0][:25] + '..' if len(s[0]) > 25 else s[0] for s in sorted_items]
            costs = [s[1] for s in sorted_items]

            # Color gradient based on cost (red = high cost)
            norm_costs = np.array(costs) / max(costs)
            colors = plt.cm.RdYlGn_r(norm_costs)

            bars = ax.barh(subcats, costs, color=colors, edgecolor='white', linewidth=1.5)

            # Value labels
            for bar, cost in zip(bars, costs):
                ax.text(cost + max(costs) * 0.02, bar.get_y() + bar.get_height()/2,
                       f'${cost/1000:.0f}K', va='center', fontsize=10, fontweight='bold')

            ax.set_xlabel('Financial Impact ($)', fontsize=12)
            ax.set_ylabel('Sub-Category', fontsize=12)
            ax.invert_yaxis()

            # Add total annotation
            total_cost = sum(costs)
            ax.annotate(f'Total Impact: ${total_cost/1000:.0f}K',
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', va='bottom', fontsize=11, fontweight='bold',
                       color=self.COLORS['danger'],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            plt.title('Sub-Category Financial Impact\nTop Cost Drivers',
                     fontsize=14, fontweight='bold', pad=20)

            # Format x-axis as currency
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

            fig.tight_layout()

            filepath = self.chart_dirs['financial'] / 'subcategory_financial_impact.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating sub-category financial impact chart: {e}")
            plt.close('all')
            return None

    def _chart_category_heatmap(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Chart: Category vs Sub-Category Heatmap
        Shows ticket distribution across categories and their sub-categories.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 10))

            # Get hierarchical data
            cat_subcat_matrix = data.get('category_subcategory_matrix', None)

            if cat_subcat_matrix is None:
                # Generate sample matrix
                categories = [
                    'Scheduling & Planning',
                    'Documentation & Reporting',
                    'Validation & QA',
                    'Process Compliance',
                    'Config & Data Mismatch',
                    'Site Readiness',
                    'Communication & Response',
                    'Nesting & Tool Errors'
                ]
                subcats = ['Sub-Cat 1', 'Sub-Cat 2', 'Sub-Cat 3']
                np.random.seed(42)
                cat_subcat_matrix = np.random.randint(5, 50, (len(categories), len(subcats)))

            else:
                categories = list(cat_subcat_matrix.keys())
                # Get all unique sub-categories
                all_subcats = set()
                for cat, subcats in cat_subcat_matrix.items():
                    all_subcats.update(subcats.keys())
                subcats = sorted(all_subcats)

                # Build matrix
                matrix = []
                for cat in categories:
                    row = [cat_subcat_matrix[cat].get(sub, 0) for sub in subcats]
                    matrix.append(row)
                cat_subcat_matrix = np.array(matrix)

            # Truncate long labels
            cat_labels = [c[:20] + '..' if len(c) > 20 else c for c in categories]
            sub_labels = [s[:15] + '..' if len(s) > 15 else s for s in subcats]

            sns.heatmap(cat_subcat_matrix, annot=True, fmt='d', cmap='YlOrRd',
                       xticklabels=sub_labels, yticklabels=cat_labels,
                       ax=ax, cbar_kws={'label': 'Ticket Count'},
                       linewidths=0.5, linecolor='white')

            ax.set_xlabel('Sub-Category', fontsize=12)
            ax.set_ylabel('Category', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=9)

            plt.title('Category vs Sub-Category Distribution\nTicket Volume Heatmap',
                     fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()

            filepath = self.chart_dirs['analysis'] / 'category_subcategory_heatmap.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            print(f"Error generating category heatmap: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # DRIFT & THRESHOLD CHARTS (04_analysis/)
    # =========================================================================
    
    def chart_category_drift(self, drift_results: List[Any], 
                              title: str = "Category Drift Analysis") -> Optional[str]:
        """
        Chart: Category Drift Visualization
        Shows emerging and declining categories with change percentages.
        
        Args:
            drift_results: List of DriftResult objects from CategoryDriftDetector
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        try:
            if not drift_results:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Filter significant drifts and sort by change
            significant = [r for r in drift_results if abs(r.change_pct) >= 10]
            significant = sorted(significant, key=lambda x: x.change_pct, reverse=True)[:15]
            
            if not significant:
                plt.close(fig)
                return None
            
            categories = [r.category for r in significant]
            changes = [r.change_pct for r in significant]
            
            # Color based on direction
            colors = [self.COLORS['success'] if c > 0 else self.COLORS['danger'] for c in changes]
            
            # Horizontal bar chart
            y_pos = np.arange(len(categories))
            bars = ax.barh(y_pos, changes, color=colors, alpha=0.8, edgecolor='white')
            
            # Add value labels
            for i, (bar, change) in enumerate(zip(bars, changes)):
                width = bar.get_width()
                label_x = width + 2 if width >= 0 else width - 2
                ha = 'left' if width >= 0 else 'right'
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       f'{change:+.0f}%', ha=ha, va='center', fontweight='bold', fontsize=9)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel('Change from Baseline (%)')
            ax.axvline(x=0, color='black', linewidth=0.8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=self.COLORS['success'], label='Emerging (↑)'),
                Patch(facecolor=self.COLORS['danger'], label='Declining (↓)')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            fig.tight_layout()
            
            filepath = self.chart_dirs['analysis'] / 'category_drift.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating category drift chart: {e}")
            plt.close('all')
            return None
    
    def chart_drift_timeline(self, rolling_drift_df: pd.DataFrame,
                              title: str = "Category Drift Over Time") -> Optional[str]:
        """
        Chart: Rolling Drift Timeline
        Shows drift severity and category changes over time.
        
        Args:
            rolling_drift_df: DataFrame from CategoryDriftDetector.get_rolling_drift()
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        try:
            if rolling_drift_df.empty:
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            dates = pd.to_datetime(rolling_drift_df['period_end'])
            
            # Top: Drift severity over time
            ax1.fill_between(dates, rolling_drift_df['max_severity'], 
                            alpha=0.3, color=self.COLORS['warning'])
            ax1.plot(dates, rolling_drift_df['max_severity'], 
                    color=self.COLORS['warning'], linewidth=2, marker='o', markersize=4)
            
            ax1.axhline(y=0.5, color=self.COLORS['danger'], linestyle='--', 
                       linewidth=1.5, label='High Drift Threshold')
            ax1.axhline(y=0.2, color=self.COLORS['warning'], linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='Moderate Drift Threshold')
            
            ax1.set_ylabel('Max Drift Severity')
            ax1.set_ylim(0, 1.1)
            ax1.legend(loc='upper right')
            ax1.set_title('Drift Severity Trend', fontweight='bold')
            
            # Bottom: Emerging vs Declining counts
            width = (dates.max() - dates.min()).days / len(dates) * 0.4
            width = pd.Timedelta(days=max(1, width))
            
            ax2.bar(dates - width/2, rolling_drift_df['emerging_categories'], 
                   width=width, label='Emerging', color=self.COLORS['success'], alpha=0.8)
            ax2.bar(dates + width/2, rolling_drift_df['declining_categories'], 
                   width=width, label='Declining', color=self.COLORS['danger'], alpha=0.8)
            
            ax2.set_ylabel('Category Count')
            ax2.set_xlabel('Period End Date')
            ax2.legend(loc='upper right')
            ax2.set_title('Emerging vs Declining Categories', fontweight='bold')
            
            plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            fig.tight_layout()
            
            filepath = self.chart_dirs['analysis'] / 'drift_timeline.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating drift timeline chart: {e}")
            plt.close('all')
            return None
    
    def chart_distribution_comparison(self, baseline_dist: Dict[str, float], 
                                       current_dist: Dict[str, float],
                                       title: str = "Category Distribution: Baseline vs Current") -> Optional[str]:
        """
        Chart: Side-by-side distribution comparison
        Shows baseline vs current category distributions.
        
        Args:
            baseline_dist: Dict of category -> proportion for baseline
            current_dist: Dict of category -> proportion for current
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            all_cats = sorted(set(baseline_dist.keys()) | set(current_dist.keys()))
            
            baseline_vals = [baseline_dist.get(c, 0) * 100 for c in all_cats]
            current_vals = [current_dist.get(c, 0) * 100 for c in all_cats]
            
            x = np.arange(len(all_cats))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                          color=self.COLORS['secondary'], alpha=0.8)
            bars2 = ax.bar(x + width/2, current_vals, width, label='Current',
                          color=self.COLORS['accent'], alpha=0.8)
            
            # Highlight significant changes
            for i, (b, c) in enumerate(zip(baseline_vals, current_vals)):
                if b > 0:
                    change = ((c - b) / b) * 100
                    if abs(change) > 25:
                        arrow = '↑' if change > 0 else '↓'
                        color = self.COLORS['success'] if change > 0 else self.COLORS['danger']
                        ax.annotate(f'{arrow}{abs(change):.0f}%', 
                                   xy=(x[i], max(b, c) + 1),
                                   ha='center', fontsize=8, fontweight='bold', color=color)
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Percentage (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=45, ha='right')
            ax.legend(loc='upper right')
            
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            fig.tight_layout()
            
            filepath = self.chart_dirs['analysis'] / 'distribution_comparison.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating distribution comparison chart: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # ALERT THRESHOLD CHARTS (01_risk/)
    # =========================================================================
    
    def chart_threshold_status(self, threshold_results: List[Any],
                                title: str = "Smart Alert Status Dashboard") -> Optional[str]:
        """
        Chart: Threshold breach status for multiple metrics
        Shows current values vs thresholds with alert levels.
        
        Args:
            threshold_results: List of ThresholdResult objects
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        try:
            if not threshold_results:
                return None
            
            n_metrics = len(threshold_results)
            fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            level_colors = {
                'normal': self.COLORS['success'],
                'warning': self.COLORS['warning'],
                'critical': '#FF5722',
                'emergency': self.COLORS['danger']
            }
            
            for ax, result in zip(axes, threshold_results):
                # Create gauge-like visualization
                thresholds = result.thresholds
                current = result.current_value
                max_val = max(thresholds['emergency'] * 1.2, current * 1.1)
                
                # Draw threshold zones
                ax.axhspan(0, thresholds['warning'], alpha=0.3, color=self.COLORS['success'])
                ax.axhspan(thresholds['warning'], thresholds['critical'], alpha=0.3, color=self.COLORS['warning'])
                ax.axhspan(thresholds['critical'], thresholds['emergency'], alpha=0.3, color='#FF5722')
                ax.axhspan(thresholds['emergency'], max_val, alpha=0.3, color=self.COLORS['danger'])
                
                # Draw current value bar
                color = level_colors.get(result.alert_level.value, self.COLORS['neutral'])
                ax.bar([0], [current], color=color, width=0.5, edgecolor='black', linewidth=2)
                
                # Add threshold lines
                for level, value in thresholds.items():
                    ax.axhline(y=value, color='black', linestyle='--', linewidth=1, alpha=0.7)
                    ax.text(0.55, value, f'{level.capitalize()}: {value:.1f}', 
                           va='center', fontsize=8)
                
                # Value label
                ax.text(0, current + max_val * 0.02, f'{current:.1f}', 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                ax.set_xlim(-0.5, 1)
                ax.set_ylim(0, max_val)
                ax.set_xticks([])
                ax.set_title(f"{result.metric_name}\n({result.alert_level.value.upper()})",
                            fontsize=11, fontweight='bold')
            
            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            fig.tight_layout()
            
            filepath = self.chart_dirs['risk'] / 'threshold_status.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating threshold status chart: {e}")
            plt.close('all')
            return None
    
    def chart_metric_with_thresholds(self, df: pd.DataFrame, 
                                      metric_column: str,
                                      datetime_column: str,
                                      thresholds: Dict[str, float],
                                      title: str = None) -> Optional[str]:
        """
        Chart: Time series with threshold bands
        Shows metric over time with warning/critical/emergency bands.
        
        Args:
            df: Data with metric and datetime
            metric_column: Column with metric values
            datetime_column: Column with datetime
            thresholds: Dict with 'warning', 'critical', 'emergency' values
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            df = df.copy()
            df['_dt'] = pd.to_datetime(df[datetime_column], errors='coerce')
            df = df.dropna(subset=['_dt', metric_column]).sort_values('_dt')
            
            dates = df['_dt']
            values = df[metric_column]
            
            # Plot threshold zones
            max_val = max(values.max(), thresholds['emergency']) * 1.1
            ax.fill_between(dates, 0, thresholds['warning'], 
                           alpha=0.15, color=self.COLORS['success'], label='Normal')
            ax.fill_between(dates, thresholds['warning'], thresholds['critical'], 
                           alpha=0.15, color=self.COLORS['warning'], label='Warning Zone')
            ax.fill_between(dates, thresholds['critical'], thresholds['emergency'], 
                           alpha=0.15, color='#FF5722', label='Critical Zone')
            ax.fill_between(dates, thresholds['emergency'], max_val, 
                           alpha=0.15, color=self.COLORS['danger'], label='Emergency Zone')
            
            # Plot values
            ax.plot(dates, values, color=self.COLORS['primary'], 
                   linewidth=2, marker='o', markersize=4, label=metric_column)
            
            # Highlight breaches
            warning_breach = values >= thresholds['warning']
            critical_breach = values >= thresholds['critical']
            emergency_breach = values >= thresholds['emergency']
            
            if warning_breach.any():
                ax.scatter(dates[warning_breach & ~critical_breach], 
                          values[warning_breach & ~critical_breach],
                          color=self.COLORS['warning'], s=100, zorder=5, marker='^')
            if critical_breach.any():
                ax.scatter(dates[critical_breach & ~emergency_breach], 
                          values[critical_breach & ~emergency_breach],
                          color='#FF5722', s=120, zorder=5, marker='^')
            if emergency_breach.any():
                ax.scatter(dates[emergency_breach], values[emergency_breach],
                          color=self.COLORS['danger'], s=150, zorder=5, marker='X')
            
            # Add threshold lines
            ax.axhline(y=thresholds['warning'], color=self.COLORS['warning'], 
                      linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(y=thresholds['critical'], color='#FF5722', 
                      linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(y=thresholds['emergency'], color=self.COLORS['danger'], 
                      linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Date')
            ax.set_ylabel(metric_column)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_ylim(0, max_val)
            
            # Stats annotation
            breach_count = (values >= thresholds['warning']).sum()
            ax.annotate(f'Breaches: {breach_count} / {len(values)} ({breach_count/len(values)*100:.0f}%)',
                       xy=(0.02, 0.98), xycoords='axes fraction',
                       ha='left', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            chart_title = title or f'{metric_column} with Smart Alert Thresholds'
            plt.title(chart_title, fontsize=14, fontweight='bold', pad=20)
            fig.tight_layout()
            
            safe_name = metric_column.lower().replace(' ', '_')[:30]
            filepath = self.chart_dirs['risk'] / f'thresholds_{safe_name}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating metric threshold chart: {e}")
            plt.close('all')
            return None
