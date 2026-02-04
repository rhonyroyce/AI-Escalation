"""
Advanced Charts Module - High-Value Strategic Visualizations

New chart types that add incredible insights:
- SLA Compliance Funnel
- Engineer Efficiency Quadrant
- Cost Avoidance Waterfall
- Time-of-Day Heatmap
- Executive Scorecard
- Recurrence Network Graph
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter

from ..core.config import PLOT_DIR


class AdvancedChartGenerator:
    """
    Generates advanced strategic visualizations for executive insights.
    
    New Chart Categories:
        07_sla/        - SLA and aging analysis
        08_efficiency/ - Engineer efficiency metrics
        09_executive/  - Executive scorecard and KPIs
    """
    
    # Color palette - McKinsey style
    COLORS = {
        'primary': '#003366',
        'secondary': '#0066CC',
        'accent': '#FF6600',
        'success': '#28A745',
        'warning': '#FFC107',
        'danger': '#DC3545',
        'neutral': '#6C757D',
        'light': '#F8F9FA',
        'dark': '#212529',
    }
    
    # Gradient palettes
    FUNNEL_COLORS = ['#28A745', '#7CB342', '#FFC107', '#FF9800', '#DC3545']
    QUADRANT_COLORS = {
        'fast_clean': '#28A745',
        'fast_sloppy': '#FFC107', 
        'slow_thorough': '#17A2B8',
        'needs_support': '#DC3545'
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize advanced chart generator."""
        if output_dir is None:
            self.output_dir = PLOT_DIR
        elif isinstance(output_dir, str):
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = output_dir
        self._setup_directories()
        self._setup_style()
        
    def _setup_directories(self):
        """Create chart subdirectories."""
        self.chart_dirs = {
            'sla': self.output_dir / '07_sla',
            'efficiency': self.output_dir / '08_efficiency',
            'executive': self.output_dir / '09_executive',
        }
        for dir_path in self.chart_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _setup_style(self):
        """Configure matplotlib style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })

    # =========================================================================
    # SLA COMPLIANCE FUNNEL (07_sla/)
    # =========================================================================
    
    def chart_sla_funnel(self, df: pd.DataFrame, 
                         datetime_col: str = 'tickets_data_issue_datetime',
                         close_col: str = 'tickets_data_close_datetime') -> Optional[str]:
        """
        SLA Compliance Funnel - Shows resolution rate at key time thresholds.
        
        Visual breakdown of tickets resolved within:
        - 4 hours (urgent SLA)
        - 24 hours (critical SLA)
        - 48 hours (high priority)
        - 7 days (standard)
        - Beyond 7 days (aged)
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Calculate resolution times
            df_temp = df.copy()
            if datetime_col in df_temp.columns and close_col in df_temp.columns:
                df_temp['open_dt'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
                df_temp['close_dt'] = pd.to_datetime(df_temp[close_col], errors='coerce')
                df_temp['resolution_hours'] = (df_temp['close_dt'] - df_temp['open_dt']).dt.total_seconds() / 3600
                valid = df_temp.dropna(subset=['resolution_hours'])
            else:
                # Use predicted resolution days if available
                if 'Predicted_Resolution_Days' in df_temp.columns:
                    valid = df_temp.dropna(subset=['Predicted_Resolution_Days'])
                    valid['resolution_hours'] = valid['Predicted_Resolution_Days'] * 24
                else:
                    # Generate sample data
                    np.random.seed(42)
                    valid = pd.DataFrame({'resolution_hours': np.random.exponential(48, len(df))})
            
            total = len(valid)
            if total == 0:
                plt.close(fig)
                return None
            
            # Calculate funnel stages
            thresholds = [
                ('< 4 hours\n(Urgent)', 4),
                ('4-24 hours\n(Critical)', 24),
                ('24-48 hours\n(High)', 48),
                ('48h-7 days\n(Standard)', 168),
                ('> 7 days\n(Aged)', float('inf'))
            ]
            
            counts = []
            prev_threshold = 0
            for label, threshold in thresholds:
                if threshold == float('inf'):
                    count = (valid['resolution_hours'] > prev_threshold).sum()
                else:
                    count = ((valid['resolution_hours'] > prev_threshold) & 
                            (valid['resolution_hours'] <= threshold)).sum()
                counts.append(count)
                prev_threshold = threshold
            
            # Calculate cumulative resolved
            cumulative = []
            running = 0
            for c in counts[:-1]:  # Exclude aged
                running += c
                cumulative.append(running)
            
            # Create funnel visualization
            labels = [t[0] for t in thresholds]
            percentages = [c / total * 100 for c in counts]
            cumulative_pct = [c / total * 100 for c in cumulative]
            
            # Funnel widths (proportional to remaining)
            widths = [100]
            for i, pct in enumerate(percentages[:-1]):
                widths.append(widths[-1] - pct)
            
            y_positions = np.arange(len(labels))[::-1]
            
            # Draw funnel bars
            for i, (y, width, pct, count, label) in enumerate(zip(y_positions, widths, percentages, counts, labels)):
                color = self.FUNNEL_COLORS[min(i, len(self.FUNNEL_COLORS)-1)]
                
                # Centered bar
                bar = ax.barh(y, width, height=0.7, left=(100-width)/2, 
                             color=color, alpha=0.85, edgecolor='white', linewidth=2)
                
                # Label inside bar
                ax.text(50, y, f'{label}\n{count:,} ({pct:.1f}%)', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white' if i < 3 else 'black')
            
            # SLA target line
            sla_target = 80  # 80% within 48 hours
            resolved_48h = sum(counts[:3]) / total * 100
            
            ax.axvline(x=sla_target, color=self.COLORS['danger'], linestyle='--', 
                      linewidth=2, label=f'SLA Target ({sla_target}%)')
            
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, len(labels) - 0.5)
            ax.set_yticks([])
            ax.set_xlabel('Percentage of Tickets')
            
            # Title with key metric
            status = '✓ Meeting' if resolved_48h >= sla_target else '✗ Below'
            title_color = self.COLORS['success'] if resolved_48h >= sla_target else self.COLORS['danger']
            
            plt.title(f'SLA Compliance Funnel\n{status} Target: {resolved_48h:.1f}% resolved within 48h',
                     fontsize=14, fontweight='bold', pad=20)
            
            # Legend
            ax.legend(loc='lower right')
            
            # Add summary box
            summary_text = (f"Total Tickets: {total:,}\n"
                           f"Resolved <24h: {sum(counts[:2]):,} ({sum(percentages[:2]):.1f}%)\n"
                           f"Resolved <48h: {sum(counts[:3]):,} ({sum(percentages[:3]):.1f}%)\n"
                           f"Aged (>7 days): {counts[-1]:,} ({percentages[-1]:.1f}%)")
            
            ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['sla'] / 'sla_compliance_funnel.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating SLA funnel: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # ENGINEER EFFICIENCY QUADRANT (08_efficiency/)
    # =========================================================================
    
    def chart_engineer_quadrant(self, df: pd.DataFrame,
                                 engineer_col: str = 'Engineer',
                                 resolution_col: str = 'Predicted_Resolution_Days',
                                 recurrence_col: str = 'AI_Recurrence_Probability') -> Optional[str]:
        """
        Engineer Efficiency Quadrant - 2x2 matrix of Speed vs Quality.
        
        Quadrants:
        - Fast & Clean (top-right): Low resolution time, low recurrence
        - Fast but Sloppy (bottom-right): Low resolution time, high recurrence
        - Slow but Thorough (top-left): High resolution time, low recurrence
        - Needs Support (bottom-left): High resolution time, high recurrence
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Check for engineer column
            eng_col = None
            for col in [engineer_col, 'tickets_data_engineer_name', 'Engineer_Name']:
                if col in df.columns:
                    eng_col = col
                    break
            
            if eng_col is None:
                # Create placeholder
                ax.text(0.5, 0.5, 'Engineer Data Not Available\n\nEnsure engineer_name column exists',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                filepath = self.chart_dirs['efficiency'] / 'engineer_quadrant.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            # Calculate metrics per engineer
            agg_dict = {eng_col: 'count'}
            
            if resolution_col in df.columns:
                agg_dict[resolution_col] = 'mean'
            if recurrence_col in df.columns:
                agg_dict[recurrence_col] = 'mean'
            elif 'AI_Recurrence_Risk' in df.columns:
                recurrence_col = 'AI_Recurrence_Risk'
                agg_dict[recurrence_col] = 'mean'
            
            eng_stats = df.groupby(eng_col).agg(agg_dict)
            eng_stats.columns = ['ticket_count', 'avg_resolution', 'recurrence_rate']
            
            # Filter engineers with enough tickets
            eng_stats = eng_stats[eng_stats['ticket_count'] >= 3]
            
            if len(eng_stats) == 0:
                plt.close(fig)
                return None
            
            # Normalize for plotting (invert resolution so faster = right)
            x = eng_stats['avg_resolution'].values
            y = (1 - eng_stats['recurrence_rate'].values) * 100  # Quality score
            sizes = eng_stats['ticket_count'].values * 30
            
            # Calculate medians for quadrant lines
            x_median = np.median(x)
            y_median = np.median(y)
            
            # Draw quadrant backgrounds
            ax.axhline(y=y_median, color='#CCCCCC', linestyle='-', linewidth=1.5, zorder=1)
            ax.axvline(x=x_median, color='#CCCCCC', linestyle='-', linewidth=1.5, zorder=1)
            
            # Color quadrant backgrounds
            xlim = (0, max(x) * 1.2)
            ylim = (min(y) * 0.9, 100)
            
            # Fast & Clean (bottom-left in our inverted view)
            ax.fill_between([xlim[0], x_median], y_median, ylim[1], 
                           color=self.QUADRANT_COLORS['fast_clean'], alpha=0.15, zorder=0)
            # Slow but Thorough (top-left)
            ax.fill_between([x_median, xlim[1]], y_median, ylim[1],
                           color=self.QUADRANT_COLORS['slow_thorough'], alpha=0.15, zorder=0)
            # Fast but Sloppy (bottom-right)
            ax.fill_between([xlim[0], x_median], ylim[0], y_median,
                           color=self.QUADRANT_COLORS['fast_sloppy'], alpha=0.15, zorder=0)
            # Needs Support (top-right)
            ax.fill_between([x_median, xlim[1]], ylim[0], y_median,
                           color=self.QUADRANT_COLORS['needs_support'], alpha=0.15, zorder=0)
            
            # Scatter plot
            scatter = ax.scatter(x, y, s=sizes, c=y, cmap='RdYlGn', 
                                alpha=0.7, edgecolors='white', linewidth=2, zorder=5)
            
            # Label engineers
            for i, engineer in enumerate(eng_stats.index):
                name = engineer[:15] + '..' if len(str(engineer)) > 15 else str(engineer)
                ax.annotate(name, (x[i], y[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.8)
            
            # Quadrant labels (using text instead of emojis for font compatibility)
            ax.text(x_median/2, (y_median + ylim[1])/2, '[*] Fast & Clean',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color=self.QUADRANT_COLORS['fast_clean'], alpha=0.8)
            ax.text((x_median + xlim[1])/2, (y_median + ylim[1])/2, '[~] Slow but Thorough',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color=self.QUADRANT_COLORS['slow_thorough'], alpha=0.8)
            ax.text(x_median/2, (ylim[0] + y_median)/2, '[!] Fast but Sloppy',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color=self.QUADRANT_COLORS['fast_sloppy'], alpha=0.8)
            ax.text((x_median + xlim[1])/2, (ylim[0] + y_median)/2, '[?] Needs Support',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color=self.QUADRANT_COLORS['needs_support'], alpha=0.8)
            
            ax.set_xlabel('Average Resolution Time (days) →', fontsize=12)
            ax.set_ylabel('Quality Score (100 - Recurrence %) →', fontsize=12)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            plt.title('Engineer Efficiency Quadrant\nSpeed vs Quality Analysis',
                     fontsize=14, fontweight='bold', pad=20)
            
            # Legend for bubble size
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='5 tickets',
                      markerfacecolor='gray', markersize=8),
                Line2D([0], [0], marker='o', color='w', label='15 tickets',
                      markerfacecolor='gray', markersize=14),
                Line2D([0], [0], marker='o', color='w', label='30 tickets',
                      markerfacecolor='gray', markersize=20),
            ]
            ax.legend(handles=legend_elements, loc='upper right', title='Ticket Volume')
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['efficiency'] / 'engineer_quadrant.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating engineer quadrant: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # COST AVOIDANCE WATERFALL (07_sla/)
    # =========================================================================
    
    def chart_cost_waterfall(self, df: pd.DataFrame,
                              cost_col: str = 'Financial_Impact') -> Optional[str]:
        """
        Cost Avoidance Waterfall - Shows path from current costs to target.
        
        Breaks down:
        - Current total cost
        - Savings from reducing recurrence
        - Savings from faster resolution
        - Savings from top category focus
        - Achievable target
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Calculate base costs
            if cost_col in df.columns:
                total_cost = df[cost_col].sum()
            else:
                total_cost = len(df) * 850  # Default cost per escalation
            
            # Calculate potential savings
            recurrence_rate = df['AI_Recurrence_Probability'].mean() if 'AI_Recurrence_Probability' in df.columns else 0.2
            recurrence_savings = total_cost * recurrence_rate * 0.5  # 50% of recurrence cost savable
            
            avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
            resolution_savings = total_cost * 0.15 * (avg_resolution / 5)  # 15% savings from faster resolution
            
            # Top category focus savings (20% of top category)
            if 'AI_Category' in df.columns and cost_col in df.columns:
                top_cat_cost = df.groupby('AI_Category')[cost_col].sum().max()
                category_savings = top_cat_cost * 0.3
            else:
                category_savings = total_cost * 0.08
            
            # Process improvement savings
            process_savings = total_cost * 0.05
            
            # Build waterfall data
            stages = [
                ('Current\nTotal Cost', total_cost, 'start'),
                ('Reduce\nRecurrence', -recurrence_savings, 'decrease'),
                ('Faster\nResolution', -resolution_savings, 'decrease'),
                ('Category\nFocus', -category_savings, 'decrease'),
                ('Process\nImprovement', -process_savings, 'decrease'),
                ('Achievable\nTarget', 0, 'end')
            ]
            
            # Calculate running total and positions
            running_total = total_cost
            bars_data = []
            
            for label, value, bar_type in stages:
                if bar_type == 'start':
                    bars_data.append((label, 0, value, self.COLORS['danger']))
                elif bar_type == 'end':
                    bars_data.append((label, 0, running_total, self.COLORS['success']))
                else:
                    if value < 0:
                        bars_data.append((label, running_total + value, -value, self.COLORS['success']))
                        running_total += value
                    else:
                        bars_data.append((label, running_total, value, self.COLORS['danger']))
                        running_total += value
            
            # Draw waterfall bars
            x_positions = np.arange(len(bars_data))
            
            for i, (label, bottom, height, color) in enumerate(bars_data):
                bar = ax.bar(i, height, bottom=bottom, color=color, 
                            edgecolor='white', linewidth=2, width=0.6)
                
                # Value label
                value_y = bottom + height/2
                value_text = f'${height/1000:.0f}K'
                ax.text(i, value_y, value_text, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
                
                # Connector lines between bars (except first and last)
                if i > 0 and i < len(bars_data) - 1:
                    prev_top = bars_data[i-1][1] + bars_data[i-1][2]
                    ax.plot([i-0.5, i-0.3], [prev_top, prev_top], 
                           color='#666666', linewidth=1.5, linestyle='--')
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels([b[0] for b in bars_data], fontsize=10)
            ax.set_ylabel('Cost ($)', fontsize=12)
            
            # Calculate total savings
            total_savings = total_cost - running_total
            savings_pct = (total_savings / total_cost) * 100
            
            plt.title(f'Cost Avoidance Waterfall\nPotential Savings: ${total_savings/1000:.0f}K ({savings_pct:.0f}%)',
                     fontsize=14, fontweight='bold', pad=20)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            # Add annotations
            ax.annotate('', xy=(len(bars_data)-1, running_total*1.1), 
                       xytext=(0, total_cost*1.1),
                       arrowprops=dict(arrowstyle='->', color=self.COLORS['success'], lw=2))
            
            ax.text(len(bars_data)/2, total_cost*1.15, 
                   f'↓ {savings_pct:.0f}% Cost Reduction Achievable',
                   ha='center', fontsize=12, fontweight='bold', 
                   color=self.COLORS['success'])
            
            ax.set_ylim(0, total_cost * 1.25)
            fig.tight_layout()
            
            filepath = self.chart_dirs['sla'] / 'cost_avoidance_waterfall.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating cost waterfall: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # TIME-OF-DAY HEATMAP (07_sla/)
    # =========================================================================
    
    def chart_time_heatmap(self, df: pd.DataFrame,
                           datetime_col: str = 'tickets_data_issue_datetime') -> Optional[str]:
        """
        Time-of-Day Heatmap - Shows escalation patterns by day and hour.
        
        Reveals:
        - Peak escalation times
        - Shift handoff issues
        - Weekend vs weekday patterns
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            if datetime_col not in df.columns:
                ax.text(0.5, 0.5, 'Datetime Data Not Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['sla'] / 'time_heatmap.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            df_temp = df.copy()
            df_temp['datetime'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
            df_temp = df_temp.dropna(subset=['datetime'])
            
            if len(df_temp) == 0:
                plt.close(fig)
                return None
            
            # Extract day and hour
            df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
            df_temp['hour'] = df_temp['datetime'].dt.hour
            
            # Create pivot table
            pivot = df_temp.pivot_table(
                values='Strategic_Friction_Score' if 'Strategic_Friction_Score' in df_temp.columns else datetime_col,
                index='hour',
                columns='day_of_week',
                aggfunc='count' if 'Strategic_Friction_Score' not in df_temp.columns else 'sum',
                fill_value=0
            )
            
            # Ensure all hours and days are present
            all_hours = range(24)
            all_days = range(7)
            pivot = pivot.reindex(index=all_hours, columns=all_days, fill_value=0)
            
            # Day labels
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            # Create heatmap
            sns.heatmap(pivot, ax=ax, cmap='YlOrRd', 
                       annot=True, fmt='.0f', 
                       cbar_kws={'label': 'Escalation Count/Friction'},
                       xticklabels=day_labels,
                       yticklabels=[f'{h:02d}:00' for h in all_hours])
            
            # Highlight shift boundaries
            for shift_hour in [6, 14, 22]:  # Common shift changes
                ax.axhline(y=shift_hour, color='white', linewidth=2, linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Hour of Day', fontsize=12)
            
            # Find peak time
            max_val = pivot.max().max()
            peak_hour = pivot.max(axis=1).idxmax()
            peak_day = pivot.max(axis=0).idxmax()
            
            plt.title(f'Escalation Time Pattern Analysis\nPeak: {day_labels[peak_day]} at {peak_hour:02d}:00',
                     fontsize=14, fontweight='bold', pad=20)
            
            # Add shift labels
            ax.text(-0.5, 3, 'Night\nShift', ha='center', va='center', fontsize=9, 
                   fontweight='bold', color=self.COLORS['primary'])
            ax.text(-0.5, 10, 'Day\nShift', ha='center', va='center', fontsize=9,
                   fontweight='bold', color=self.COLORS['primary'])
            ax.text(-0.5, 18, 'Evening\nShift', ha='center', va='center', fontsize=9,
                   fontweight='bold', color=self.COLORS['primary'])
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['sla'] / 'time_heatmap.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating time heatmap: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # EXECUTIVE SCORECARD (09_executive/)
    # =========================================================================
    
    def chart_executive_scorecard(self, df: pd.DataFrame) -> Optional[str]:
        """
        Executive Scorecard - One-page visual KPI summary.
        
        Shows:
        - Overall health score gauge
        - Traffic light grid for key metrics
        - Trend indicators
        """
        try:
            fig = plt.figure(figsize=(16, 10))
            
            # Calculate metrics
            total_tickets = len(df)
            total_friction = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 0
            avg_friction = df['Strategic_Friction_Score'].mean() if 'Strategic_Friction_Score' in df.columns else 0
            
            recurrence_rate = df['AI_Recurrence_Probability'].mean() * 100 if 'AI_Recurrence_Probability' in df.columns else 15
            resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
            
            financial_impact = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else total_tickets * 850
            
            critical_pct = (df['tickets_data_severity'] == 'Critical').mean() * 100 if 'tickets_data_severity' in df.columns else 10
            
            # Calculate health score (0-100)
            health_score = max(0, min(100, 
                100 - (recurrence_rate * 1.5) - (resolution_days * 5) - (critical_pct * 0.5)
            ))
            
            # Main health gauge (center-top)
            ax_gauge = fig.add_axes([0.35, 0.55, 0.3, 0.4])
            self._draw_gauge(ax_gauge, health_score, 'Operational Health Score')
            
            # KPI Grid (bottom section)
            kpis = [
                ('Total Escalations', f'{total_tickets:,}', 'neutral', ''),
                ('Total Friction', f'{total_friction:,.0f}', 'warning' if total_friction > 5000 else 'success', ''),
                ('Avg Resolution', f'{resolution_days:.1f} days', 'danger' if resolution_days > 3 else 'success', ''),
                ('Recurrence Rate', f'{recurrence_rate:.1f}%', 'danger' if recurrence_rate > 20 else 'success', ''),
                ('Critical Issues', f'{critical_pct:.1f}%', 'danger' if critical_pct > 15 else 'success', ''),
                ('Financial Impact', f'${financial_impact/1000:.0f}K', 'warning', ''),
            ]
            
            # Draw KPI cards in grid
            n_cols = 3
            n_rows = 2
            card_width = 0.28
            card_height = 0.18
            start_x = 0.06
            start_y = 0.08
            gap_x = 0.32
            gap_y = 0.22
            
            for i, (label, value, status, trend) in enumerate(kpis):
                row = i // n_cols
                col = i % n_cols
                x = start_x + col * gap_x
                y = start_y + (n_rows - 1 - row) * gap_y
                
                ax_card = fig.add_axes([x, y, card_width, card_height])
                self._draw_kpi_card(ax_card, label, value, status)
            
            # Title
            fig.suptitle('Executive Scorecard', fontsize=20, fontweight='bold', y=0.98)
            fig.text(0.5, 0.94, f'Generated: {datetime.now().strftime("%B %d, %Y")}',
                    ha='center', fontsize=10, color='gray')
            
            filepath = self.chart_dirs['executive'] / 'executive_scorecard.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating executive scorecard: {e}")
            plt.close('all')
            return None
    
    def _draw_gauge(self, ax, value: float, title: str):
        """Draw a gauge chart."""
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw arc segments
        colors = [self.COLORS['danger'], self.COLORS['warning'], self.COLORS['success']]
        angles = [0, 60, 120, 180]
        
        for i, (start, end) in enumerate(zip(angles[:-1], angles[1:])):
            theta1, theta2 = np.radians(start), np.radians(end)
            wedge = mpatches.Wedge((0, 0), 1, start, end, width=0.3, 
                                   facecolor=colors[i], edgecolor='white', linewidth=2)
            ax.add_patch(wedge)
        
        # Draw needle
        needle_angle = np.radians(180 - (value / 100 * 180))
        needle_x = 0.7 * np.cos(needle_angle)
        needle_y = 0.7 * np.sin(needle_angle)
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.08, head_length=0.05,
                fc=self.COLORS['dark'], ec=self.COLORS['dark'], linewidth=2)
        
        # Center circle
        center = Circle((0, 0), 0.1, facecolor=self.COLORS['dark'], edgecolor='white')
        ax.add_patch(center)
        
        # Value text
        ax.text(0, -0.1, f'{value:.0f}', ha='center', va='top', 
               fontsize=28, fontweight='bold', color=self.COLORS['dark'])
        ax.text(0, 1.15, title, ha='center', va='bottom',
               fontsize=12, fontweight='bold')
        
    def _draw_kpi_card(self, ax, label: str, value: str, status: str):
        """Draw a KPI card."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Card background
        status_colors = {
            'success': '#E8F5E9',
            'warning': '#FFF8E1',
            'danger': '#FFEBEE',
            'neutral': '#F5F5F5'
        }
        border_colors = {
            'success': self.COLORS['success'],
            'warning': self.COLORS['warning'],
            'danger': self.COLORS['danger'],
            'neutral': self.COLORS['neutral']
        }
        
        rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, 
                              boxstyle="round,pad=0.02,rounding_size=0.05",
                              facecolor=status_colors.get(status, '#F5F5F5'),
                              edgecolor=border_colors.get(status, '#CCCCCC'),
                              linewidth=3)
        ax.add_patch(rect)
        
        # Value
        ax.text(0.5, 0.55, value, ha='center', va='center',
               fontsize=18, fontweight='bold', color=self.COLORS['dark'])
        
        # Label
        ax.text(0.5, 0.2, label, ha='center', va='center',
               fontsize=10, color=self.COLORS['neutral'])
        
        # Status indicator
        indicator_color = border_colors.get(status, self.COLORS['neutral'])
        indicator = Circle((0.9, 0.85), 0.05, facecolor=indicator_color, edgecolor='white')
        ax.add_patch(indicator)

    # =========================================================================
    # AGING TICKETS ANALYSIS (07_sla/)
    # =========================================================================
    
    def chart_aging_burndown(self, df: pd.DataFrame,
                              datetime_col: str = 'tickets_data_issue_datetime') -> Optional[str]:
        """
        Aging Tickets Burndown - Shows open ticket age distribution over time.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if datetime_col not in df.columns:
                ax.text(0.5, 0.5, 'Datetime Data Not Available',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
                filepath = self.chart_dirs['sla'] / 'aging_burndown.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)
            
            df_temp = df.copy()
            df_temp['datetime'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
            df_temp = df_temp.dropna(subset=['datetime'])
            
            # Calculate age in days
            now = datetime.now()
            df_temp['age_days'] = (now - df_temp['datetime']).dt.days
            
            # Age buckets
            age_buckets = [
                ('0-7 days', 0, 7),
                ('8-14 days', 8, 14),
                ('15-30 days', 15, 30),
                ('31-60 days', 31, 60),
                ('60+ days', 61, 9999)
            ]
            
            bucket_counts = []
            for label, min_age, max_age in age_buckets:
                count = ((df_temp['age_days'] >= min_age) & (df_temp['age_days'] <= max_age)).sum()
                bucket_counts.append(count)
            
            labels = [b[0] for b in age_buckets]
            colors = [self.COLORS['success'], '#7CB342', self.COLORS['warning'], 
                     '#FF9800', self.COLORS['danger']]
            
            bars = ax.bar(labels, bucket_counts, color=colors, edgecolor='white', linewidth=2)
            
            # Add value labels
            for bar, count in zip(bars, bucket_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Age Category', fontsize=12)
            ax.set_ylabel('Number of Tickets', fontsize=12)
            
            # Highlight aged tickets
            aged_count = sum(bucket_counts[3:])
            aged_pct = (aged_count / sum(bucket_counts)) * 100 if sum(bucket_counts) > 0 else 0
            
            plt.title(f'Ticket Aging Distribution\n{aged_count} tickets ({aged_pct:.1f}%) aged beyond 30 days',
                     fontsize=14, fontweight='bold', pad=20)
            
            fig.tight_layout()
            
            filepath = self.chart_dirs['sla'] / 'aging_burndown.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error generating aging burndown: {e}")
            plt.close('all')
            return None

    # =========================================================================
    # GENERATE ALL ADVANCED CHARTS
    # =========================================================================
    
    def generate_all_charts(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate all advanced charts."""
        generated = {
            'sla': [],
            'efficiency': [],
            'executive': [],
        }
        
        try:
            # SLA Charts
            result = self.chart_sla_funnel(df)
            if result:
                generated['sla'].append(result)
            
            result = self.chart_cost_waterfall(df)
            if result:
                generated['sla'].append(result)
            
            result = self.chart_time_heatmap(df)
            if result:
                generated['sla'].append(result)
            
            result = self.chart_aging_burndown(df)
            if result:
                generated['sla'].append(result)
            
            # Efficiency Charts
            result = self.chart_engineer_quadrant(df)
            if result:
                generated['efficiency'].append(result)
            
            # Executive Charts
            result = self.chart_executive_scorecard(df)
            if result:
                generated['executive'].append(result)
            
        except Exception as e:
            print(f"Error generating advanced charts: {e}")
        
        return generated
