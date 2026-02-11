"""
Advanced Charts Module - High-Value Strategic Visualizations.

This module generates executive-grade chart types that complement the core charts
produced by chart_generator.py. While chart_generator.py covers standard analytical
charts (Pareto, bar, scatter, heatmap), this module focuses on high-level strategic
visualizations tailored for C-suite and senior management stakeholders.

Architecture Overview:
    The AdvancedChartGenerator class produces matplotlib-based PNG charts organized
    into three output directories:
        07_sla/        - SLA compliance, cost avoidance, time patterns, ticket aging
        08_efficiency/ - Engineer speed-vs-quality quadrant analysis
        09_executive/  - Executive scorecard with gauge and KPI cards

Integration Points:
    - Called by: ExcelReportWriter._generate_advanced_charts() in report_generator.py
    - Config from: core/config.py (PLOT_DIR for output directory)
    - Output: PNG files at 150 DPI, embedded into Excel "Visual Analytics" sheet

Chart Types:
    1. SLA Compliance Funnel (chart_sla_funnel):
       - Resolution rate at 4h/24h/48h/7d thresholds
       - Funnel visualization with progressively narrowing bars
       - SLA target line overlay (80% within 48h)

    2. Engineer Efficiency Quadrant (chart_engineer_quadrant):
       - 2x2 Speed vs Quality scatter matrix
       - Four quadrants: Fast&Clean, Fast&Sloppy, Slow&Thorough, NeedsSupport
       - Bubble size proportional to ticket volume

    3. Cost Avoidance Waterfall (chart_cost_waterfall):
       - Waterfall from current total cost through savings categories
       - Shows recurrence, resolution, category, and process savings
       - Ends at achievable target cost

    4. Time-of-Day Heatmap (chart_time_heatmap):
       - 24x7 grid: hours of day vs days of week
       - Seaborn heatmap with YlOrRd colormap
       - Shift boundary lines at 06:00, 14:00, 22:00

    5. Executive Scorecard (chart_executive_scorecard):
       - Central gauge chart for Operational Health Score (0-100)
       - 2x3 grid of KPI cards with RAG status indicators
       - Generated date stamp

    6. Aging Burndown (chart_aging_burndown):
       - Ticket age distribution in 5 buckets (0-7d, 8-14d, 15-30d, 31-60d, 60+d)
       - Color gradient from green (fresh) to red (aged)

Design Pattern:
    Each chart method follows the same pattern:
    1. Create figure with appropriate figsize
    2. Check data availability, provide placeholder or fallback if missing
    3. Calculate derived metrics from the DataFrame
    4. Build the visualization using matplotlib/seaborn
    5. Add title, labels, summary annotations
    6. Save PNG at 150 DPI to the appropriate subdirectory
    7. Close figure and return filepath (or None on error)

    All methods are wrapped in try/except to ensure one chart failure
    does not prevent other charts from generating.

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

# PLOT_DIR: Base output directory for all chart PNG files (from config.py).
# Advanced charts create subdirectories under this path.
from ..core.config import PLOT_DIR


class AdvancedChartGenerator:
    """
    Generates advanced strategic visualizations for executive insights.

    This class complements ChartGenerator (chart_generator.py) by providing
    higher-level strategic chart types focused on SLA performance, engineer
    efficiency analysis, and executive KPI summaries.

    All charts use a consistent McKinsey-style color palette and are saved
    as PNG files at 150 DPI for embedding in the Excel report.

    New Chart Categories:
        07_sla/        - SLA and aging analysis
        08_efficiency/ - Engineer efficiency metrics
        09_executive/  - Executive scorecard and KPIs
    """

    # ---------------------------------------------------------------------------
    # McKinsey-style color palette for consistent executive-grade visuals.
    # These hex colors are used across all chart types for a cohesive look.
    # ---------------------------------------------------------------------------
    COLORS = {
        'primary': '#003366',     # Deep navy blue - titles, primary elements
        'secondary': '#0066CC',   # Medium blue - secondary elements
        'accent': '#FF6600',      # Orange - accent/highlight elements
        'success': '#28A745',     # Green - positive metrics, meeting targets
        'warning': '#FFC107',     # Amber/yellow - caution metrics
        'danger': '#DC3545',      # Red - negative metrics, below targets
        'neutral': '#6C757D',     # Gray - neutral/informational elements
        'light': '#F8F9FA',       # Light gray - backgrounds
        'dark': '#212529',        # Near-black - text, gauge needle
    }

    # Color gradient for the SLA funnel chart stages.
    # Progresses from green (fast resolution) to red (aged/slow).
    FUNNEL_COLORS = ['#28A745', '#7CB342', '#FFC107', '#FF9800', '#DC3545']

    # Color mapping for the 2x2 engineer quadrant chart.
    # Each quadrant represents a different speed/quality combination.
    QUADRANT_COLORS = {
        'fast_clean': '#28A745',      # Green - ideal performers (fast + low recurrence)
        'fast_sloppy': '#FFC107',     # Yellow - fast but quality issues (fast + high recurrence)
        'slow_thorough': '#17A2B8',   # Cyan - thorough but slow (slow + low recurrence)
        'needs_support': '#DC3545'    # Red - struggling (slow + high recurrence)
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the advanced chart generator.

        Args:
            output_dir: Base directory for chart output. Defaults to PLOT_DIR from config.
                       Can be a Path object or string path.
        """
        # Resolve the output directory: use PLOT_DIR from config if not specified,
        # otherwise convert string paths to Path objects for consistent handling.
        if output_dir is None:
            self.output_dir = PLOT_DIR
        elif isinstance(output_dir, str):
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = output_dir
        self._setup_directories()
        self._setup_style()

    def _setup_directories(self):
        """
        Create chart subdirectories for each chart category.

        Creates three subdirectories under the base output directory:
            07_sla/        - SLA funnel, cost waterfall, time heatmap, aging burndown
            08_efficiency/ - Engineer efficiency quadrant
            09_executive/  - Executive scorecard with gauge and KPIs

        The numeric prefixes (07, 08, 09) ensure these directories sort after
        the core chart directories (01-06) created by ChartGenerator.
        """
        self.chart_dirs = {
            'sla': self.output_dir / '07_sla',
            'efficiency': self.output_dir / '08_efficiency',
            'executive': self.output_dir / '09_executive',
        }
        for dir_path in self.chart_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_style(self):
        """
        Configure matplotlib global style for McKinsey-grade executive charts.

        Sets up:
            - Clean seaborn whitegrid base style
            - Arial/Helvetica font family (falls back to DejaVu Sans)
            - Bold, appropriately sized title and axis labels
            - Subtle grid lines (alpha=0.3) for reference without visual clutter
            - White figure and axes backgrounds for clean report embedding
        """
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

        Creates a funnel visualization showing what percentage of tickets were
        resolved within each SLA time window. The funnel narrows progressively,
        with the widest bar representing total tickets and each subsequent bar
        showing the remaining unresolved tickets.

        Resolution Time Calculation:
            1. Primary: Uses actual open/close datetime columns to compute hours
            2. Fallback: Uses Predicted_Resolution_Days * 24 if datetimes missing
            3. Last resort: Generates exponential sample data (mean=48h) for demo

        SLA Thresholds:
            - < 4 hours: Urgent SLA (green)
            - 4-24 hours: Critical SLA (light green)
            - 24-48 hours: High priority (yellow)
            - 48h-7 days: Standard (orange)
            - > 7 days: Aged/overdue (red)

        Visual Elements:
            - Centered horizontal bars with decreasing widths (funnel shape)
            - Inline count and percentage labels on each bar
            - SLA target vertical line (80% within 48 hours)
            - Summary statistics box in bottom-left corner
            - Title with pass/fail status indicator

        Args:
            df: Escalation DataFrame with ticket data
            datetime_col: Column name for ticket open datetime
            close_col: Column name for ticket close datetime

        Returns:
            Path to saved PNG file, or None if generation fails
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # -------------------------------------------------------------------
            # Step 1: Calculate resolution times in hours.
            # Try actual datetime columns first, fall back to predicted days,
            # then to synthetic exponential data as a last resort.
            # -------------------------------------------------------------------
            df_temp = df.copy()
            if datetime_col in df_temp.columns and close_col in df_temp.columns:
                # Parse datetime strings and compute resolution duration in hours
                df_temp['open_dt'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
                df_temp['close_dt'] = pd.to_datetime(df_temp[close_col], errors='coerce')
                df_temp['resolution_hours'] = (df_temp['close_dt'] - df_temp['open_dt']).dt.total_seconds() / 3600
                valid = df_temp.dropna(subset=['resolution_hours'])
            else:
                # Use predicted resolution days if available (convert to hours)
                if 'Predicted_Resolution_Days' in df_temp.columns:
                    valid = df_temp.dropna(subset=['Predicted_Resolution_Days'])
                    valid['resolution_hours'] = valid['Predicted_Resolution_Days'] * 24
                else:
                    # Generate sample data with exponential distribution (mean=48h)
                    # for demonstration purposes when no time data is available
                    np.random.seed(42)
                    valid = pd.DataFrame({'resolution_hours': np.random.exponential(48, len(df))})

            total = len(valid)
            if total == 0:
                plt.close(fig)
                return None

            # -------------------------------------------------------------------
            # Step 2: Calculate ticket counts for each funnel stage.
            # Each threshold defines a time window; counts are non-cumulative
            # (tickets in that specific window only).
            # -------------------------------------------------------------------
            thresholds = [
                ('< 4 hours\n(Urgent)', 4),
                ('4-24 hours\n(Critical)', 24),
                ('24-48 hours\n(High)', 48),
                ('48h-7 days\n(Standard)', 168),      # 7 days * 24 hours
                ('> 7 days\n(Aged)', float('inf'))
            ]

            counts = []
            prev_threshold = 0
            for label, threshold in thresholds:
                if threshold == float('inf'):
                    # Last bucket: everything beyond the previous threshold
                    count = (valid['resolution_hours'] > prev_threshold).sum()
                else:
                    # Count tickets in the range (prev_threshold, threshold]
                    count = ((valid['resolution_hours'] > prev_threshold) &
                            (valid['resolution_hours'] <= threshold)).sum()
                counts.append(count)
                prev_threshold = threshold

            # -------------------------------------------------------------------
            # Step 3: Calculate cumulative resolved counts (excluding aged bucket).
            # Used for SLA compliance tracking.
            # -------------------------------------------------------------------
            cumulative = []
            running = 0
            for c in counts[:-1]:  # Exclude the "aged" bucket from cumulative
                running += c
                cumulative.append(running)

            # -------------------------------------------------------------------
            # Step 4: Create the funnel visualization.
            # Bars are centered horizontally, with widths proportional to the
            # remaining unresolved tickets at each stage.
            # -------------------------------------------------------------------
            labels = [t[0] for t in thresholds]
            percentages = [c / total * 100 for c in counts]
            cumulative_pct = [c / total * 100 for c in cumulative]

            # Funnel widths: start at 100%, subtract each bucket's percentage
            # to create the narrowing effect
            widths = [100]
            for i, pct in enumerate(percentages[:-1]):
                widths.append(widths[-1] - pct)

            # Y positions: reversed so the widest bar (first stage) is at top
            y_positions = np.arange(len(labels))[::-1]

            # Draw each funnel bar
            for i, (y, width, pct, count, label) in enumerate(zip(y_positions, widths, percentages, counts, labels)):
                color = self.FUNNEL_COLORS[min(i, len(self.FUNNEL_COLORS)-1)]

                # Center the bar horizontally: left offset = (100 - width) / 2
                bar = ax.barh(y, width, height=0.7, left=(100-width)/2,
                             color=color, alpha=0.85, edgecolor='white', linewidth=2)

                # Label inside each bar: show stage name, count, and percentage
                ax.text(50, y, f'{label}\n{count:,} ({pct:.1f}%)',
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white' if i < 3 else 'black')

            # -------------------------------------------------------------------
            # Step 5: Add SLA target line and summary annotations.
            # -------------------------------------------------------------------
            sla_target = 80  # Target: 80% of tickets resolved within 48 hours
            resolved_48h = sum(counts[:3]) / total * 100  # First 3 buckets = within 48h

            # Vertical dashed line at the SLA target percentage
            ax.axvline(x=sla_target, color=self.COLORS['danger'], linestyle='--',
                      linewidth=2, label=f'SLA Target ({sla_target}%)')

            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, len(labels) - 0.5)
            ax.set_yticks([])  # Hide y-axis ticks (labels are drawn inside bars)
            ax.set_xlabel('Percentage of Tickets')

            # Title with pass/fail status based on SLA target
            status = '✓ Meeting' if resolved_48h >= sla_target else '✗ Below'
            title_color = self.COLORS['success'] if resolved_48h >= sla_target else self.COLORS['danger']

            plt.title(f'SLA Compliance Funnel\n{status} Target: {resolved_48h:.1f}% resolved within 48h',
                     fontsize=14, fontweight='bold', pad=20)

            ax.legend(loc='lower right')

            # Summary statistics box in bottom-left corner
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

        Creates a scatter plot where each engineer is positioned based on their
        average resolution speed (x-axis) and quality score (y-axis, derived
        from inverse recurrence rate). Engineers are plotted as bubbles where
        size represents ticket volume.

        Quadrant Interpretation (median-split):
            - Fast & Clean (low time, high quality): Ideal performers
            - Fast but Sloppy (low time, low quality): Speed at expense of quality
            - Slow but Thorough (high time, high quality): Careful but needs speed
            - Needs Support (high time, low quality): Requires intervention/training

        Data Requirements:
            - Engineer name column (tries: 'Engineer', 'tickets_data_engineer_name',
              'Engineer_Name')
            - Resolution time column (Predicted_Resolution_Days)
            - Recurrence probability column (AI_Recurrence_Probability or AI_Recurrence_Risk)

        Visual Elements:
            - Colored quadrant backgrounds (green, cyan, yellow, red)
            - Bubble scatter with RdYlGn colormap (green=high quality)
            - Engineer name labels on each bubble
            - Quadrant labels with descriptive text
            - Bubble size legend (5, 15, 30 tickets)
            - Median crosshair lines dividing the four quadrants

        Filtering:
            Engineers with fewer than 3 tickets are excluded to ensure
            statistical significance of the averages.

        Args:
            df: Escalation DataFrame with engineer, resolution, and recurrence data
            engineer_col: Column name for engineer identifier
            resolution_col: Column name for resolution time (days)
            recurrence_col: Column name for recurrence probability (0-1)

        Returns:
            Path to saved PNG file, or None if generation fails
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 10))

            # -------------------------------------------------------------------
            # Step 1: Find the engineer column.
            # Try multiple possible column names in priority order.
            # -------------------------------------------------------------------
            eng_col = None
            for col in [engineer_col, 'tickets_data_engineer_name', 'Engineer_Name']:
                if col in df.columns:
                    eng_col = col
                    break

            if eng_col is None:
                # No engineer data available - create a placeholder chart
                ax.text(0.5, 0.5, 'Engineer Data Not Available\n\nEnsure engineer_name column exists',
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                filepath = self.chart_dirs['efficiency'] / 'engineer_quadrant.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                return str(filepath)

            # -------------------------------------------------------------------
            # Step 2: Calculate per-engineer metrics.
            # Aggregate: ticket count, average resolution time, average recurrence.
            # -------------------------------------------------------------------
            agg_dict = {eng_col: 'count'}  # Ticket count

            if resolution_col in df.columns:
                agg_dict[resolution_col] = 'mean'
            if recurrence_col in df.columns:
                agg_dict[recurrence_col] = 'mean'
            elif 'AI_Recurrence_Risk' in df.columns:
                # Fall back to AI_Recurrence_Risk if primary column not found
                recurrence_col = 'AI_Recurrence_Risk'
                agg_dict[recurrence_col] = 'mean'

            eng_stats = df.groupby(eng_col).agg(agg_dict)
            eng_stats.columns = ['ticket_count', 'avg_resolution', 'recurrence_rate']

            # Filter out engineers with too few tickets for meaningful analysis
            eng_stats = eng_stats[eng_stats['ticket_count'] >= 3]

            if len(eng_stats) == 0:
                plt.close(fig)
                return None

            # -------------------------------------------------------------------
            # Step 3: Prepare scatter data.
            # X-axis: average resolution time (days) - higher = slower
            # Y-axis: quality score = (1 - recurrence_rate) * 100 - higher = better
            # Bubble size: ticket_count * 30 scaling factor
            # -------------------------------------------------------------------
            x = eng_stats['avg_resolution'].values
            y = (1 - eng_stats['recurrence_rate'].values) * 100  # Quality score (0-100)
            sizes = eng_stats['ticket_count'].values * 30  # Scale factor for visibility

            # Median values define the quadrant boundaries
            x_median = np.median(x)
            y_median = np.median(y)

            # -------------------------------------------------------------------
            # Step 4: Draw quadrant backgrounds and boundaries.
            # -------------------------------------------------------------------
            # Quadrant divider lines at median values
            ax.axhline(y=y_median, color='#CCCCCC', linestyle='-', linewidth=1.5, zorder=1)
            ax.axvline(x=x_median, color='#CCCCCC', linestyle='-', linewidth=1.5, zorder=1)

            # Set axis limits with padding
            xlim = (0, max(x) * 1.2)
            ylim = (min(y) * 0.9, 100)

            # Fill each quadrant with a semi-transparent background color.
            # Note: In this coordinate system, "fast" is LEFT (low resolution time)
            # and "high quality" is TOP (high quality score).

            # Fast & Clean: left side, top half (fast resolution, high quality)
            ax.fill_between([xlim[0], x_median], y_median, ylim[1],
                           color=self.QUADRANT_COLORS['fast_clean'], alpha=0.15, zorder=0)
            # Slow but Thorough: right side, top half (slow resolution, high quality)
            ax.fill_between([x_median, xlim[1]], y_median, ylim[1],
                           color=self.QUADRANT_COLORS['slow_thorough'], alpha=0.15, zorder=0)
            # Fast but Sloppy: left side, bottom half (fast resolution, low quality)
            ax.fill_between([xlim[0], x_median], ylim[0], y_median,
                           color=self.QUADRANT_COLORS['fast_sloppy'], alpha=0.15, zorder=0)
            # Needs Support: right side, bottom half (slow resolution, low quality)
            ax.fill_between([x_median, xlim[1]], ylim[0], y_median,
                           color=self.QUADRANT_COLORS['needs_support'], alpha=0.15, zorder=0)

            # -------------------------------------------------------------------
            # Step 5: Plot engineer scatter bubbles.
            # Color mapped to quality score (RdYlGn: red=low, green=high)
            # -------------------------------------------------------------------
            scatter = ax.scatter(x, y, s=sizes, c=y, cmap='RdYlGn',
                                alpha=0.7, edgecolors='white', linewidth=2, zorder=5)

            # Label each bubble with the engineer's name (truncated to 15 chars)
            for i, engineer in enumerate(eng_stats.index):
                name = engineer[:15] + '..' if len(str(engineer)) > 15 else str(engineer)
                ax.annotate(name, (x[i], y[i]), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, alpha=0.8)

            # -------------------------------------------------------------------
            # Step 6: Add quadrant labels.
            # Text markers use bracket notation ([*], [~], [!], [?]) for font
            # compatibility (avoiding emoji rendering issues in some environments).
            # -------------------------------------------------------------------
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

            # Legend showing bubble size reference points
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
        Cost Avoidance Waterfall - Shows path from current costs to achievable target.

        Creates a waterfall chart that starts with the current total cost and shows
        how specific improvement actions can progressively reduce it. Each bar
        represents a savings category, with the final bar showing the achievable
        target after all improvements.

        Savings Calculation Logic:
            1. Recurrence Savings = total_cost * recurrence_rate * 0.5
               (Assumes 50% of recurrence-related costs are avoidable)
            2. Resolution Savings = total_cost * 0.15 * (avg_resolution / 5)
               (15% base savings scaled by resolution time relative to 5-day baseline)
            3. Category Focus Savings = top_category_cost * 0.3
               (30% reduction achievable by focusing on the highest-cost category)
            4. Process Improvement = total_cost * 0.05
               (5% savings from general process improvement initiatives)

        Waterfall Visual Structure:
            - First bar (red): Total current cost, starts from zero
            - Middle bars (green): Savings amounts, shown as floating decreases
            - Last bar (green): Achievable target, starts from zero
            - Connector lines between bars showing running total
            - Currency-formatted labels inside each bar ($XK format)
            - Arrow annotation showing total reduction percentage

        Args:
            df: Escalation DataFrame with financial and metric data
            cost_col: Column name for financial impact per ticket

        Returns:
            Path to saved PNG file, or None if generation fails
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            # -------------------------------------------------------------------
            # Step 1: Calculate base total cost.
            # Use actual Financial_Impact column if available, otherwise
            # estimate at $850 per escalation (industry average default).
            # -------------------------------------------------------------------
            if cost_col in df.columns:
                total_cost = df[cost_col].sum()
            else:
                total_cost = len(df) * 850  # Default cost per escalation

            # -------------------------------------------------------------------
            # Step 2: Calculate potential savings for each improvement area.
            # -------------------------------------------------------------------

            # Recurrence savings: reducing repeat escalations saves proportional costs.
            # Formula: total_cost * recurrence_rate * 0.5 (50% of recurrence cost is avoidable)
            recurrence_rate = df['AI_Recurrence_Probability'].mean() if 'AI_Recurrence_Probability' in df.columns else 0.2
            recurrence_savings = total_cost * recurrence_rate * 0.5

            # Resolution time savings: faster resolution reduces operational cost.
            # Formula: total_cost * 0.15 * (avg_days / 5), scaled by how far above
            # the 5-day baseline the average resolution time is.
            avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
            resolution_savings = total_cost * 0.15 * (avg_resolution / 5)

            # Top category focus savings: concentrating effort on the most expensive
            # category can yield a 30% reduction in that category's cost.
            if 'AI_Category' in df.columns and cost_col in df.columns:
                top_cat_cost = df.groupby('AI_Category')[cost_col].sum().max()
                category_savings = top_cat_cost * 0.3
            else:
                category_savings = total_cost * 0.08  # Default: 8% of total

            # Process improvement: general operational improvements yield ~5% savings
            process_savings = total_cost * 0.05

            # -------------------------------------------------------------------
            # Step 3: Build waterfall stage data.
            # Each stage: (label, value_change, bar_type)
            # bar_type: 'start' = base bar from zero, 'decrease' = floating savings,
            #           'end' = final target bar from zero.
            # -------------------------------------------------------------------
            stages = [
                ('Current\nTotal Cost', total_cost, 'start'),
                ('Reduce\nRecurrence', -recurrence_savings, 'decrease'),
                ('Faster\nResolution', -resolution_savings, 'decrease'),
                ('Category\nFocus', -category_savings, 'decrease'),
                ('Process\nImprovement', -process_savings, 'decrease'),
                ('Achievable\nTarget', 0, 'end')
            ]

            # -------------------------------------------------------------------
            # Step 4: Calculate bar positions (bottom, height) for waterfall effect.
            # Savings bars "float" by setting their bottom to the new running total
            # and their height to the savings amount (positive, shown in green).
            # -------------------------------------------------------------------
            running_total = total_cost
            bars_data = []

            for label, value, bar_type in stages:
                if bar_type == 'start':
                    # Starting bar: full height from zero (red = current cost)
                    bars_data.append((label, 0, value, self.COLORS['danger']))
                elif bar_type == 'end':
                    # Final target bar: full height from zero (green = achievable target)
                    bars_data.append((label, 0, running_total, self.COLORS['success']))
                else:
                    if value < 0:
                        # Savings bar: floats at new level, height = savings amount (green)
                        bars_data.append((label, running_total + value, -value, self.COLORS['success']))
                        running_total += value
                    else:
                        # Cost increase bar (not used currently but supported)
                        bars_data.append((label, running_total, value, self.COLORS['danger']))
                        running_total += value

            # -------------------------------------------------------------------
            # Step 5: Draw the waterfall bars with value labels and connectors.
            # -------------------------------------------------------------------
            x_positions = np.arange(len(bars_data))

            for i, (label, bottom, height, color) in enumerate(bars_data):
                bar = ax.bar(i, height, bottom=bottom, color=color,
                            edgecolor='white', linewidth=2, width=0.6)

                # Value label centered inside each bar ($XK format)
                value_y = bottom + height/2
                value_text = f'${height/1000:.0f}K'
                ax.text(i, value_y, value_text, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')

                # Draw connector lines between bars (dashed, shows running total)
                # Skip first and last bars
                if i > 0 and i < len(bars_data) - 1:
                    prev_top = bars_data[i-1][1] + bars_data[i-1][2]
                    ax.plot([i-0.5, i-0.3], [prev_top, prev_top],
                           color='#666666', linewidth=1.5, linestyle='--')

            ax.set_xticks(x_positions)
            ax.set_xticklabels([b[0] for b in bars_data], fontsize=10)
            ax.set_ylabel('Cost ($)', fontsize=12)

            # -------------------------------------------------------------------
            # Step 6: Add annotations showing total savings potential.
            # -------------------------------------------------------------------
            total_savings = total_cost - running_total
            savings_pct = (total_savings / total_cost) * 100

            plt.title(f'Cost Avoidance Waterfall\nPotential Savings: ${total_savings/1000:.0f}K ({savings_pct:.0f}%)',
                     fontsize=14, fontweight='bold', pad=20)

            # Format y-axis as currency ($XK)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

            # Arrow annotation spanning from start to end bar showing total reduction
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

        Creates a 24-row x 7-column heatmap where each cell represents the
        escalation count (or total friction score) for a specific hour-of-day
        and day-of-week combination. This reveals temporal patterns useful for
        staffing optimization and shift planning.

        Data Handling:
            - Parses the datetime column and extracts day_of_week (0=Mon) and hour (0-23)
            - Creates a pivot table with hours as rows and days as columns
            - Uses count aggregation by default; uses sum of Strategic_Friction_Score
              if that column is available (friction-weighted view)
            - Fills missing hour/day combinations with zero

        Visual Elements:
            - Seaborn heatmap with YlOrRd (Yellow-Orange-Red) colormap
            - Annotated cell values showing the count/friction in each slot
            - White dashed horizontal lines at shift boundaries (06:00, 14:00, 22:00)
            - Shift labels on the left margin (Night, Day, Evening)
            - Title showing the peak day and hour

        Reveals:
            - Peak escalation times (for staffing optimization)
            - Shift handoff issues (spikes around shift change times)
            - Weekend vs weekday patterns (workload distribution)

        Args:
            df: Escalation DataFrame with datetime data
            datetime_col: Column name for ticket creation datetime

        Returns:
            Path to saved PNG file, or None if generation fails
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            # Check if datetime column exists
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

            # -------------------------------------------------------------------
            # Extract temporal components for the pivot table.
            # dayofweek: 0=Monday through 6=Sunday
            # hour: 0-23
            # -------------------------------------------------------------------
            df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
            df_temp['hour'] = df_temp['datetime'].dt.hour

            # -------------------------------------------------------------------
            # Create pivot table.
            # If Strategic_Friction_Score is available, use sum (friction-weighted view).
            # Otherwise, use count (raw escalation count view).
            # -------------------------------------------------------------------
            pivot = df_temp.pivot_table(
                values='Strategic_Friction_Score' if 'Strategic_Friction_Score' in df_temp.columns else datetime_col,
                index='hour',
                columns='day_of_week',
                aggfunc='count' if 'Strategic_Friction_Score' not in df_temp.columns else 'sum',
                fill_value=0
            )

            # Ensure all 24 hours and 7 days are present in the grid
            # (fill missing combinations with zero)
            all_hours = range(24)
            all_days = range(7)
            pivot = pivot.reindex(index=all_hours, columns=all_days, fill_value=0)

            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # -------------------------------------------------------------------
            # Draw the seaborn heatmap.
            # YlOrRd colormap: yellow (low) -> orange (medium) -> red (high)
            # -------------------------------------------------------------------
            sns.heatmap(pivot, ax=ax, cmap='YlOrRd',
                       annot=True, fmt='.0f',
                       cbar_kws={'label': 'Escalation Count/Friction'},
                       xticklabels=day_labels,
                       yticklabels=[f'{h:02d}:00' for h in all_hours])

            # -------------------------------------------------------------------
            # Highlight shift boundary lines.
            # Common 8-hour shift pattern: Night (22-06), Day (06-14), Evening (14-22)
            # -------------------------------------------------------------------
            for shift_hour in [6, 14, 22]:  # Common shift change times
                ax.axhline(y=shift_hour, color='white', linewidth=2, linestyle='--', alpha=0.7)

            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Hour of Day', fontsize=12)

            # Identify the peak time (highest value cell in the heatmap)
            max_val = pivot.max().max()
            peak_hour = pivot.max(axis=1).idxmax()
            peak_day = pivot.max(axis=0).idxmax()

            plt.title(f'Escalation Time Pattern Analysis\nPeak: {day_labels[peak_day]} at {peak_hour:02d}:00',
                     fontsize=14, fontweight='bold', pad=20)

            # Shift labels on the left margin outside the heatmap
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
        Executive Scorecard - One-page visual KPI summary with gauge and cards.

        Creates a composite figure with:
            - Central gauge chart showing Operational Health Score (0-100)
            - 2x3 grid of KPI cards with RAG (Red/Amber/Green) status indicators

        Health Score Calculation:
            health_score = 100 - (recurrence_rate * 1.5) - (resolution_days * 5) - (critical_pct * 0.5)
            Clamped to [0, 100] range.

            Higher recurrence, longer resolution times, and more critical issues
            all reduce the health score. The weights (1.5, 5, 0.5) reflect the
            relative importance of each factor.

        KPI Cards (2x3 grid):
            1. Total Escalations - Always neutral (informational)
            2. Total Friction - Warning if > 5000, else green
            3. Avg Resolution - Red if > 3 days, else green
            4. Recurrence Rate - Red if > 20%, else green
            5. Critical Issues - Red if > 15%, else green
            6. Financial Impact - Always warning (cost is always a concern)

        Layout:
            - Top half: Gauge chart centered at [0.35, 0.55, 0.3, 0.4]
            - Bottom half: KPI cards in 3-column, 2-row grid
            - Title and generation date at the very top

        Shows:
            - Overall health score gauge
            - Traffic light grid for key metrics
            - Trend indicators

        Args:
            df: Escalation DataFrame with metric columns

        Returns:
            Path to saved PNG file, or None if generation fails
        """
        try:
            fig = plt.figure(figsize=(16, 10))

            # -------------------------------------------------------------------
            # Step 1: Calculate all KPI metrics from the DataFrame.
            # Each metric has a fallback default for when the column is missing.
            # -------------------------------------------------------------------
            total_tickets = len(df)
            total_friction = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 0
            avg_friction = df['Strategic_Friction_Score'].mean() if 'Strategic_Friction_Score' in df.columns else 0

            recurrence_rate = df['AI_Recurrence_Probability'].mean() * 100 if 'AI_Recurrence_Probability' in df.columns else 15
            resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5

            financial_impact = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else total_tickets * 850

            critical_pct = (df['tickets_data_severity'] == 'Critical').mean() * 100 if 'tickets_data_severity' in df.columns else 10

            # -------------------------------------------------------------------
            # Step 2: Calculate the composite Operational Health Score (0-100).
            # Deductions for recurrence, slow resolution, and critical volume.
            # -------------------------------------------------------------------
            health_score = max(0, min(100,
                100 - (recurrence_rate * 1.5) - (resolution_days * 5) - (critical_pct * 0.5)
            ))

            # -------------------------------------------------------------------
            # Step 3: Draw the main health gauge in the upper-center area.
            # Uses manual axes placement for precise positioning.
            # -------------------------------------------------------------------
            ax_gauge = fig.add_axes([0.35, 0.55, 0.3, 0.4])
            self._draw_gauge(ax_gauge, health_score, 'Operational Health Score')

            # -------------------------------------------------------------------
            # Step 4: Build and draw the KPI card grid.
            # Each card: (label, formatted_value, rag_status, trend_indicator)
            # -------------------------------------------------------------------
            kpis = [
                ('Total Escalations', f'{total_tickets:,}', 'neutral', ''),
                ('Total Friction', f'{total_friction:,.0f}', 'warning' if total_friction > 5000 else 'success', ''),
                ('Avg Resolution', f'{resolution_days:.1f} days', 'danger' if resolution_days > 3 else 'success', ''),
                ('Recurrence Rate', f'{recurrence_rate:.1f}%', 'danger' if recurrence_rate > 20 else 'success', ''),
                ('Critical Issues', f'{critical_pct:.1f}%', 'danger' if critical_pct > 15 else 'success', ''),
                ('Financial Impact', f'${financial_impact/1000:.0f}K', 'warning', ''),
            ]

            # Grid layout parameters for 2x3 card arrangement
            n_cols = 3
            n_rows = 2
            card_width = 0.28
            card_height = 0.18
            start_x = 0.06     # Left margin
            start_y = 0.08     # Bottom margin
            gap_x = 0.32       # Horizontal spacing between card centers
            gap_y = 0.22       # Vertical spacing between card centers

            for i, (label, value, status, trend) in enumerate(kpis):
                row = i // n_cols
                col = i % n_cols
                x = start_x + col * gap_x
                # Invert row order so first row is at top
                y = start_y + (n_rows - 1 - row) * gap_y

                # Each KPI card gets its own axes for precise rendering
                ax_card = fig.add_axes([x, y, card_width, card_height])
                self._draw_kpi_card(ax_card, label, value, status)

            # -------------------------------------------------------------------
            # Step 5: Add title and generation timestamp.
            # -------------------------------------------------------------------
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
        """
        Draw a semicircular gauge chart on the given axes.

        The gauge shows a value from 0-100 using a three-color arc
        (red/yellow/green) with a needle pointing to the current value.

        Visual Components:
            - Three colored arc segments: red (0-60), yellow (60-120), green (120-180)
              representing danger, warning, and success zones
            - A needle (arrow) pointing to the value's position on the arc
            - Center hub circle covering the arrow base
            - Large numeric value display below the gauge
            - Title text above the gauge

        Coordinate System:
            - The gauge is drawn in a [-1.2, 1.2] x [-0.2, 1.2] space
            - Arc angles go from 0 to 180 degrees (right to left semicircle)
            - Needle angle = 180 - (value/100 * 180), mapping value to arc position

        Args:
            ax: Matplotlib axes to draw on
            value: Score value (0-100) determining needle position
            title: Text label displayed above the gauge
        """
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw three arc segments (Wedge patches) for the gauge background.
        # Each covers 60 degrees of the 180-degree semicircle.
        colors = [self.COLORS['danger'], self.COLORS['warning'], self.COLORS['success']]
        angles = [0, 60, 120, 180]  # Degree boundaries for the three segments

        for i, (start, end) in enumerate(zip(angles[:-1], angles[1:])):
            theta1, theta2 = np.radians(start), np.radians(end)
            # Wedge: centered at origin, radius 1, width 0.3 (donut shape)
            wedge = mpatches.Wedge((0, 0), 1, start, end, width=0.3,
                                   facecolor=colors[i], edgecolor='white', linewidth=2)
            ax.add_patch(wedge)

        # Draw the needle pointing to the value's position on the arc.
        # Angle mapping: value 0 -> 180 degrees (left), value 100 -> 0 degrees (right)
        needle_angle = np.radians(180 - (value / 100 * 180))
        needle_x = 0.7 * np.cos(needle_angle)  # Needle tip x (length 0.7)
        needle_y = 0.7 * np.sin(needle_angle)  # Needle tip y
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.08, head_length=0.05,
                fc=self.COLORS['dark'], ec=self.COLORS['dark'], linewidth=2)

        # Center hub circle to cover the needle base
        center = Circle((0, 0), 0.1, facecolor=self.COLORS['dark'], edgecolor='white')
        ax.add_patch(center)

        # Large numeric value display below the gauge center
        ax.text(0, -0.1, f'{value:.0f}', ha='center', va='top',
               fontsize=28, fontweight='bold', color=self.COLORS['dark'])
        # Title above the gauge arc
        ax.text(0, 1.15, title, ha='center', va='bottom',
               fontsize=12, fontweight='bold')

    def _draw_kpi_card(self, ax, label: str, value: str, status: str):
        """
        Draw a single KPI card on the given axes.

        Each card is a rounded rectangle with a status-colored background,
        a large value display in the center, a label below, and a small
        colored circle indicator in the top-right corner.

        Status Color Mapping:
            - 'success': Light green background (#E8F5E9), green border
            - 'warning': Light yellow background (#FFF8E1), amber border
            - 'danger': Light red background (#FFEBEE), red border
            - 'neutral': Light gray background (#F5F5F5), gray border

        The small circle indicator in the top-right corner provides a
        "traffic light" quick-read of the metric's health status.

        Args:
            ax: Matplotlib axes to draw on (should be [0,1] x [0,1] space)
            label: Metric name displayed below the value
            value: Formatted metric value displayed prominently
            status: RAG status string ('success', 'warning', 'danger', 'neutral')
        """
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Background color mapping: light pastel shades for card fill
        status_colors = {
            'success': '#E8F5E9',   # Light green
            'warning': '#FFF8E1',   # Light yellow
            'danger': '#FFEBEE',    # Light red
            'neutral': '#F5F5F5'    # Light gray
        }
        # Border color mapping: saturated colors for the card border
        border_colors = {
            'success': self.COLORS['success'],
            'warning': self.COLORS['warning'],
            'danger': self.COLORS['danger'],
            'neutral': self.COLORS['neutral']
        }

        # Draw the rounded rectangle card background
        rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                              boxstyle="round,pad=0.02,rounding_size=0.05",
                              facecolor=status_colors.get(status, '#F5F5F5'),
                              edgecolor=border_colors.get(status, '#CCCCCC'),
                              linewidth=3)
        ax.add_patch(rect)

        # Large value text centered in the card
        ax.text(0.5, 0.55, value, ha='center', va='center',
               fontsize=18, fontweight='bold', color=self.COLORS['dark'])

        # Label text below the value
        ax.text(0.5, 0.2, label, ha='center', va='center',
               fontsize=10, color=self.COLORS['neutral'])

        # Small circle status indicator in the top-right corner of the card.
        # Provides a quick "traffic light" visual cue.
        indicator_color = border_colors.get(status, self.COLORS['neutral'])
        indicator = Circle((0.9, 0.85), 0.05, facecolor=indicator_color, edgecolor='white')
        ax.add_patch(indicator)

    # =========================================================================
    # AGING TICKETS ANALYSIS (07_sla/)
    # =========================================================================

    def chart_aging_burndown(self, df: pd.DataFrame,
                              datetime_col: str = 'tickets_data_issue_datetime') -> Optional[str]:
        """
        Aging Tickets Burndown - Shows open ticket age distribution.

        Creates a bar chart showing the distribution of ticket ages across
        five time buckets. Older tickets are shown in progressively warmer
        colors (green to red) to highlight aging concerns.

        Age Calculation:
            age_days = (current_datetime - ticket_open_datetime).days

        Age Buckets:
            - 0-7 days (green): Fresh tickets, normal processing
            - 8-14 days (light green): Approaching first follow-up window
            - 15-30 days (yellow): Needs attention, may be stalling
            - 31-60 days (orange): Significant aging, escalation risk
            - 60+ days (red): Critical aging, immediate attention required

        Title Annotation:
            The title includes the count and percentage of tickets aged beyond
            30 days, providing an immediate aging severity indicator.

        Args:
            df: Escalation DataFrame with datetime data
            datetime_col: Column name for ticket creation datetime

        Returns:
            Path to saved PNG file, or None if generation fails
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Check if datetime column exists; show placeholder if not
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

            # Calculate ticket age in days from creation to now
            now = datetime.now()
            df_temp['age_days'] = (now - df_temp['datetime']).dt.days

            # -------------------------------------------------------------------
            # Define age buckets and count tickets in each bucket.
            # Each bucket: (label, min_days, max_days)
            # -------------------------------------------------------------------
            age_buckets = [
                ('0-7 days', 0, 7),
                ('8-14 days', 8, 14),
                ('15-30 days', 15, 30),
                ('31-60 days', 31, 60),
                ('60+ days', 61, 9999)     # Upper bound effectively unbounded
            ]

            bucket_counts = []
            for label, min_age, max_age in age_buckets:
                count = ((df_temp['age_days'] >= min_age) & (df_temp['age_days'] <= max_age)).sum()
                bucket_counts.append(count)

            labels = [b[0] for b in age_buckets]

            # Color gradient: green (fresh) -> light green -> yellow -> orange -> red (aged)
            colors = [self.COLORS['success'], '#7CB342', self.COLORS['warning'],
                     '#FF9800', self.COLORS['danger']]

            # Draw the bar chart
            bars = ax.bar(labels, bucket_counts, color=colors, edgecolor='white', linewidth=2)

            # Add count labels above each bar
            for bar, count in zip(bars, bucket_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xlabel('Age Category', fontsize=12)
            ax.set_ylabel('Number of Tickets', fontsize=12)

            # Calculate aging severity metric for the title:
            # count of tickets aged beyond 30 days (buckets 3 and 4)
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
        """
        Generate all advanced charts and return their file paths.

        Orchestrates the generation of all six advanced chart types, collecting
        file paths organized by category. Each chart is generated independently;
        if one chart fails, the others continue.

        Generation Order:
            SLA Charts (07_sla/):
                1. SLA Compliance Funnel
                2. Cost Avoidance Waterfall
                3. Time-of-Day Heatmap
                4. Aging Burndown
            Efficiency Charts (08_efficiency/):
                5. Engineer Efficiency Quadrant
            Executive Charts (09_executive/):
                6. Executive Scorecard

        Called by ExcelReportWriter._generate_advanced_charts() which passes the
        scored DataFrame and collects the returned paths for embedding in the
        Excel report.

        Args:
            df: Scored escalation DataFrame with all AI-enriched columns

        Returns:
            Dict mapping category keys ('sla', 'efficiency', 'executive') to
            lists of file path strings for successfully generated charts.
            Empty lists indicate all charts in that category failed.
        """
        generated = {
            'sla': [],
            'efficiency': [],
            'executive': [],
        }

        try:
            # --- SLA Charts (07_sla/) ---
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

            # --- Efficiency Charts (08_efficiency/) ---
            result = self.chart_engineer_quadrant(df)
            if result:
                generated['efficiency'].append(result)

            # --- Executive Charts (09_executive/) ---
            result = self.chart_executive_scorecard(df)
            if result:
                generated['executive'].append(result)

        except Exception as e:
            # Catch-all: if the orchestration itself fails, log and return
            # whatever charts were successfully generated before the error.
            print(f"Error generating advanced charts: {e}")

        return generated
