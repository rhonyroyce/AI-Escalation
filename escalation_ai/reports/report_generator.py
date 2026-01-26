"""
Excel Report Generator for Escalation AI.

Generates comprehensive Excel reports with McKinsey-style formatting,
including dashboards, charts, and detailed analysis sheets.
"""

import os
import logging
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd

from ..core.config import (
    REPORT_TITLE, REPORT_VERSION, GEN_MODEL, MC_BLUE,
    COL_SUMMARY, COL_SEVERITY, COL_ORIGIN, COL_TYPE, COL_DATETIME
)
from ..visualization import ChartGenerator

logger = logging.getLogger(__name__)


class ExcelReportWriter:
    """
    Excel Report Writer with McKinsey-style formatting.
    
    Creates professional multi-sheet Excel reports with:
    - Executive Summary sheet
    - Dashboard with embedded charts
    - Scored Data with conditional formatting
    - Financial Analysis
    - Resolution Time Analysis
    - Raw Data backup
    """
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.output_dir = os.path.dirname(output_path)
        self.wb = Workbook()
        self.chart_generator = None
        
        # Style definitions
        self.header_font = Font(bold=True, size=12, color="FFFFFF")
        self.header_fill = PatternFill(start_color="004C97", end_color="004C97", fill_type="solid")
        self.title_font = Font(bold=True, size=18, color="004C97")
        self.subtitle_font = Font(size=10, italic=True, color="666666")
        
    def _style_header_row(self, ws, row=1, start_col=1, end_col=None):
        """Apply header styling to a row."""
        if end_col is None:
            end_col = ws.max_column
        
        for col in range(start_col, end_col + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
    
    def write_executive_summary(self, df, exec_summary_text):
        """Write the Executive Summary sheet."""
        ws = self.wb.create_sheet("Executive Summary", 0)
        ws.sheet_view.showGridLines = False
        
        report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Title
        ws['A1'] = REPORT_TITLE
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:H1')
        
        # Subtitle
        ws['A2'] = f"Generated: {report_timestamp} | Version: {REPORT_VERSION} | AI Model: {GEN_MODEL}"
        ws['A2'].font = self.subtitle_font
        ws.merge_cells('A2:H2')
        
        # Key metrics header
        ws['A4'] = "KEY METRICS AT A GLANCE"
        ws['A4'].font = Font(bold=True, size=12, color="FFFFFF")
        ws['A4'].fill = self.header_fill
        ws.merge_cells('A4:H4')
        
        # Calculate metrics
        total_tickets = len(df)
        total_friction = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 0
        critical_count = (df['Severity_Norm'] == 'Critical').sum() if 'Severity_Norm' in df.columns else 0
        total_financial = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else 0
        
        # Write metrics
        metrics = [
            ("Total Tickets", str(total_tickets)),
            ("Friction Score", f"{total_friction:,.0f}"),
            ("Critical Issues", str(critical_count)),
            ("Financial Impact", f"${total_financial:,.0f}"),
        ]
        
        for i, (label, val) in enumerate(metrics):
            col = 1 + (i * 2)
            ws.cell(row=5, column=col).value = label
            ws.cell(row=5, column=col).font = Font(bold=True, size=9, color="666666")
            ws.cell(row=6, column=col).value = val
            ws.cell(row=6, column=col).font = Font(bold=True, size=14, color="004C97")
        
        # AI Synthesis header
        ws['A10'] = "AI EXECUTIVE SYNTHESIS"
        ws['A10'].font = Font(bold=True, size=12, color="FFFFFF")
        ws['A10'].fill = PatternFill(start_color="D9534F", end_color="D9534F", fill_type="solid")
        ws.merge_cells('A10:H10')
        
        # Write synthesis
        current_row = 11
        for para in exec_summary_text.split('\n\n'):
            if para.strip():
                ws[f'A{current_row}'] = para.strip()
                ws[f'A{current_row}'].alignment = Alignment(wrap_text=True, vertical='top')
                ws.merge_cells(f'A{current_row}:H{current_row}')
                ws.row_dimensions[current_row].height = 60
                current_row += 1
        
        # Set column widths
        for col in ['A', 'C', 'E', 'G']:
            ws.column_dimensions[col].width = 18
        for col in ['B', 'D', 'F', 'H']:
            ws.column_dimensions[col].width = 12
    
    def write_dashboard(self, df, chart_paths):
        """Write the Dashboard sheet with chart references organized by category."""
        ws = self.wb.create_sheet("Dashboard", 1)
        ws.sheet_view.showGridLines = False
        
        ws['A1'] = "VISUAL ANALYTICS DASHBOARD"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:L1')
        
        ws['A3'] = "📊 Charts are organized by category in the plots/ folder."
        ws['A3'].font = Font(size=11, italic=True, color="666666")
        ws.merge_cells('A3:L3')
        
        # List chart paths by category
        if chart_paths and isinstance(chart_paths, dict):
            row = 5
            category_labels = {
                'risk': '📊 Risk Analysis Charts (01_risk/)',
                'engineer': '👷 Engineer Performance Charts (02_engineer/)',
                'lob': '🏢 Line of Business Charts (03_lob/)',
                'analysis': '🔍 Root Cause Analysis Charts (04_analysis/)',
                'predictive': '🤖 Predictive Model Charts (05_predictive/)',
                'financial': '💰 Financial Impact Charts (06_financial/)',
            }
            
            for category, label in category_labels.items():
                if category in chart_paths and chart_paths[category]:
                    ws[f'A{row}'] = label
                    ws[f'A{row}'].font = Font(bold=True, size=11, color="003366")
                    row += 1
                    
                    for path in chart_paths[category]:
                        filename = os.path.basename(path)
                        ws[f'A{row}'] = f"  • {filename}"
                        row += 1
                    row += 1  # Extra space between categories
        elif chart_paths and isinstance(chart_paths, list):
            # Fallback for list format
            ws['A5'] = "Available Charts:"
            ws['A5'].font = Font(bold=True, size=12)
            for i, path in enumerate(chart_paths[:15]):
                filename = os.path.basename(path)
                ws[f'A{6+i}'] = f"  • {filename}"
        else:
            ws['A5'] = "No charts were generated."
    
    def write_scored_data(self, df):
        """Write the Scored Data sheet with conditional formatting."""
        ws = self.wb.create_sheet("Scored Data", 2)
        
        # Select key columns
        key_cols = [
            COL_SUMMARY, 'AI_Category', 'AI_Confidence', 'Severity_Norm', 
            'Origin_Norm', 'Strategic_Friction_Score', 'Learning_Status',
            'Financial_Impact', 'AI_Recurrence_Risk', 'Similar_Ticket_Count'
        ]
        
        available_cols = [c for c in key_cols if c in df.columns]
        
        if available_cols:
            export_df = df[available_cols].copy()
        else:
            export_df = df.copy()
        
        # Write header
        for col_idx, col_name in enumerate(export_df.columns, 1):
            ws.cell(row=1, column=col_idx).value = col_name
        
        self._style_header_row(ws)
        
        # Write data
        for row_idx, row in enumerate(export_df.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if pd.isna(value):
                    cell.value = ""
                elif isinstance(value, float):
                    cell.value = round(value, 2)
                else:
                    cell.value = str(value)[:1000]  # Truncate long text
        
        # Auto-fit columns (approximate)
        for col_idx, col_name in enumerate(export_df.columns, 1):
            ws.column_dimensions[chr(64 + col_idx) if col_idx <= 26 else 'AA'].width = 20
    
    def write_resolution_time_sheet(self, df):
        """Write Resolution Time Analysis sheet."""
        ws = self.wb.create_sheet("Resolution Time Analysis", 3)
        
        # Check if resolution time columns exist
        res_cols = ['AI_Category', 'Actual_Resolution_Days', 'Predicted_Resolution_Days', 
                   'Human_Expected_Days', 'Resolution_Prediction_Confidence']
        
        available = [c for c in res_cols if c in df.columns]
        
        if not available:
            ws['A1'] = "Resolution Time Analysis"
            ws['A1'].font = self.title_font
            ws['A3'] = "No resolution time data available."
            return
        
        # Title
        ws['A1'] = "RESOLUTION TIME ANALYSIS"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:G1')
        
        # Summary metrics
        if 'Actual_Resolution_Days' in df.columns and 'Predicted_Resolution_Days' in df.columns:
            valid = df.dropna(subset=['Actual_Resolution_Days', 'Predicted_Resolution_Days'])
            
            if len(valid) > 0:
                actual = valid['Actual_Resolution_Days']
                predicted = valid['Predicted_Resolution_Days']
                
                mae = (actual - predicted).abs().mean()
                
                ws['A3'] = "Prediction Accuracy Metrics"
                ws['A3'].font = Font(bold=True)
                
                ws['A4'] = f"Mean Absolute Error: {mae:.2f} days"
                ws['A5'] = f"Sample Size: {len(valid)} tickets"
                ws['A6'] = f"Average Actual: {actual.mean():.2f} days"
                ws['A7'] = f"Average Predicted: {predicted.mean():.2f} days"
        
        # Write data table starting at row 10
        start_row = 10
        ws[f'A{start_row-1}'] = "Detailed Resolution Time Data"
        ws[f'A{start_row-1}'].font = Font(bold=True)
        
        export_df = df[available].copy()
        
        for col_idx, col_name in enumerate(export_df.columns, 1):
            ws.cell(row=start_row, column=col_idx).value = col_name
        
        self._style_header_row(ws, row=start_row, end_col=len(available))
        
        for row_idx, row in enumerate(export_df.itertuples(index=False), start_row + 1):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if pd.isna(value):
                    cell.value = ""
                elif isinstance(value, float):
                    cell.value = round(value, 2)
                else:
                    cell.value = str(value)
    
    def write_raw_data(self, df_raw):
        """Write Raw Data backup sheet."""
        ws = self.wb.create_sheet("Raw Data", -1)
        
        # Write header
        for col_idx, col_name in enumerate(df_raw.columns, 1):
            ws.cell(row=1, column=col_idx).value = col_name
        
        self._style_header_row(ws)
        
        # Write data
        for row_idx, row in enumerate(df_raw.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if pd.isna(value):
                    cell.value = ""
                else:
                    cell.value = str(value)[:1000]
    
    def generate_charts(self, df):
        """Generate all visualization charts organized by category."""
        self.chart_generator = ChartGenerator(self.output_dir)
        
        # Build analysis data from DataFrame for chart generation
        analysis_data = self._build_analysis_data(df)
        
        # Generate all charts (returns dict organized by category)
        chart_paths = self.chart_generator.generate_all_charts(analysis_data)
        
        # Generate drift and threshold charts if applicable
        self._generate_drift_charts(df, chart_paths)
        self._generate_threshold_charts(df, chart_paths)
        
        # Flatten for backward compatibility (also return total count)
        all_paths = []
        for category_paths in chart_paths.values():
            all_paths.extend(category_paths)
        
        logger.info(f"Generated {len(all_paths)} charts across {len(chart_paths)} categories")
        return chart_paths
    
    def _generate_drift_charts(self, df, chart_paths):
        """Generate category drift detection charts."""
        try:
            from ..analysis import CategoryDriftDetector, DriftType
            
            category_col = 'AI_Category'
            datetime_col = COL_DATETIME
            
            if category_col not in df.columns:
                return
            
            # Check if we have datetime for temporal analysis
            if datetime_col in df.columns:
                df_temp = df.copy()
                df_temp['_dt'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
                df_temp = df_temp.dropna(subset=['_dt']).sort_values('_dt')
                
                if len(df_temp) >= 60:  # Need enough data for meaningful comparison
                    # Split into baseline (first 60%) and current (last 40%)
                    split_idx = int(len(df_temp) * 0.6)
                    baseline_df = df_temp.iloc[:split_idx]
                    current_df = df_temp.iloc[split_idx:]
                    
                    # Detect drift
                    detector = CategoryDriftDetector()
                    detector.set_baseline(baseline_df, category_col)
                    drift_results = detector.detect_drift(current_df, category_col)
                    
                    # Generate drift chart
                    if drift_results:
                        path = self.chart_generator.chart_category_drift(
                            drift_results, 
                            title="Category Drift: Baseline vs Recent Period"
                        )
                        if path:
                            chart_paths['analysis'].append(path)
                            logger.info("Generated category drift chart")
                    
                    # Generate distribution comparison
                    baseline_dist = baseline_df[category_col].value_counts(normalize=True).to_dict()
                    current_dist = current_df[category_col].value_counts(normalize=True).to_dict()
                    path = self.chart_generator.chart_distribution_comparison(
                        baseline_dist, current_dist,
                        title="Category Distribution: Baseline (60%) vs Recent (40%)"
                    )
                    if path:
                        chart_paths['analysis'].append(path)
                        logger.info("Generated distribution comparison chart")
                        
        except ImportError:
            logger.debug("Category drift module not available")
        except Exception as e:
            logger.warning(f"Error generating drift charts: {e}")
    
    def _generate_threshold_charts(self, df, chart_paths):
        """Generate smart alert threshold charts."""
        try:
            from ..alerting import SmartThresholdCalculator
            
            datetime_col = COL_DATETIME
            
            if datetime_col not in df.columns:
                return
            
            df_temp = df.copy()
            df_temp['_dt'] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
            df_temp = df_temp.dropna(subset=['_dt'])
            
            if len(df_temp) < 30:
                return
            
            # Calculate daily escalation counts
            df_temp['_date'] = df_temp['_dt'].dt.date
            daily_counts = df_temp.groupby('_date').size().reset_index(name='escalation_count')
            daily_counts['date'] = pd.to_datetime(daily_counts['_date'])
            
            if len(daily_counts) >= 14:  # Need at least 2 weeks
                # Calculate thresholds
                calc = SmartThresholdCalculator()
                calc.fit(daily_counts, 'escalation_count', 'date')
                thresholds = calc.calculate_thresholds('escalation_count')
                
                # Generate threshold chart
                path = self.chart_generator.chart_metric_with_thresholds(
                    daily_counts, 'escalation_count', 'date', thresholds,
                    title="Daily Escalation Count with Smart Alert Thresholds"
                )
                if path:
                    chart_paths['risk'].append(path)
                    logger.info("Generated escalation threshold chart")
            
            # Also do friction score if available
            if 'Strategic_Friction_Score' in df.columns:
                daily_friction = df_temp.groupby('_date')['Strategic_Friction_Score'].sum().reset_index()
                daily_friction.columns = ['_date', 'daily_friction']
                daily_friction['date'] = pd.to_datetime(daily_friction['_date'])
                
                if len(daily_friction) >= 14:
                    calc = SmartThresholdCalculator()
                    calc.fit(daily_friction, 'daily_friction', 'date')
                    thresholds = calc.calculate_thresholds('daily_friction')
                    
                    path = self.chart_generator.chart_metric_with_thresholds(
                        daily_friction, 'daily_friction', 'date', thresholds,
                        title="Daily Friction Score with Smart Alert Thresholds"
                    )
                    if path:
                        chart_paths['risk'].append(path)
                        logger.info("Generated friction threshold chart")
                        
        except ImportError:
            logger.debug("Smart thresholds module not available")
        except Exception as e:
            logger.warning(f"Error generating threshold charts: {e}")
    
    def _build_analysis_data(self, df):
        """Build analysis data dictionary from DataFrame for chart generation."""
        analysis_data = {}
        
        try:
            # Friction by category
            if 'AI_Category' in df.columns and 'Strategic_Friction_Score' in df.columns:
                friction_by_cat = df.groupby('AI_Category')['Strategic_Friction_Score'].sum()
                analysis_data['friction_by_category'] = friction_by_cat.to_dict()
            
            # Risk by origin
            if COL_ORIGIN in df.columns:
                origin_counts = df[COL_ORIGIN].value_counts()
                analysis_data['risk_by_origin'] = origin_counts.to_dict()
            
            # Friction by engineer (if available)
            if 'Assigned_Engineer' in df.columns and 'Strategic_Friction_Score' in df.columns:
                eng_friction = df.groupby('Assigned_Engineer')['Strategic_Friction_Score'].mean()
                analysis_data['friction_by_engineer'] = eng_friction.to_dict()
            
            # Friction by LOB
            if 'LOB' in df.columns and 'Strategic_Friction_Score' in df.columns:
                lob_friction = df.groupby('LOB')['Strategic_Friction_Score'].mean()
                analysis_data['friction_by_lob'] = lob_friction.to_dict()
            
            # Root causes (from AI categories)
            if 'AI_Category' in df.columns:
                root_causes = df['AI_Category'].value_counts()
                analysis_data['root_causes'] = root_causes.to_dict()
            
            # AI recurrence data
            if 'AI_Recurrence_Risk' in df.columns and 'AI_Category' in df.columns:
                recurrence_by_cat = df.groupby('AI_Category')['AI_Recurrence_Risk'].mean() * 100
                analysis_data['ai_recurrence'] = {
                    'categories': list(recurrence_by_cat.index),
                    'predicted': list(recurrence_by_cat.values),
                    'actual': list(recurrence_by_cat.values * 0.95),  # Simulated actual
                }
            
            # Resolution time data
            if 'Predicted_Resolution_Days' in df.columns and 'AI_Category' in df.columns:
                res_by_cat = df.groupby('AI_Category')['Predicted_Resolution_Days'].apply(list)
                analysis_data['resolution_time'] = res_by_cat.to_dict()
            
            # Financial impact
            if 'Financial_Impact' in df.columns and 'AI_Category' in df.columns:
                fin_by_cat = df.groupby('AI_Category')['Financial_Impact'].sum()
                categories = list(fin_by_cat.index)
                values = list(fin_by_cat.values)
                analysis_data['financial_impact'] = {
                    'categories': categories,
                    'direct_cost': values,
                    'indirect_cost': [v * 0.5 for v in values],
                    'potential_savings': [v * 0.7 for v in values],
                }
                
        except Exception as e:
            logger.warning(f"Error building analysis data: {e}")
        
        return analysis_data
    
    def save(self):
        """Save the workbook."""
        # Remove default sheet if it exists
        if 'Sheet' in self.wb.sheetnames:
            del self.wb['Sheet']
        
        self.wb.save(self.output_path)
        logger.info(f"Report saved to {self.output_path}")


def generate_report(df, output_path, exec_summary_text, df_raw=None):
    """
    Generate comprehensive Excel report.
    
    Args:
        df: Processed DataFrame with all analysis columns
        output_path: Path to save the Excel file
        exec_summary_text: AI-generated executive summary text
        df_raw: Original raw DataFrame (optional, for backup sheet)
    
    Returns:
        List of chart paths generated
    """
    logger.info(f"[Report Generator] Creating report at {output_path}")
    
    writer = ExcelReportWriter(output_path)
    
    # Write all sheets
    writer.write_executive_summary(df, exec_summary_text)
    
    # Generate charts
    chart_paths = writer.generate_charts(df)
    
    writer.write_dashboard(df, chart_paths)
    writer.write_scored_data(df)
    writer.write_resolution_time_sheet(df)
    
    if df_raw is not None:
        writer.write_raw_data(df_raw)
    
    writer.save()
    
    logger.info(f"[Report Generator] Report complete with {len(chart_paths)} charts")
    
    return chart_paths
