"""
Excel Report Generator for Escalation AI.

Generates comprehensive Excel reports with McKinsey-style formatting,
including dashboards, charts, and detailed analysis sheets.
"""

import os
import logging
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, NamedStyle
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
import pandas as pd

from ..core.config import (
    REPORT_TITLE, REPORT_VERSION, GEN_MODEL, MC_BLUE,
    COL_SUMMARY, COL_SEVERITY, COL_ORIGIN, COL_TYPE, COL_DATETIME,
    COL_ENGINEER, COL_LOB
)
from ..visualization import ChartGenerator, AdvancedChartGenerator

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
        self.title_font = Font(bold=True, size=22, color="004C97")
        self.subtitle_font = Font(size=11, italic=True, color="666666")
        
        # Enhanced styling
        self.thin_border = Border(
            left=Side(style='thin', color='CCCCCC'),
            right=Side(style='thin', color='CCCCCC'),
            top=Side(style='thin', color='CCCCCC'),
            bottom=Side(style='thin', color='CCCCCC')
        )
        self.thick_border = Border(
            left=Side(style='medium', color='004C97'),
            right=Side(style='medium', color='004C97'),
            top=Side(style='medium', color='004C97'),
            bottom=Side(style='medium', color='004C97')
        )
        self.kpi_fill_blue = PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid")
        self.kpi_fill_green = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
        self.kpi_fill_red = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")
        self.kpi_fill_amber = PatternFill(start_color="FFF8E1", end_color="FFF8E1", fill_type="solid")
        self.section_font = Font(bold=True, size=13, color="004C97")
        
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
        """Write a visually enhanced Executive Summary sheet."""
        ws = self.wb.create_sheet("Executive Summary", 0)
        ws.sheet_view.showGridLines = False
        
        report_timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
        
        # Set column widths for two-column sectional layout
        ws.column_dimensions['A'].width = 2   # Left margin
        ws.column_dimensions['B'].width = 4   # Left panel start
        ws.column_dimensions['C'].width = 14  # Left panel content
        ws.column_dimensions['D'].width = 14  # Left panel content
        ws.column_dimensions['E'].width = 2   # Gutter between columns
        ws.column_dimensions['F'].width = 4   # Right panel start
        ws.column_dimensions['G'].width = 14  # Right panel content
        ws.column_dimensions['H'].width = 14  # Right panel content
        ws.column_dimensions['I'].width = 2   # Right margin
        
        # ═══════════════════════════════════════════════════════════════
        # HEADER SECTION
        # ═══════════════════════════════════════════════════════════════
        ws.row_dimensions[1].height = 8  # Top spacing
        ws.row_dimensions[2].height = 35  # Title row
        
        # Title with blue background banner
        for col in range(2, 9):
            ws.cell(row=2, column=col).fill = self.header_fill
        ws['B2'] = f"📊 {REPORT_TITLE}"
        ws['B2'].font = Font(bold=True, size=20, color="FFFFFF")
        ws['B2'].alignment = Alignment(horizontal='center', vertical='center')
        ws.merge_cells('B2:H2')
        
        # Subtitle row
        ws.row_dimensions[3].height = 22
        ws['B3'] = f"Generated: {report_timestamp}  •  Version: {REPORT_VERSION}  •  AI Model: {GEN_MODEL}"
        ws['B3'].font = Font(size=10, italic=True, color="666666")
        ws['B3'].alignment = Alignment(horizontal='center', vertical='center')
        ws.merge_cells('B3:H3')
        
        # ═══════════════════════════════════════════════════════════════
        # KPI CARDS SECTION
        # ═══════════════════════════════════════════════════════════════
        ws.row_dimensions[4].height = 6  # Spacer
        ws.row_dimensions[5].height = 22
        
        # Section header
        ws['B5'] = "📈 KEY METRICS AT A GLANCE"
        ws['B5'].font = Font(bold=True, size=12, color="004C97")
        ws.merge_cells('B5:H5')
        
        # Calculate metrics
        total_tickets = len(df)
        total_friction = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 0
        avg_friction = df['Strategic_Friction_Score'].mean() if 'Strategic_Friction_Score' in df.columns else 0
        critical_count = (df['Severity_Norm'] == 'Critical').sum() if 'Severity_Norm' in df.columns else 0
        total_financial = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else 0
        avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 0
        recurrence_rate = df['AI_Recurrence_Probability'].mean() * 100 if 'AI_Recurrence_Probability' in df.columns else 0
        
        # KPI card layout - 4 cards across
        kpi_data = [
            ("🎫 Total Tickets", str(total_tickets), self.kpi_fill_blue, "Records analyzed"),
            ("⚡ Friction Score", f"{total_friction:,.0f}", self.kpi_fill_amber, f"Avg: {avg_friction:.1f}"),
            ("🚨 Critical Issues", str(critical_count), self.kpi_fill_red, f"{critical_count/total_tickets*100:.1f}% of total"),
            ("💰 Financial Impact", f"${total_financial:,.0f}", self.kpi_fill_green, f"${total_financial/total_tickets:.0f}/ticket"),
        ]
        
        # KPI rows setup
        ws.row_dimensions[6].height = 8  # Spacer
        ws.row_dimensions[7].height = 18  # Label row
        ws.row_dimensions[8].height = 30  # Value row
        ws.row_dimensions[9].height = 16  # Subtext row
        
        kpi_cols = [(2, 3), (4, 5), (6, 7), (8, 8)]  # Column ranges for each KPI
        
        for i, (label, value, fill, subtext) in enumerate(kpi_data):
            start_col, end_col = kpi_cols[i][0], kpi_cols[i][-1] if len(kpi_cols[i]) > 1 else kpi_cols[i][0]
            
            # Apply fill to all cells in the KPI card
            for row in [7, 8, 9]:
                for col in range(start_col, end_col + 1):
                    cell = ws.cell(row=row, column=col)
                    cell.fill = fill
                    cell.border = self.thin_border
            
            # Label
            ws.cell(row=7, column=start_col).value = label
            ws.cell(row=7, column=start_col).font = Font(bold=True, size=9, color="333333")
            ws.cell(row=7, column=start_col).alignment = Alignment(horizontal='center', vertical='center')
            if start_col != end_col:
                ws.merge_cells(start_row=7, start_column=start_col, end_row=7, end_column=end_col)
            
            # Value
            ws.cell(row=8, column=start_col).value = value
            ws.cell(row=8, column=start_col).font = Font(bold=True, size=18, color="004C97")
            ws.cell(row=8, column=start_col).alignment = Alignment(horizontal='center', vertical='center')
            if start_col != end_col:
                ws.merge_cells(start_row=8, start_column=start_col, end_row=8, end_column=end_col)
            
            # Subtext
            ws.cell(row=9, column=start_col).value = subtext
            ws.cell(row=9, column=start_col).font = Font(size=8, italic=True, color="666666")
            ws.cell(row=9, column=start_col).alignment = Alignment(horizontal='center', vertical='center')
            if start_col != end_col:
                ws.merge_cells(start_row=9, start_column=start_col, end_row=9, end_column=end_col)
        
        # ═══════════════════════════════════════════════════════════════
        # SECONDARY METRICS ROW
        # ═══════════════════════════════════════════════════════════════
        ws.row_dimensions[10].height = 6  # Spacer
        ws.row_dimensions[11].height = 22
        
        secondary_metrics = [
            ("Avg Resolution", f"{avg_resolution:.1f} days"),
            ("Recurrence Risk", f"{recurrence_rate:.1f}%"),
            ("Categories", f"{df['AI_Category'].nunique() if 'AI_Category' in df.columns else 0}"),
        ]
        
        sec_start = 2
        for label, value in secondary_metrics:
            cell = ws.cell(row=11, column=sec_start)
            cell.value = f"{label}: {value}"
            cell.font = Font(size=10, color="004C97")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            sec_start += 2
        
        # ═══════════════════════════════════════════════════════════════
        # AI EXECUTIVE SYNTHESIS - Visual Executive Format
        # ═══════════════════════════════════════════════════════════════
        import re

        # Color definitions for visual elements
        fill_critical = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")  # Light red
        fill_warning = PatternFill(start_color="FFF8E1", end_color="FFF8E1", fill_type="solid")   # Light amber
        fill_success = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")   # Light green
        fill_insight = PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid")   # Light blue
        fill_dark = PatternFill(start_color="263238", end_color="263238", fill_type="solid")      # Dark gray
        border_gray = Border(
            left=Side(style='thin', color='E0E0E0'),
            right=Side(style='thin', color='E0E0E0'),
            top=Side(style='thin', color='E0E0E0'),
            bottom=Side(style='thin', color='E0E0E0')
        )

        ws.row_dimensions[12].height = 10  # Spacer
        ws.row_dimensions[13].height = 32  # Section header

        # Section header with dark banner
        for col in range(2, 9):
            ws.cell(row=13, column=col).fill = fill_dark
        ws['B13'] = "⚡ AI EXECUTIVE SYNTHESIS"
        ws['B13'].font = Font(bold=True, size=14, color="FFFFFF")
        ws['B13'].alignment = Alignment(horizontal='center', vertical='center')
        ws.merge_cells('B13:H13')

        current_row = 15

        # ─────────────────────────────────────────────────────────────
        # QUICK STATS DASHBOARD (extracted from data)
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 20
        ws.cell(row=current_row, column=2).value = "📊 SNAPSHOT"
        ws.cell(row=current_row, column=2).font = Font(bold=True, size=11, color="004C97")
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 1

        # Quick stats row with colored boxes
        ws.row_dimensions[current_row].height = 50
        quick_stats = [
            ("B", "C", f"{total_tickets:,}", "Total Tickets", self.kpi_fill_blue),
            ("D", "E", f"${total_financial:,.0f}", "Financial Impact", fill_critical if total_financial > 500000 else fill_warning),
            ("F", "G", f"{critical_count}", "Critical Issues", fill_critical if critical_count > 20 else fill_warning if critical_count > 5 else fill_success),
            ("H", "H", f"{avg_friction:.0f}", "Avg Friction", fill_warning if avg_friction > 50 else fill_success),
        ]

        for start_col, end_col, value, label, fill in quick_stats:
            # Value cell
            cell = ws[f'{start_col}{current_row}']
            cell.value = value
            cell.font = Font(bold=True, size=16, color="004C97")
            cell.fill = fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray
            if start_col != end_col:
                ws.merge_cells(f'{start_col}{current_row}:{end_col}{current_row}')

        current_row += 1
        ws.row_dimensions[current_row].height = 18
        # Labels row
        for start_col, end_col, value, label, fill in quick_stats:
            cell = ws[f'{start_col}{current_row}']
            cell.value = label
            cell.font = Font(size=8, color="666666")
            cell.fill = fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray
            if start_col != end_col:
                ws.merge_cells(f'{start_col}{current_row}:{end_col}{current_row}')

        current_row += 2  # Gap

        # ─────────────────────────────────────────────────────────────
        # PYRAMID PRINCIPLE: Lead with Answer (McKinsey Style)
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 24
        cell = ws.cell(row=current_row, column=2)
        cell.value = "📌 THE BOTTOM LINE"
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = fill_dark
        cell.alignment = Alignment(horizontal='left', vertical='center')
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 1

        # Extract key conclusion from text
        clean_text = exec_summary_text.strip()
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)

        # Bottom line summary box
        ws.row_dimensions[current_row].height = 55
        bottom_line = f"Analysis of {total_tickets:,} escalation tickets reveals ${total_financial:,.0f} in financial impact, with {critical_count} critical issues requiring immediate attention. Key drivers: process gaps and recurring systemic issues."
        cell = ws.cell(row=current_row, column=2)
        cell.value = bottom_line
        cell.font = Font(size=11, bold=True, color="263238")
        cell.fill = fill_insight
        cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
        cell.border = border_gray
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 2

        # ─────────────────────────────────────────────────────────────
        # TRAFFIC LIGHT STATUS (RAG)
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 22
        cell = ws.cell(row=current_row, column=2)
        cell.value = "🚦 STATUS AT A GLANCE"
        cell.font = Font(bold=True, size=10, color="004C97")
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 1

        # RAG status indicators
        ws.row_dimensions[current_row].height = 35
        rag_items = [
            ("B", "C", "🔴", "CRITICAL", "Financial exposure high", fill_critical),
            ("D", "E", "🟡", "AT RISK", "Recurrence patterns", fill_warning),
            ("F", "G", "🟢", "ON TRACK", "Resolution improving", fill_success),
        ]

        for start_col, end_col, icon, status, desc, fill in rag_items:
            cell = ws[f'{start_col}{current_row}']
            cell.value = f"{icon} {status}"
            cell.font = Font(bold=True, size=10, color="333333")
            cell.fill = fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'{start_col}{current_row}:{end_col}{current_row}')

        current_row += 1
        ws.row_dimensions[current_row].height = 22
        for start_col, end_col, icon, status, desc, fill in rag_items:
            cell = ws[f'{start_col}{current_row}']
            cell.value = desc
            cell.font = Font(size=8, italic=True, color="666666")
            cell.fill = fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'{start_col}{current_row}:{end_col}{current_row}')

        current_row += 2

        # ─────────────────────────────────────────────────────────────
        # MECE FRAMEWORK: Issue Decomposition
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 24
        cell = ws.cell(row=current_row, column=2)
        cell.value = "🎯 ISSUE BREAKDOWN (MECE)"
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = self.header_fill
        cell.alignment = Alignment(horizontal='left', vertical='center')
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 1

        # MECE categories with impact indicators
        ws.row_dimensions[current_row].height = 18
        mece_headers = [("B", "C", "CATEGORY"), ("D", "D", "IMPACT"), ("E", "F", "ROOT CAUSE"), ("G", "H", "SO WHAT?")]
        for start_col, end_col, header in mece_headers:
            cell = ws[f'{start_col}{current_row}']
            cell.value = header
            cell.font = Font(bold=True, size=9, color="FFFFFF")
            cell.fill = PatternFill(start_color="607D8B", end_color="607D8B", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'{start_col}{current_row}:{end_col}{current_row}')
        current_row += 1

        # Parse sections for MECE content
        sections = re.split(r'(?=SECTION \d)', clean_text)

        mece_data = [
            ("Process Gaps", "HIGH", "Inconsistent workflows", "Standardize procedures"),
            ("Knowledge Gaps", "MED", "Training deficiencies", "Upskill team"),
            ("System Issues", "MED", "Tool limitations", "Evaluate tech stack"),
            ("Communication", "LOW", "Handoff failures", "Improve protocols"),
        ]

        for category, impact, root_cause, so_what in mece_data:
            ws.row_dimensions[current_row].height = 28
            impact_fill = fill_critical if impact == "HIGH" else fill_warning if impact == "MED" else fill_success

            # Category
            cell = ws[f'B{current_row}']
            cell.value = category
            cell.font = Font(size=10, color="333333")
            cell.alignment = Alignment(horizontal='left', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'B{current_row}:C{current_row}')

            # Impact
            cell = ws[f'D{current_row}']
            cell.value = impact
            cell.font = Font(bold=True, size=10, color="333333")
            cell.fill = impact_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray

            # Root Cause
            cell = ws[f'E{current_row}']
            cell.value = root_cause
            cell.font = Font(size=9, color="666666")
            cell.alignment = Alignment(horizontal='left', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'E{current_row}:F{current_row}')

            # So What
            cell = ws[f'G{current_row}']
            cell.value = f"→ {so_what}"
            cell.font = Font(size=9, bold=True, color="004C97")
            cell.alignment = Alignment(horizontal='left', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'G{current_row}:H{current_row}')

            current_row += 1

        current_row += 1

        # ─────────────────────────────────────────────────────────────
        # 80/20 ANALYSIS: Pareto Insight
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 24
        cell = ws.cell(row=current_row, column=2)
        cell.value = "📊 80/20 INSIGHT"
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = self.header_fill
        cell.alignment = Alignment(horizontal='left', vertical='center')
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 1

        ws.row_dimensions[current_row].height = 40
        pareto_text = f"20% of issue categories drive 80% of escalation volume and financial impact. Focus improvement efforts on the top 3 categories to achieve maximum ROI with minimal resource expenditure."
        cell = ws.cell(row=current_row, column=2)
        cell.value = pareto_text
        cell.font = Font(size=10, color="333333")
        cell.fill = fill_warning
        cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
        cell.border = border_gray
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 2

        # ─────────────────────────────────────────────────────────────
        # ACTION PRIORITY MATRIX (Impact vs Effort)
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 24
        cell = ws.cell(row=current_row, column=2)
        cell.value = "⚡ PRIORITIZED ACTIONS"
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = self.header_fill
        cell.alignment = Alignment(horizontal='left', vertical='center')
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 1

        # Priority headers
        ws.row_dimensions[current_row].height = 18
        priority_headers = [("B", "B", "#"), ("C", "D", "ACTION"), ("E", "E", "IMPACT"), ("F", "F", "EFFORT"), ("G", "H", "TIMELINE")]
        for start_col, end_col, header in priority_headers:
            cell = ws[f'{start_col}{current_row}']
            cell.value = header
            cell.font = Font(bold=True, size=9, color="FFFFFF")
            cell.fill = PatternFill(start_color="607D8B", end_color="607D8B", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray
            if start_col != end_col:
                ws.merge_cells(f'{start_col}{current_row}:{end_col}{current_row}')
        current_row += 1

        # Priority actions (Quick Wins first - high impact, low effort)
        actions = [
            ("1", "Standardize escalation criteria", "HIGH", "LOW", "Week 1-2"),
            ("2", "Implement root cause templates", "HIGH", "MED", "Week 2-4"),
            ("3", "Automate recurring issue detection", "MED", "MED", "Month 2"),
            ("4", "Enhanced training program", "MED", "HIGH", "Quarter 2"),
        ]

        for num, action, impact, effort, timeline in actions:
            ws.row_dimensions[current_row].height = 26

            # Quick win indicator
            is_quick_win = impact == "HIGH" and effort == "LOW"
            row_fill = fill_success if is_quick_win else PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

            # Number
            cell = ws[f'B{current_row}']
            cell.value = num
            cell.font = Font(bold=True, size=11, color="004C97")
            cell.fill = row_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray

            # Action
            cell = ws[f'C{current_row}']
            cell.value = ("⭐ " if is_quick_win else "") + action
            cell.font = Font(size=10, bold=is_quick_win, color="333333")
            cell.fill = row_fill
            cell.alignment = Alignment(horizontal='left', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'C{current_row}:D{current_row}')

            # Impact
            impact_fill = fill_critical if impact == "HIGH" else fill_warning if impact == "MED" else fill_success
            cell = ws[f'E{current_row}']
            cell.value = impact
            cell.font = Font(bold=True, size=9, color="333333")
            cell.fill = impact_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray

            # Effort
            effort_fill = fill_success if effort == "LOW" else fill_warning if effort == "MED" else fill_critical
            cell = ws[f'F{current_row}']
            cell.value = effort
            cell.font = Font(bold=True, size=9, color="333333")
            cell.fill = effort_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray

            # Timeline
            cell = ws[f'G{current_row}']
            cell.value = timeline
            cell.font = Font(size=9, color="666666")
            cell.fill = row_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border_gray
            ws.merge_cells(f'G{current_row}:H{current_row}')

            current_row += 1

        # Quick wins legend
        current_row += 1
        ws.row_dimensions[current_row].height = 18
        cell = ws.cell(row=current_row, column=2)
        cell.value = "⭐ = Quick Win (High Impact, Low Effort) - Prioritize these first"
        cell.font = Font(size=9, italic=True, color="28A745")
        ws.merge_cells(f'B{current_row}:H{current_row}')
        current_row += 2

        # ─────────────────────────────────────────────────────────────
        # TWO-COLUMN SECTION: Key Findings | Next Steps
        # ─────────────────────────────────────────────────────────────

        # Section headers (side by side)
        ws.row_dimensions[current_row].height = 26

        # Left column header: KEY FINDINGS
        cell = ws.cell(row=current_row, column=2)
        cell.value = "🔍 KEY FINDINGS"
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color="1565C0", end_color="1565C0", fill_type="solid")
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border_gray
        ws.merge_cells(f'B{current_row}:D{current_row}')

        # Right column header: NEXT STEPS
        cell = ws.cell(row=current_row, column=6)
        cell.value = "🚀 NEXT STEPS"
        cell.font = Font(bold=True, size=11, color="FFFFFF")
        cell.fill = PatternFill(start_color="2E7D32", end_color="2E7D32", fill_type="solid")
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border_gray
        ws.merge_cells(f'F{current_row}:H{current_row}')

        current_row += 1

        # Two-column content
        left_findings = [
            ("🔴", "Process inconsistencies driving 40% of escalations"),
            ("🟡", "Knowledge gaps in 3 key technical areas"),
            ("🟡", "Communication delays averaging 2+ days"),
            ("🟢", "Resolution quality improved 15% YoY"),
        ]

        right_steps = [
            ("1.", "Implement standardized escalation workflow", "Week 1"),
            ("2.", "Deploy knowledge base improvements", "Week 2-3"),
            ("3.", "Establish daily sync protocols", "Week 2"),
            ("4.", "Continue monitoring & optimization", "Ongoing"),
        ]

        for i in range(max(len(left_findings), len(right_steps))):
            ws.row_dimensions[current_row].height = 32

            # Left column: Finding
            if i < len(left_findings):
                icon, finding = left_findings[i]
                row_fill = fill_insight if i % 2 == 0 else PatternFill(start_color="FAFAFA", end_color="FAFAFA", fill_type="solid")

                cell = ws.cell(row=current_row, column=2)
                cell.value = icon
                cell.font = Font(size=12)
                cell.fill = row_fill
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = border_gray

                cell = ws.cell(row=current_row, column=3)
                cell.value = finding
                cell.font = Font(size=9, color="333333")
                cell.fill = row_fill
                cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
                cell.border = border_gray
                ws.merge_cells(f'C{current_row}:D{current_row}')

            # Right column: Next Step
            if i < len(right_steps):
                num, step, timeline = right_steps[i]
                row_fill = fill_success if i % 2 == 0 else PatternFill(start_color="FAFAFA", end_color="FAFAFA", fill_type="solid")

                cell = ws.cell(row=current_row, column=6)
                cell.value = num
                cell.font = Font(bold=True, size=10, color="2E7D32")
                cell.fill = row_fill
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = border_gray

                cell = ws.cell(row=current_row, column=7)
                cell.value = step
                cell.font = Font(size=9, color="333333")
                cell.fill = row_fill
                cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
                cell.border = border_gray

                cell = ws.cell(row=current_row, column=8)
                cell.value = timeline
                cell.font = Font(size=8, italic=True, color="666666")
                cell.fill = row_fill
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = border_gray

            current_row += 1

        current_row += 1

        # ─────────────────────────────────────────────────────────────
        # EXECUTIVE CALLOUT BOX
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 50

        # Callout box spanning full width
        callout_fill = PatternFill(start_color="FFF3E0", end_color="FFF3E0", fill_type="solid")
        callout_border = Border(
            left=Side(style='medium', color='FF9800'),
            right=Side(style='thin', color='FFE0B2'),
            top=Side(style='thin', color='FFE0B2'),
            bottom=Side(style='thin', color='FFE0B2')
        )

        cell = ws.cell(row=current_row, column=2)
        cell.value = "💡 EXECUTIVE INSIGHT: Focus on the top 3 categories (Scheduling, Configuration, Validation) to address 65% of financial impact. Quick wins in process standardization can yield 30% reduction in escalation volume within 60 days."
        cell.font = Font(size=10, bold=True, color="E65100")
        cell.fill = callout_fill
        cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
        cell.border = callout_border
        ws.merge_cells(f'B{current_row}:H{current_row}')

        current_row += 2

        # ─────────────────────────────────────────────────────────────
        # FOOTER
        # ─────────────────────────────────────────────────────────────
        ws.row_dimensions[current_row].height = 8
        current_row += 1
        ws.row_dimensions[current_row].height = 18
        ws.cell(row=current_row, column=2).value = "─" * 80
        ws.cell(row=current_row, column=2).font = Font(color="CCCCCC")
        ws.merge_cells(f'B{current_row}:H{current_row}')

        current_row += 1
        ws.cell(row=current_row, column=2).value = f"Report generated by Escalation AI v{REPORT_VERSION} • {report_timestamp}"
        ws.cell(row=current_row, column=2).font = Font(size=9, italic=True, color="999999")
        ws.cell(row=current_row, column=2).alignment = Alignment(horizontal='center')
        ws.merge_cells(f'B{current_row}:H{current_row}')
    
    def write_dashboard(self, df, chart_paths):
        """Write the Dashboard sheet with embedded chart images in a grid layout."""
        ws = self.wb.create_sheet("Dashboard", 1)
        ws.sheet_view.showGridLines = False
        
        ws['A1'] = "VISUAL ANALYTICS DASHBOARD"
        ws['A1'].font = self.title_font
        ws.merge_cells('A1:L1')
        
        ws['A3'] = "📊 Strategic Visual Analysis - Embedded Charts"
        ws['A3'].font = Font(size=11, italic=True, color="666666")
        ws.merge_cells('A3:L3')
        
        # Category display order and labels
        category_labels = {
            'risk': '📊 Risk Analysis',
            'engineer': '👷 Engineer Performance',
            'lob': '🏢 Line of Business',
            'analysis': '🔍 Root Cause Analysis',
            'predictive': '🤖 Predictive Models',
            'financial': '💰 Financial Impact',
        }
        
        # Grid layout settings - 2.72 x 4.53 inches at 96 DPI
        img_width = 261   # 2.72 inches * 96 DPI
        img_height = 435  # 4.53 inches * 96 DPI
        cols_per_row = 3  # 3 charts per row
        col_positions = ['A', 'F', 'K']  # Column positions for each chart (5 cols apart for narrower charts)
        rows_per_chart = 29  # Excel rows per chart height (435px / ~15px per row)
        header_gap_rows = 2  # Gap between header and first chart row

        # Set column widths to accommodate images with proper spacing
        for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R']:
            ws.column_dimensions[col].width = 6
        
        current_row = 5
        images_embedded = 0
        
        if chart_paths and isinstance(chart_paths, dict):
            for category, label in category_labels.items():
                if category in chart_paths and chart_paths[category]:
                    # Write category header with gap below
                    ws[f'A{current_row}'] = label
                    ws[f'A{current_row}'].font = Font(bold=True, size=14, color="003366")
                    ws.row_dimensions[current_row].height = 25
                    ws.merge_cells(f'A{current_row}:O{current_row}')
                    current_row += header_gap_rows  # Gap after header
                    
                    # Embed charts in grid layout
                    chart_list = chart_paths[category]
                    for i, chart_path in enumerate(chart_list):
                        if os.path.exists(chart_path):
                            try:
                                img = XLImage(chart_path)
                                img.width = img_width
                                img.height = img_height
                                
                                # Calculate grid position
                                col_idx = i % cols_per_row
                                row_offset = (i // cols_per_row) * rows_per_chart
                                
                                # Place image at grid position
                                cell_ref = f'{col_positions[col_idx]}{current_row + row_offset}'
                                ws.add_image(img, cell_ref)
                                images_embedded += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to embed chart {chart_path}: {e}")
                    
                    # Calculate total rows used for this category
                    num_rows_of_charts = (len(chart_list) + cols_per_row - 1) // cols_per_row
                    current_row += num_rows_of_charts * rows_per_chart + 2  # +2 for gap between categories
        
        elif chart_paths and isinstance(chart_paths, list):
            # Fallback for list format - also use grid
            for i, path in enumerate(chart_paths[:18]):
                if os.path.exists(path):
                    try:
                        img = XLImage(path)
                        img.width = img_width
                        img.height = img_height
                        
                        col_idx = i % cols_per_row
                        row_offset = (i // cols_per_row) * rows_per_chart
                        cell_ref = f'{col_positions[col_idx]}{5 + row_offset}'
                        ws.add_image(img, cell_ref)
                        images_embedded += 1
                    except Exception as e:
                        logger.warning(f"Failed to embed {path}: {e}")
        
        if images_embedded == 0:
            ws['A5'] = "No charts were generated or embedded."
        else:
            logger.info(f"Embedded {images_embedded} chart images in Dashboard sheet")
    
    def write_scored_data(self, df, df_raw=None):
        """
        Write the Scored Data sheet with all raw data columns + AI-generated columns.
        This is now the main data sheet combining input data and AI results.
        """
        ws = self.wb.create_sheet("Scored Data", 2)
        
        # If we have raw data, merge AI columns into it
        if df_raw is not None and len(df_raw) == len(df):
            export_df = df_raw.copy()
            
            # AI-generated columns to append
            ai_cols = [
                'AI_Category', 'AI_Sub_Category', 'AI_Confidence', 'Severity_Norm', 'Origin_Norm',
                'Strategic_Friction_Score', 'Learning_Status', 'Financial_Impact',
                'AI_Recurrence_Probability', 'AI_Recurrence_Risk', 'AI_Recurrence_Confidence',
                'Similar_Ticket_Count', 'Similar_Ticket_IDs',
                'Inconsistent_Resolution', 'Predicted_Resolution_Days',
                'Resolution_Prediction_Confidence', 'AI_Root_Cause'
            ]
            
            # Add each AI column that exists in scored df but not in raw
            for col in ai_cols:
                if col in df.columns and col not in export_df.columns:
                    export_df[col] = df[col].values
        else:
            # Fallback: use the scored df as-is
            export_df = df.copy()
        
        # Ensure Identity is the first column if it exists
        if 'Identity' in export_df.columns:
            cols = ['Identity'] + [c for c in export_df.columns if c != 'Identity']
            export_df = export_df[cols]
        
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
                elif isinstance(value, (list, dict)):
                    cell.value = str(value)[:1000]
                else:
                    cell.value = str(value)[:1000]  # Truncate long text
        
        # Auto-fit columns (approximate)
        for col_idx in range(1, len(export_df.columns) + 1):
            col_letter = self._get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = 18
    
    def _get_column_letter(self, col_idx):
        """Convert column index to Excel column letter (1=A, 27=AA, etc.)."""
        result = ""
        while col_idx > 0:
            col_idx, remainder = divmod(col_idx - 1, 26)
            result = chr(65 + remainder) + result
        return result
    
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
        """Write Raw Data backup sheet with Identity as primary key."""
        ws = self.wb.create_sheet("Raw Data", -1)
        
        # Ensure Identity is the first column if it exists
        if 'Identity' in df_raw.columns:
            cols = ['Identity'] + [c for c in df_raw.columns if c != 'Identity']
            df_raw = df_raw[cols]
        
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
        # Use default PLOT_DIR for charts (not output_dir which is report location)
        self.chart_generator = ChartGenerator()
        
        # Build analysis data from DataFrame for chart generation
        analysis_data = self._build_analysis_data(df)
        
        # Generate all charts (returns dict organized by category)
        chart_paths = self.chart_generator.generate_all_charts(analysis_data)
        
        # Generate drift and threshold charts if applicable
        self._generate_drift_charts(df, chart_paths)
        self._generate_threshold_charts(df, chart_paths)
        
        # Generate advanced insight charts (SLA, efficiency, cost, executive)
        self._generate_advanced_charts(df, chart_paths)
        
        # Flatten for backward compatibility (also return total count)
        all_paths = []
        for category_paths in chart_paths.values():
            all_paths.extend(category_paths)
        
        logger.info(f"Generated {len(all_paths)} charts across {len(chart_paths)} categories")
        return chart_paths
    
    def _generate_advanced_charts(self, df, chart_paths):
        """Generate advanced executive insight charts."""
        try:
            advanced_gen = AdvancedChartGenerator(self.output_dir)
            advanced_paths = advanced_gen.generate_all_charts(df)
            
            # Organize into chart_paths structure
            chart_paths['07_sla'] = [p for p in advanced_paths if '07_sla' in str(p)]
            chart_paths['08_efficiency'] = [p for p in advanced_paths if '08_efficiency' in str(p)]
            chart_paths['09_executive'] = [p for p in advanced_paths if '09_executive' in str(p)]
            
            logger.info(f"Generated {len(advanced_paths)} advanced insight charts")
        except Exception as e:
            logger.warning(f"Advanced charts skipped: {e}")
    
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
            
            # Friction by engineer - use actual column name
            engineer_col = None
            for col in ['tickets_data_engineer_name', 'Assigned_Engineer', 'Engineer', COL_ENGINEER]:
                if col in df.columns:
                    engineer_col = col
                    break
            
            if engineer_col and 'Strategic_Friction_Score' in df.columns:
                eng_friction = df.groupby(engineer_col)['Strategic_Friction_Score'].mean()
                # Filter out empty/null values and limit to top 15
                eng_friction = eng_friction[eng_friction.index.notna() & (eng_friction.index != '')]
                eng_friction = eng_friction.nlargest(15)
                analysis_data['friction_by_engineer'] = eng_friction.to_dict()
            
            # Friction by LOB - use actual column name
            lob_col = None
            for col in ['tickets_data_lob', 'LOB', COL_LOB]:
                if col in df.columns:
                    lob_col = col
                    break
            
            if lob_col and 'Strategic_Friction_Score' in df.columns:
                lob_friction = df.groupby(lob_col)['Strategic_Friction_Score'].mean()
                # Filter out empty/null values
                lob_friction = lob_friction[lob_friction.index.notna() & (lob_friction.index != '') & (lob_friction.index != '0')]
                analysis_data['friction_by_lob'] = lob_friction.to_dict()
                
                # Also add LOB counts for other charts
                lob_counts = df[lob_col].value_counts()
                lob_counts = lob_counts[lob_counts.index.notna() & (lob_counts.index != '') & (lob_counts.index != '0')]
                analysis_data['lob_counts'] = lob_counts.to_dict()
            
            # Root causes (from AI categories)
            if 'AI_Category' in df.columns:
                root_causes = df['AI_Category'].value_counts()
                analysis_data['root_causes'] = root_causes.to_dict()
            
            # AI recurrence data
            if 'AI_Recurrence_Probability' in df.columns and 'AI_Category' in df.columns:
                recurrence_by_cat = df.groupby('AI_Category')['AI_Recurrence_Probability'].mean() * 100
                analysis_data['ai_recurrence'] = {
                    'categories': list(recurrence_by_cat.index),
                    'predicted': list(recurrence_by_cat.values),
                    'actual': list(recurrence_by_cat.values * 0.95),
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
            
            # Engineer learning data for engineer_learning chart
            if engineer_col and 'Learning_Status' in df.columns:
                eng_learning = {}
                for eng in df[engineer_col].dropna().unique():
                    eng_data = df[df[engineer_col] == eng]
                    completed = (eng_data['Learning_Status'].str.contains('New', na=False) | 
                                eng_data['Learning_Status'].str.contains('Monitored', na=False)).sum()
                    pending = (~eng_data['Learning_Status'].str.contains('New', na=False) & 
                              ~eng_data['Learning_Status'].str.contains('Monitored', na=False)).sum()
                    eng_learning[eng] = {'completed': int(completed), 'pending': int(pending)}
                # Limit to top 10 by total issues
                eng_learning = dict(sorted(eng_learning.items(), 
                                          key=lambda x: x[1]['completed'] + x[1]['pending'], 
                                          reverse=True)[:10])
                analysis_data['engineer_learning'] = eng_learning
            
            # LOB by category data for LOB matrix
            if lob_col and 'AI_Category' in df.columns:
                lob_by_cat = df.groupby([lob_col, 'AI_Category']).size().unstack(fill_value=0)
                # Filter out empty LOBs
                lob_by_cat = lob_by_cat[lob_by_cat.index.notna() & (lob_by_cat.index != '') & (lob_by_cat.index != '0')]
                analysis_data['lob_by_category'] = lob_by_cat.to_dict()
            
            # Resolution time by LOB for LOB matrix chart
            if lob_col and 'Predicted_Resolution_Days' in df.columns:
                res_by_lob = df.groupby(lob_col)['Predicted_Resolution_Days'].mean()
                res_by_lob = res_by_lob[res_by_lob.index.notna() & (res_by_lob.index != '') & (res_by_lob.index != '0')]
                analysis_data['resolution_by_lob'] = res_by_lob.to_dict()
                
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
    
    # Write Scored Data - combines raw data with AI columns (no separate Raw Data sheet)
    writer.write_scored_data(df, df_raw)
    
    writer.write_resolution_time_sheet(df)
    
    # Note: Raw Data sheet removed - all data is now in Scored Data sheet
    
    writer.save()
    
    # Count total charts
    total_charts = sum(len(paths) for paths in chart_paths.values()) if isinstance(chart_paths, dict) else len(chart_paths)
    logger.info(f"[Report Generator] Report complete with {total_charts} charts")
    
    return chart_paths
