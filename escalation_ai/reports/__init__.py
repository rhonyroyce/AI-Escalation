"""
Reports module for Escalation AI.

Contains report generation for Excel output with charts.
"""

from .report_generator import generate_report, ExcelReportWriter

__all__ = [
    'generate_report',
    'ExcelReportWriter'
]
