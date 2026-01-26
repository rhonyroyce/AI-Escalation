"""
Escalation AI Dashboard - Streamlit Web Interface.

Modern, interactive dashboard for escalation analysis with:
- Real-time KPI metrics and trends
- Interactive Plotly charts
- Category drift visualization
- Smart alert monitoring
- What-If scenario simulator
"""

from .app import run_dashboard, get_dashboard_path

__all__ = ['run_dashboard', 'get_dashboard_path']
