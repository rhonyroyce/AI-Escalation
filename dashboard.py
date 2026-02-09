"""
Unified Dashboard Launcher

Serves the Pulse Dashboard as a multi-page Streamlit app.
Escalation AI can be run separately via run.py on a different port.

Usage:
    streamlit run dashboard.py --server.port 8501   # Pulse Dashboard
    python run.py --dashboard-only --port 8502      # Escalation AI (separate)
"""

import sys
from pathlib import Path

# Ensure pulse_dashboard utils are importable
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'pulse_dashboard'))

import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Pulse Dashboard | Project Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# NAVIGATION
# ============================================================================
pulse_pages = [
    st.Page(
        str(project_root / "pulse_dashboard" / "app.py"),
        title="Pulse Home",
        icon="🏠",
        default=True,
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "1_Executive_Summary.py"),
        title="Executive Summary",
        icon="📊",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "2_Drill_Down.py"),
        title="Drill Down",
        icon="🔍",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "3_Trends.py"),
        title="Trends",
        icon="📈",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "4_Prioritization.py"),
        title="Prioritization",
        icon="🎯",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "5_AI_Insights.py"),
        title="AI Insights",
        icon="🤖",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "6_Project_Details.py"),
        title="Project Details",
        icon="📋",
    ),
]

pg = st.navigation({"Pulse Dashboard": pulse_pages})
pg.run()
