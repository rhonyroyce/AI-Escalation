"""
Unified Dashboard Launcher (LEGACY)

WARNING: This entry point is DEPRECATED. Use one of:
    streamlit run unified_app.py   # Unified dashboard (recommended)
    python run.py --dashboard-only # Escalation AI dashboard only

This file is maintained only for backward compatibility and will be
removed in a future release.
"""

import warnings
warnings.warn(
    "dashboard.py is deprecated. Use 'streamlit run unified_app.py' or "
    "'python run.py --dashboard-only' instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
