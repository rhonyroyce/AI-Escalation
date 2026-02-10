"""
CSE Intelligence Platform — Unified Dashboard

Single Streamlit entry point serving both Project Pulse and Escalation AI
as two navigation groups with sub-pages.

Usage:
    streamlit run unified_app.py --server.port 8501
    python run.py                                     # pipeline + dashboard
    python run.py --dashboard-only                    # dashboard only
"""

import sys
from pathlib import Path

# Ensure all imports work
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'pulse_dashboard'))

import streamlit as st

# ============================================================================
# SINGLE PAGE CONFIG (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="CSE Intelligence Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS — inject both Pulse and Escalation styles
# ============================================================================
from pulse_dashboard.utils.styles import inject_css as inject_pulse_css

inject_pulse_css()

from escalation_ai.dashboard.esc_bridge import inject_escalation_css

inject_escalation_css()

# ============================================================================
# AUTO-LOAD PULSE DATA
# ============================================================================
from pulse_dashboard.utils.data_loader import load_pulse_data, get_default_file_path

# Initialize Pulse session state defaults
PULSE_DEFAULTS = {
    'df': None,
    'filtered_df': None,
    'selected_year': None,
    'selected_week': None,
    'selected_regions': [],
    'pulse_target': 17.0,
    'pulse_stretch': 19.0,
    'green_pct_target': 80,
    'max_red_target': 3,
    'embeddings_index': None,
    'selected_project': None,
    'ollama_available': None,
    'selected_drill': None,
}
for key, default in PULSE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Auto-load ProjectPulse.xlsx
if st.session_state.df is None:
    default_path = get_default_file_path()
    if default_path:
        st.session_state.df = load_pulse_data(str(default_path))

# ============================================================================
# AUTO-RUN AI INSIGHTS — check Ollama + pre-build embeddings
# ============================================================================
if st.session_state.get('ollama_available') is None:
    try:
        from pulse_dashboard.utils.pulse_insights import check_ollama
        st.session_state.ollama_available = check_ollama()
    except Exception:
        st.session_state.ollama_available = False

if (
    st.session_state.ollama_available
    and st.session_state.df is not None
    and st.session_state.get('embeddings_index') is None
):
    try:
        from pulse_dashboard.utils.pulse_insights import build_embeddings_index
        with st.spinner("Building AI embeddings index..."):
            st.session_state.embeddings_index = build_embeddings_index(
                st.session_state.df
            )
    except Exception:
        pass

# ============================================================================
# NAVIGATION — two groups
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

esc_pages = [
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "1_Executive_Dashboard.py"),
        title="Executive Dashboard",
        icon="📊",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "2_Deep_Analysis.py"),
        title="Deep Analysis",
        icon="📈",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "3_Financial_Intelligence.py"),
        title="Financial Intelligence",
        icon="💰",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "4_Benchmarking_Monitoring.py"),
        title="Benchmarking & Monitoring",
        icon="🏆",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "5_Planning_Actions.py"),
        title="Planning & Actions",
        icon="🎯",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "6_Presentation_Mode.py"),
        title="Presentation Mode",
        icon="📽️",
    ),
]

pg = st.navigation({
    "Project Pulse": pulse_pages,
    "Escalation AI": esc_pages,
})
pg.run()
