"""
Main Streamlit Application Entry Point.

This module provides the main dashboard interface and utility functions
for launching the Streamlit server.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_dashboard_path() -> Path:
    """Get the path to the main streamlit app file."""
    return Path(__file__).parent / "streamlit_app.py"


def run_dashboard(data_path: str = None, port: int = 8501):
    """
    Launch the Streamlit dashboard.
    
    Args:
        data_path: Path to processed data file (optional)
        port: Port to run on (default 8501)
    """
    import subprocess
    
    app_path = get_dashboard_path()
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    if data_path:
        cmd.extend(["--", "--data", data_path])
    
    subprocess.run(cmd)


# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Escalation AI Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# Escalation AI\nStrategic Friction Analysis Dashboard"
        }
    )


def load_custom_css():
    """Load custom CSS for modern styling."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark mode compatible glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 51, 102, 0.2) 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #0066CC;
        margin: 8px 0;
    }
    
    .metric-card.critical {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(139, 0, 0, 0.2) 100%);
        border-left-color: #DC3545;
    }
    
    .metric-card.warning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 140, 0, 0.2) 100%);
        border-left-color: #FFC107;
    }
    
    .metric-card.success {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(0, 100, 0, 0.2) 100%);
        border-left-color: #28A745;
    }
    
    /* KPI value styling */
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0066CC 0%, #00BFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .kpi-delta.positive { color: #28A745; }
    .kpi-delta.negative { color: #DC3545; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001f3f 0%, #003366 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066CC 0%, #004C97 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.4);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0066CC 0%, #00BFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #6C757D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Alert badges */
    .alert-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .alert-badge.critical {
        background: #DC3545;
        color: white;
    }
    
    .alert-badge.warning {
        background: #FFC107;
        color: #212529;
    }
    
    .alert-badge.normal {
        background: #28A745;
        color: white;
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(0, 102, 204, 0.1);
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 102, 204, 0.1);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066CC 0%, #004C97 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main dashboard header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<p class="main-header">🚀 Escalation AI Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Strategic Friction Analysis & Predictive Intelligence</p>', unsafe_allow_html=True)
    
    with col2:
        # Theme toggle placeholder
        theme = st.selectbox("Theme", ["🌙 Dark", "☀️ Light"], label_visibility="collapsed")


def get_sample_data() -> pd.DataFrame:
    """Load or generate sample data for dashboard."""
    # Try to load processed data
    data_files = list(Path(".").glob("**/Escalation_Analysis_*.xlsx"))
    
    if data_files:
        latest = max(data_files, key=lambda x: x.stat().st_mtime)
        try:
            df = pd.read_excel(latest, sheet_name="Detailed Analysis")
            return df
        except:
            pass
    
    # Generate sample data if no file found
    np.random.seed(42)
    n = 200
    
    categories = ['RF & Antenna Issues', 'Transmission & Backhaul', 'Power & Environment',
                  'Site Access & Logistics', 'Contractor & Vendor Issues', 
                  'Configuration & Integration', 'OSS/NMS & Systems']
    
    df = pd.DataFrame({
        'AI_Category': np.random.choice(categories, n),
        'Strategic_Friction_Score': np.random.exponential(50, n),
        'AI_Recurrence_Risk': np.random.random(n),
        'Predicted_Resolution_Days': np.random.exponential(3, n),
        'tickets_data_severity': np.random.choice(['Critical', 'Major', 'Minor'], n, p=[0.2, 0.5, 0.3]),
        'tickets_data_escalation_origin': np.random.choice(['External', 'Internal'], n, p=[0.3, 0.7]),
        'tickets_data_issue_datetime': pd.date_range('2025-01-01', periods=n, freq='4H'),
    })
    
    return df


if __name__ == "__main__":
    # This file can be run directly with: streamlit run app.py
    configure_page()
    load_custom_css()
    render_header()
    
    st.info("Loading dashboard components...")
    
    # Load data
    df = get_sample_data()
    st.success(f"Loaded {len(df)} records")
