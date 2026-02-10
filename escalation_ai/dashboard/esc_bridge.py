"""
Escalation AI Bridge Module

Provides data loading, sidebar rendering, CSS injection, and session state
initialization for Escalation AI page shims in the unified dashboard.

This module allows the 9,260-line monolith (streamlit_app.py) to be imported
for its render functions without triggering top-level Streamlit side effects.
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime


# ============================================================================
# SESSION STATE
# ============================================================================

def init_escalation_state():
    """Initialize Escalation AI session state keys."""
    defaults = {
        'presentation_mode': False,
        'current_slide': 0,
        'action_items': [],
        'uploaded_file_path': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================================
# CSS INJECTION
# ============================================================================

def inject_escalation_css():
    """Inject Escalation AI executive styling CSS."""
    st.markdown(_ESCALATION_CSS, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING & SIDEBAR
# ============================================================================

def esc_load_and_filter(page_name: str = "Executive Dashboard") -> pd.DataFrame:
    """Load Escalation AI data and render sidebar filters.

    Mirrors the sidebar logic from streamlit_app.py main() (lines 8074-8384).

    Args:
        page_name: Current page name, controls which filters are shown.

    Returns:
        Filtered DataFrame, or None if no data available.
    """
    from escalation_ai.dashboard.streamlit_app import load_data

    # Load data (cached internally)
    df, data_source = load_data()

    if df is None:
        st.error("No escalation data found. Run the pipeline first: `python run.py`")
        return None

    with st.sidebar:
        st.markdown("---")

        # Excel-style filters (shown for Executive Dashboard)
        if page_name == "Executive Dashboard":
            st.markdown("### Add filter(s)")

            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Categories</div>', unsafe_allow_html=True)
            if 'AI_Category' in df.columns:
                all_categories = sorted(df['AI_Category'].unique().tolist())
                selected_categories = st.multiselect(
                    "Select Categories",
                    options=all_categories,
                    default=all_categories,
                    key="excel_cat_filter",
                    label_visibility="collapsed"
                )
                if selected_categories:
                    df = df[df['AI_Category'].isin(selected_categories)]
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Year</div>', unsafe_allow_html=True)
            if 'tickets_data_issue_datetime' in df.columns:
                df_temp_year = df.copy()
                df_temp_year['year'] = pd.to_datetime(
                    df_temp_year['tickets_data_issue_datetime']
                ).dt.year
                all_years = sorted(df_temp_year['year'].unique().tolist())
                selected_years = st.multiselect(
                    "Select Years",
                    options=all_years,
                    default=all_years,
                    key="excel_year_filter",
                    label_visibility="collapsed"
                )
                if selected_years:
                    df['year'] = pd.to_datetime(df['tickets_data_issue_datetime']).dt.year
                    df = df[df['year'].isin(selected_years)]
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Severity</div>', unsafe_allow_html=True)
            if 'tickets_data_severity' in df.columns:
                all_severities = sorted(df['tickets_data_severity'].unique().tolist())
                selected_severities = st.multiselect(
                    "Select Severities",
                    options=all_severities,
                    default=all_severities,
                    key="excel_sev_filter",
                    label_visibility="collapsed"
                )
                if selected_severities:
                    df = df[df['tickets_data_severity'].isin(selected_severities)]
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Clear Filter(s)", key="clear_filters", type="secondary"):
                st.rerun()

        st.markdown("---")
        st.markdown(f"**Data Source:**")
        st.caption(str(data_source))
        st.markdown(f"**Records:** {len(df):,}")

        if 'Financial_Impact' in df.columns:
            total_cost = df['Financial_Impact'].sum()
            st.markdown(f"**Total Cost:** ${total_cost:,.0f}")

        if st.button("Refresh Data", key="esc_refresh"):
            st.cache_data.clear()
            st.rerun()

        # Settings — Date filter
        st.markdown("---")
        st.markdown("### Settings")
        if 'tickets_data_issue_datetime' in df.columns:
            try:
                dates = pd.to_datetime(df['tickets_data_issue_datetime'])
                min_date = dates.min().date()
                max_date = dates.max().date()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="esc_date_range",
                )
                if len(date_range) == 2:
                    df = df[
                        (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date >= date_range[0])
                        & (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date <= date_range[1])
                    ]
            except Exception:
                pass

    return df


# ============================================================================
# CSS CONSTANT — extracted from streamlit_app.py lines 346-1007
# ============================================================================

_ESCALATION_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* Force Plotly charts to have adequate height and prevent overlap */
.stPlotlyChart {
    min-height: 400px !important;
}

.stPlotlyChart > div {
    min-height: 400px !important;
}

/* Ensure columns don't overlap */
.stColumns {
    gap: 1rem;
}

.stColumn {
    padding: 0 0.5rem;
}

/* Executive Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    margin: 10px 0;
}

/* Executive Summary Card */
.exec-card {
    background: linear-gradient(145deg, rgba(0, 40, 85, 0.9) 0%, rgba(0, 20, 40, 0.95) 100%);
    border-radius: 20px;
    padding: 32px;
    border: 1px solid rgba(0, 150, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin: 16px 0;
}

/* Strategic Recommendation Cards */
.strategy-card {
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 51, 102, 0.2) 100%);
    border-left: 5px solid #00BFFF;
    border-radius: 0 16px 16px 0;
    padding: 20px 24px;
    margin: 12px 0;
}

.strategy-card.high-priority {
    border-left-color: #FF6B6B;
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(139, 0, 0, 0.15) 100%);
}

.strategy-card.medium-priority {
    border-left-color: #FFB347;
    background: linear-gradient(135deg, rgba(255, 179, 71, 0.1) 0%, rgba(255, 140, 0, 0.15) 100%);
}

/* KPI Cards - Enhanced */
.kpi-container {
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
    border-radius: 16px;
    padding: 24px;
    border-left: 4px solid #0066CC;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.kpi-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 102, 204, 0.3);
}

.kpi-container.critical {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-left-color: #DC3545;
}

.kpi-container.warning {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 140, 0, 0.25) 100%);
    border-left-color: #FFC107;
}

.kpi-container.success {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
    border-left-color: #28A745;
}

/* Executive KPI - Larger */
.exec-kpi {
    background: linear-gradient(145deg, rgba(0, 102, 204, 0.2) 0%, rgba(0, 40, 80, 0.4) 100%);
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    border: 2px solid rgba(0, 191, 255, 0.3);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
}

.exec-kpi-value {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 50%, #004080 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.exec-kpi-value.money {
    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.exec-kpi-value.alert {
    background: linear-gradient(135deg, #FF6B6B 0%, #DC3545 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-value {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-label {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
}

.kpi-delta {
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 4px;
}

.delta-up { color: #DC3545; }
.delta-down { color: #28A745; }

/* Pulse Indicator */
.pulse-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

.pulse-dot.green { background: #28A745; box-shadow: 0 0 8px #28A745; }
.pulse-dot.yellow { background: #FFC107; box-shadow: 0 0 8px #FFC107; }
.pulse-dot.red { background: #DC3545; box-shadow: 0 0 8px #DC3545; }

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
    100% { opacity: 1; transform: scale(1); }
}

/* Main header - Executive */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.exec-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #00BFFF 50%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    text-align: center;
}

.sub-header {
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Benchmark Meter */
.benchmark-container {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.benchmark-bar {
    height: 24px;
    background: linear-gradient(90deg, #28A745 0%, #FFC107 50%, #DC3545 100%);
    border-radius: 12px;
    position: relative;
    margin: 10px 0;
}

.benchmark-marker {
    position: absolute;
    width: 4px;
    height: 32px;
    background: white;
    top: -4px;
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
}

/* Action Item Cards */
.action-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.1);
}

.action-card.completed {
    border-left: 4px solid #28A745;
    opacity: 0.7;
}

.action-card.in-progress {
    border-left: 4px solid #0066CC;
}

.action-card.blocked {
    border-left: 4px solid #DC3545;
}

/* Hide streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Chart container fix - prevent overlap */
[data-testid="stPlotlyChart"] {
    min-height: 360px;
    max-height: 450px;
    overflow: visible !important;
}

[data-testid="column"] {
    overflow: visible !important;
}

/* Ensure charts scale properly */
.js-plotly-plot {
    width: 100% !important;
}

/* Chart styling */
.chart-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #E0E0E0;
    margin-bottom: 12px;
}

/* Alert badges */
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-critical { background: #DC3545; color: white; }
.badge-warning { background: #FFC107; color: #212529; }
.badge-success { background: #28A745; color: white; }
.badge-info { background: #0066CC; color: white; }

/* Priority Tags */
.priority-p1 { background: linear-gradient(135deg, #DC3545, #8B0000); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.priority-p2 { background: linear-gradient(135deg, #FFC107, #FF8C00); color: #212529; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.priority-p3 { background: linear-gradient(135deg, #0066CC, #004080); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.05);
    padding: 4px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0066CC 0%, #004C97 100%);
}

/* Slider styling */
.stSlider > div > div {
    background: linear-gradient(90deg, #0066CC, #00BFFF);
}

/* Metric delta fix */
[data-testid="stMetricDelta"] {
    font-size: 0.9rem;
}

/* Presentation Mode */
.presentation-slide {
    min-height: 80vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 40px;
}

/* Confidence Score */
.confidence-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(0, 102, 204, 0.2);
    border: 1px solid rgba(0, 191, 255, 0.3);
}

/* Impact Cards */
.impact-positive {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
    border-color: #28A745;
}

.impact-negative {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-color: #DC3545;
}

/* Table Styling */
.exec-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
}

.exec-table th {
    background: rgba(0, 102, 204, 0.3);
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 1px;
}

.exec-table td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.exec-table tr:hover {
    background: rgba(0, 102, 204, 0.1);
}

/* EXCEL-STYLE DASHBOARD CSS */
.excel-dashboard-header {
    background: linear-gradient(135deg, #0a2540 0%, #003366 100%);
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.excel-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2px;
    margin: 0;
}

.excel-title-accent {
    color: #dc3545;
    font-weight: 800;
}

.excel-subtitle {
    color: #87ceeb;
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}

/* Filter Sidebar Styling */
.excel-filter-section {
    background: linear-gradient(180deg, #003366 0%, #002244 100%);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.excel-filter-title {
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.excel-clear-btn {
    background: #dc3545;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    margin-top: 10px;
}

/* KPI Cards - Excel Style */
.excel-kpi-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.excel-kpi-card.primary {
    background: linear-gradient(145deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
    border-left: 4px solid #0066cc;
}

.excel-kpi-card.accent {
    background: linear-gradient(145deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-left: 4px solid #dc3545;
}

.excel-kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}

.excel-kpi-value.large {
    font-size: 2.5rem;
}

.excel-kpi-value.money {
    color: #4ade80;
}

.excel-kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

.excel-kpi-sublabel {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}

/* Chart Card Container - Excel Style */
.excel-chart-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.005) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

.excel-chart-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

/* Legend Container */
.excel-legend {
    padding: 10px;
    font-size: 0.75rem;
}

.excel-legend-item {
    display: flex;
    align-items: center;
    margin: 6px 0;
    color: #94a3b8;
}

.excel-legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.excel-legend-dot.blue { background: #3b82f6; }
.excel-legend-dot.red { background: #dc3545; }
.excel-legend-dot.green { background: #22c55e; }
.excel-legend-dot.orange { background: #f97316; }

/* Progress Bar - Excel Style */
.excel-progress-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    height: 24px;
    overflow: hidden;
    margin: 8px 0;
}

.excel-progress-bar {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
}

.excel-progress-bar.male {
    background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
}

.excel-progress-bar.female {
    background: linear-gradient(90deg, #dc3545 0%, #ef4444 100%);
}

/* Comparison Grid */
.excel-comparison-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}

/* Donut Chart Value Overlay */
.excel-donut-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}

/* Horizontal Bar Chart Labels */
.excel-bar-row {
    display: flex;
    align-items: center;
    margin: 4px 0;
}

.excel-bar-value {
    font-size: 0.7rem;
    color: #94a3b8;
    min-width: 60px;
    text-align: right;
    padding-right: 8px;
}

.excel-bar-name {
    font-size: 0.75rem;
    color: #cbd5e1;
    margin-left: 8px;
}

/* Product Revenue Cards */
.excel-product-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}

.excel-product-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
}

.excel-product-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: #ffffff;
}

.excel-product-label {
    font-size: 0.7rem;
    color: #64748b;
}

/* Stores Analysis Scatter */
.excel-scatter-container {
    position: relative;
    height: 200px;
}

/* Quarter Donut Grid */
.excel-quarter-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}

.excel-quarter-card {
    text-align: center;
    padding: 8px;
}

.excel-quarter-label {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}
</style>
"""
