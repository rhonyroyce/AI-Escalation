"""
Escalation AI - Executive Intelligence Dashboard

McKinsey-grade executive dashboard with:
- C-Suite Executive Summary with strategic recommendations
- Financial Impact Analysis with ROI calculations
- Predictive Intelligence with 30/60/90 day forecasts
- Competitive Benchmarking vs industry standards
- Root Cause Analysis with Pareto & driver trees
- Action Tracker with RACI and progress monitoring
- Executive Presentation Mode with auto-cycling slides
- Real-time KPI metrics with pulse indicators
- Interactive Plotly charts
- Category drift visualization
- Smart alert monitoring
- What-If scenario simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json
import time
import base64
import io

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import advanced charts
from escalation_ai.dashboard.advanced_plotly_charts import (
    chart_sla_funnel, chart_engineer_quadrant, chart_cost_waterfall,
    chart_time_heatmap, chart_aging_analysis, chart_health_gauge,
    chart_resolution_consistency, chart_recurrence_patterns,
    # Sub-category drill-down charts
    chart_category_sunburst as advanced_category_sunburst,
    chart_category_treemap, chart_subcategory_breakdown,
    chart_category_financial_drilldown, chart_subcategory_comparison_table
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Escalation AI | Executive Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'presentation_mode' not in st.session_state:
    st.session_state.presentation_mode = False
if 'current_slide' not in st.session_state:
    st.session_state.current_slide = 0
if 'action_items' not in st.session_state:
    st.session_state.action_items = []

# ============================================================================
# ACTION ITEMS PERSISTENCE (JSON)
# ============================================================================

ACTION_ITEMS_FILE = Path(__file__).parent / 'action_items.json'

def load_action_items():
    """Load action items from JSON file."""
    if ACTION_ITEMS_FILE.exists():
        try:
            with open(ACTION_ITEMS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def save_action_items(items):
    """Save action items to JSON file."""
    try:
        with open(ACTION_ITEMS_FILE, 'w') as f:
            json.dump(items, f, indent=2)
    except IOError as e:
        st.warning(f"Could not save action items: {e}")

# ============================================================================
# CUSTOM CSS - EXECUTIVE STYLING
# ============================================================================

st.markdown("""
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

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1929 0%, #001e3c 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load the most recent analysis data."""
    # Get project root directory (2 levels up from dashboard)
    project_root = Path(__file__).parent.parent.parent

    # Try multiple locations for Strategic_Report.xlsx
    search_paths = [
        Path("Strategic_Report.xlsx"),  # Current directory
        project_root / "Strategic_Report.xlsx",  # Project root
        Path.cwd() / "Strategic_Report.xlsx",  # Working directory
    ]

    for strategic_report in search_paths:
        if strategic_report.exists():
            try:
                df = pd.read_excel(strategic_report, sheet_name="Scored Data")

                # Map column names to expected format
                if 'tickets_data_engineer_name' in df.columns and 'Engineer' not in df.columns:
                    df['Engineer'] = df['tickets_data_engineer_name']
                if 'tickets_data_lob' in df.columns and 'LOB' not in df.columns:
                    df['LOB'] = df['tickets_data_lob']

                # Use AI_Recurrence_Probability as the numeric recurrence risk
                # (AI_Recurrence_Risk in file is string like "Elevated (50-70%)")
                if 'AI_Recurrence_Probability' in df.columns:
                    df['AI_Recurrence_Risk'] = pd.to_numeric(df['AI_Recurrence_Probability'], errors='coerce').fillna(0.15)

                # Ensure numeric columns are numeric
                numeric_cols = ['Strategic_Friction_Score', 'Financial_Impact', 'Predicted_Resolution_Days',
                               'AI_Confidence', 'Resolution_Prediction_Confidence']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                print(f"✅ Loaded {len(df)} records from {strategic_report}")
                return df, str(strategic_report)
            except Exception as e:
                st.warning(f"Could not load {strategic_report}: {e}")
                continue

    # Look for other processed Excel files in project root
    data_files = list(project_root.glob("Escalation_Analysis_*.xlsx"))

    if data_files:
        latest = max(data_files, key=lambda x: x.stat().st_mtime)
        try:
            df = pd.read_excel(latest, sheet_name="Detailed Analysis")
            print(f"✅ Loaded {len(df)} records from {latest}")
            return df, str(latest)
        except Exception as e:
            st.warning(f"Could not load {latest}: {e}")

    # Generate sample data as fallback
    st.warning("⚠️ No data file found. Showing sample data. Please ensure Strategic_Report.xlsx is in the project root.")
    return generate_sample_data(), "Sample Data"


def generate_sample_data():
    """Generate realistic sample data for demo."""
    np.random.seed(42)
    n = 250

    # 8-category system for telecom escalation analysis
    categories = [
        'Scheduling & Planning', 'Documentation & Reporting', 'Validation & QA',
        'Process Compliance', 'Configuration & Data Mismatch',
        'Site Readiness', 'Communication & Response', 'Nesting & Tool Errors'
    ]

    # Sub-categories for each main category
    sub_categories = {
        'Scheduling & Planning': ['TI/Calendar Issues', 'FE Coordination', 'Closeout/Bucket Issues'],
        'Documentation & Reporting': ['Snapshot/Screenshot Issues', 'E911/CBN Reports', 'Email/Attachment Issues'],
        'Validation & QA': ['Precheck/Postcheck Failures', 'Measurement Issues', 'Escalation Gaps'],
        'Process Compliance': ['SOP Violations', 'Improper Escalations', 'Release Procedure Issues'],
        'Configuration & Data Mismatch': ['Port Matrix Issues', 'RET/TAC Naming', 'SCF/CIQ/RFDS Mismatch'],
        'Site Readiness': ['Backhaul Issues', 'MW/Transmission Issues', 'Material/Equipment Issues'],
        'Communication & Response': ['Delayed Responses', 'Follow-up Issues', 'Distro/Routing Issues'],
        'Nesting & Tool Errors': ['Nesting Type Errors', 'RIOT/FCI Tool Issues', 'Market Guideline Violations']
    }

    engineers = ['Alice Chen', 'Bob Smith', 'Carlos Rodriguez', 'Diana Patel',
                 'Eric Johnson', 'Fatima Ahmed', 'George Kim', 'Hannah Lee']

    lobs = ['Network Operations', 'Field Services', 'Customer Support',
            'Infrastructure', 'Enterprise', 'Residential']

    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast', 'Central']

    root_causes = ['Training Gap', 'Process Failure', 'Tool Limitation', 'Resource Constraint',
                   'Vendor Issue', 'Communication Breakdown', 'Technical Debt', 'Policy Conflict']

    # Generate dates over 90 days
    dates = pd.date_range(end=datetime.now(), periods=n, freq='8H')

    # Generate categories
    cat_values = np.random.choice(categories, n, p=[0.18, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10, 0.08])

    # Generate sub-categories based on main category
    sub_cat_values = [np.random.choice(sub_categories[cat]) for cat in cat_values]

    df = pd.DataFrame({
        'AI_Category': cat_values,
        'AI_Sub_Category': sub_cat_values,
        'AI_Confidence': np.clip(np.random.beta(8, 2, n), 0.4, 0.99),
        'Strategic_Friction_Score': np.clip(np.random.exponential(45, n), 5, 200),
        'AI_Recurrence_Risk': np.clip(np.random.beta(2, 5, n), 0, 1),
        'AI_Recurrence_Probability': np.clip(np.random.beta(2, 5, n), 0, 1),
        'Predicted_Resolution_Days': np.clip(np.random.exponential(2.5, n), 0.5, 15),
        'tickets_data_severity': np.random.choice(['Critical', 'Major', 'Minor'], n, p=[0.15, 0.45, 0.40]),
        'tickets_data_escalation_origin': np.random.choice(['External', 'Internal'], n, p=[0.35, 0.65]),
        'tickets_data_issue_datetime': dates,
        'Engineer': np.random.choice(engineers, n),
        'LOB': np.random.choice(lobs, n),
        'Region': np.random.choice(regions, n),
        'Root_Cause': np.random.choice(root_causes, n),
        'Financial_Impact': np.clip(np.random.exponential(2500, n), 100, 25000),
        'Customer_Impact_Score': np.clip(np.random.exponential(50, n), 5, 100),
        'SLA_Breached': np.random.choice([True, False], n, p=[0.12, 0.88]),
        'Repeat_Customer': np.random.choice([True, False], n, p=[0.25, 0.75]),
        'Contract_Value': np.clip(np.random.exponential(50000, n), 5000, 500000),
        'Customer_Tenure_Years': np.clip(np.random.exponential(3, n), 0.5, 15),
        'NPS_Impact': np.random.choice([-3, -2, -1, 0], n, p=[0.1, 0.2, 0.4, 0.3]),
    })

    # Derived metrics
    df['Revenue_At_Risk'] = df['Contract_Value'] * df['AI_Recurrence_Risk'] * 0.15
    df['Churn_Probability'] = np.clip(df['Customer_Impact_Score'] / 100 * df['AI_Recurrence_Risk'], 0, 0.5)

    return df


# ============================================================================
# INDUSTRY BENCHMARKS
# ============================================================================

INDUSTRY_BENCHMARKS = {
    'resolution_days': {'best_in_class': 1.2, 'industry_avg': 2.8, 'laggard': 5.5},
    'recurrence_rate': {'best_in_class': 8, 'industry_avg': 18, 'laggard': 32},
    'sla_breach_rate': {'best_in_class': 3, 'industry_avg': 12, 'laggard': 25},
    'first_contact_resolution': {'best_in_class': 72, 'industry_avg': 55, 'laggard': 38},
    'cost_per_escalation': {'best_in_class': 450, 'industry_avg': 850, 'laggard': 1500},
    'customer_satisfaction': {'best_in_class': 92, 'industry_avg': 78, 'laggard': 62},
}

# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_plotly_theme():
    """Get consistent Plotly theme settings."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#E0E0E0'),
        margin=dict(l=40, r=40, t=50, b=40),
    )


def chart_friction_by_category(df):
    """Interactive bar chart of friction by category."""
    friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=True)
    
    # Create gradient colors
    colors = px.colors.sequential.Blues_r[:len(friction)]
    
    fig = go.Figure(go.Bar(
        x=friction.values,
        y=friction.index,
        orientation='h',
        marker=dict(
            color=friction.values,
            colorscale='Blues',
            line=dict(width=0)
        ),
        hovertemplate='<b>%{y}</b><br>Friction: %{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Strategic Friction by Category', font=dict(size=16)),
        xaxis_title='Total Friction Score',
        yaxis_title='',
        height=450,
        showlegend=False
    )
    
    return fig


def chart_severity_distribution(df):
    """Donut chart of severity distribution."""
    severity_counts = df['tickets_data_severity'].value_counts()
    
    colors = {'Critical': '#DC3545', 'Major': '#FFC107', 'Minor': '#28A745'}
    
    fig = go.Figure(go.Pie(
        labels=severity_counts.index,
        values=severity_counts.values,
        hole=0.6,
        marker=dict(colors=[colors.get(s, '#6C757D') for s in severity_counts.index]),
        textinfo='label+percent',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Severity Distribution', font=dict(size=16)),
        height=450,
        showlegend=True,
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center')
    )
    
    # Add center text
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=20, color='#E0E0E0'),
        showarrow=False
    )
    
    return fig


def chart_trend_timeline(df):
    """Animated area chart of escalations over time."""
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime']).dt.date
    daily = df_temp.groupby('date').agg({
        'Strategic_Friction_Score': 'sum',
        'AI_Category': 'count'
    }).rename(columns={'AI_Category': 'count'}).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    
    # 7-day rolling average
    daily['friction_ma'] = daily['Strategic_Friction_Score'].rolling(7, min_periods=1).mean()
    daily['count_ma'] = daily['count'].rolling(7, min_periods=1).mean()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Friction area
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['friction_ma'],
        name='Friction Score',
        fill='tozeroy',
        fillcolor='rgba(0, 102, 204, 0.3)',
        line=dict(color='#0066CC', width=2),
        hovertemplate='Date: %{x}<br>Friction: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)
    
    # Count line
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['count_ma'],
        name='Escalation Count',
        line=dict(color='#FF6B6B', width=2, dash='dot'),
        hovertemplate='Date: %{x}<br>Count: %{y:.1f}<extra></extra>'
    ), secondary_y=True)
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Escalation Trend (7-Day Moving Average)', font=dict(size=16)),
        height=450,
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='Friction Score', secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text='Count', secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def chart_recurrence_risk(df):
    """Gauge chart for average recurrence risk."""
    # AI_Recurrence_Risk is now guaranteed to be numeric from load_data()
    avg_risk = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 15
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_risk,
        number={'suffix': '%', 'font': {'size': 40}},
        delta={'reference': 15, 'relative': False, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#0066CC'},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'steps': [
                {'range': [0, 20], 'color': 'rgba(40, 167, 69, 0.3)'},
                {'range': [20, 40], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [40, 100], 'color': 'rgba(220, 53, 69, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#FF6B6B', 'width': 3},
                'thickness': 0.8,
                'value': 30
            }
        }
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Avg Recurrence Risk', font=dict(size=16)),
        height=450
    )
    
    return fig


def chart_resolution_distribution(df):
    """Histogram of predicted resolution times."""
    fig = go.Figure(go.Histogram(
        x=df['Predicted_Resolution_Days'],
        nbinsx=20,
        marker=dict(
            color='rgba(0, 191, 255, 0.7)',
            line=dict(color='#00BFFF', width=1)
        ),
        hovertemplate='Days: %{x:.1f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_days = df['Predicted_Resolution_Days'].mean()
    fig.add_vline(x=mean_days, line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"Avg: {mean_days:.1f} days")
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Resolution Time Distribution', font=dict(size=16)),
        xaxis_title='Predicted Days',
        yaxis_title='Count',
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def chart_category_sunburst(df):
    """Interactive sunburst chart of categories and sub-categories with drill-down."""
    # Check if AI_Sub_Category column exists
    sub_cat_col = 'AI_Sub_Category' if 'AI_Sub_Category' in df.columns else None
    cost_col = 'Financial_Impact' if 'Financial_Impact' in df.columns else None

    if sub_cat_col:
        # Category → Sub-Category drill-down
        sunburst_data = df.groupby(['AI_Category', sub_cat_col]).size().reset_index(name='count')
        path_cols = ['AI_Category', sub_cat_col]
        title_text = 'Category & Sub-Category Drill-Down<br><span style="font-size:12px">Click to expand categories</span>'
    else:
        # Fallback to Category → Severity
        sunburst_data = df.groupby(['AI_Category', 'tickets_data_severity']).size().reset_index(name='count')
        path_cols = ['AI_Category', 'tickets_data_severity']
        title_text = 'Category & Severity Breakdown'

    # Add financial data if available
    if cost_col and sub_cat_col:
        cost_data = df.groupby(['AI_Category', sub_cat_col])[cost_col].sum().reset_index()
        sunburst_data = sunburst_data.merge(cost_data, on=['AI_Category', sub_cat_col], how='left')
        sunburst_data[cost_col] = sunburst_data[cost_col].fillna(0)

    fig = px.sunburst(
        sunburst_data,
        path=path_cols,
        values='count',
        color='count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text=title_text, font=dict(size=16)),
        height=500
    )

    # Enhanced hover template
    if cost_col and sub_cat_col:
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Tickets: %{value}<extra></extra>'
        )
    else:
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        )

    return fig


def chart_engineer_performance(df):
    """Horizontal bar chart of engineer friction."""
    if 'Engineer' not in df.columns:
        return None
    
    # Build agg dict with available columns
    agg_dict = {'AI_Category': 'count'}
    if 'Strategic_Friction_Score' in df.columns:
        agg_dict['Strategic_Friction_Score'] = 'mean'
    if 'AI_Recurrence_Risk' in df.columns:
        agg_dict['AI_Recurrence_Risk'] = 'mean'
    
    eng_stats = df.groupby('Engineer').agg(agg_dict).rename(columns={'AI_Category': 'ticket_count'})
    if 'Strategic_Friction_Score' in eng_stats.columns:
        eng_stats = eng_stats.sort_values('Strategic_Friction_Score')
        x_vals = eng_stats['Strategic_Friction_Score']
        x_title = 'Average Friction Score'
        text_vals = [f"{v:.0f} ({c} tickets)" for v, c in zip(eng_stats['Strategic_Friction_Score'], eng_stats['ticket_count'])]
    else:
        eng_stats = eng_stats.sort_values('ticket_count')
        x_vals = eng_stats['ticket_count']
        x_title = 'Ticket Count'
        text_vals = [f"{c} tickets" for c in eng_stats['ticket_count']]
    
    fig = go.Figure(go.Bar(
        x=x_vals,
        y=eng_stats.index,
        orientation='h',
        marker=dict(
            color=x_vals,
            colorscale='RdYlGn_r',
            line=dict(width=0)
        ),
        text=text_vals,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Value: %{x:.1f}<extra></extra>'
    ))
    
    # Get theme without margin, then add custom margin
    theme = create_plotly_theme()
    theme.pop('margin', None)
    
    fig.update_layout(
        **theme,
        title=dict(text='Engineer Performance', font=dict(size=16)),
        xaxis_title=x_title,
        height=400,
        margin=dict(l=150, r=100, t=60, b=40)  # Room for names and values
    )
    
    return fig


# ============================================================================
# EXECUTIVE CHARTS
# ============================================================================

def chart_pareto_analysis(df):
    """Pareto chart showing 80/20 rule for escalation causes."""
    category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
    cumulative_pct = category_friction.cumsum() / category_friction.sum() * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    colors = ['#FF6B6B' if pct <= 80 else '#6C757D' for pct in cumulative_pct]
    
    fig.add_trace(go.Bar(
        x=category_friction.index,
        y=category_friction.values,
        name='Friction Score',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Friction: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=category_friction.index,
        y=cumulative_pct.values,
        name='Cumulative %',
        mode='lines+markers',
        line=dict(color='#00BFFF', width=3),
        marker=dict(size=8),
        hovertemplate='%{x}<br>Cumulative: %{y:.1f}%<extra></extra>'
    ), secondary_y=True)
    
    fig.add_hline(y=80, line_dash="dash", line_color="#FFC107", 
                  annotation_text="80% Threshold", secondary_y=True)
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='🎯 Pareto Analysis: Focus on the Vital Few', font=dict(size=18)),
        height=400,
        xaxis_tickangle=-45,
        legend=dict(orientation='h', y=1.15)
    )
    
    fig.update_yaxes(title_text='Friction Score', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative %', secondary_y=True, range=[0, 105])
    
    return fig


def chart_driver_tree(df):
    """Create a driver tree showing impact decomposition."""
    total_friction = df['Strategic_Friction_Score'].sum()
    
    # Level 1: By Origin
    origin_data = df.groupby('tickets_data_escalation_origin')['Strategic_Friction_Score'].sum()
    
    # Level 2: By Severity within Origin
    severity_origin = df.groupby(['tickets_data_escalation_origin', 'tickets_data_severity'])['Strategic_Friction_Score'].sum()
    
    labels = ['Total Friction']
    parents = ['']
    values = [total_friction]
    
    for origin in origin_data.index:
        labels.append(origin)
        parents.append('Total Friction')
        values.append(origin_data[origin])
        
        for severity in ['Critical', 'Major', 'Minor']:
            if (origin, severity) in severity_origin.index:
                labels.append(f"{origin} - {severity}")
                parents.append(origin)
                values.append(severity_origin[(origin, severity)])
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        marker=dict(
            colors=values,
            colorscale='RdYlBu_r',
            showscale=True
        ),
        textinfo='label+value+percent parent',
        hovertemplate='<b>%{label}</b><br>Friction: %{value:,.0f}<br>%{percentParent:.1%} of parent<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='🌳 Friction Driver Tree', font=dict(size=18)),
        height=500
    )
    
    return fig


def chart_forecast_projection(df):
    """Create 30/60/90 day forecast visualization."""
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime']).dt.date
    daily = df_temp.groupby('date').agg({
        'Strategic_Friction_Score': 'sum',
        'AI_Category': 'count'
    }).rename(columns={'AI_Category': 'count'}).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Calculate trend
    daily['day_num'] = (daily['date'] - daily['date'].min()).dt.days
    z = np.polyfit(daily['day_num'], daily['count'], 1)
    slope = z[0]
    
    # Forecast
    last_date = daily['date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
    forecast_day_nums = np.arange(daily['day_num'].max() + 1, daily['day_num'].max() + 91)
    forecast_values = np.polyval(z, forecast_day_nums)
    
    # Add uncertainty cone
    std = daily['count'].std()
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['count'],
        mode='lines',
        name='Historical',
        line=dict(color='#0066CC', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 102, 204, 0.2)'
    ))
    
    # Forecast cone (upper)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values + 2*std,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255, 107, 107, 0.3)', width=0),
        showlegend=False
    ))
    
    # Forecast cone (lower)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=np.maximum(0, forecast_values - 2*std),
        mode='lines',
        name='Forecast Range',
        line=dict(color='rgba(255, 107, 107, 0.3)', width=0),
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    # Add 30/60/90 markers
    for days, label in [(30, '30D'), (60, '60D'), (90, '90D')]:
        if days <= len(forecast_dates):
            fig.add_vline(x=forecast_dates[days-1], line_dash="dot", line_color="#FFC107")
            fig.add_annotation(x=forecast_dates[days-1], y=forecast_values[days-1]*1.1,
                             text=label, showarrow=False, font=dict(color='#FFC107'))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='📈 90-Day Escalation Forecast', font=dict(size=18)),
        height=400,
        xaxis_title='Date',
        yaxis_title='Daily Escalations',
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig, slope


def chart_risk_heatmap(df):
    """Create risk heatmap by category and severity."""
    pivot = df.pivot_table(
        values='Strategic_Friction_Score',
        index='AI_Category',
        columns='tickets_data_severity',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder columns
    cols = ['Critical', 'Major', 'Minor']
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        text=pivot.values.astype(int),
        texttemplate='%{text}',
        textfont=dict(size=12),
        hovertemplate='<b>%{y}</b><br>Severity: %{x}<br>Friction: %{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='🔥 Risk Heatmap: Category × Severity', font=dict(size=18)),
        height=450,
        xaxis_title='Severity',
        yaxis_title=''
    )
    
    return fig


def chart_benchmark_gauge(metric_name, current_value, benchmark_data, unit=''):
    """Create a benchmark gauge showing position vs industry."""
    best = benchmark_data['best_in_class']
    avg = benchmark_data['industry_avg']
    laggard = benchmark_data['laggard']
    
    # Determine if lower is better
    lower_better = best < laggard
    
    if lower_better:
        min_val, max_val = best * 0.5, laggard * 1.2
    else:
        min_val, max_val = laggard * 0.8, best * 1.1
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        number={'suffix': unit, 'font': {'size': 32}},
        delta={'reference': avg, 'relative': False, 'suffix': unit,
               'increasing': {'color': '#DC3545' if lower_better else '#28A745'},
               'decreasing': {'color': '#28A745' if lower_better else '#DC3545'}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': '#0066CC'},
            'steps': [
                {'range': [min_val, best if lower_better else laggard], 'color': 'rgba(40, 167, 69, 0.3)' if lower_better else 'rgba(220, 53, 69, 0.3)'},
                {'range': [best if lower_better else laggard, avg], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [avg, max_val], 'color': 'rgba(220, 53, 69, 0.3)' if lower_better else 'rgba(40, 167, 69, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#00BFFF', 'width': 4},
                'thickness': 0.75,
                'value': current_value
            }
        }
    ))
    
    # Add benchmark annotations
    fig.add_annotation(x=0.15, y=-0.15, text=f"Best: {best}{unit}", showarrow=False, font=dict(size=10, color='#28A745'))
    fig.add_annotation(x=0.5, y=-0.15, text=f"Avg: {avg}{unit}", showarrow=False, font=dict(size=10, color='#FFC107'))
    fig.add_annotation(x=0.85, y=-0.15, text=f"Laggard: {laggard}{unit}", showarrow=False, font=dict(size=10, color='#DC3545'))
    
    # Get theme without margin, then add specific margin
    theme = create_plotly_theme()
    theme.pop('margin', None)  # Remove margin from theme
    
    fig.update_layout(
        **theme,
        title=dict(text=metric_name, font=dict(size=14)),
        height=250,
        margin=dict(t=50, b=50, l=30, r=30)
    )
    
    return fig

# ============================================================================
# WHAT-IF SIMULATOR
# ============================================================================

def generate_strategic_recommendations(df):
    """Generate AI-powered strategic recommendations with confidence scores."""
    recommendations = []
    
    # Analyze data patterns
    category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
    top_category = category_friction.index[0]
    top_category_pct = category_friction.iloc[0] / category_friction.sum() * 100
    
    # Get safe values
    avg_recurrence = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 15
    critical_pct = (df['tickets_data_severity'] == 'Critical').mean() * 100 if 'tickets_data_severity' in df.columns else 10
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    
    sla_breach_rate = df['SLA_Breached'].mean() * 100 if 'SLA_Breached' in df.columns else 12
    
    # Recommendation 1: Category Focus
    if top_category_pct > 15:
        recommendations.append({
            'priority': 'P1',
            'title': f'Establish {top_category} Tiger Team',
            'description': f'{top_category} accounts for {top_category_pct:.0f}% of total friction. Create dedicated cross-functional team to address root causes.',
            'impact': f'Reduce total friction by up to {top_category_pct * 0.4:.0f}%',
            'confidence': 92,
            'timeline': '30 days',
            'investment': '$25,000 - $50,000',
            'roi': '340%'
        })
    
    # Recommendation 2: Recurrence Prevention
    if avg_recurrence > 20:
        recommendations.append({
            'priority': 'P1',
            'title': 'Implement Predictive Maintenance Program',
            'description': f'Recurrence risk at {avg_recurrence:.0f}% indicates systemic issues. Deploy ML-based early warning system.',
            'impact': f'Reduce recurring escalations by 35-50%',
            'confidence': 87,
            'timeline': '60 days',
            'investment': '$75,000 - $120,000',
            'roi': '280%'
        })
    
    # Recommendation 3: SLA Improvement
    if sla_breach_rate > 10:
        recommendations.append({
            'priority': 'P2',
            'title': 'SLA Recovery Initiative',
            'description': f'Current SLA breach rate of {sla_breach_rate:.1f}% exceeds industry benchmark. Implement escalation fast-track protocol.',
            'impact': 'Reduce SLA breaches to <5%',
            'confidence': 85,
            'timeline': '45 days',
            'investment': '$30,000 - $60,000',
            'roi': '420%'
        })
    
    # Recommendation 4: Resolution Optimization
    if avg_resolution > 2.5:
        recommendations.append({
            'priority': 'P2',
            'title': 'Resolution Time Optimization',
            'description': f'Average {avg_resolution:.1f} day resolution time above benchmark. Implement automated triage and parallel processing.',
            'impact': f'Reduce resolution time by {min(40, (avg_resolution - 1.5) / avg_resolution * 100):.0f}%',
            'confidence': 88,
            'timeline': '90 days',
            'investment': '$50,000 - $100,000',
            'roi': '250%'
        })
    
    # Recommendation 5: Training Investment
    if critical_pct > 12:
        recommendations.append({
            'priority': 'P2',
            'title': 'Targeted Skill Development Program',
            'description': f'Critical severity rate of {critical_pct:.0f}% suggests training gaps. Deploy category-specific certification program.',
            'impact': 'Reduce critical escalations by 25-40%',
            'confidence': 79,
            'timeline': '120 days',
            'investment': '$40,000 - $80,000',
            'roi': '200%'
        })
    
    # Recommendation 6: Process Automation
    recommendations.append({
        'priority': 'P3',
        'title': 'Intelligent Process Automation',
        'description': 'Deploy RPA for repetitive escalation handling tasks. Integrate with existing ticketing systems.',
        'impact': 'Reduce manual effort by 30-45%',
        'confidence': 83,
        'timeline': '180 days',
        'investment': '$100,000 - $200,000',
        'roi': '180%'
    })
    
    return recommendations


def render_executive_summary(df):
    """Render the C-Suite Executive Summary page."""
    st.markdown('<p class="exec-title">🎯 Executive Intelligence Brief</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="text-align: center;">Strategic insights for leadership decision-making</p>', unsafe_allow_html=True)
    
    # Top-line executive KPIs
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_impact = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * 850
    # Revenue at risk should be a percentage of financial impact, not a multiplier
    # Using 20% as reasonable estimate for churn risk impact
    revenue_at_risk = df['Revenue_At_Risk'].sum() if 'Revenue_At_Risk' in df.columns else total_impact * 0.20
    cost_per_esc = total_impact / len(df)
    
    with col1:
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value money">${total_impact/1000:,.0f}K</p>
            <p class="kpi-label">Total Operational Cost</p>
            <p class="kpi-delta">90-day period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value alert">${revenue_at_risk/1000:,.0f}K</p>
            <p class="kpi-label">Revenue at Risk</p>
            <p class="kpi-delta delta-up">Due to churn risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        potential_savings = total_impact * 0.35
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value money">${potential_savings/1000:,.0f}K</p>
            <p class="kpi-label">Savings Opportunity</p>
            <p class="kpi-delta delta-down">35% reduction achievable</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Get safe values
        recurrence_rate = df['AI_Recurrence_Risk'].mean() * 100 if 'AI_Recurrence_Risk' in df.columns else 15
        friction_mean = df['Strategic_Friction_Score'].mean() if 'Strategic_Friction_Score' in df.columns else 50
        health_score = max(20, 100 - recurrence_rate - (friction_mean / 2))
        pulse_color = 'green' if health_score > 70 else 'yellow' if health_score > 50 else 'red'
        st.markdown(f"""
        <div class="exec-kpi">
            <p class="exec-kpi-value">{health_score:.0f}</p>
            <p class="kpi-label">Operational Health Score</p>
            <p><span class="pulse-dot {pulse_color}"></span>{'Healthy' if health_score > 70 else 'At Risk' if health_score > 50 else 'Critical'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    st.markdown("*AI-generated insights with confidence scoring*")
    
    recommendations = generate_strategic_recommendations(df)
    
    for i, rec in enumerate(recommendations[:4]):  # Top 4 recommendations
        priority_class = f"priority-{rec['priority'].lower()}"
        card_class = 'high-priority' if rec['priority'] == 'P1' else 'medium-priority' if rec['priority'] == 'P2' else ''
        
        st.markdown(f"""
        <div class="strategy-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span class="{priority_class}">{rec['priority']}</span>
                    <strong style="font-size: 1.1rem; margin-left: 12px;">{rec['title']}</strong>
                </div>
                <div class="confidence-badge">
                    <span style="margin-right: 6px;">🎯</span> {rec['confidence']}% confidence
                </div>
            </div>
            <p style="color: #B0B0B0; margin: 8px 0;">{rec['description']}</p>
            <div style="display: flex; gap: 24px; margin-top: 12px;">
                <div><strong style="color: #28A745;">Impact:</strong> <span style="color: #E0E0E0;">{rec['impact']}</span></div>
                <div><strong style="color: #0066CC;">Timeline:</strong> <span style="color: #E0E0E0;">{rec['timeline']}</span></div>
                <div><strong style="color: #FFC107;">Investment:</strong> <span style="color: #E0E0E0;">{rec['investment']}</span></div>
                <div><strong style="color: #00BFFF;">ROI:</strong> <span style="color: #E0E0E0;">{rec['roi']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(chart_pareto_analysis(df), use_container_width=True)
    
    with col2:
        forecast_fig, slope = chart_forecast_projection(df)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        trend_direction = "increasing" if slope > 0 else "decreasing"
        trend_color = "#DC3545" if slope > 0 else "#28A745"
        st.markdown(f"""
        <div style="text-align: center; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            <span style="color: {trend_color}; font-weight: 600;">
                {'📈' if slope > 0 else '📉'} Trend: {abs(slope):.2f} escalations/day {trend_direction}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_financial_analysis(df):
    """Render the Enhanced Financial Impact Analysis page with advanced metrics."""
    from escalation_ai.financial import (
        calculate_financial_metrics,
        calculate_roi_metrics,
        calculate_cost_avoidance,
        calculate_efficiency_metrics,
        calculate_financial_forecasts,
        generate_financial_insights,
        create_financial_waterfall,
        create_roi_opportunity_chart,
        create_cost_avoidance_breakdown,
        create_cost_trend_forecast,
        create_efficiency_scorecard,
        create_category_cost_comparison,
        create_engineer_cost_efficiency_matrix,
        create_financial_kpi_cards,
        create_insights_table
    )

    st.markdown('<p class="main-header">💰 Financial Impact Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive financial metrics, ROI analysis, and cost optimization</p>', unsafe_allow_html=True)

    # Ensure Financial_Impact column exists
    if 'Financial_Impact' not in df.columns:
        df = df.copy()
        df['Financial_Impact'] = df['Strategic_Friction_Score'] * 50 if 'Strategic_Friction_Score' in df.columns else 1500

    # Calculate comprehensive metrics
    with st.spinner('Calculating advanced financial metrics...'):
        financial_metrics = calculate_financial_metrics(df)
        roi_metrics = calculate_roi_metrics(df)
        cost_avoidance = calculate_cost_avoidance(df)
        efficiency_metrics = calculate_efficiency_metrics(df)
        forecasts = calculate_financial_forecasts(df)
        insights = generate_financial_insights(df)

    # KPI Cards
    st.markdown("### 📊 Key Financial Indicators")
    kpi_data = create_financial_kpi_cards(financial_metrics)

    # Display in 3x2 grid
    for row_idx in range(2):
        cols = st.columns(3)
        for col_idx, col in enumerate(cols):
            kpi_idx = row_idx * 3 + col_idx
            if kpi_idx < len(kpi_data):
                kpi = kpi_data[kpi_idx]
                with col:
                    # Proper color logic: green = good, red = bad
                    delta_color = "off"  # Default: no color

                    # Get numeric delta for proper color calculation
                    delta_value = kpi.get('delta')
                    delta_display = kpi.get('delta_text')

                    # For costs: lower is better (inverse)
                    if 'Cost' in kpi['title'] or 'Revenue at Risk' in kpi['title']:
                        delta_color = "inverse"

                    # For positive metrics: higher is better (normal)
                    elif 'ROI' in kpi['title'] or 'Efficiency' in kpi['title'] or 'Avoidance' in kpi['title']:
                        delta_color = "normal"

                    # Use numeric delta if available for proper coloring, otherwise use text
                    display_delta = delta_value if delta_value is not None else delta_display

                    st.metric(
                        label=f"{kpi['icon']} {kpi['title']}",
                        value=kpi['value'],
                        delta=display_delta,
                        delta_color=delta_color
                    )

    st.markdown("---")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "💹 ROI Opportunities",
        "💡 Cost Avoidance",
        "📈 Trends & Forecast",
        "🎯 Insights & Actions"
    ])

    with tab1:
        st.markdown("### Financial Impact Overview")

        col1, col2 = st.columns(2)

        with col1:
            # Financial waterfall
            try:
                waterfall_data = {
                    'total_cost': financial_metrics.total_cost,
                    'recurring_issue_cost': financial_metrics.recurring_issue_cost,
                    'preventable_cost': financial_metrics.preventable_cost,
                    'customer_impact_cost': financial_metrics.customer_impact_cost,
                    'sla_penalty_exposure': financial_metrics.sla_penalty_exposure
                }
                fig = create_financial_waterfall(waterfall_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating waterfall chart: {e}")

        with col2:
            # Efficiency scorecard
            try:
                fig = create_efficiency_scorecard(efficiency_metrics, financial_metrics)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating efficiency scorecard: {e}")

        # Category cost comparison
        st.markdown("### Cost Analysis by Category")
        try:
            fig = create_category_cost_comparison(df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating category comparison: {e}")

        # Cost concentration
        st.markdown("### Cost Concentration (Pareto Analysis)")
        from escalation_ai.financial.visualizations import create_cost_concentration_chart
        try:
            fig = create_cost_concentration_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating concentration chart: {e}")

        st.info(f"""
        **Cost Concentration**: {financial_metrics.cost_concentration_ratio*100:.0f}% of total costs come from the top 20% of tickets.
        {'🔴 High concentration - focus on top cost drivers' if financial_metrics.cost_concentration_ratio > 0.8 else '🟢 Good cost distribution'}
        """)

    with tab2:
        st.markdown("### 💹 ROI Investment Opportunities")

        if roi_metrics['top_opportunities']:
            # ROI opportunity chart
            try:
                fig = create_roi_opportunity_chart(roi_metrics)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating ROI chart: {e}")

            # ROI opportunity table
            st.markdown("### Top ROI Opportunities")
            roi_df = pd.DataFrame(roi_metrics['top_opportunities']).round(2)
            roi_df_display = pd.DataFrame({
                'Category': roi_df['category'],
                'Incidents': roi_df['incident_count'],
                'Total Cost': roi_df['total_cost'].apply(lambda x: f"${x:,.0f}"),
                'Investment': roi_df['investment_required'].apply(lambda x: f"${x:,.0f}"),
                'Annual Savings': roi_df['annual_savings'].apply(lambda x: f"${x:,.0f}"),
                'ROI %': roi_df['roi_percentage'].apply(lambda x: f"{x:.0f}%"),
                'Payback (mo)': roi_df['payback_months'].apply(lambda x: f"{x:.1f}")
            })

            st.dataframe(roi_df_display, use_container_width=True, hide_index=True)

            # ROI summary
            st.success(f"""
            **Investment Summary:**
            - Total Investment: **${roi_metrics['total_investment_required']:,.0f}**
            - Expected Annual Savings: **${roi_metrics['expected_annual_savings']:,.0f}**
            - Overall ROI: **{roi_metrics['roi_percentage']:.0f}%**
            - Payback Period: **{roi_metrics['payback_months']:.1f} months**
            """)
        else:
            st.info("Not enough recurring patterns to identify ROI opportunities. Need at least 3 similar incidents per category.")

    with tab3:
        st.markdown("### 💡 Cost Avoidance Potential")

        col1, col2 = st.columns(2)

        with col1:
            # Cost avoidance breakdown
            try:
                fig = create_cost_avoidance_breakdown(cost_avoidance)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating cost avoidance chart: {e}")

        with col2:
            # Cost avoidance details
            st.markdown("#### Avoidance Opportunities")

            avoidance_items = [
                ("🔄 Recurring Issues", cost_avoidance['recurring_issues'], "Fix root causes to prevent repeat incidents"),
                ("📋 Preventable Categories", cost_avoidance['preventable_categories'], "Improve processes and documentation"),
                ("📚 Knowledge Sharing", cost_avoidance['knowledge_sharing'], "Leverage similar ticket solutions"),
                ("🤖 Automation", cost_avoidance['automation'], "Automate repetitive tasks")
            ]

            for label, value, description in avoidance_items:
                st.markdown(f"""
                <div style="padding: 15px; background: #0a1929; border-left: 4px solid #2ca02c; margin-bottom: 10px;">
                    <div style="font-size: 1.1rem; font-weight: 600;">{label}</div>
                    <div style="font-size: 1.5rem; color: #2ca02c; font-weight: 700;">${value:,.0f}</div>
                    <div style="color: #999; font-size: 0.9rem;">{description}</div>
                </div>
                """, unsafe_allow_html=True)

            st.success(f"**Total Avoidance Potential: ${cost_avoidance['total_avoidance']:,.0f}**")

    with tab4:
        st.markdown("### 📈 Cost Trends & Financial Forecast")

        # Check if we have required data
        has_dates = any(col for col in df.columns if 'date' in col.lower() or 'time' in col.lower())

        if not has_dates:
            st.warning("""
            **No date information available in this dataset.**

            To enable trend analysis and forecasting:
            1. Regenerate the report with the latest pipeline
            2. Ensure your input data has an 'Issue Date' or 'Created Date' column

            The current report was generated without date information.
            """)

            # Show basic cost summary instead
            if 'AI_Category' in df.columns:
                st.markdown("#### Current Cost Summary by Category")
                cost_summary = df.groupby('AI_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])
                cost_summary.columns = ['Total Cost', 'Avg Cost', 'Count']
                cost_summary = cost_summary.sort_values('Total Cost', ascending=False)
                st.dataframe(cost_summary.style.format({
                    'Total Cost': '${:,.0f}',
                    'Avg Cost': '${:,.0f}'
                }), use_container_width=True)
        else:
            # Forecast chart
            try:
                fig = create_cost_trend_forecast(df, forecasts)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating trend forecast: {e}")
                st.info("Try regenerating the report with the latest pipeline.")

        # Forecast metrics (only show if we have forecast data)
        if forecasts.get('monthly_projection'):
            col1, col2, col3 = st.columns(3)
        else:
            st.info("Run the pipeline with date information to see forecasts.")
            col1, col2, col3 = st.columns(3)

        with col1:
            trend_icon = "📈" if forecasts['trend'] == 'increasing' else "📉" if forecasts['trend'] == 'decreasing' else "➡️"
            st.metric("Cost Trend", f"{trend_icon} {forecasts['trend'].title()}",
                     delta=f"{forecasts['confidence'].title()} confidence")

        with col2:
            st.metric("30-Day Projection", f"${financial_metrics.cost_forecast_30d:,.0f}")

        with col3:
            st.metric("Annual Projection", f"${forecasts.get('annual_projection', 0):,.0f}")

        # Risk scenarios
        if forecasts.get('risk_scenarios'):
            st.markdown("#### 📊 Financial Scenarios")
            scenarios_df = pd.DataFrame({
                'Scenario': ['Best Case (20% reduction)', 'Expected', 'Worst Case (30% increase)'],
                'Annual Cost': [
                    f"${forecasts['risk_scenarios']['best_case']:,.0f}",
                    f"${forecasts['risk_scenarios']['expected']:,.0f}",
                    f"${forecasts['risk_scenarios']['worst_case']:,.0f}"
                ],
                'Monthly': [
                    f"${forecasts['risk_scenarios']['best_case']/12:,.0f}",
                    f"${forecasts['risk_scenarios']['expected']/12:,.0f}",
                    f"${forecasts['risk_scenarios']['worst_case']/12:,.0f}"
                ]
            })
            st.dataframe(scenarios_df, use_container_width=True, hide_index=True)

    with tab5:
        st.markdown("### 🎯 Financial Insights & Action Items")

        if insights:
            # Display insights as cards
            for insight in insights:
                priority_colors = {
                    'high': '#d62728',
                    'medium': '#ff7f0e',
                    'low': '#2ca02c'
                }
                color = priority_colors.get(insight['priority'], '#999')

                st.markdown(f"""
                <div style="padding: 20px; background: #0a1929; border-left: 5px solid {color}; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <div style="font-size: 0.8rem; color: {color}; font-weight: 600; text-transform: uppercase;">
                                {insight['priority']} Priority
                            </div>
                            <div style="font-size: 1.2rem; font-weight: 700; margin: 8px 0;">
                                {insight['title']}
                            </div>
                            <div style="color: #bbb; margin-bottom: 10px;">
                                {insight['description']}
                            </div>
                            <div style="background: #001e3c; padding: 10px; border-radius: 5px;">
                                <strong>💡 Recommendation:</strong> {insight['recommendation']}
                            </div>
                        </div>
                        <div style="text-align: right; margin-left: 20px;">
                            <div style="font-size: 0.8rem; color: #888;">Potential Savings</div>
                            <div style="font-size: 1.8rem; color: #2ca02c; font-weight: 700;">
                                ${insight.get('potential_savings', 0):,.0f}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Insights summary table
            st.markdown("### 📋 Insights Summary Table")
            insights_df = create_insights_table(insights)
            st.dataframe(insights_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant financial insights identified. Continue monitoring.")

        # Engineer efficiency matrix
        if 'Engineer_Assigned' in df.columns and 'Resolution_Days' in df.columns:
            st.markdown("### 👥 Engineer Cost Efficiency Analysis")
            try:
                fig = create_engineer_cost_efficiency_matrix(df)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating engineer efficiency matrix: {e}")

    # Bottom section: Interactive ROI Calculator
    st.markdown("---")
    st.markdown("### 💹 ROI Scenario Calculator")

    col1, col2 = st.columns([1, 2])

    with col1:
        reduction_pct = st.slider("Target Friction Reduction %", 10, 50, 25)
        investment = st.number_input("Proposed Investment ($)", 50000, 500000, 100000, step=25000)
        timeline_months = st.slider("Implementation Timeline (months)", 3, 18, 6)

    with col2:
        # Calculate ROI
        total_cost = financial_metrics.total_cost
        annual_savings = (total_cost * 4) * (reduction_pct / 100)  # Annualized
        roi = ((annual_savings - investment) / investment) * 100 if investment > 0 else 0
        payback_months = investment / (annual_savings / 12) if annual_savings > 0 else float('inf')
        npv = sum([(annual_savings - investment if i == 0 else annual_savings) / (1.08 ** i) for i in range(3)])

        st.markdown(f"""
        <div class="exec-card">
            <h4 style="color: #00BFFF; margin-bottom: 20px;">📈 Projected Financial Outcomes</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #28A745; margin: 0;">${annual_savings:,.0f}</p>
                    <p style="color: #888; font-size: 0.85rem;">Annual Savings</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #00BFFF; margin: 0;">{roi:.0f}%</p>
                    <p style="color: #888; font-size: 0.85rem;">First Year ROI</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #FFC107; margin: 0;">{f'{payback_months:.1f}' if payback_months != float('inf') else 'N/A'}</p>
                    <p style="color: #888; font-size: 0.85rem;">Payback (Months)</p>
                </div>
                <div style="text-align: center;">
                    <p style="font-size: 2rem; font-weight: 700; color: #4CAF50; margin: 0;">${npv:,.0f}</p>
                    <p style="color: #888; font-size: 0.85rem;">3-Year NPV</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_benchmarking(df):
    """Render the Competitive Benchmarking page."""
    st.markdown('<p class="main-header">🏆 Competitive Benchmarking</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">How you compare against industry standards</p>', unsafe_allow_html=True)
    
    # Get safe values with defaults
    recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
    resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    
    # Calculate current metrics
    current_metrics = {
        'resolution_days': resolution_days,
        'recurrence_rate': recurrence_rate * 100,
        'sla_breach_rate': df['SLA_Breached'].mean() * 100 if 'SLA_Breached' in df.columns else 12,
        'first_contact_resolution': 100 - (recurrence_rate * 100 * 2),  # Proxy
        'cost_per_escalation': df['Financial_Impact'].mean() if 'Financial_Impact' in df.columns else 850,
        'customer_satisfaction': 100 - (df['Customer_Impact_Score'].mean() * 0.3) if 'Customer_Impact_Score' in df.columns else 75,
    }
    
    # Benchmark gauges
    col1, col2, col3 = st.columns(3)
    
    gauge_configs = [
        ('Resolution Time', 'resolution_days', current_metrics['resolution_days'], ' days'),
        ('Recurrence Rate', 'recurrence_rate', current_metrics['recurrence_rate'], '%'),
        ('SLA Breach Rate', 'sla_breach_rate', current_metrics['sla_breach_rate'], '%'),
        ('First Contact Resolution', 'first_contact_resolution', current_metrics['first_contact_resolution'], '%'),
        ('Cost per Escalation', 'cost_per_escalation', current_metrics['cost_per_escalation'], '$'),
        ('Customer Satisfaction', 'customer_satisfaction', current_metrics['customer_satisfaction'], '%'),
    ]
    
    for i, (name, key, value, unit) in enumerate(gauge_configs):
        col = [col1, col2, col3][i % 3]
        with col:
            fig = chart_benchmark_gauge(name, value, INDUSTRY_BENCHMARKS[key], unit)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Competitive Position Summary
    st.markdown("### 📊 Competitive Position Summary")
    
    position_data = []
    for name, key, value, unit in gauge_configs:
        bench = INDUSTRY_BENCHMARKS[key]
        lower_better = bench['best_in_class'] < bench['laggard']
        
        if lower_better:
            if value <= bench['best_in_class']:
                position = "Best-in-Class"
                gap = 0
                color = "#28A745"
            elif value <= bench['industry_avg']:
                position = "Above Average"
                gap = value - bench['best_in_class']
                color = "#28A745"
            elif value <= bench['laggard']:
                position = "Below Average"
                gap = value - bench['industry_avg']
                color = "#FFC107"
            else:
                position = "Laggard"
                gap = value - bench['laggard']
                color = "#DC3545"
        else:
            if value >= bench['best_in_class']:
                position = "Best-in-Class"
                gap = 0
                color = "#28A745"
            elif value >= bench['industry_avg']:
                position = "Above Average"
                gap = bench['best_in_class'] - value
                color = "#28A745"
            elif value >= bench['laggard']:
                position = "Below Average"
                gap = bench['industry_avg'] - value
                color = "#FFC107"
            else:
                position = "Laggard"
                gap = bench['laggard'] - value
                color = "#DC3545"
        
        position_data.append({
            'Metric': name,
            'Current': f"{value:.1f}{unit}",
            'Best-in-Class': f"{bench['best_in_class']}{unit}",
            'Industry Avg': f"{bench['industry_avg']}{unit}",
            'Position': position,
            'Gap to Best': f"{gap:.1f}{unit}" if gap > 0 else "—",
            'Color': color
        })
    
    # Display as a proper dataframe with styling
    display_df = pd.DataFrame(position_data)
    display_df = display_df[['Metric', 'Current', 'Best-in-Class', 'Industry Avg', 'Position', 'Gap to Best']]
    
    # Use Streamlit's dataframe with custom styling
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Current": st.column_config.TextColumn("Current", width="small"),
            "Best-in-Class": st.column_config.TextColumn("Best-in-Class", width="small"),
            "Industry Avg": st.column_config.TextColumn("Industry Avg", width="small"),
            "Position": st.column_config.TextColumn("Position", width="medium"),
            "Gap to Best": st.column_config.TextColumn("Gap to Best", width="small"),
        }
    )


def render_root_cause(df):
    """Render Root Cause Analysis page."""
    st.markdown('<p class="main-header">🔬 Root Cause Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identify and quantify the drivers of escalation friction</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(chart_pareto_analysis(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(chart_driver_tree(df), use_container_width=True)
    
    st.markdown("---")
    
    # Root cause breakdown
    st.markdown("### 🎯 Root Cause Impact Quantification")
    
    if 'Root_Cause' in df.columns:
        root_cause_analysis = df.groupby('Root_Cause').agg({
            'Strategic_Friction_Score': 'sum',
            'Financial_Impact': 'sum',
            'AI_Category': 'count'
        }).rename(columns={'AI_Category': 'count'}).sort_values('Strategic_Friction_Score', ascending=False)
        
        root_cause_analysis['Friction %'] = root_cause_analysis['Strategic_Friction_Score'] / root_cause_analysis['Strategic_Friction_Score'].sum() * 100
        root_cause_analysis['Avg Cost'] = root_cause_analysis['Financial_Impact'] / root_cause_analysis['count']
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Friction by Root Cause', 'Cost Impact by Root Cause'),
                            horizontal_spacing=0.15)  # Add spacing between subplots
        
        fig.add_trace(go.Bar(
            y=root_cause_analysis.index,
            x=root_cause_analysis['Strategic_Friction_Score'],
            orientation='h',
            marker_color=px.colors.sequential.Blues_r[:len(root_cause_analysis)],
            name='Friction'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            y=root_cause_analysis.index,
            x=root_cause_analysis['Financial_Impact'],
            orientation='h',
            marker_color=px.colors.sequential.Reds_r[:len(root_cause_analysis)],
            name='Cost'
        ), row=1, col=2)
        
        fig.update_layout(
            **create_plotly_theme(),
            height=500,  # Taller to prevent overlap
            showlegend=False,
            margin=dict(l=150, r=40, t=60, b=40)  # More left margin for labels
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk heatmap
    st.plotly_chart(chart_risk_heatmap(df), use_container_width=True)


def render_action_tracker(df):
    """Render the Action Tracker page."""
    st.markdown('<p class="main-header">📋 Action Tracker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Strategic initiatives monitoring and accountability</p>', unsafe_allow_html=True)
    
    # Initialize action items - load from JSON, then merge with AI recommendations
    if not st.session_state.action_items:
        saved_items = load_action_items() or []
        st.session_state.action_items = saved_items
    
    # Always generate fresh AI recommendations and merge new ones
    recommendations = generate_strategic_recommendations(df)
    existing_titles = {item['title'] for item in st.session_state.action_items}
    
    new_items_added = False
    for rec in recommendations[:5]:
        if rec['title'] not in existing_titles:
            # This is a new AI recommendation - add it
            new_id = max((item['id'] for item in st.session_state.action_items), default=-1) + 1
            st.session_state.action_items.append({
                'id': new_id,
                'title': rec['title'],
                'priority': rec['priority'],
                'owner': 'Unassigned',
                'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'status': 'Not Started',
                'progress': 0,
                'notes': rec['description'],
                'ai_generated': True  # Mark as AI-generated
            })
            new_items_added = True
    
    if new_items_added:
        save_action_items(st.session_state.action_items)
        st.toast("🤖 New AI recommendations added!", icon="✨")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_actions = len(st.session_state.action_items)
    completed = sum(1 for a in st.session_state.action_items if a['status'] == 'Completed')
    in_progress = sum(1 for a in st.session_state.action_items if a['status'] == 'In Progress')
    blocked = sum(1 for a in st.session_state.action_items if a['status'] == 'Blocked')
    
    with col1:
        st.metric("Total Initiatives", total_actions)
    with col2:
        st.metric("Completed", completed, delta=f"{completed/total_actions*100:.0f}%" if total_actions > 0 else "0%")
    with col3:
        st.metric("In Progress", in_progress)
    with col4:
        st.metric("Blocked", blocked, delta_color="inverse" if blocked > 0 else "normal")
    
    st.markdown("---")
    
    # Add new action
    with st.expander("➕ Add New Initiative"):
        col1, col2 = st.columns(2)
        with col1:
            new_title = st.text_input("Initiative Title")
            new_priority = st.selectbox("Priority", ['P1', 'P2', 'P3'])
        with col2:
            new_owner = st.text_input("Owner")
            new_due = st.date_input("Due Date", value=datetime.now() + timedelta(days=30))
        
        new_notes = st.text_area("Description/Notes")
        
        if st.button("Add Initiative"):
            st.session_state.action_items.append({
                'id': len(st.session_state.action_items),
                'title': new_title,
                'priority': new_priority,
                'owner': new_owner,
                'due_date': new_due.strftime('%Y-%m-%d'),
                'status': 'Not Started',
                'progress': 0,
                'notes': new_notes
            })
            save_action_items(st.session_state.action_items)
            st.rerun()
    
    # Action items list
    st.markdown("### 📝 Initiative Status")
    
    # Track items to delete (can't modify list while iterating)
    items_to_delete = []
    
    for i, action in enumerate(st.session_state.action_items):
        status_class = 'completed' if action['status'] == 'Completed' else 'in-progress' if action['status'] == 'In Progress' else 'blocked' if action['status'] == 'Blocked' else ''
        priority_class = f"priority-{action['priority'].lower()}"
        
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 0.5])
            
            with col1:
                ai_badge = "🤖 " if action.get('ai_generated') else ""
                st.markdown(f"""
                <div class="action-card {status_class}">
                    <span class="{priority_class}">{action['priority']}</span>
                    <strong style="margin-left: 12px;">{ai_badge}{action['title']}</strong>
                    <p style="color: #888; font-size: 0.85rem; margin: 8px 0 0 0;">{action['notes'][:100]}{'...' if len(action['notes']) > 100 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                new_status = st.selectbox(
                    "Status", 
                    ['Not Started', 'In Progress', 'Completed', 'Blocked'],
                    index=['Not Started', 'In Progress', 'Completed', 'Blocked'].index(action['status']),
                    key=f"status_{i}"
                )
                if st.session_state.action_items[i]['status'] != new_status:
                    st.session_state.action_items[i]['status'] = new_status
                    save_action_items(st.session_state.action_items)
            
            with col3:
                new_owner = st.text_input("Owner", value=action['owner'], key=f"owner_{i}")
                if st.session_state.action_items[i]['owner'] != new_owner:
                    st.session_state.action_items[i]['owner'] = new_owner
                    save_action_items(st.session_state.action_items)
            
            with col4:
                progress = st.slider("Progress", 0, 100, action['progress'], key=f"progress_{i}")
                if st.session_state.action_items[i]['progress'] != progress:
                    st.session_state.action_items[i]['progress'] = progress
                    save_action_items(st.session_state.action_items)
            
            with col5:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                col_close, col_del = st.columns(2)
                with col_close:
                    if action['status'] != 'Completed':
                        if st.button("✅", key=f"close_{i}", help="Mark as Completed"):
                            st.session_state.action_items[i]['status'] = 'Completed'
                            st.session_state.action_items[i]['progress'] = 100
                            save_action_items(st.session_state.action_items)
                            st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete Initiative"):
                        items_to_delete.append(i)
        
        st.markdown("---")
    
    # Process deletions after loop
    if items_to_delete:
        for idx in sorted(items_to_delete, reverse=True):
            st.session_state.action_items.pop(idx)
        save_action_items(st.session_state.action_items)
        st.rerun()


def render_presentation_mode(df):
    """Render Executive Presentation Mode with auto-cycling slides."""
    st.markdown('<p class="exec-title">📽️ Executive Presentation</p>', unsafe_allow_html=True)
    
    slides = [
        "executive_summary",
        "financial_impact", 
        "benchmarking",
        "recommendations",
        "forecast"
    ]
    
    slide_titles = [
        "Executive Summary",
        "Financial Impact",
        "Competitive Benchmarking", 
        "Strategic Recommendations",
        "90-Day Forecast"
    ]
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.current_slide = (st.session_state.current_slide - 1) % len(slides)
            st.rerun()
    
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>Slide {st.session_state.current_slide + 1} of {len(slides)}: {slide_titles[st.session_state.current_slide]}</h3>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️"):
            st.session_state.current_slide = (st.session_state.current_slide + 1) % len(slides)
            st.rerun()
    
    auto_play = st.checkbox("Auto-advance slides (10 seconds)")
    
    st.markdown("---")
    
    current = slides[st.session_state.current_slide]
    
    if current == "executive_summary":
        # Condensed executive summary
        total_impact = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * 850
        savings = total_impact * 0.35
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cost", f"${total_impact:,.0f}")
        with col2:
            st.metric("Savings Opportunity", f"${savings:,.0f}")
        with col3:
            st.metric("Escalations", f"{len(df):,}")
        
        st.plotly_chart(chart_pareto_analysis(df), use_container_width=True)
    
    elif current == "financial_impact":
        col1, col2 = st.columns(2)
        with col1:
            cost_by_cat = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False)
            fig = go.Figure(go.Pie(
                labels=cost_by_cat.index,
                values=cost_by_cat.values,
                hole=0.5,
                marker=dict(colors=px.colors.sequential.Reds_r)
            ))
            fig.update_layout(**create_plotly_theme(), title='Cost Distribution', height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(chart_driver_tree(df), use_container_width=True)
    
    elif current == "benchmarking":
        # Get safe values
        recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
        resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
        
        current_metrics = {
            'resolution_days': resolution_days,
            'recurrence_rate': recurrence_rate * 100,
            'cost_per_escalation': df['Financial_Impact'].mean() if 'Financial_Impact' in df.columns else 850,
        }
        
        col1, col2, col3 = st.columns(3)
        for i, (key, title) in enumerate([('resolution_days', 'Resolution Time'), ('recurrence_rate', 'Recurrence Rate'), ('cost_per_escalation', 'Cost/Escalation')]):
            unit = ' days' if 'days' in key else '%' if 'rate' in key else '$'
            with [col1, col2, col3][i]:
                fig = chart_benchmark_gauge(title, current_metrics[key], INDUSTRY_BENCHMARKS[key], unit)
                st.plotly_chart(fig, use_container_width=True)
    
    elif current == "recommendations":
        recommendations = generate_strategic_recommendations(df)
        for rec in recommendations[:3]:
            st.markdown(f"""
            <div class="strategy-card {'high-priority' if rec['priority'] == 'P1' else ''}">
                <span class="priority-{rec['priority'].lower()}">{rec['priority']}</span>
                <strong style="margin-left: 12px;">{rec['title']}</strong>
                <p style="color: #888;">{rec['description']}</p>
                <p><strong>ROI:</strong> {rec['roi']} | <strong>Timeline:</strong> {rec['timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif current == "forecast":
        forecast_fig, slope = chart_forecast_projection(df)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        trend = "📈 Escalations trending UP" if slope > 0 else "📉 Escalations trending DOWN"
        color = "#DC3545" if slope > 0 else "#28A745"
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{trend} ({abs(slope):.2f}/day)</h3>", unsafe_allow_html=True)
    
    if auto_play:
        time.sleep(10)
        st.session_state.current_slide = (st.session_state.current_slide + 1) % len(slides)
        st.rerun()


def render_whatif_simulator(df):
    """Render the What-If Scenario Simulator page."""
    st.markdown("### 🔮 What-If Scenario Simulator")
    st.markdown("Adjust parameters to simulate impact on escalation metrics.")
    
    # Get safe values with defaults
    recurrence_rate = df['AI_Recurrence_Risk'].mean() if 'AI_Recurrence_Risk' in df.columns else 0.15
    avg_resolution = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
    friction_sum = df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in df.columns else 3000
    cost_sum = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else 375000
    
    # Current baseline metrics
    baseline = {
        'avg_resolution': avg_resolution,
        'recurrence_rate': recurrence_rate * 100,
        'monthly_friction': friction_sum / 3,  # Assume 3 months data
        'monthly_cost': cost_sum / 3
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Scenario Parameters")
        
        # Staffing
        st.markdown("**👥 Staffing Changes**")
        engineer_change = st.slider("Add/Remove Engineers", -3, 5, 0, 
                                     help="Positive = add engineers, Negative = reduce")
        
        # Training
        st.markdown("**📚 Training Impact**")
        training_effect = st.slider("Error Reduction from Training", 0, 50, 0,
                                     help="Expected % reduction in human errors")
        
        # Volume
        st.markdown("**📈 Volume Changes**")
        volume_change = st.slider("Escalation Volume Change %", -30, 50, 0,
                                   help="Positive = more escalations expected")
        
        # Process
        st.markdown("**⚙️ Process Improvements**")
        process_improvement = st.slider("Process Efficiency Gain %", 0, 40, 0,
                                         help="Expected efficiency improvement")
    
    with col2:
        st.markdown("#### 📈 Projected Impact")
        
        # Calculate projections (simplified model)
        # Staffing effect: more engineers = faster resolution, less recurrence
        resolution_factor = 1 - (engineer_change * 0.08)  # Each engineer reduces time by 8%
        recurrence_factor = 1 - (engineer_change * 0.05)  # Each engineer reduces recurrence by 5%
        
        # Training effect
        recurrence_factor *= (1 - training_effect / 100 * 0.5)  # Training reduces recurrence
        resolution_factor *= (1 - training_effect / 100 * 0.2)  # Training speeds resolution
        
        # Volume effect
        cost_factor = 1 + (volume_change / 100)
        friction_factor = 1 + (volume_change / 100 * 0.8)
        
        # Process effect
        resolution_factor *= (1 - process_improvement / 100)
        friction_factor *= (1 - process_improvement / 100 * 0.7)
        
        # Calculate projected values
        projected = {
            'avg_resolution': baseline['avg_resolution'] * max(0.3, resolution_factor),
            'recurrence_rate': baseline['recurrence_rate'] * max(0.2, recurrence_factor),
            'monthly_friction': baseline['monthly_friction'] * max(0.3, friction_factor),
            'monthly_cost': baseline['monthly_cost'] * max(0.3, cost_factor * resolution_factor)
        }
        
        # Display comparison
        metrics = [
            ('Resolution Time', 'avg_resolution', 'days', True),
            ('Recurrence Rate', 'recurrence_rate', '%', True),
            ('Monthly Friction', 'monthly_friction', 'pts', True),
            ('Monthly Cost', 'monthly_cost', '$', True)
        ]
        
        for label, key, unit, lower_better in metrics:
            base_val = baseline[key]
            proj_val = projected[key]
            delta = ((proj_val - base_val) / base_val) * 100
            
            if unit == '$':
                base_str = f"${base_val:,.0f}"
                proj_str = f"${proj_val:,.0f}"
            elif unit == 'days':
                base_str = f"{base_val:.1f} {unit}"
                proj_str = f"{proj_val:.1f} {unit}"
            else:
                base_str = f"{base_val:.1f}{unit}"
                proj_str = f"{proj_val:.1f}{unit}"
            
            is_improvement = (delta < 0) if lower_better else (delta > 0)
            delta_color = "delta-down" if is_improvement else "delta-up"
            arrow = "↓" if delta < 0 else "↑"
            
            st.markdown(f"""
            <div class="kpi-container {'success' if is_improvement else 'warning'}">
                <div style="font-size: 0.8rem; color: #888;">{label}</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{base_str} → {proj_str}</div>
                <div class="{delta_color}">{arrow} {abs(delta):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ROI Calculation
    st.markdown("---")
    st.markdown("#### 💰 Return on Investment")
    
    # Costs of changes
    engineer_cost = max(0, engineer_change) * 85000  # Annual salary
    training_cost = training_effect * 500 * len(df['Engineer'].unique()) if 'Engineer' in df.columns else training_effect * 5000
    process_cost = process_improvement * 2000
    
    total_investment = engineer_cost + training_cost + process_cost
    
    # Savings (annualized)
    monthly_savings = baseline['monthly_cost'] - projected['monthly_cost']
    annual_savings = monthly_savings * 12
    
    if total_investment > 0:
        roi = (annual_savings / total_investment) * 100
        payback_months = total_investment / monthly_savings if monthly_savings > 0 else float('inf')
    else:
        roi = float('inf') if annual_savings > 0 else 0
        payback_months = 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Investment", f"${total_investment:,.0f}")
    with col2:
        st.metric("Annual Savings", f"${annual_savings:,.0f}", 
                  delta=f"{annual_savings/baseline['monthly_cost']/12*100:.0f}% of current cost")
    with col3:
        if payback_months < 24:
            st.metric("Payback Period", f"{payback_months:.1f} months")
        else:
            st.metric("ROI", f"{roi:.0f}%" if roi != float('inf') else "∞")

# ============================================================================
# DRIFT DETECTION
# ============================================================================

def render_drift_page(df):
    """Render the Category Drift Detection page."""
    try:
        st.markdown("### 📊 Category Drift Detection")
        st.markdown("Analyze how escalation patterns are changing over time.")

        # Debug: Show available columns
        if st.checkbox("Show debug info", value=False):
            st.write("Available columns:", df.columns.tolist())
            st.write("DataFrame shape:", df.shape)

        # Find date column
        date_col = None
        for col in ['Issue_Date', 'Issue Date', 'tickets_data_issue_datetime', 'Created_Date', 'Date', 'Timestamp']:
            if col in df.columns:
                date_col = col
                break

        if not date_col:
            st.warning("""
            **No date information available for drift detection.**

            To enable drift analysis:
            1. Regenerate the report with the latest pipeline (`python run.py`)
            2. Ensure your input data has an 'Issue Date' or 'Created Date' column

            **Current dataset does not contain date information.**
            """)

            # Show what we DO have
            st.info(f"Dataset contains {len(df)} records with {len(df.columns)} columns.")
            return

        # Check for AI_Category column
        if 'AI_Category' not in df.columns:
            st.error("Missing 'AI_Category' column required for drift detection.")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            return

        # Split data into baseline and recent
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=['date'])
        df_temp = df_temp.sort_values('date')

        if len(df_temp) < 10:
            st.warning(f"Not enough data points for drift detection. Found {len(df_temp)} tickets with dates (need at least 10).")
            return

        st.success(f"✓ Found {len(df_temp)} tickets with valid dates for drift analysis")

        split_idx = int(len(df_temp) * 0.6)
        baseline_df = df_temp.iloc[:split_idx]
        current_df = df_temp.iloc[split_idx:]

        # Calculate distributions
        baseline_dist = baseline_df['AI_Category'].value_counts(normalize=True)
        current_dist = current_df['AI_Category'].value_counts(normalize=True)

    except Exception as e:
        st.error(f"""
        **Error in Drift Detection:**

        {str(e)}

        Please try:
        1. Regenerating the report with `python run.py`
        2. Ensuring your input data has proper date columns
        """)

        import traceback
        if st.checkbox("Show technical details"):
            st.code(traceback.format_exc())
        return

    # Create comparison chart
    all_cats = sorted(set(baseline_dist.index) | set(current_dist.index))
    
    comparison_data = pd.DataFrame({
        'Category': all_cats,
        'Baseline': [baseline_dist.get(c, 0) * 100 for c in all_cats],
        'Current': [current_dist.get(c, 0) * 100 for c in all_cats]
    })
    comparison_data['Change'] = comparison_data['Current'] - comparison_data['Baseline']
    comparison_data = comparison_data.sort_values('Change')
    
    # Grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline (First 60%)',
        x=comparison_data['Category'],
        y=comparison_data['Baseline'],
        marker_color='rgba(100, 149, 237, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        name='Current (Last 40%)',
        x=comparison_data['Category'],
        y=comparison_data['Current'],
        marker_color='rgba(255, 107, 107, 0.7)'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title='Category Distribution: Baseline vs Current',
        barmode='group',
        xaxis_tickangle=-45,
        height=400,
        legend=dict(orientation='h', y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Change analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Emerging Categories")
        emerging = comparison_data[comparison_data['Change'] > 2].sort_values('Change', ascending=False)
        for _, row in emerging.iterrows():
            st.markdown(f"""
            <div class="kpi-container warning">
                <b>{row['Category']}</b><br>
                <span class="delta-up">↑ {row['Change']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        if emerging.empty:
            st.info("No significantly emerging categories detected.")
    
    with col2:
        st.markdown("#### 📉 Declining Categories")
        declining = comparison_data[comparison_data['Change'] < -2].sort_values('Change')
        for _, row in declining.iterrows():
            st.markdown(f"""
            <div class="kpi-container success">
                <b>{row['Category']}</b><br>
                <span class="delta-down">↓ {abs(row['Change']):.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        if declining.empty:
            st.info("No significantly declining categories detected.")

# ============================================================================
# ALERTS PAGE
# ============================================================================

def render_alerts_page(df):
    """Render the Smart Alerts page."""
    st.markdown("### ⚠️ Smart Alert Thresholds")
    st.markdown("Real-time monitoring of key metrics against dynamic thresholds.")

    # Find date column
    date_col = None
    for col in ['Issue_Date', 'Issue Date', 'tickets_data_issue_datetime', 'Created_Date', 'Date', 'Timestamp']:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        st.warning("No date information available. Showing current state alerts only.")
        date_col = None

    # Calculate current metrics
    df_temp = df.copy()
    if date_col:
        try:
            df_temp['date'] = pd.to_datetime(df_temp[date_col], errors='coerce').dt.date
        except:
            date_col = None
    
    # Build agg dict with available columns
    agg_dict = {'AI_Category': 'count'}
    if 'Strategic_Friction_Score' in df.columns:
        agg_dict['Strategic_Friction_Score'] = 'sum'
    if 'AI_Recurrence_Risk' in df.columns:
        agg_dict['AI_Recurrence_Risk'] = 'mean'
    
    daily = df_temp.groupby('date').agg(agg_dict).rename(columns={'AI_Category': 'count'})
    
    # Ensure Strategic_Friction_Score exists
    if 'Strategic_Friction_Score' not in daily.columns:
        daily['Strategic_Friction_Score'] = 50
    
    # Calculate thresholds (simplified)
    metrics_config = {
        'Daily Escalations': {
            'values': daily['count'],
            'current': daily['count'].iloc[-1] if len(daily) > 0 else 0,
            'warning': daily['count'].quantile(0.75),
            'critical': daily['count'].quantile(0.90)
        },
        'Daily Friction': {
            'values': daily['Strategic_Friction_Score'],
            'current': daily['Strategic_Friction_Score'].iloc[-1] if len(daily) > 0 else 0,
            'warning': daily['Strategic_Friction_Score'].quantile(0.75),
            'critical': daily['Strategic_Friction_Score'].quantile(0.90)
        },
        'Recurrence Risk': {
            'values': daily['AI_Recurrence_Risk'] * 100,
            'current': daily['AI_Recurrence_Risk'].iloc[-1] * 100 if len(daily) > 0 else 0,
            'warning': 25,
            'critical': 40
        }
    }
    
    cols = st.columns(3)
    
    for i, (metric_name, config) in enumerate(metrics_config.items()):
        with cols[i]:
            current = config['current']
            warning = config['warning']
            critical = config['critical']
            
            if current >= critical:
                status = 'critical'
                badge_class = 'badge-critical'
                status_text = 'CRITICAL'
            elif current >= warning:
                status = 'warning'
                badge_class = 'badge-warning'
                status_text = 'WARNING'
            else:
                status = 'success'
                badge_class = 'badge-success'
                status_text = 'NORMAL'
            
            st.markdown(f"""
            <div class="kpi-container {status}">
                <span class="badge {badge_class}">{status_text}</span>
                <h3 style="margin: 10px 0; font-size: 2rem;">{current:.1f}</h3>
                <p style="color: #888; margin: 0;">{metric_name}</p>
                <small>Warning: {warning:.1f} | Critical: {critical:.1f}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Threshold timeline chart
    st.markdown("---")
    st.markdown("#### 📈 Metric Timeline with Thresholds")
    
    selected_metric = st.selectbox("Select Metric", list(metrics_config.keys()))
    config = metrics_config[selected_metric]
    
    fig = go.Figure()
    
    dates = daily.index
    values = config['values']
    
    # Add threshold zones
    fig.add_hrect(y0=0, y1=config['warning'], fillcolor="rgba(40,167,69,0.1)", line_width=0)
    fig.add_hrect(y0=config['warning'], y1=config['critical'], fillcolor="rgba(255,193,7,0.1)", line_width=0)
    fig.add_hrect(y0=config['critical'], y1=values.max()*1.2, fillcolor="rgba(220,53,69,0.1)", line_width=0)
    
    # Add threshold lines
    fig.add_hline(y=config['warning'], line_dash="dash", line_color="#FFC107", 
                  annotation_text="Warning")
    fig.add_hline(y=config['critical'], line_dash="dash", line_color="#DC3545",
                  annotation_text="Critical")
    
    # Add values
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines+markers',
        line=dict(color='#0066CC', width=2),
        marker=dict(size=6),
        hovertemplate='%{x}<br>Value: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        height=400,
        showlegend=False,
        xaxis_title='Date',
        yaxis_title=selected_metric
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PDF EXPORT FUNCTIONALITY
# ============================================================================

def generate_executive_pdf_report(df):
    """Generate a comprehensive PDF executive report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.graphics.shapes import Drawing, Rect
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.piecharts import Pie
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0066CC')
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#0066CC')
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=13,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#333333')
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        )
        
        # Build content
        story = []
        
        # ===== COVER PAGE =====
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("ESCALATION AI", title_style))
        story.append(Paragraph("Executive Intelligence Report", subtitle_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))
        story.append(Paragraph(f"Analysis Period: 90 Days | {len(df):,} Escalations Analyzed", subtitle_style))
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("CONFIDENTIAL - FOR EXECUTIVE REVIEW ONLY", 
                              ParagraphStyle('Confidential', parent=body_style, 
                                           alignment=TA_CENTER, textColor=colors.HexColor('#DC3545'))))
        story.append(PageBreak())
        
        # ===== EXECUTIVE SUMMARY =====
        story.append(Paragraph("1. Executive Summary", heading_style))
        
        # Calculate key metrics
        total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * 850
        revenue_risk = df['Revenue_At_Risk'].sum() if 'Revenue_At_Risk' in df.columns else total_cost * 0.20
        avg_resolution = df['Predicted_Resolution_Days'].mean()
        recurrence_rate = df['AI_Recurrence_Risk'].mean() * 100
        critical_count = len(df[df['tickets_data_severity'] == 'Critical'])
        
        summary_text = f"""
        This report provides a comprehensive analysis of escalation patterns and their business impact 
        over the past 90 days. The analysis covers {len(df):,} records with a total operational 
        cost of ${total_cost:,.0f} and revenue at risk of ${revenue_risk:,.0f}.
        """
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Key Metrics Table
        story.append(Paragraph("Key Performance Indicators", subheading_style))
        
        # Count by type
        type_col = 'tickets_data_type_1'
        escalations_count = len(df[df[type_col].astype(str).str.contains('Escalation', case=False, na=False)]) if type_col in df.columns else len(df)
        concerns_count = len(df[df[type_col].astype(str).str.contains('Concern', case=False, na=False)]) if type_col in df.columns else 0
        lessons_count = len(df[df[type_col].astype(str).str.contains('Lesson', case=False, na=False)]) if type_col in df.columns else 0
        
        kpi_data = [
            ['Metric', 'Current Value', 'Industry Benchmark', 'Status'],
            ['Total Records', f'{len(df):,}', '—', '—'],
            ['Escalations', f'{escalations_count:,}', '—', '—'],
            ['Concerns', f'{concerns_count:,}', '—', '—'],
            ['Lessons Learned', f'{lessons_count:,}', '—', '—'],
            ['Critical Issues', f'{critical_count}', '—', 'High' if critical_count > 20 else 'Normal'],
            ['Total Operational Cost', f'${total_cost:,.0f}', '—', '—'],
            ['Revenue at Risk', f'${revenue_risk:,.0f}', '—', '—'],
            ['Avg Resolution Time', f'{avg_resolution:.1f} days', '2.8 days', 'Above' if avg_resolution > 2.8 else 'Below'],
            ['Recurrence Rate', f'{recurrence_rate:.1f}%', '18%', 'Above' if recurrence_rate > 18 else 'Below'],
            ['Potential Savings', f'${total_cost * 0.35:,.0f}', '—', '35% reduction achievable'],
        ]
        
        kpi_table = Table(kpi_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.2*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 0.3*inch))
        
        # ===== STRATEGIC RECOMMENDATIONS =====
        story.append(Paragraph("2. Strategic Recommendations", heading_style))
        
        recommendations = generate_strategic_recommendations(df)
        
        for i, rec in enumerate(recommendations[:4], 1):
            story.append(Paragraph(f"<b>{rec['priority']}: {rec['title']}</b>", subheading_style))
            story.append(Paragraph(rec['description'], body_style))
            
            rec_details = f"""
            <b>Expected Impact:</b> {rec['impact']}<br/>
            <b>Timeline:</b> {rec['timeline']} | <b>Investment:</b> {rec['investment']} | <b>ROI:</b> {rec['roi']}<br/>
            <b>Confidence Score:</b> {rec['confidence']}%
            """
            story.append(Paragraph(rec_details, body_style))
            story.append(Spacer(1, 0.15*inch))
        
        story.append(PageBreak())
        
        # ===== CATEGORY ANALYSIS =====
        story.append(Paragraph("3. Category Analysis", heading_style))
        
        category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
        cumulative_pct = category_friction.cumsum() / category_friction.sum() * 100
        
        story.append(Paragraph("Pareto Analysis: Focus on the Vital Few", subheading_style))
        story.append(Paragraph(
            "The following table shows escalation categories ranked by friction score. "
            "Categories highlighted contribute to 80% of total friction and should be prioritized.",
            body_style
        ))
        
        pareto_data = [['Rank', 'Category', 'Friction Score', 'Cumulative %', 'Priority']]
        for i, (cat, friction) in enumerate(category_friction.items(), 1):
            cum_pct = cumulative_pct[cat]
            priority = '🔴 HIGH' if cum_pct <= 80 else '🟡 MEDIUM' if cum_pct <= 95 else '🟢 LOW'
            pareto_data.append([str(i), cat, f'{friction:,.0f}', f'{cum_pct:.1f}%', priority])
        
        pareto_table = Table(pareto_data, colWidths=[0.5*inch, 2.5*inch, 1.2*inch, 1*inch, 1*inch])
        pareto_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(pareto_table)
        story.append(Spacer(1, 0.3*inch))
        
        # ===== FINANCIAL IMPACT =====
        story.append(Paragraph("4. Financial Impact Analysis", heading_style))
        
        cost_by_cat = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False)
        
        fin_data = [['Category', 'Total Cost', '% of Total', 'Avg per Ticket']]
        for cat, cost in cost_by_cat.items():
            cat_count = len(df[df['AI_Category'] == cat])
            pct = cost / total_cost * 100
            avg_cost = cost / cat_count if cat_count > 0 else 0
            fin_data.append([cat, f'${cost:,.0f}', f'{pct:.1f}%', f'${avg_cost:,.0f}'])
        
        fin_table = Table(fin_data, colWidths=[2.5*inch, 1.3*inch, 1*inch, 1.2*inch])
        fin_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28A745')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(fin_table)
        story.append(Spacer(1, 0.2*inch))
        
        # ROI Summary
        story.append(Paragraph("Investment Opportunity", subheading_style))
        roi_text = f"""
        Based on our analysis, a targeted improvement program could yield significant returns:
        <br/><br/>
        <b>Projected Annual Savings:</b> ${total_cost * 4 * 0.25:,.0f} (25% reduction scenario)<br/>
        <b>Recommended Investment:</b> $100,000 - $150,000<br/>
        <b>Expected ROI:</b> 250-350%<br/>
        <b>Payback Period:</b> 4-6 months
        """
        story.append(Paragraph(roi_text, body_style))
        
        story.append(PageBreak())
        
        # ===== BENCHMARKING =====
        story.append(Paragraph("5. Competitive Benchmarking", heading_style))
        
        story.append(Paragraph(
            "The following table compares your current performance against industry benchmarks. "
            "Best-in-class performers are in the top 10th percentile of the industry.",
            body_style
        ))
        
        bench_data = [['Metric', 'Your Performance', 'Best-in-Class', 'Industry Avg', 'Gap to Best']]
        
        metrics_for_bench = [
            ('Resolution Time', f'{avg_resolution:.1f} days', '1.2 days', '2.8 days', f'{max(0, avg_resolution - 1.2):.1f} days'),
            ('Recurrence Rate', f'{recurrence_rate:.1f}%', '8%', '18%', f'{max(0, recurrence_rate - 8):.1f}%'),
            ('Cost per Escalation', f'${total_cost/len(df):,.0f}', '$450', '$850', f'${max(0, total_cost/len(df) - 450):,.0f}'),
        ]
        
        for m in metrics_for_bench:
            bench_data.append(list(m))
        
        bench_table = Table(bench_data, colWidths=[1.5*inch, 1.3*inch, 1.2*inch, 1.2*inch, 1*inch])
        bench_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFC107')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(bench_table)
        story.append(Spacer(1, 0.3*inch))
        
        # ===== APPENDIX =====
        story.append(Paragraph("6. Appendix: Methodology", heading_style))
        
        methodology_text = """
        <b>Data Sources:</b> Escalation tickets, CRM data, financial systems<br/>
        <b>Analysis Period:</b> Rolling 90 days<br/>
        <b>AI Models Used:</b> Random Forest classification, XGBoost regression, NLP categorization<br/>
        <b>Confidence Level:</b> 95% for all statistical measures<br/>
        <b>Benchmark Sources:</b> Industry reports, peer comparisons, historical performance<br/><br/>
        
        <b>Key Definitions:</b><br/>
        • <b>Strategic Friction Score:</b> Composite metric measuring operational impact (0-200)<br/>
        • <b>Recurrence Risk:</b> ML-predicted probability of issue recurring within 30 days<br/>
        • <b>Revenue at Risk:</b> Estimated revenue impact based on churn probability and contract value<br/>
        """
        story.append(Paragraph(methodology_text, body_style))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("—— END OF REPORT ——", 
                              ParagraphStyle('Footer', parent=body_style, alignment=TA_CENTER, 
                                           textColor=colors.HexColor('#888888'))))
        story.append(Paragraph(f"Generated by Escalation AI v2.2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                              ParagraphStyle('Version', parent=body_style, alignment=TA_CENTER,
                                           fontSize=8, textColor=colors.HexColor('#AAAAAA'))))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        return None


def generate_html_report(df):
    """Generate an HTML report that can be converted to PDF."""
    total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * 850
    revenue_risk = df['Revenue_At_Risk'].sum() if 'Revenue_At_Risk' in df.columns else total_cost * 0.20
    avg_resolution = df['Predicted_Resolution_Days'].mean()
    recurrence_rate = df['AI_Recurrence_Risk'].mean() * 100
    critical_count = len(df[df['tickets_data_severity'] == 'Critical'])
    
    category_friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
    
    recommendations = generate_strategic_recommendations(df)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Escalation AI Executive Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Inter', sans-serif; color: #333; line-height: 1.6; padding: 40px; max-width: 1000px; margin: 0 auto; }}
            
            .header {{ text-align: center; margin-bottom: 40px; padding: 40px; background: linear-gradient(135deg, #0066CC 0%, #004080 100%); color: white; border-radius: 12px; }}
            .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
            .header p {{ opacity: 0.9; }}
            
            .section {{ margin: 30px 0; }}
            .section h2 {{ color: #0066CC; border-bottom: 2px solid #0066CC; padding-bottom: 10px; margin-bottom: 20px; }}
            .section h3 {{ color: #333; margin: 20px 0 10px 0; }}
            
            .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
            .kpi-card {{ background: #f8f9fa; border-radius: 12px; padding: 24px; text-align: center; border-left: 4px solid #0066CC; }}
            .kpi-card.alert {{ border-left-color: #DC3545; }}
            .kpi-card.success {{ border-left-color: #28A745; }}
            .kpi-value {{ font-size: 2rem; font-weight: 700; color: #0066CC; }}
            .kpi-value.money {{ color: #28A745; }}
            .kpi-value.alert {{ color: #DC3545; }}
            .kpi-label {{ font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }}
            
            .rec-card {{ background: #f0f7ff; border-left: 4px solid #0066CC; padding: 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
            .rec-card.p1 {{ border-left-color: #DC3545; background: #fff5f5; }}
            .rec-card.p2 {{ border-left-color: #FFC107; background: #fffbf0; }}
            .rec-card h4 {{ color: #333; margin-bottom: 8px; }}
            .rec-card p {{ color: #666; margin-bottom: 10px; }}
            .rec-meta {{ display: flex; gap: 20px; font-size: 0.9rem; }}
            .rec-meta span {{ color: #0066CC; font-weight: 500; }}
            
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #0066CC; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 12px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background: #f8f9fa; }}
            
            .priority-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
            .priority-badge.p1 {{ background: #DC3545; color: white; }}
            .priority-badge.p2 {{ background: #FFC107; color: #333; }}
            .priority-badge.p3 {{ background: #0066CC; color: white; }}
            
            .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #888; font-size: 0.85rem; border-top: 1px solid #eee; }}
            
            @media print {{
                body {{ padding: 20px; }}
                .header {{ break-after: page; }}
                .section {{ break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 ESCALATION AI</h1>
            <p>Executive Intelligence Report</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
            <p>Analysis Period: 90 Days | {len(df):,} Records Analyzed</p>
        </div>
        
        <div class="section">
            <h2>1. Executive Summary</h2>
            
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{len(df):,}</div>
                    <div class="kpi-label">Total Records</div>
                </div>
                <div class="kpi-card alert">
                    <div class="kpi-value alert">{critical_count}</div>
                    <div class="kpi-label">Critical Issues</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value money">${total_cost:,.0f}</div>
                    <div class="kpi-label">Total Cost</div>
                </div>
                <div class="kpi-card success">
                    <div class="kpi-value money">${total_cost * 0.35:,.0f}</div>
                    <div class="kpi-label">Savings Opportunity</div>
                </div>
            </div>
            
            <table>
                <tr><th>Category</th><th>Count</th><th>Description</th></tr>
                <tr><td>📋 Escalations</td><td>{len(df[df['tickets_data_type_1'].astype(str).str.contains('Escalation', case=False, na=False)]) if 'tickets_data_type_1' in df.columns else len(df)}</td><td>Active escalation tickets</td></tr>
                <tr><td>⚠️ Concerns</td><td>{len(df[df['tickets_data_type_1'].astype(str).str.contains('Concern', case=False, na=False)]) if 'tickets_data_type_1' in df.columns else 0}</td><td>Potential issues flagged</td></tr>
                <tr><td>📚 Lessons Learned</td><td>{len(df[df['tickets_data_type_1'].astype(str).str.contains('Lesson', case=False, na=False)]) if 'tickets_data_type_1' in df.columns else 0}</td><td>Historical learnings</td></tr>
            </table>
            
            <table>
                <tr><th>Metric</th><th>Current</th><th>Benchmark</th><th>Status</th></tr>
                <tr><td>Avg Resolution Time</td><td>{avg_resolution:.1f} days</td><td>2.8 days</td><td>{'⚠️ Above' if avg_resolution > 2.8 else '✅ Below'}</td></tr>
                <tr><td>Recurrence Rate</td><td>{recurrence_rate:.1f}%</td><td>18%</td><td>{'⚠️ Above' if recurrence_rate > 18 else '✅ Below'}</td></tr>
                <tr><td>Revenue at Risk</td><td>${revenue_risk:,.0f}</td><td>—</td><td>—</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>2. Strategic Recommendations</h2>
            {''.join([f'''
            <div class="rec-card {rec['priority'].lower()}">
                <h4><span class="priority-badge {rec['priority'].lower()}">{rec['priority']}</span> {rec['title']}</h4>
                <p>{rec['description']}</p>
                <div class="rec-meta">
                    <span>Impact: {rec['impact']}</span>
                    <span>Timeline: {rec['timeline']}</span>
                    <span>Investment: {rec['investment']}</span>
                    <span>ROI: {rec['roi']}</span>
                </div>
            </div>
            ''' for rec in recommendations[:4]])}
        </div>
        
        <div class="section">
            <h2>3. Category Analysis (Pareto)</h2>
            <table>
                <tr><th>Rank</th><th>Category</th><th>Friction Score</th><th>% of Total</th></tr>
                {''.join([f"<tr><td>{i+1}</td><td>{cat}</td><td>{friction:,.0f}</td><td>{friction/category_friction.sum()*100:.1f}%</td></tr>" for i, (cat, friction) in enumerate(category_friction.items())])}
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by Escalation AI v2.2.0</p>
            <p>CONFIDENTIAL - FOR EXECUTIVE REVIEW ONLY</p>
        </div>
    </body>
    </html>
    """
    
    return html


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Escalation AI")
        st.markdown("*Executive Intelligence Platform*")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🎯 Executive Summary", "📊 Dashboard", "📈 Analytics",
             "💰 Financial Analysis", "🏆 Benchmarking", "🔬 Root Cause",
             "🚀 Advanced Insights", "🔍 Drift Detection", "⚠️ Alerts",
             "🔮 What-If Simulator", "📋 Action Tracker", "📽️ Presentation Mode"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Data info
        df, data_source = load_data()
        st.markdown(f"**Data Source:**")
        st.caption(data_source)
        st.markdown(f"**Records:** {len(df):,}")
        
        # Quick stats
        if 'Financial_Impact' in df.columns:
            total_cost = df['Financial_Impact'].sum()
            st.markdown(f"**Total Cost:** ${total_cost:,.0f}")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # Date filter
        if 'tickets_data_issue_datetime' in df.columns:
            min_date = pd.to_datetime(df['tickets_data_issue_datetime']).min().date()
            max_date = pd.to_datetime(df['tickets_data_issue_datetime']).max().date()
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                df = df[
                    (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date >= date_range[0]) &
                    (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date <= date_range[1])
                ]
        
        st.markdown("---")
        st.markdown("### 📤 Export")
        
        export_format = st.selectbox("Format", ["PDF Report", "HTML Report", "Excel Data", "CSV Data"])
        
        if st.button("📥 Generate Report"):
            with st.spinner("Generating report..."):
                if export_format == "PDF Report":
                    pdf_data = generate_executive_pdf_report(df)
                    if pdf_data:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_data,
                            file_name=f"Escalation_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("PDF generation requires reportlab. Generating HTML instead...")
                        html_data = generate_html_report(df)
                        st.download_button(
                            label="⬇️ Download HTML",
                            data=html_data,
                            file_name=f"Escalation_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                            mime="text/html"
                        )
                
                elif export_format == "HTML Report":
                    html_data = generate_html_report(df)
                    st.download_button(
                        label="⬇️ Download HTML",
                        data=html_data,
                        file_name=f"Escalation_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html"
                    )
                
                elif export_format == "Excel Data":
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='All Records', index=False)
                        
                        # Add summary sheet with type breakdown
                        type_col = 'tickets_data_type_1'
                        escalations = len(df[df[type_col].astype(str).str.contains('Escalation', case=False, na=False)]) if type_col in df.columns else len(df)
                        concerns = len(df[df[type_col].astype(str).str.contains('Concern', case=False, na=False)]) if type_col in df.columns else 0
                        lessons = len(df[df[type_col].astype(str).str.contains('Lesson', case=False, na=False)]) if type_col in df.columns else 0
                        
                        summary_df = pd.DataFrame({
                            'Metric': ['Total Records', 'Escalations', 'Concerns', 'Lessons Learned', 
                                       'Critical Issues', 'Avg Resolution', 'Recurrence Rate'],
                            'Value': [len(df), escalations, concerns, lessons,
                                     len(df[df['tickets_data_severity']=='Critical']), 
                                     f"{df['Predicted_Resolution_Days'].mean():.1f} days",
                                     f"{df['AI_Recurrence_Risk'].mean()*100:.1f}%"]
                        })
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"Escalation_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif export_format == "CSV Data":
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=f"Escalation_Data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
    
    # Main content - Route to appropriate page
    if page == "🎯 Executive Summary":
        render_executive_summary(df)
    elif page == "📊 Dashboard":
        render_dashboard(df)
    elif page == "📈 Analytics":
        render_analytics(df)
    elif page == "💰 Financial Analysis":
        render_financial_analysis(df)
    elif page == "🏆 Benchmarking":
        render_benchmarking(df)
    elif page == "🔬 Root Cause":
        render_root_cause(df)
    elif page == "🚀 Advanced Insights":
        render_advanced_insights(df)
    elif page == "🔍 Drift Detection":
        render_drift_page(df)
    elif page == "⚠️ Alerts":
        render_alerts_page(df)
    elif page == "🔮 What-If Simulator":
        render_whatif_simulator(df)
    elif page == "📋 Action Tracker":
        render_action_tracker(df)
    elif page == "📽️ Presentation Mode":
        render_presentation_mode(df)


# ============================================================================
# ADVANCED INSIGHTS PAGE
# ============================================================================

def render_advanced_insights(df):
    """Render the Advanced Insights page with high-value visualizations."""
    st.markdown('<p class="main-header">🚀 Advanced Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Strategic visualizations for executive decision-making</p>', unsafe_allow_html=True)
    
    # Create tabs for different insight categories
    tabs = st.tabs(["📊 SLA & Aging", "👥 Engineer Efficiency", "💰 Cost Analysis", "🔄 Patterns"])
    
    with tabs[0]:
        st.markdown("### SLA Compliance & Ticket Aging Analysis")
        st.markdown("*Track resolution performance against service level agreements*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(chart_sla_funnel(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(chart_aging_analysis(df), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Time Pattern Analysis")
        st.markdown("*Identify peak escalation times and shift handoff issues*")
        st.plotly_chart(chart_time_heatmap(df), use_container_width=True)
    
    with tabs[1]:
        st.markdown("### Engineer Efficiency Quadrant")
        st.markdown("*Speed vs Quality: Identify top performers and those needing support*")
        
        st.plotly_chart(chart_engineer_quadrant(df), use_container_width=True)
        
        # Quadrant legend explanation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(40,167,69,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #28A745;">
                <strong>⭐ Fast & Clean</strong><br>
                <small>Low resolution time, low recurrence. Top performers.</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(23,162,184,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #17A2B8;">
                <strong>🐢 Slow but Thorough</strong><br>
                <small>Higher resolution time, but quality work.</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: rgba(255,193,7,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107;">
                <strong>⚡ Fast but Sloppy</strong><br>
                <small>Quick fixes that may recur. Need coaching.</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: rgba(220,53,69,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #DC3545;">
                <strong>🆘 Needs Support</strong><br>
                <small>Slow and issues recur. Priority for training.</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Resolution Consistency")
        st.markdown("*Categories with high variability may indicate process gaps or training needs*")
        st.plotly_chart(chart_resolution_consistency(df), use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Cost Avoidance Waterfall")
        st.markdown("*Path from current costs to achievable target through strategic interventions*")
        
        st.plotly_chart(chart_cost_waterfall(df), use_container_width=True)
        
        # Cost insights
        st.markdown("---")
        st.markdown("### 💡 Cost Reduction Opportunities")
        
        total_cost = df['Financial_Impact'].sum() if 'Financial_Impact' in df.columns else len(df) * 850
        recurrence_rate = df['AI_Recurrence_Probability'].mean() if 'AI_Recurrence_Probability' in df.columns else 0.2
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recurrence_savings = total_cost * recurrence_rate * 0.5
            st.markdown(f"""
            <div class="kpi-container success">
                <h3 style="color: #28A745;">${recurrence_savings/1000:.0f}K</h3>
                <p>Recurrence Prevention</p>
                <small>50% of recurring issue costs</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            resolution_savings = total_cost * 0.15
            st.markdown(f"""
            <div class="kpi-container success">
                <h3 style="color: #28A745;">${resolution_savings/1000:.0f}K</h3>
                <p>Faster Resolution</p>
                <small>15% from reduced handling time</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            process_savings = total_cost * 0.05
            st.markdown(f"""
            <div class="kpi-container success">
                <h3 style="color: #28A745;">${process_savings/1000:.0f}K</h3>
                <p>Process Improvement</p>
                <small>5% from automation & efficiency</small>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("### Operational Health Score")
        st.markdown("*Composite score based on recurrence, resolution time, and critical issues*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(chart_health_gauge(df), use_container_width=True)
        
        with col2:
            # Health score breakdown
            recurrence_rate = df['AI_Recurrence_Probability'].mean() * 100 if 'AI_Recurrence_Probability' in df.columns else 15
            resolution_days = df['Predicted_Resolution_Days'].mean() if 'Predicted_Resolution_Days' in df.columns else 5
            critical_pct = (df['tickets_data_severity'] == 'Critical').mean() * 100 if 'tickets_data_severity' in df.columns else 10
            
            st.markdown("#### Score Components")
            st.markdown(f"""
            | Component | Value | Impact |
            |-----------|-------|--------|
            | Recurrence Rate | {recurrence_rate:.1f}% | -{ recurrence_rate * 1.5:.0f} pts |
            | Resolution Time | {resolution_days:.1f} days | -{resolution_days * 5:.0f} pts |
            | Critical Issues | {critical_pct:.1f}% | -{critical_pct * 0.5:.0f} pts |
            """)
            
            st.info("💡 **Tip:** Focus on reducing recurrence rate for the biggest health score improvement.")
        
        st.markdown("---")
        st.markdown("### Category to Recurrence Flow")
        st.markdown("*Which categories are driving high-risk outcomes?*")
        st.plotly_chart(chart_recurrence_patterns(df), use_container_width=True)


def render_dashboard(df):
    """Render the main dashboard page."""
    st.markdown('<p class="main-header">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time escalation intelligence at a glance</p>', unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Count by ticket type
    type_col = 'tickets_data_type_1'
    total = len(df)
    escalations_count = len(df[df[type_col].astype(str).str.contains('Escalation', case=False, na=False)]) if type_col in df.columns else total
    concerns_count = len(df[df[type_col].astype(str).str.contains('Concern', case=False, na=False)]) if type_col in df.columns else 0
    lessons_count = len(df[df[type_col].astype(str).str.contains('Lesson', case=False, na=False)]) if type_col in df.columns else 0
    
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{total:,}</p>
            <p class="kpi-label">Total Records</p>
            <p class="kpi-delta" style="font-size: 0.7rem; color: #888;">📋 {escalations_count} Escalations | ⚠️ {concerns_count} Concerns | 📚 {lessons_count} Lessons</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical = len(df[df['tickets_data_severity'] == 'Critical'])
        critical_pct = (critical / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div class="kpi-container critical">
            <p class="kpi-value">{critical}</p>
            <p class="kpi-label">Critical Issues</p>
            <p class="kpi-delta delta-up">{critical_pct:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_friction = df['Strategic_Friction_Score'].mean()
        st.markdown(f"""
        <div class="kpi-container warning">
            <p class="kpi-value">{avg_friction:.0f}</p>
            <p class="kpi-label">Avg Friction Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_days = df['Predicted_Resolution_Days'].mean()
        st.markdown(f"""
        <div class="kpi-container success">
            <p class="kpi-value">{avg_days:.1f}</p>
            <p class="kpi-label">Avg Resolution (Days)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(chart_trend_timeline(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(chart_severity_distribution(df), use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(chart_friction_by_category(df), use_container_width=True)
    
    with col2:
        st.plotly_chart(chart_recurrence_risk(df), use_container_width=True)


def render_analytics(df):
    """Render the analytics page with detailed charts."""
    st.markdown('<p class="main-header">📈 Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep dive into escalation patterns and performance</p>', unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Categories", "👥 Engineers", "📊 Distributions", "💰 Financial"])

    with tabs[0]:
        # Sub-tabs for different category views
        cat_tabs = st.tabs(["📊 Overview", "🔍 Drill-Down", "📈 Treemap", "📋 Details"])

        with cat_tabs[0]:
            # Overview - Sunburst and friction chart
            st.plotly_chart(chart_category_sunburst(df), use_container_width=True)
            st.plotly_chart(chart_friction_by_category(df), use_container_width=True)

        with cat_tabs[1]:
            # Drill-down - Category selector with sub-category breakdown
            st.markdown("### Sub-Category Drill-Down")
            st.markdown("Select a category to view detailed sub-category breakdown")

            # Category selector
            categories = ['All Categories'] + sorted(df['AI_Category'].unique().tolist())
            selected_cat = st.selectbox("Select Category", categories, key="cat_drilldown_select")

            if selected_cat == 'All Categories':
                # Show overall sub-category breakdown
                st.plotly_chart(chart_subcategory_breakdown(df, None), use_container_width=True)
            else:
                # Show breakdown for selected category
                st.plotly_chart(chart_subcategory_breakdown(df, selected_cat), use_container_width=True)

            # Sub-category comparison table
            st.markdown("### Sub-Category Comparison")
            comparison_df = chart_subcategory_comparison_table(df)
            if not comparison_df.empty:
                if selected_cat != 'All Categories':
                    comparison_df = comparison_df[comparison_df['Category'] == selected_cat]
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        with cat_tabs[2]:
            # Treemap view
            st.markdown("### Category Treemap")
            st.markdown("*Click on categories to drill down into sub-categories*")
            st.plotly_chart(chart_category_treemap(df), use_container_width=True)

        with cat_tabs[3]:
            # Detailed category statistics
            st.markdown("### Category Statistics")

            # Summary table
            if 'AI_Sub_Category' in df.columns:
                cat_stats = df.groupby('AI_Category').agg({
                    'AI_Sub_Category': 'count',
                    'AI_Confidence': 'mean',
                    'Strategic_Friction_Score': 'sum' if 'Strategic_Friction_Score' in df.columns else 'count'
                }).round(2)
                cat_stats.columns = ['Ticket Count', 'Avg Confidence', 'Total Friction']
                cat_stats = cat_stats.sort_values('Ticket Count', ascending=False)

                # Add financial if available
                if 'Financial_Impact' in df.columns:
                    fin_stats = df.groupby('AI_Category')['Financial_Impact'].sum()
                    cat_stats['Total Impact'] = fin_stats
                    cat_stats['Total Impact'] = cat_stats['Total Impact'].apply(lambda x: f"${x:,.0f}")

                st.dataframe(cat_stats, use_container_width=True)

                # Sub-category distribution within each category
                st.markdown("### Sub-Category Distribution")
                for cat in df['AI_Category'].unique():
                    with st.expander(f"📁 {cat}"):
                        cat_df = df[df['AI_Category'] == cat]
                        sub_counts = cat_df['AI_Sub_Category'].value_counts()

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            fig = go.Figure(go.Bar(
                                x=sub_counts.values,
                                y=sub_counts.index,
                                orientation='h',
                                marker_color='#0066CC'
                            ))
                            fig.update_layout(
                                **create_plotly_theme(),
                                height=200,
                                margin=dict(l=10, r=10, t=10, b=10),
                                yaxis_title='',
                                xaxis_title='Count'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.metric("Total Tickets", len(cat_df))
                            if 'Financial_Impact' in df.columns:
                                st.metric("Total Impact", f"${cat_df['Financial_Impact'].sum():,.0f}")
            else:
                st.info("Sub-category data not available. Run classification with the updated system.")
    
    with tabs[1]:
        fig = chart_engineer_performance(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Engineer data not available.")
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_resolution_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(chart_recurrence_risk(df), use_container_width=True)
    
    with tabs[3]:
        if 'Financial_Impact' in df.columns:
            # Sub-tabs for financial drill-down
            fin_tabs = st.tabs(["📊 By Category", "🔍 Sub-Category Drill-Down", "📋 Summary Table"])

            with fin_tabs[0]:
                # Financial by category
                fin_by_cat = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False)

                fig = go.Figure(go.Bar(
                    x=fin_by_cat.index,
                    y=fin_by_cat.values,
                    marker=dict(
                        color=fin_by_cat.values,
                        colorscale='Reds'
                    ),
                    text=[f"${v/1000:.0f}K" for v in fin_by_cat.values],
                    textposition='outside'
                ))

                # Get theme without margin
                theme = create_plotly_theme()
                theme.pop('margin', None)

                fig.update_layout(
                    **theme,
                    title='Financial Impact by Category',
                    xaxis_tickangle=-45,
                    height=400,
                    margin=dict(l=40, r=60, t=60, b=100)  # Room for labels
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Impact", f"${df['Financial_Impact'].sum():,.0f}")
                with col2:
                    st.metric("Average per Ticket", f"${df['Financial_Impact'].mean():,.0f}")
                with col3:
                    st.metric("Max Single Ticket", f"${df['Financial_Impact'].max():,.0f}")

            with fin_tabs[1]:
                # Sub-category financial drill-down
                st.markdown("### Financial Impact by Sub-Category")
                st.markdown("*Click on categories in the chart to drill down*")

                # Financial drill-down chart
                st.plotly_chart(chart_category_financial_drilldown(df), use_container_width=True)

                # Sub-category breakdown if available
                if 'AI_Sub_Category' in df.columns:
                    st.markdown("### Sub-Category Cost Breakdown")

                    # Category selector for detailed view
                    categories = ['All Categories'] + sorted(df['AI_Category'].unique().tolist())
                    selected_cat = st.selectbox("Select Category for Details", categories, key="fin_cat_select")

                    if selected_cat == 'All Categories':
                        subcat_fin = df.groupby('AI_Sub_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])
                    else:
                        subcat_fin = df[df['AI_Category'] == selected_cat].groupby('AI_Sub_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])

                    subcat_fin.columns = ['Total', 'Average', 'Count']
                    subcat_fin = subcat_fin.sort_values('Total', ascending=False)

                    # Bar chart
                    fig = go.Figure(go.Bar(
                        x=subcat_fin['Total'],
                        y=subcat_fin.index,
                        orientation='h',
                        marker=dict(
                            color=subcat_fin['Total'],
                            colorscale='Reds'
                        ),
                        text=[f"${v:,.0f}" for v in subcat_fin['Total']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Total: $%{x:,.0f}<extra></extra>'
                    ))

                    theme = create_plotly_theme()
                    theme.pop('margin', None)

                    fig.update_layout(
                        **theme,
                        title=f'Financial Impact: {selected_cat}',
                        height=max(300, len(subcat_fin) * 35),
                        margin=dict(l=200, r=80, t=60, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with fin_tabs[2]:
                # Summary table
                st.markdown("### Financial Summary Table")

                if 'AI_Sub_Category' in df.columns:
                    summary = df.groupby(['AI_Category', 'AI_Sub_Category']).agg({
                        'Financial_Impact': ['sum', 'mean', 'count']
                    }).round(2)
                    summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
                    summary = summary.reset_index()
                    summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f"${x:,.0f}")
                    summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f"${x:,.0f}")
                    summary = summary.sort_values(['AI_Category', 'Ticket Count'], ascending=[True, False])
                else:
                    summary = df.groupby('AI_Category').agg({
                        'Financial_Impact': ['sum', 'mean', 'count']
                    }).round(2)
                    summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
                    summary = summary.reset_index()
                    summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f"${x:,.0f}")
                    summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f"${x:,.0f}")

                st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.info("Financial impact data not available.")


if __name__ == "__main__":
    main()
