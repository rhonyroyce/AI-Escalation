"""
Escalation AI - Streamlit Dashboard

Modern, interactive dashboard with:
- Real-time KPI metrics
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Escalation AI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    margin: 10px 0;
}

/* KPI Cards */
.kpi-container {
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
    border-radius: 16px;
    padding: 24px;
    border-left: 4px solid #0066CC;
    text-align: center;
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

/* Main header */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.sub-header {
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 2rem;
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load the most recent analysis data."""
    # Look for processed Excel files
    data_files = list(Path(".").glob("**/Escalation_Analysis_*.xlsx"))
    
    if data_files:
        latest = max(data_files, key=lambda x: x.stat().st_mtime)
        try:
            df = pd.read_excel(latest, sheet_name="Detailed Analysis")
            return df, str(latest)
        except Exception as e:
            st.warning(f"Could not load {latest}: {e}")
    
    # Generate sample data
    return generate_sample_data(), "Sample Data"


def generate_sample_data():
    """Generate realistic sample data for demo."""
    np.random.seed(42)
    n = 250
    
    categories = [
        'RF & Antenna Issues', 'Transmission & Backhaul', 'Power & Environment',
        'Site Access & Logistics', 'Contractor & Vendor Issues', 
        'Configuration & Integration', 'OSS/NMS & Systems',
        'Process & Documentation', 'Communication & Coordination'
    ]
    
    engineers = ['Alice Chen', 'Bob Smith', 'Carlos Rodriguez', 'Diana Patel', 
                 'Eric Johnson', 'Fatima Ahmed', 'George Kim', 'Hannah Lee']
    
    lobs = ['Network Operations', 'Field Services', 'Customer Support', 
            'Infrastructure', 'Enterprise', 'Residential']
    
    # Generate dates over 90 days
    dates = pd.date_range(end=datetime.now(), periods=n, freq='8H')
    
    df = pd.DataFrame({
        'AI_Category': np.random.choice(categories, n, p=[0.18, 0.15, 0.12, 0.12, 0.11, 0.1, 0.08, 0.08, 0.06]),
        'Strategic_Friction_Score': np.clip(np.random.exponential(45, n), 5, 200),
        'AI_Recurrence_Risk': np.clip(np.random.beta(2, 5, n), 0, 1),
        'Predicted_Resolution_Days': np.clip(np.random.exponential(2.5, n), 0.5, 15),
        'tickets_data_severity': np.random.choice(['Critical', 'Major', 'Minor'], n, p=[0.15, 0.45, 0.40]),
        'tickets_data_escalation_origin': np.random.choice(['External', 'Internal'], n, p=[0.35, 0.65]),
        'tickets_data_issue_datetime': dates,
        'Engineer': np.random.choice(engineers, n),
        'LOB': np.random.choice(lobs, n),
        'Financial_Impact': np.clip(np.random.exponential(2500, n), 100, 25000),
    })
    
    return df

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
        height=400,
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
        height=350,
        showlegend=True,
        legend=dict(orientation='h', y=-0.1)
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
        height=350,
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='Friction Score', secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text='Count', secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def chart_recurrence_risk(df):
    """Gauge chart for average recurrence risk."""
    avg_risk = df['AI_Recurrence_Risk'].mean() * 100
    
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
        height=280
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
        height=300,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def chart_category_sunburst(df):
    """Interactive sunburst chart of categories and severity."""
    # Prepare data
    sunburst_data = df.groupby(['AI_Category', 'tickets_data_severity']).size().reset_index(name='count')
    
    fig = px.sunburst(
        sunburst_data,
        path=['AI_Category', 'tickets_data_severity'],
        values='count',
        color='count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Category & Severity Breakdown', font=dict(size=16)),
        height=450
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    )
    
    return fig


def chart_engineer_performance(df):
    """Horizontal bar chart of engineer friction."""
    if 'Engineer' not in df.columns:
        return None
    
    eng_stats = df.groupby('Engineer').agg({
        'Strategic_Friction_Score': 'mean',
        'AI_Recurrence_Risk': 'mean',
        'AI_Category': 'count'
    }).rename(columns={'AI_Category': 'ticket_count'}).sort_values('Strategic_Friction_Score')
    
    fig = go.Figure(go.Bar(
        x=eng_stats['Strategic_Friction_Score'],
        y=eng_stats.index,
        orientation='h',
        marker=dict(
            color=eng_stats['Strategic_Friction_Score'],
            colorscale='RdYlGn_r',
            line=dict(width=0)
        ),
        text=[f"{v:.0f} ({c} tickets)" for v, c in zip(eng_stats['Strategic_Friction_Score'], eng_stats['ticket_count'])],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Avg Friction: %{x:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        **create_plotly_theme(),
        title=dict(text='Engineer Performance (Avg Friction)', font=dict(size=16)),
        xaxis_title='Average Friction Score',
        height=350
    )
    
    return fig

# ============================================================================
# WHAT-IF SIMULATOR
# ============================================================================

def render_whatif_simulator(df):
    """Render the What-If Scenario Simulator page."""
    st.markdown("### 🔮 What-If Scenario Simulator")
    st.markdown("Adjust parameters to simulate impact on escalation metrics.")
    
    # Current baseline metrics
    baseline = {
        'avg_resolution': df['Predicted_Resolution_Days'].mean(),
        'recurrence_rate': df['AI_Recurrence_Risk'].mean() * 100,
        'monthly_friction': df['Strategic_Friction_Score'].sum() / 3,  # Assume 3 months data
        'monthly_cost': df['Financial_Impact'].sum() / 3 if 'Financial_Impact' in df.columns else 125000
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
    st.markdown("### 📊 Category Drift Detection")
    st.markdown("Analyze how escalation patterns are changing over time.")
    
    # Split data into baseline and recent
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime'])
    df_temp = df_temp.sort_values('date')
    
    split_idx = int(len(df_temp) * 0.6)
    baseline_df = df_temp.iloc[:split_idx]
    current_df = df_temp.iloc[split_idx:]
    
    # Calculate distributions
    baseline_dist = baseline_df['AI_Category'].value_counts(normalize=True)
    current_dist = current_df['AI_Category'].value_counts(normalize=True)
    
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
    
    # Calculate current metrics
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime']).dt.date
    daily = df_temp.groupby('date').agg({
        'Strategic_Friction_Score': 'sum',
        'AI_Category': 'count',
        'AI_Recurrence_Risk': 'mean'
    }).rename(columns={'AI_Category': 'count'})
    
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
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🚀 Escalation AI")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "📈 Analytics", "🔍 Drift Detection", 
             "⚠️ Alerts", "🔮 What-If Simulator"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Data info
        df, data_source = load_data()
        st.markdown(f"**Data Source:**")
        st.caption(data_source)
        st.markdown(f"**Records:** {len(df):,}")
        
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
    
    # Main content
    if page == "📊 Dashboard":
        render_dashboard(df)
    elif page == "📈 Analytics":
        render_analytics(df)
    elif page == "🔍 Drift Detection":
        render_drift_page(df)
    elif page == "⚠️ Alerts":
        render_alerts_page(df)
    elif page == "🔮 What-If Simulator":
        render_whatif_simulator(df)


def render_dashboard(df):
    """Render the main dashboard page."""
    st.markdown('<p class="main-header">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time escalation intelligence at a glance</p>', unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(df)
        st.markdown(f"""
        <div class="kpi-container">
            <p class="kpi-value">{total:,}</p>
            <p class="kpi-label">Total Escalations</p>
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
        st.plotly_chart(chart_category_sunburst(df), use_container_width=True)
        st.plotly_chart(chart_friction_by_category(df), use_container_width=True)
    
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
            
            fig.update_layout(
                **create_plotly_theme(),
                title='Financial Impact by Category',
                xaxis_tickangle=-45,
                height=400
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
        else:
            st.info("Financial impact data not available.")


if __name__ == "__main__":
    main()
