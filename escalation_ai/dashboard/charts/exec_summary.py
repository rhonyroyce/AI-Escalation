"""
Escalation AI Executive Summary — SCR Format
=============================================
Mirrors the Pulse Executive Summary page pattern:
- Situation: Volume overview and current state
- Complication: Risk factors and friction scores
- Resolution: AI-generated recommendations and predictions
"""
import streamlit as st
import pandas as pd
from shared_theme import COLORS


def render_exec_kpis(df: pd.DataFrame):
    """Render hero KPI cards for escalation overview."""
    cols = st.columns(5)

    with cols[0]:
        st.markdown(f"""
        <div class="shared-kpi">
            <div class="value">{len(df):,}</div>
            <div class="label">Total Tickets</div>
        </div>""", unsafe_allow_html=True)

    with cols[1]:
        critical = len(df[df['tickets_data_severity'] == 'Critical']) if 'tickets_data_severity' in df.columns else 0
        st.markdown(f"""
        <div class="shared-kpi" style="border-left-color:{COLORS['status_red']};">
            <div class="value" style="background:linear-gradient(135deg,#ef4444,#dc2626);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{critical}</div>
            <div class="label">Critical Severity</div>
        </div>""", unsafe_allow_html=True)

    with cols[2]:
        if 'Strategic_Friction_Score' in df.columns:
            avg_sfs = df['Strategic_Friction_Score'].mean()
            st.markdown(f"""
            <div class="shared-kpi" style="border-left-color:{COLORS['chart_amber']};">
                <div class="value" style="background:linear-gradient(135deg,#f59e0b,#d97706);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{avg_sfs:.1f}</div>
                <div class="label">Avg Friction Score</div>
            </div>""", unsafe_allow_html=True)

    with cols[3]:
        if 'Financial_Impact' in df.columns:
            total_cost = df['Financial_Impact'].sum()
            st.markdown(f"""
            <div class="shared-kpi" style="border-left-color:#22c55e;">
                <div class="value" style="background:linear-gradient(135deg,#22c55e,#16a34a);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">${total_cost:,.0f}</div>
                <div class="label">Financial Impact</div>
            </div>""", unsafe_allow_html=True)

    with cols[4]:
        if 'Recurrence_Probability' in df.columns:
            high_recur = len(df[df['Recurrence_Probability'] > 0.7])
            st.markdown(f"""
            <div class="shared-kpi" style="border-left-color:#8b5cf6;">
                <div class="value" style="background:linear-gradient(135deg,#8b5cf6,#7c3aed);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{high_recur}</div>
                <div class="label">High Recurrence Risk</div>
            </div>""", unsafe_allow_html=True)


def render_scr_narrative(df: pd.DataFrame):
    """Render Situation-Complication-Resolution narrative."""
    n_tickets = len(df)
    n_markets = df['ticket_market_name'].nunique() if 'ticket_market_name' in df.columns else 0
    n_critical = len(df[df['tickets_data_severity'] == 'Critical']) if 'tickets_data_severity' in df.columns else 0

    # Situation
    st.markdown(f"""
    <div style="background:rgba(59,130,246,0.08);border-left:4px solid #3b82f6;
        border-radius:0 8px 8px 0;padding:16px 20px;margin:16px 0;">
        <div style="color:#3b82f6;font-weight:600;font-size:0.8rem;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:8px;">Situation</div>
        <p style="color:#e2e8f0;margin:0;">
            The escalation portfolio contains <strong>{n_tickets:,} tickets</strong> across
            <strong>{n_markets} markets</strong>, with <strong>{n_critical} critical-severity</strong>
            issues requiring immediate attention.</p>
    </div>
    """, unsafe_allow_html=True)

    # Complication
    if 'Strategic_Friction_Score' in df.columns:
        high_friction = len(df[df['Strategic_Friction_Score'] > df['Strategic_Friction_Score'].quantile(0.75)])
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.08);border-left:4px solid #f59e0b;
            border-radius:0 8px 8px 0;padding:16px 20px;margin:16px 0;">
            <div style="color:#f59e0b;font-weight:600;font-size:0.8rem;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:8px;">Complication</div>
            <p style="color:#e2e8f0;margin:0;">
                <strong>{high_friction} tickets</strong> are in the top quartile of Strategic Friction Scores,
                indicating systemic process issues that compound across LOBs and geographies.</p>
        </div>
        """, unsafe_allow_html=True)

    # Resolution
    st.markdown("""
    <div style="background:rgba(34,197,94,0.08);border-left:4px solid #22c55e;
        border-radius:0 8px 8px 0;padding:16px 20px;margin:16px 0;">
        <div style="color:#22c55e;font-weight:600;font-size:0.8rem;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:8px;">Resolution</div>
        <p style="color:#e2e8f0;margin:0;">
            Use the Deep Analysis page to identify root cause clusters, the Financial Intelligence page
            to quantify cost exposure, and the Planning & Actions page to track remediation progress.</p>
    </div>
    """, unsafe_allow_html=True)
