"""Deep Analysis — Engineers tab.

Renders engineer performance matrix (friction vs volume quadrant) and
engineer comparison / workload distribution.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Pure figure builders
# ---------------------------------------------------------------------------

def engineer_quadrant(df: pd.DataFrame) -> go.Figure | None:
    """Scatter plot classifying engineers into performance quadrants."""
    eng_col = (
        'tickets_data_engineer_name'
        if 'tickets_data_engineer_name' in df.columns
        else 'tickets_data_issue_resolved_by'
    )
    if eng_col not in df.columns or 'Predicted_Resolution_Days' not in df.columns:
        return None

    eng_metrics = df.groupby(eng_col).agg({
        'Predicted_Resolution_Days': 'mean',
        'Financial_Impact': ['sum', 'count'],
        'AI_Recurrence_Risk': 'mean'
    }).reset_index()
    eng_metrics.columns = ['Engineer', 'Avg_Resolution', 'Total_Cost', 'Ticket_Count', 'Recurrence_Risk']
    eng_metrics = eng_metrics[eng_metrics['Ticket_Count'] >= 3]

    if eng_metrics.empty:
        return None

    avg_res_median = eng_metrics['Avg_Resolution'].median()
    avg_risk_median = eng_metrics['Recurrence_Risk'].median()

    def _quadrant(row):
        fast = row['Avg_Resolution'] <= avg_res_median
        quality = row['Recurrence_Risk'] <= avg_risk_median
        if fast and quality:
            return 'Fast & Clean'
        if not fast and quality:
            return 'Slow but Thorough'
        if fast and not quality:
            return 'Fast but Sloppy'
        return 'Needs Support'

    eng_metrics['Quadrant'] = eng_metrics.apply(_quadrant, axis=1)
    quadrant_colors = {
        'Fast & Clean': '#22c55e',
        'Slow but Thorough': '#3b82f6',
        'Fast but Sloppy': '#f97316',
        'Needs Support': '#ef4444',
    }

    fig = go.Figure()
    for quadrant, color in quadrant_colors.items():
        q_data = eng_metrics[eng_metrics['Quadrant'] == quadrant]
        if len(q_data) > 0:
            fig.add_trace(go.Scatter(
                x=q_data['Avg_Resolution'],
                y=q_data['Recurrence_Risk'] * 100,
                mode='markers+text',
                name=quadrant,
                text=q_data['Engineer'].str.split().str[0],
                textposition='top center',
                marker=dict(size=q_data['Ticket_Count'] * 2, color=color, opacity=0.7),
                hovertemplate='<b>%{text}</b><br>Resolution: %{x:.1f}d<br>Recurrence: %{y:.1f}%<extra></extra>'
            ))

    fig.add_vline(x=avg_res_median, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_hline(y=avg_risk_median * 100, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        title=dict(text='Engineer Performance Quadrant', font=dict(size=14, color='#e2e8f0')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis=dict(title='Avg Resolution (days)', gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(title='Recurrence Risk (%)', gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='#94a3b8', size=10))
    )
    return fig


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

_QUADRANT_LEGEND = """
<div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; margin-top: 10px;">
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.8rem;">
        <div><span style="color: #22c55e;">●</span> <b>Fast & Clean:</b> High performers</div>
        <div><span style="color: #3b82f6;">●</span> <b>Slow but Thorough:</b> Quality focused</div>
        <div><span style="color: #f97316;">●</span> <b>Fast but Sloppy:</b> Speed over quality</div>
        <div><span style="color: #ef4444;">●</span> <b>Needs Support:</b> Training required</div>
    </div>
</div>
"""


def render_tab(df: pd.DataFrame) -> None:
    """Render the Engineers tab inside an already-active st.tabs context."""
    from escalation_ai.dashboard.streamlit_app import chart_engineer_performance

    col1, col2 = st.columns(2)
    with col1:
        with st.spinner("Generating visualization..."):
            st.plotly_chart(chart_engineer_performance(df), use_container_width=True)
    with col2:
        fig = engineer_quadrant(df)
        if fig:
            with st.spinner("Generating visualization..."):
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(_QUADRANT_LEGEND, unsafe_allow_html=True)
        else:
            st.info("Engineer performance data not available")
