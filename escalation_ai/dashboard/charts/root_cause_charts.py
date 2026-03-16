"""Deep Analysis — Root Cause tab.

Renders Pareto analysis, driver tree, root cause quantification (financial
impact by category), and risk heatmap.
"""

import plotly.graph_objects as go
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Pure figure builders
# ---------------------------------------------------------------------------

def financial_impact_by_category(df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar chart of financial impact per AI_Category."""
    if 'AI_Category' not in df.columns or 'Financial_Impact' not in df.columns:
        return None

    impact_data = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=True)

    fig = go.Figure(data=[go.Bar(
        y=impact_data.index,
        x=impact_data.values,
        orientation='h',
        marker_color='#ef4444',
        text=[f'${v:,.0f}' for v in impact_data.values],
        textposition='outside'
    )])
    fig.update_layout(
        title="Financial Impact by Root Cause",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=10, r=80, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='$,.0f'),
        yaxis=dict(showgrid=False)
    )
    return fig


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_tab(df: pd.DataFrame) -> None:
    """Render the Root Cause tab inside an already-active st.tabs context."""
    from escalation_ai.dashboard.streamlit_app import (
        chart_pareto_analysis, chart_driver_tree, chart_risk_heatmap,
    )
    from escalation_ai.dashboard.shared_helpers import render_chart_with_insight

    col1, col2 = st.columns(2)
    with col1:
        render_chart_with_insight('pareto_analysis', chart_pareto_analysis(df), df)
    with col2:
        with st.spinner("Generating visualization..."):
            st.plotly_chart(chart_driver_tree(df), use_container_width=True)

    st.markdown("#### 📊 Root Cause Impact Quantification")
    col3, col4 = st.columns(2)
    with col3:
        fig = financial_impact_by_category(df)
        if fig:
            with st.spinner("Generating visualization..."):
                st.plotly_chart(fig, use_container_width=True)
    with col4:
        render_chart_with_insight('risk_heatmap', chart_risk_heatmap(df), df)
