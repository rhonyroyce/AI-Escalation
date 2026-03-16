"""Deep Analysis — Patterns & SLA tab.

Renders SLA funnel, aging analysis, time heatmap, and recurrence patterns
from the advanced_insights module (with a fallback to basic charts).
"""

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_tab(df: pd.DataFrame) -> None:
    """Render the Patterns & SLA tab inside an already-active st.tabs context."""
    try:
        from escalation_ai.advanced_insights import (
            chart_sla_funnel, chart_aging_analysis, chart_time_heatmap,
            chart_recurrence_patterns,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📊 SLA Compliance Funnel")
            with st.spinner("Generating visualization..."):
                st.plotly_chart(chart_sla_funnel(df), use_container_width=True)
        with col2:
            st.markdown("##### ⏱️ Ticket Aging Analysis")
            with st.spinner("Generating visualization..."):
                st.plotly_chart(chart_aging_analysis(df), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### 🕐 Peak Escalation Times")
            with st.spinner("Generating visualization..."):
                st.plotly_chart(chart_time_heatmap(df), use_container_width=True)
        with col4:
            st.markdown("##### 🔄 Recurrence Patterns")
            with st.spinner("Generating visualization..."):
                st.plotly_chart(chart_recurrence_patterns(df), use_container_width=True)

    except ImportError:
        from escalation_ai.dashboard.streamlit_app import (
            chart_recurrence_risk, chart_resolution_distribution,
        )
        with st.spinner("Generating visualization..."):
            st.plotly_chart(chart_recurrence_risk(df), use_container_width=True)
        with st.spinner("Generating visualization..."):
            st.plotly_chart(chart_resolution_distribution(df), use_container_width=True)
