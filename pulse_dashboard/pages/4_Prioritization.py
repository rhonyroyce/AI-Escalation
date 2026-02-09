"""
Pulse Dashboard - Prioritization (2x2 Matrix, Quick Wins, Pareto)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from utils.sidebar import render_sidebar
from utils.styles import inject_css, SCORE_DIMENSIONS
from utils.mckinsey_charts import chart_impact_effort_matrix, chart_pareto

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

target = st.session_state.get('pulse_target', 17.0)

st.markdown('<p class="main-header">Prioritization</p>', unsafe_allow_html=True)

# ============================================================================
# 2x2 IMPACT-EFFORT MATRIX
# ============================================================================
fig = chart_impact_effort_matrix(filtered_df, target)
st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# QUICK WINS PANEL
# ============================================================================
st.markdown("### Quick Wins")
st.markdown("Projects with high impact potential and low effort (few dimensions to fix).")

plot_df = filtered_df.copy()
plot_df['Impact'] = (target - plot_df['Total Score']).clip(lower=0)

impact_mid = max(plot_df['Impact'].median(), 0.5)
effort_mid = max(plot_df['Effort'].median(), 0.5)

quick_wins = plot_df[(plot_df['Impact'] >= impact_mid) & (plot_df['Effort'] < effort_mid)]
quick_wins = quick_wins.sort_values('Impact', ascending=False).head(10)

if not quick_wins.empty:
    for _, row in quick_wins.iterrows():
        # Find the primary dimension to fix
        dim_scores = {d: row[d] for d in SCORE_DIMENSIONS}
        worst_dim = min(dim_scores, key=dim_scores.get)
        worst_score = dim_scores[worst_dim]

        st.markdown(f"""
        <div class="glass-card" style="padding: 12px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <b style="color: #E0E0E0;">{row['Project']}</b>
                    <span style="color: #94a3b8; margin-left: 12px;">{row['Region']} / {row['Area']}</span>
                </div>
                <div style="text-align: right;">
                    <span class="badge badge-warning">Score: {row['Total Score']}</span>
                    <span style="color: #94a3b8; margin-left: 8px;">Fix: <b style="color: #f59e0b;">{worst_dim}</b> ({worst_score}/3)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No quick wins identified — all projects are either on track or require multi-dimension effort.")

# ============================================================================
# PARETO BY AREA
# ============================================================================
st.markdown("### Pareto: Below-Target by Area")
fig_pareto = chart_pareto(filtered_df, target, groupby_col='Area')
st.plotly_chart(fig_pareto, use_container_width=True)

# ============================================================================
# PARETO BY PM
# ============================================================================
st.markdown("### Pareto: Below-Target by PM")
fig_pareto_pm = chart_pareto(filtered_df, target, groupby_col='PM Name')
st.plotly_chart(fig_pareto_pm, use_container_width=True)
