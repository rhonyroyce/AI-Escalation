"""
Pulse Dashboard - Prioritization (2x2 Matrix, Quick Wins, Pareto)

This page helps leadership decide WHERE to invest remediation effort
by ranking and segmenting projects along two key axes:
    - Impact: how far below target the project sits (bigger gap = more impact)
    - Effort: how many dimensions need fixing (more dimensions = more effort)

PURPOSE
-------
The classic consulting "2x2 matrix" segments projects into four quadrants:
    High Impact / Low Effort   = "Quick Wins" (fix these first!)
    High Impact / High Effort  = "Major Projects" (plan carefully)
    Low Impact  / Low Effort   = "Fill-Ins" (low priority)
    Low Impact  / High Effort  = "Thankless Tasks" (deprioritize)

Below the matrix, a "Quick Wins" panel highlights the top candidates,
and two Pareto charts identify which Areas and PMs account for the
majority of below-target score gaps.

SCORING SYSTEM CONTEXT
----------------------
Each project is rated across 8 dimensions (SCORE_DIMENSIONS):
    Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential

Each dimension is scored 0-3 (integer):
    0 = Critical / Escalation
    1 = Issues / Complaints
    2 = Acceptable / On-track
    3 = Exceptional / Positive

The Total Score ("Pulse Score") is the sum of all 8 dimensions (0-24).
Status thresholds:
    Red        :  1-13  (Critical)
    Yellow     : 14-15  (At Risk)
    Green      : 16-19  (On Track)
    Dark Green : 20-24  (Exceptional)

Impact is computed as: max(0, target - Total Score)
    A project AT or ABOVE target has zero impact (nothing to gain).
    A project 5 points below target has impact = 5.

Effort is computed externally (likely the count of dimensions scoring
below a threshold, e.g. how many dimensions scored 0 or 1).  This
column is expected to already exist on filtered_df as 'Effort'.

PAGE LAYOUT (top to bottom)
---------------------------
1. Impact-Effort 2x2 Matrix  - Scatter plot with quadrant labels
2. Quick Wins panel          - Cards for high-impact / low-effort projects
3. Pareto by Area            - Which areas contribute most to the gap
4. Pareto by PM              - Which PMs contribute most to the gap
"""

# ---------------------------------------------------------------------------
# PATH SETUP: Allow imports from the pulse_dashboard package root.
# This insert ensures `from utils.sidebar import ...` resolves correctly
# even when Streamlit runs this file directly as a "page".
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# UTILITY IMPORTS
# render_sidebar              - shared sidebar; returns the filtered DataFrame
# inject_css                  - injects the global dark-theme CSS
# SCORE_DIMENSIONS            - list of the 8 scoring dimension column names
# chart_impact_effort_matrix  - Plotly scatter: 2x2 impact vs effort matrix
# chart_pareto                - Plotly bar+line: Pareto chart for gap analysis
# ---------------------------------------------------------------------------
from utils.sidebar import render_sidebar
from utils.styles import inject_css, SCORE_DIMENSIONS
from utils.mckinsey_charts import chart_impact_effort_matrix, chart_pareto

# Inject the global dark-theme CSS into the Streamlit page
inject_css()

# ---------------------------------------------------------------------------
# SIDEBAR & DATA LOADING
# render_sidebar() draws the sidebar widgets and returns the filtered DataFrame.
# If no data is loaded or nothing matches the filters, bail out early.
# ---------------------------------------------------------------------------
filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

# Target pulse score: used to compute the "Impact" gap for each project
target = st.session_state.get('pulse_target', 17.0)

# Page title rendered as styled HTML
st.markdown('<p class="main-header">Prioritization</p>', unsafe_allow_html=True)

# ============================================================================
# 2x2 IMPACT-EFFORT MATRIX
#
# A scatter plot where each dot is a project:
#   X-axis = Effort (number of dimensions needing improvement)
#   Y-axis = Impact (gap between target and current Total Score)
#
# Quadrant lines are drawn at the median of each axis.  Projects in the
# upper-left quadrant (high impact, low effort) are the "quick wins".
# The chart_impact_effort_matrix function handles the quadrant labels,
# coloring, and hover tooltips.
# ============================================================================
fig = chart_impact_effort_matrix(filtered_df, target)
st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# QUICK WINS PANEL
#
# Filters the data to find projects in the "quick win" quadrant:
#   Impact >= median (above the midline = meaningful improvement potential)
#   Effort <  median (below the midline = fewer dimensions to fix)
#
# For each quick-win project, a card shows:
#   - Project name and region/area
#   - Current Total Score
#   - The single worst-scoring dimension (the one thing to fix first)
# ============================================================================
st.markdown("### Quick Wins")
st.markdown("Projects with high impact potential and low effort (few dimensions to fix).")

# Compute Impact: how many points below target (clamped to zero minimum)
plot_df = filtered_df.copy()
plot_df['Impact'] = (target - plot_df['Total Score']).clip(lower=0)

# Determine quadrant boundaries using medians (with 0.5 floor to avoid
# degenerate cases where median is 0 and all projects end up in one quadrant)
impact_mid = max(plot_df['Impact'].median(), 0.5)
effort_mid = max(plot_df['Effort'].median(), 0.5)

# Filter to the "quick win" quadrant: high impact AND low effort
quick_wins = plot_df[(plot_df['Impact'] >= impact_mid) & (plot_df['Effort'] < effort_mid)]
# Sort by impact descending (biggest improvement opportunities first), limit to 10
quick_wins = quick_wins.sort_values('Impact', ascending=False).head(10)

if not quick_wins.empty:
    for _, row in quick_wins.iterrows():
        # Identify the single worst dimension for this project (the "fix this first" action)
        dim_scores = {d: row[d] for d in SCORE_DIMENSIONS}
        worst_dim = min(dim_scores, key=dim_scores.get)
        worst_score = dim_scores[worst_dim]

        # Render a card for each quick-win project
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
#
# Pareto chart (bar + cumulative line) showing which Areas contribute the
# most to the total below-target gap.  The 80/20 rule typically applies:
# a small number of areas account for most of the portfolio's gap.
# This helps leadership focus area-level interventions.
# ============================================================================
st.markdown("### Pareto: Below-Target by Area")
fig_pareto = chart_pareto(filtered_df, target, groupby_col='Area')
st.plotly_chart(fig_pareto, use_container_width=True)

# ============================================================================
# PARETO BY PM
#
# Same Pareto analysis but grouped by PM Name.  Identifies which project
# managers are responsible for the largest share of below-target gaps.
# Useful for coaching / resource reallocation decisions.
# ============================================================================
st.markdown("### Pareto: Below-Target by PM")
fig_pareto_pm = chart_pareto(filtered_df, target, groupby_col='PM Name')
st.plotly_chart(fig_pareto_pm, use_container_width=True)
