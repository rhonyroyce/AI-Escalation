"""
Pulse Dashboard - Week-over-Week / Region Comparison
=====================================================
Side-by-side comparison of two weeks or two regions,
showing exec-level KPI cards and dimension-level deltas.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.sidebar import render_sidebar
from utils.styles import inject_css, SCORE_DIMENSIONS, get_pulse_color
from utils.wow_utils import get_previous_week

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

df = st.session_state.df
target = st.session_state.get('pulse_target', 17.0)

st.markdown('<p class="main-header">Comparison</p>', unsafe_allow_html=True)

# ── Mode selector ─────────────────────────────────────────────────────────
compare_mode = st.radio(
    "Compare", ["Weeks", "Regions"],
    horizontal=True, key="compare_mode", label_visibility="collapsed",
)


def _exec_card(label: str, subset: pd.DataFrame) -> str:
    """Build an HTML exec-card summarizing a subset."""
    if subset.empty:
        return f'<div class="glass-card" style="padding:16px;"><b>{label}</b><p>No data</p></div>'
    avg = subset['Total Score'].mean()
    n = len(subset)
    n_proj = subset['Project'].nunique()
    green = ((subset['Pulse_Status'] == 'Green') | (subset['Pulse_Status'] == 'Dark Green')).sum()
    red = (subset['Pulse_Status'] == 'Red').sum()
    green_pct = green / n * 100 if n else 0
    color = get_pulse_color(avg)
    return f"""
    <div class="glass-card" style="padding:16px;">
        <b style="color:#E0E0E0; font-size:1.05rem;">{label}</b>
        <div style="display:flex; gap:24px; margin-top:10px;">
            <div>
                <span style="font-size:1.8rem; font-weight:bold; color:{color};">{avg:.1f}</span>
                <span style="color:#94a3b8;"> / 24</span>
            </div>
            <div style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">
                {n_proj} projects, {n} entries<br>
                Green: {green_pct:.0f}% | Red: {red}
            </div>
        </div>
    </div>"""


def _dim_table(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str):
    """Render a dimension comparison table."""
    rows = []
    a_means = a[SCORE_DIMENSIONS].mean() if not a.empty else pd.Series(0, index=SCORE_DIMENSIONS)
    b_means = b[SCORE_DIMENSIONS].mean() if not b.empty else pd.Series(0, index=SCORE_DIMENSIONS)
    for dim in SCORE_DIMENSIONS:
        av = a_means.get(dim, 0)
        bv = b_means.get(dim, 0)
        delta = av - bv
        trend = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "→")
        rows.append({
            'Dimension': dim,
            label_a: f"{av:.2f}",
            label_b: f"{bv:.2f}",
            'Delta': f"{delta:+.2f}",
            'Trend': trend,
        })
    return pd.DataFrame(rows)


# ============================================================================
# WEEK COMPARISON
# ============================================================================
if compare_mode == "Weeks":
    years = sorted(df['Year'].dropna().unique())
    all_weeks = sorted(df['Wk'].dropna().unique())

    sel_year = st.session_state.get('selected_year', years[-1] if years else None)

    if sel_year:
        yr_weeks = sorted(df[df['Year'] == sel_year]['Wk'].dropna().unique())
    else:
        yr_weeks = all_weeks

    col1, col2 = st.columns(2)
    with col1:
        wk_a = st.selectbox("Week A", yr_weeks, index=len(yr_weeks) - 1, key="cmp_wk_a")
    with col2:
        prev_wk, _ = get_previous_week(wk_a, sel_year) if wk_a else (yr_weeks[0], sel_year)
        default_b = yr_weeks.index(prev_wk) if prev_wk in yr_weeks else 0
        wk_b = st.selectbox("Week B", yr_weeks, index=default_b, key="cmp_wk_b")

    subset_a = df[(df['Year'] == sel_year) & (df['Wk'] == wk_a)]
    subset_b = df[(df['Year'] == sel_year) & (df['Wk'] == wk_b)]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_exec_card(f"Week {wk_a}", subset_a), unsafe_allow_html=True)
    with c2:
        st.markdown(_exec_card(f"Week {wk_b}", subset_b), unsafe_allow_html=True)

    st.markdown("### Dimension Comparison")
    dim_df = _dim_table(subset_a, subset_b, f"Wk {wk_a}", f"Wk {wk_b}")
    st.dataframe(dim_df, use_container_width=True, hide_index=True)

    # Project-level delta table
    st.markdown("### Project Movers")
    if not subset_a.empty and not subset_b.empty:
        a_proj = subset_a.groupby('Project')['Total Score'].mean()
        b_proj = subset_b.groupby('Project')['Total Score'].mean()
        proj_delta = (a_proj - b_proj).dropna().sort_values()
        movers = pd.DataFrame({
            'Project': proj_delta.index,
            f'Wk {wk_a}': [f"{a_proj.get(p, 0):.1f}" for p in proj_delta.index],
            f'Wk {wk_b}': [f"{b_proj.get(p, 0):.1f}" for p in proj_delta.index],
            'Delta': [f"{d:+.1f}" for d in proj_delta.values],
        })
        st.dataframe(movers, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient data for project comparison.")

# ============================================================================
# REGION COMPARISON
# ============================================================================
else:
    regions = sorted(filtered_df['Region'].dropna().unique())
    if len(regions) < 2:
        st.info("Need at least 2 regions for comparison. Adjust sidebar filters.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        reg_a = st.selectbox("Region A", regions, index=0, key="cmp_reg_a")
    with col2:
        default_b = 1 if len(regions) > 1 else 0
        reg_b = st.selectbox("Region B", regions, index=default_b, key="cmp_reg_b")

    subset_a = filtered_df[filtered_df['Region'] == reg_a]
    subset_b = filtered_df[filtered_df['Region'] == reg_b]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_exec_card(reg_a, subset_a), unsafe_allow_html=True)
    with c2:
        st.markdown(_exec_card(reg_b, subset_b), unsafe_allow_html=True)

    st.markdown("### Dimension Comparison")
    dim_df = _dim_table(subset_a, subset_b, reg_a, reg_b)
    st.dataframe(dim_df, use_container_width=True, hide_index=True)
