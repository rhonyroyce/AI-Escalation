"""
Pulse Dashboard - Project Details (Filterable Table + Deep Dive)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.sidebar import render_sidebar
from utils.styles import inject_css, SCORE_DIMENSIONS, STATUS_CONFIG
from utils.mckinsey_charts import chart_radar, chart_project_trend

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

df = st.session_state.df
target = st.session_state.get('pulse_target', 17.0)

st.markdown('<p class="main-header">Project Details</p>', unsafe_allow_html=True)

# ============================================================================
# FILTERABLE DATA TABLE
# ============================================================================
display_cols = ['Project', 'PM Name', 'Region', 'Area', 'Total Score', 'Pulse_Status'] + SCORE_DIMENSIONS
available_cols = [c for c in display_cols if c in filtered_df.columns]

st.dataframe(
    filtered_df[available_cols].sort_values('Total Score', ascending=True),
    column_config={
        'Total Score': st.column_config.ProgressColumn(
            'Total Score',
            min_value=0, max_value=24,
            format='%d',
        ),
        'Project': st.column_config.TextColumn('Project', width='large'),
        'Pulse_Status': st.column_config.TextColumn('Status'),
    },
    use_container_width=True,
    hide_index=True,
    height=400,
)

# ============================================================================
# PROJECT DEEP DIVE
# ============================================================================
st.markdown("---")
st.markdown("### Project Deep Dive")

projects = sorted(filtered_df['Project'].dropna().unique())
selected_project = st.selectbox("Select a project for detailed view", projects, key="detail_project")

if selected_project:
    # Get all data for this project (current week + history)
    proj_current = filtered_df[filtered_df['Project'] == selected_project]
    proj_history = df[df['Project'] == selected_project]

    if proj_current.empty:
        st.info("No data for selected project in current filters.")
    else:
        latest_row = proj_current.iloc[0]

        # ── Status Badge ──
        status = latest_row['Pulse_Status']
        color = STATUS_CONFIG[status]['color']
        label = STATUS_CONFIG[status]['label']

        st.markdown(f"""
        <div class="exec-card" style="padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #E0E0E0; margin: 0;">{selected_project}</h3>
                    <p style="color: #94a3b8;">PM: {latest_row.get('PM Name', 'N/A')} | {latest_row.get('Region', '')} / {latest_row.get('Area', '')}</p>
                </div>
                <div>
                    <span class="badge" style="background: {color}; color: white; font-size: 1rem; padding: 8px 16px;">
                        {latest_row['Total Score']} / 24 — {label}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts ──
        col1, col2 = st.columns(2)

        with col1:
            fig_radar = chart_radar(latest_row)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            fig_trend = chart_project_trend(proj_history, selected_project, target)
            st.plotly_chart(fig_trend, use_container_width=True)

        # ── Text Fields ──
        st.markdown("#### Comments & Issues")

        text_fields = [
            ('Comments', 'comments_exp'),
            ('Pain Points', 'pain_exp'),
            ('Resolution Plan', 'resolution_exp'),
            ('Issue', 'issue_exp'),
            ('Pending Action', 'pending_exp'),
        ]

        for field_name, key in text_fields:
            if field_name in proj_current.columns:
                values = proj_current[field_name].dropna()
                if not values.empty:
                    with st.expander(f"{field_name} ({len(values)} entries)"):
                        for val in values:
                            st.markdown(f"- {val}")

        # ── Dimension Score Table ──
        st.markdown("#### Dimension Scores")
        dim_data = []
        for dim in SCORE_DIMENSIONS:
            score = latest_row.get(dim, 0)
            if score >= 3:
                rating = "Excellent"
            elif score >= 2:
                rating = "Good"
            elif score >= 1:
                rating = "Needs Improvement"
            else:
                rating = "Critical"
            dim_data.append({'Dimension': dim, 'Score': score, 'Max': 3, 'Rating': rating})

        dim_df = pd.DataFrame(dim_data)
        st.dataframe(
            dim_df,
            column_config={
                'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=3, format='%d'),
            },
            use_container_width=True,
            hide_index=True,
        )
