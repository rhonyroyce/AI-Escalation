"""
Pulse Dashboard - Page 6: Project Details (Filterable Table + Deep Dive)

This page provides two views for inspecting individual projects:

1. **Filterable Data Table** -- A sortable, scrollable Streamlit dataframe of
   all projects that match the current sidebar filters.  Columns shown include
   identifying information (Project, PM Name, Region, Area), the Pulse status
   label, Total Score (rendered as a progress bar 0-24), and each of the 8
   scoring dimensions.  The table is sorted ascending by Total Score so the
   worst-performing (most at-risk) projects appear first.

2. **Project Deep Dive** -- When the user selects a project from a dropdown:
   - A styled status badge card shows the project name, PM, region/area, Total
     Score out of 24, and a color-coded status label (Critical / At Risk /
     On Track / Exceptional).
   - Two side-by-side charts:
     * A **radar chart** of the 8 dimension scores for the latest data point,
       showing the project's "shape" across Design, IX, PAG, RF Opt, Field,
       CSAT, PM Performance, and Potential.
     * A **trend line chart** of the project's Total Score over all available
       weeks, with a horizontal target line.
   - **Text field expanders**: collapsible sections for each narrative column
     (Comments, Pain Points, Resolution Plan, Issue, Pending Action) listing
     all non-null entries.
   - A **Dimension Score Table** that maps each of the 8 dimensions to a
     human-readable rating: 3 = Excellent, 2 = Good, 1 = Needs Improvement,
     0 = Critical.  The Score column is rendered as a progress bar (0-3).

**Scoring Context**
The Pulse scoring system rates each project across 8 dimensions, each scored
0-3 (where 0 is worst, 3 is best).  The Total Score (sum of all dimensions)
ranges from 0-24.  The 4-tier Pulse Status thresholds are:
- Red (Critical):       1-13
- Yellow (At Risk):     14-15
- Green (On Track):     16-19
- Dark Green (Exceptional): 20-24

These thresholds and color mappings come from ``STATUS_CONFIG`` in
``utils/styles.py``.

**AI Cache / Session State Context**
This page does not directly use the AI cache keys (``ai_exec_summary``,
``ai_issue_categories``, ``embeddings_index``).  However, it reads from
``st.session_state.df`` (the full unfiltered DataFrame) and
``st.session_state.pulse_target`` (the user-configurable target score,
defaulting to 17.0), both of which are set during app initialisation.
"""

# ---------------------------------------------------------------------------
# Path setup -- allow imports from the parent ``pulse_dashboard`` package.
# Streamlit's ``pages/`` directory does not inherit the parent's import
# context, so we prepend it to sys.path.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

# render_sidebar: applies Region/Area/Year/Week filters, returns filtered DF.
# inject_css: injects the shared dark-theme CSS.
# SCORE_DIMENSIONS: the list of 8 dimension column names.
# STATUS_CONFIG: dict mapping status names (Red/Yellow/Green/Dark Green) to
#   their hex color, human label, and score range.
from utils.sidebar import render_sidebar
from utils.styles import inject_css, SCORE_DIMENSIONS, STATUS_CONFIG

# chart_radar: creates a Plotly radar (spider) chart of the 8 dimension scores.
# chart_project_trend: creates a Plotly line chart of Total Score over weeks
#   for a single project, with a horizontal target line.
from utils.mckinsey_charts import chart_radar, chart_project_trend

# ---------------------------------------------------------------------------
# Page initialisation
# ---------------------------------------------------------------------------

# Inject the shared dark-theme CSS into this Streamlit page.
inject_css()

# Apply sidebar filters and retrieve the filtered DataFrame.  If nothing
# matches (or no data was loaded), stop rendering the page.
filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

# Full (unfiltered) DataFrame -- used to show historical trend data for a
# project even if the current sidebar filters only select a single week.
df = st.session_state.df

# The configurable Pulse target score (default 17.0 / 24).  This is drawn as
# a horizontal reference line on the project trend chart.
target = st.session_state.get('pulse_target', 17.0)

# Page title rendered via the shared ``main-header`` CSS class.
st.markdown('<p class="main-header">Project Details</p>', unsafe_allow_html=True)

# ============================================================================
# FILTERABLE DATA TABLE
# ============================================================================
# Define the desired column order.  We start with identifiers, then Total
# Score and Pulse Status, followed by the 8 individual dimension scores.
display_cols = ['Project', 'PM Name', 'Region', 'Area', 'Total Score', 'Pulse_Status'] + SCORE_DIMENSIONS
# Guard against missing columns (e.g. if the loaded Excel lacks a dimension).
available_cols = [c for c in display_cols if c in filtered_df.columns]

# Render the interactive table.  Sorted ascending by Total Score so the
# lowest-scoring (most concerning) projects appear at the top.
st.dataframe(
    filtered_df[available_cols].sort_values('Total Score', ascending=True),
    column_config={
        # ProgressColumn renders Total Score as a horizontal bar (0-24).
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
    height=400,  # Fixed height with internal scroll
)

# ============================================================================
# PROJECT DEEP DIVE
# ============================================================================
st.markdown("---")
st.markdown("### Project Deep Dive")

# Dropdown to select a single project for detailed inspection.  The list is
# derived from the *filtered* DataFrame so only visible projects appear.
projects = sorted(filtered_df['Project'].dropna().unique())
selected_project = st.selectbox("Select a project for detailed view", projects, key="detail_project")

if selected_project:
    # ``proj_current``: rows for this project under the current sidebar
    # filters (usually just the selected week).
    # ``proj_history``: ALL rows for this project across ALL weeks (unfiltered),
    # used for the trend chart.
    proj_current = filtered_df[filtered_df['Project'] == selected_project]
    proj_history = df[df['Project'] == selected_project]

    if proj_current.empty:
        st.info("No data for selected project in current filters.")
    else:
        # Use the first (and typically only) row for the current filter scope.
        latest_row = proj_current.iloc[0]

        # ── Status Badge Card ──
        # Look up the status key (e.g. "Red") and retrieve its display color
        # and human-friendly label (e.g. "Critical") from STATUS_CONFIG.
        status = latest_row['Pulse_Status']
        color = STATUS_CONFIG[status]['color']
        label = STATUS_CONFIG[status]['label']

        # Render a styled executive-card div with the project header, PM info,
        # region/area, and a color-coded score badge.
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

        # ── Charts: Radar + Trend ──
        # Side-by-side layout: left column = radar chart, right column = trend.
        col1, col2 = st.columns(2)

        with col1:
            # Radar chart plots the 8 dimension scores (0-3 each) as a filled
            # polygon on a spider-web axis.  It gives a quick visual "shape"
            # for the project's strengths and weaknesses.
            fig_radar = chart_radar(latest_row)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Trend chart plots Total Score over all available Year/Week pairs
            # for this project.  A dashed horizontal line marks the target
            # score (typically 17).  This uses the unfiltered ``proj_history``
            # so the full trajectory is visible.
            fig_trend = chart_project_trend(proj_history, selected_project, target)
            st.plotly_chart(fig_trend, use_container_width=True)

        # ── Text Fields (Narrative Columns) ──
        # Display each narrative text column as a collapsible expander.
        # Each expander shows a bullet list of all non-null values for this
        # project within the current filter scope.
        st.markdown("#### Comments & Issues")

        # Pairs of (column_name, streamlit_expander_key).  The keys must be
        # unique across the page to prevent Streamlit widget conflicts.
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
                    # Show the count of entries in the expander header.
                    with st.expander(f"{field_name} ({len(values)} entries)"):
                        for val in values:
                            st.markdown(f"- {val}")

        # ── Dimension Score Table ──
        # A table showing each of the 8 dimensions with its numeric score
        # (0-3), the maximum possible (3), and a qualitative rating label.
        st.markdown("#### Dimension Scores")
        dim_data = []
        for dim in SCORE_DIMENSIONS:
            score = latest_row.get(dim, 0)
            # Map the numeric 0-3 score to a human-readable rating.
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
                # ProgressColumn renders each dimension score as a mini bar (0-3).
                'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=3, format='%d'),
            },
            use_container_width=True,
            hide_index=True,
        )
