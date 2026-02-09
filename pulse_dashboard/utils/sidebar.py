"""
Pulse Dashboard - Shared Sidebar

Renders filters and targets in the sidebar. Called by every page.
Returns the filtered DataFrame.
"""

import streamlit as st
import pandas as pd

from utils.styles import STATUS_CONFIG, STATUS_ORDER, inject_css


def render_sidebar() -> pd.DataFrame | None:
    """Render sidebar filters and return filtered DataFrame.

    Returns None if no data is loaded yet.
    """
    inject_css()

    if st.session_state.get('df') is None:
        return None

    df = st.session_state.df

    with st.sidebar:
        st.markdown("### Pulse Dashboard")
        st.markdown("---")

        # ── Week Filter ──
        weeks = sorted(df['Wk'].dropna().unique())
        years = sorted(df['Year'].dropna().unique())

        # If multiple years, show year selector
        if len(years) > 1:
            selected_year = st.selectbox("Year", years, index=len(years) - 1)
            year_df = df[df['Year'] == selected_year]
            year_weeks = sorted(year_df['Wk'].dropna().unique())
        else:
            selected_year = years[0]
            year_weeks = weeks

        selected_week = st.selectbox(
            "Week",
            year_weeks,
            index=len(year_weeks) - 1,
            key='sidebar_week',
        )

        # ── Region Filter ──
        regions = sorted(df['Region'].dropna().unique())
        selected_regions = st.multiselect(
            "Regions",
            regions,
            default=list(regions),
            key='sidebar_regions',
        )

        # ── Area Filter (filtered by selected regions) ──
        region_mask = df['Region'].isin(selected_regions) if selected_regions else pd.Series(True, index=df.index)
        available_areas = sorted(df.loc[region_mask, 'Area'].dropna().unique())
        selected_areas = st.multiselect(
            "Areas",
            available_areas,
            default=list(available_areas),
            key='sidebar_areas',
        )

        # ── Pulse Status Filter ──
        status_options = [f"{s} ({STATUS_CONFIG[s]['range']})" for s in STATUS_ORDER]
        selected_statuses = st.multiselect(
            "Pulse Status",
            status_options,
            default=status_options,
            key='sidebar_statuses',
        )
        # Map back to status names
        active_statuses = [STATUS_ORDER[i] for i, opt in enumerate(status_options) if opt in selected_statuses]

        st.markdown("---")

        # ── Targets ──
        st.markdown("### Targets")
        pulse_target = st.number_input("Pulse Target", value=17.0, step=0.5, key='sidebar_target')
        pulse_stretch = st.number_input("Stretch Target", value=19.0, step=0.5, key='sidebar_stretch')
        green_pct_target = st.slider("Green % Target", 0, 100, 80, key='sidebar_green_pct')
        max_red_target = st.number_input("Max Red Projects", value=3, min_value=0, step=1, key='sidebar_max_red')

        # Store targets in session state
        st.session_state['pulse_target'] = pulse_target
        st.session_state['pulse_stretch'] = pulse_stretch
        st.session_state['green_pct_target'] = green_pct_target
        st.session_state['max_red_target'] = int(max_red_target)

        st.markdown("---")

        # ── Ollama Status ──
        st.markdown("### AI Status")
        try:
            from utils.pulse_insights import check_ollama
            if check_ollama():
                st.success("Ollama: Connected")
            else:
                st.error("Ollama: Not running")
        except Exception:
            st.warning("Ollama: Unknown")

    # ── Apply Filters ──
    filtered = df.copy()
    filtered = filtered[filtered['Year'] == selected_year]
    filtered = filtered[filtered['Wk'] == selected_week]

    if selected_regions:
        filtered = filtered[filtered['Region'].isin(selected_regions)]
    if selected_areas:
        filtered = filtered[filtered['Area'].isin(selected_areas)]
    if active_statuses:
        filtered = filtered[filtered['Pulse_Status'].isin(active_statuses)]

    # Store filter state
    st.session_state['selected_year'] = selected_year
    st.session_state['selected_week'] = selected_week
    st.session_state['selected_regions'] = selected_regions
    st.session_state['filtered_df'] = filtered

    return filtered
