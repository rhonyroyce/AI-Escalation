"""
Pulse Dashboard - Shared Sidebar
=================================

This module renders the sidebar that appears on every page of the Pulse
Dashboard.  The sidebar provides a unified set of filters and configuration
controls that narrow the data shown on whatever page the user is viewing.

Architecture / Design Decision
------------------------------
Rather than embedding filter logic in each page, a single `render_sidebar()`
function is called at the top of every page script.  This guarantees:
  - Consistent filter options across all pages (no page shows stale filters).
  - Shared session state -- filter selections are written to
    `st.session_state` so that other components (charts, tables, AI prompts)
    can read them without being passed explicitly.
  - A single place to modify filter behavior when requirements change.

Filter Hierarchy
----------------
Filters are applied in a cascading/hierarchical order:

    Year  -->  Week  -->  Region  -->  Area  -->  Pulse Status

1. **Year**: Which calendar year to view.  Defaults to the most recent year
   available in the data.  Only shown as a dropdown if multiple years exist.
2. **Week**: Which reporting week within the selected year.  Defaults to the
   latest week.  The available weeks update dynamically based on the year.
3. **Region**: Multi-select of geographic regions (e.g. Central, NE, South,
   West).  Defaults to all regions selected.
4. **Area**: Multi-select of sub-areas within the selected regions.  The
   options dynamically filter to only show areas that belong to the selected
   regions -- this prevents the user from selecting an area that would
   produce zero results.
5. **Pulse Status**: Multi-select of status buckets (Red, Yellow, Green,
   Dark Green).  Each option displays the score range in parentheses for
   clarity.

Target Configuration
--------------------
Below the filters, the sidebar exposes four tunable target parameters that
are used by KPI cards, charts, and conditional formatting throughout the
dashboard:

- **Pulse Target** (default 17.0):  The "good enough" average score.
- **Stretch Target** (default 19.0):  The aspirational average score.
- **Green % Target** (default 80%):  Desired percentage of projects in
  Green or Dark Green status.
- **Max Red Projects** (default 3):  Maximum acceptable count of Red
  (critical) projects.

These are stored in `st.session_state` so any page can read them.

Ollama Status
-------------
At the bottom of the sidebar, a status indicator shows whether the local
Ollama LLM server is reachable.  This is a UX convenience so the user knows
whether AI-powered features (insights, semantic search) will work before
navigating to those pages.

Data Flow
---------
1. Read the full (unfiltered) DataFrame from `st.session_state['df']`.
2. Render filter widgets and capture user selections.
3. Apply all filters to produce a filtered DataFrame.
4. Store filtered DataFrame and filter values back into `st.session_state`.
5. Return the filtered DataFrame to the calling page.

Returns None if no data has been loaded yet (the page should show an upload
prompt or error message in that case).
"""

import streamlit as st
import pandas as pd

# STATUS_CONFIG: dict mapping status name -> {color, label, range, min, max}
# STATUS_ORDER: canonical ordering ['Red', 'Yellow', 'Green', 'Dark Green']
# inject_css: injects the shared dark-theme CSS into the page
from utils.styles import STATUS_CONFIG, STATUS_ORDER, inject_css


def render_sidebar() -> pd.DataFrame | None:
    """Render sidebar filters and return the filtered DataFrame.

    This function is designed to be called once at the top of every Streamlit
    page script.  It draws all sidebar widgets, reads the user's selections,
    applies them to the full dataset, and returns the filtered result.

    Streamlit Re-run Behavior:
        Every time a widget value changes, Streamlit re-runs the entire page
        script from top to bottom.  Because this function runs on every rerun,
        filter state is always fresh.  The `key=` parameter on each widget
        ties it to a stable session_state key so that values persist across
        reruns and across page navigations.

    Returns:
        Filtered pd.DataFrame matching all active filter criteria, or None
        if no data has been loaded into session_state yet.
    """
    # Inject the shared CSS stylesheet into the page.  This is called here
    # (rather than in each page) because the sidebar renders on every page,
    # ensuring CSS is always present regardless of which page is active.
    inject_css()

    # ── Guard: no data loaded yet ────────────────────────────────────────
    # If the user hasn't uploaded a file and no default file was found,
    # session_state['df'] will be None.  Return early so the calling page
    # can show an appropriate upload prompt.
    if st.session_state.get('df') is None:
        return None

    # Grab the FULL (unfiltered) DataFrame.  All filter operations below
    # operate on this complete dataset so that filter options always reflect
    # the full range of available values (not a previously filtered subset).
    df = st.session_state.df

    with st.sidebar:
        # Sidebar header -- consistent branding across all pages.
        st.markdown("### Pulse Dashboard")
        st.markdown("---")

        # ── Year / Week Filter ───────────────────────────────────────────
        # These two filters work together: selecting a Year constrains which
        # Weeks are available.  This prevents the user from selecting a week
        # that doesn't exist in the chosen year.

        # Extract all unique weeks and years from the full dataset.
        # sorted() ensures the dropdowns display values in ascending order.
        weeks = sorted(df['Wk'].dropna().unique())
        years = sorted(df['Year'].dropna().unique())

        # If the data spans multiple years, show a Year dropdown.
        # index=len(years)-1 selects the LAST (most recent) year by default,
        # which is what users want 99% of the time -- they care about the
        # current reporting period, not historical data.
        if len(years) > 1:
            selected_year = st.selectbox("Year", years, index=len(years) - 1)
            # Filter weeks to only those present in the selected year.
            year_df = df[df['Year'] == selected_year]
            year_weeks = sorted(year_df['Wk'].dropna().unique())
        else:
            # Only one year in the data -- no need for a year selector.
            selected_year = years[0]
            year_weeks = weeks

        # Week dropdown: defaults to the most recent week (last in sorted list).
        # key='sidebar_week' ensures this widget's state persists across reruns
        # and doesn't collide with any other selectbox on the page.
        selected_week = st.selectbox(
            "Week",
            year_weeks,
            index=len(year_weeks) - 1,
            key='sidebar_week',
        )

        # ── Region Filter ────────────────────────────────────────────────
        # Multi-select widget showing all unique regions (e.g. Central, NE,
        # South, West).  Defaults to ALL regions selected so the user sees
        # the full picture and can deselect to drill down.
        regions = sorted(df['Region'].dropna().unique())
        selected_regions = st.multiselect(
            "Regions",
            regions,
            default=list(regions),      # Start with everything selected
            key='sidebar_regions',
        )

        # ── Area Filter (cascaded from Regions) ──────────────────────────
        # Areas are sub-divisions within Regions.  To prevent showing areas
        # that belong to unselected regions (which would produce zero results),
        # we first build a boolean mask for the selected regions, then extract
        # only the areas that appear in those regions.
        #
        # If no regions are selected (empty multiselect), the mask becomes
        # True for all rows so that all areas remain available -- this avoids
        # an empty Area dropdown that would confuse the user.
        region_mask = df['Region'].isin(selected_regions) if selected_regions else pd.Series(True, index=df.index)
        available_areas = sorted(df.loc[region_mask, 'Area'].dropna().unique())
        selected_areas = st.multiselect(
            "Areas",
            available_areas,
            default=list(available_areas),  # Start with all areas in the selected regions
            key='sidebar_areas',
        )

        # ── Pulse Status Filter ──────────────────────────────────────────
        # Each option is formatted as "Status (range)" for clarity, e.g.:
        #   "Red (1-13)", "Yellow (14-15)", "Green (16-19)", "Dark Green (20-24)"
        # This helps users understand the scoring thresholds without needing
        # to consult documentation.
        status_options = [f"{s} ({STATUS_CONFIG[s]['range']})" for s in STATUS_ORDER]
        selected_statuses = st.multiselect(
            "Pulse Status",
            status_options,
            default=status_options,         # Start with all statuses visible
            key='sidebar_statuses',
        )
        # Map the display strings back to the plain status names ('Red', etc.)
        # that match the Pulse_Status column in the DataFrame.  We do this by
        # checking which formatted option strings are in the user's selection
        # and looking up the corresponding status name by index position.
        active_statuses = [STATUS_ORDER[i] for i, opt in enumerate(status_options) if opt in selected_statuses]

        st.markdown("---")

        # ── Target Configuration ─────────────────────────────────────────
        # These numeric inputs define performance thresholds that are used
        # throughout the dashboard for KPI cards, goal lines on charts,
        # and conditional formatting (e.g. "are we meeting our targets?").
        #
        # They are deliberately placed in the sidebar (not hardcoded) so
        # that different stakeholders can adjust them to match their own
        # expectations during live presentations or reviews.

        st.markdown("### Targets")

        # Pulse Target: the minimum "acceptable" average Total Score.
        # Step of 0.5 allows fine-grained tuning without being overwhelming.
        pulse_target = st.number_input("Pulse Target", value=17.0, step=0.5, key='sidebar_target')

        # Stretch Target: the aspirational average -- a higher bar that teams
        # aim for but are not penalised for missing.
        pulse_stretch = st.number_input("Stretch Target", value=19.0, step=0.5, key='sidebar_stretch')

        # Green % Target: what percentage of projects should be Green or
        # Dark Green.  Expressed as an integer 0-100 for intuitive slider UX.
        green_pct_target = st.slider("Green % Target", 0, 100, 80, key='sidebar_green_pct')

        # Max Red Projects: the maximum count of Red (critical) projects
        # that is considered acceptable.  Anything above this triggers alerts.
        max_red_target = st.number_input("Max Red Projects", value=3, min_value=0, step=1, key='sidebar_max_red')

        # Persist all target values into session_state so that any page or
        # component can read them via st.session_state['pulse_target'], etc.
        # This decouples the sidebar from the pages -- pages don't need to
        # know how targets are configured, they just read the values.
        st.session_state['pulse_target'] = pulse_target
        st.session_state['pulse_stretch'] = pulse_stretch
        st.session_state['green_pct_target'] = green_pct_target
        st.session_state['max_red_target'] = int(max_red_target)  # Ensure integer type

        st.markdown("---")

        # ── Ollama AI Status Indicator ───────────────────────────────────
        # Quick health check for the local Ollama LLM server.  This gives
        # the user immediate feedback about whether AI-powered features
        # (e.g. PulseInsights narrative generation, semantic search) will
        # be available.  The import is done lazily inside the try block to
        # avoid circular imports and to keep the sidebar functional even if
        # the pulse_insights module has issues.
        st.markdown("### AI Status")
        try:
            from utils.pulse_insights import check_ollama
            if check_ollama():
                st.success("Ollama: Connected")    # Green banner
            else:
                st.error("Ollama: Not running")    # Red banner
        except Exception:
            st.warning("Ollama: Unknown")          # Yellow banner (import failed)

    # ── Apply All Filters to Build the Filtered DataFrame ────────────────
    # Start with a copy of the full data to avoid mutating the original.
    # Filters are applied sequentially -- each one narrows the result further.
    filtered = df.copy()

    # Year + Week: these are always applied (single-select, always have a value).
    # This gives us one snapshot week of data to display.
    filtered = filtered[filtered['Year'] == selected_year]
    filtered = filtered[filtered['Wk'] == selected_week]

    # Region: only apply if the user has at least one region selected.
    # An empty selection means "show nothing" which is handled gracefully
    # by downstream components (they check for empty DataFrames).
    if selected_regions:
        filtered = filtered[filtered['Region'].isin(selected_regions)]

    # Area: same logic as Region -- only filter if selections exist.
    if selected_areas:
        filtered = filtered[filtered['Area'].isin(selected_areas)]

    # Pulse Status: filter by the active status categories.
    if active_statuses:
        filtered = filtered[filtered['Pulse_Status'].isin(active_statuses)]

    # ── Persist Filter State ─────────────────────────────────────────────
    # Store the current filter selections and the resulting filtered DataFrame
    # in session_state.  This allows:
    #   - Other components to know which year/week/regions are active
    #     (e.g. for building chart titles or AI prompts).
    #   - The filtered DataFrame to be accessed by components that don't
    #     receive it as a direct argument.
    st.session_state['selected_year'] = selected_year
    st.session_state['selected_week'] = selected_week
    st.session_state['selected_regions'] = selected_regions
    st.session_state['filtered_df'] = filtered

    return filtered
