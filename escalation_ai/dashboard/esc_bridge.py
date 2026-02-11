"""
Escalation AI Bridge Module
============================

Provides data loading, sidebar rendering, CSS injection, and session state
initialization for Escalation AI page shims in the unified dashboard.

This module allows the 9,260-line monolith (streamlit_app.py) to be imported
for its render functions without triggering top-level Streamlit side effects.

Architecture Overview
---------------------
The Escalation AI dashboard was originally a single monolithic Streamlit app
(``streamlit_app.py``, ~9,265 lines) that ran standalone.  To integrate it
into a unified multi-page Streamlit application without a massive refactor,
a **Bridge Pattern** was adopted:

    1.  **This bridge module** (``esc_bridge.py``) centralizes all shared
        setup that each page needs: session-state initialization, CSS
        injection, data loading, and sidebar filter rendering.

    2.  **Page shim files** (``pages/1_Executive_Dashboard.py``, etc.) are
        thin ~18-line wrappers.  Each shim:
            a) calls the bridge to initialize state, inject CSS, and load /
               filter data;
            b) imports a single ``render_*`` function from the monolith;
            c) passes the already-filtered DataFrame to that render function.

    3.  **The monolith** (``streamlit_app.py``) is *never executed directly*
        in the unified app.  Only its ``render_*`` functions and ``load_data``
        are imported, so its ``if __name__ == "__main__"`` guard prevents any
        standalone execution side effects.

Data Flow
---------
::

    Streamlit MPA Router
          |
          v
    pages/N_PageName.py   (page shim)
          |
          |-- init_escalation_state()    -> sets session-state defaults
          |-- inject_escalation_css()    -> injects CSS into the page
          |-- esc_load_and_filter(name)  -> loads data & renders sidebar
          |       |
          |       |-- streamlit_app.load_data()  (cached)
          |       |-- sidebar filters (category, year, severity, date)
          |       |-- returns filtered DataFrame
          |       v
          |-- render_<page>(df)  <- imported from streamlit_app.py
          v
    Rendered Streamlit page

Why a Bridge?
-------------
- Avoids duplicating sidebar / CSS / state logic across six page shims.
- Keeps each shim under 20 lines so they are trivially auditable.
- Lets the monolith stay untouched: no structural edits required.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: ensure the project root (two levels above this file) is on
# sys.path so that ``escalation_ai.*`` imports work regardless of where
# Streamlit is launched from.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime


# ============================================================================
# SESSION STATE
# ============================================================================

def init_escalation_state():
    """Initialize Escalation AI session state keys.

    Streamlit session state is a dictionary-like object that persists across
    reruns of a page within the same browser session.  Each key below
    supports a specific feature:

    - ``presentation_mode`` (bool): toggles the full-screen slide deck view
      used by the Presentation Mode page.
    - ``current_slide`` (int): zero-based index of the currently displayed
      slide in Presentation Mode.
    - ``action_items`` (list): accumulated user-created action items from
      the Planning & Actions page, persisted across page navigations.
    - ``uploaded_file_path`` (str | None): filesystem path of a user-
      uploaded Excel file (allows re-runs without re-uploading).

    This function is **idempotent** -- calling it multiple times (e.g. once
    per page shim) will not overwrite values that the user has already
    changed during the current session.
    """
    # Default values for every session-state key that the Escalation AI
    # pages depend on.  Only keys that do NOT already exist are written,
    # preserving any user-modified state from prior interactions.
    defaults = {
        'presentation_mode': False,
        'current_slide': 0,
        'action_items': [],
        'uploaded_file_path': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================================
# CSS INJECTION
# ============================================================================

def inject_escalation_css():
    """Inject Escalation AI executive styling CSS into the current page.

    Uses ``st.markdown`` with ``unsafe_allow_html=True`` to insert a
    ``<style>`` block.  The CSS constant ``_ESCALATION_CSS`` (defined at the
    bottom of this file) was extracted verbatim from the monolith
    (``streamlit_app.py``, lines 346-1007) so that every page shim renders
    with the same visual identity without needing to re-import the monolith's
    CSS-injection helpers.

    This must be called **once per page load** (i.e. at the top of each page
    shim) because Streamlit does not carry injected HTML across page
    navigations in a multi-page app.
    """
    st.markdown(_ESCALATION_CSS, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING & SIDEBAR
# ============================================================================

def esc_load_and_filter(page_name: str = "Executive Dashboard") -> pd.DataFrame:
    """Load Escalation AI data and render sidebar filters.

    This function is the heart of the bridge module.  It performs three jobs:

    1. **Data loading** -- delegates to ``streamlit_app.load_data()`` which
       is decorated with ``@st.cache_data`` inside the monolith, so repeated
       calls across pages or reruns are fast.

    2. **Sidebar filter rendering** -- builds a sidebar with interactive
       Streamlit widgets that mirror the original monolith's sidebar
       (``streamlit_app.py main()``, lines 8074-8384).  Filters are applied
       *sequentially* (category -> year -> severity -> date range), each
       narrowing the DataFrame further.

    3. **Metadata display** -- shows the data source path, current record
       count, total financial impact, and a "Refresh Data" button that
       busts the Streamlit cache.

    Filter Mechanics
    ~~~~~~~~~~~~~~~~
    - **Category filter** (Executive Dashboard only): multi-select on the
      ``AI_Category`` column.  Defaults to all categories selected.
    - **Year filter** (Executive Dashboard only): extracts year from the
      ``tickets_data_issue_datetime`` column, multi-select.
    - **Severity filter** (Executive Dashboard only): multi-select on
      ``tickets_data_severity``.
    - **Date range filter** (all pages): a date-input widget that clips the
      DataFrame to [start, end] based on ``tickets_data_issue_datetime``.

    The "Clear Filter(s)" button simply triggers ``st.rerun()``, which
    resets all widget states to their defaults (all items selected).

    Args:
        page_name: The name of the current page.  When set to
            ``"Executive Dashboard"``, the Excel-style category / year /
            severity filters are shown.  For all other pages only the date
            range filter and metadata are displayed.

    Returns:
        A pandas DataFrame filtered by the user's sidebar selections, or
        ``None`` if no data could be loaded (e.g. the pipeline has not been
        run yet).
    """
    # ---- Step 1: Load raw data via the monolith's cached loader ----------
    # Import is deferred (inside the function body) to avoid importing the
    # monolith at module level.  This prevents Streamlit side effects that
    # occur at import time and keeps startup fast when the bridge is
    # imported but not yet needed.
    from escalation_ai.dashboard.streamlit_app import load_data

    # load_data() returns (DataFrame, data_source_path).  The DataFrame
    # contains every escalation record; data_source is a Path or string
    # describing where the data came from (e.g. the output Excel file).
    df, data_source = load_data()

    # If no data is available, show a user-friendly error and bail out.
    # Page shims check for None and skip rendering accordingly.
    if df is None:
        st.error("No escalation data found. Run the pipeline first: `python run.py`")
        return None

    # ---- Step 2: Build the sidebar with filters and metadata -------------
    with st.sidebar:
        # Visual divider between the navigation links and the filters
        st.markdown("---")

        # ---- Excel-style categorical filters (Executive Dashboard only) --
        # These three multiselects (Category, Year, Severity) give the
        # Executive Dashboard an Excel-like pivot-table feel.  They are
        # intentionally omitted on other pages to reduce sidebar clutter.
        if page_name == "Executive Dashboard":
            st.markdown("### Add filter(s)")

            # -- Category filter -------------------------------------------
            # Wraps the widget in styled HTML divs for the dark-themed
            # filter sections defined in _ESCALATION_CSS.
            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Categories</div>', unsafe_allow_html=True)
            if 'AI_Category' in df.columns:
                # Build the list of unique categories, sorted alphabetically
                all_categories = sorted(df['AI_Category'].unique().tolist())
                # Default: all categories selected (no filtering)
                selected_categories = st.multiselect(
                    "Select Categories",
                    options=all_categories,
                    default=all_categories,
                    key="excel_cat_filter",
                    label_visibility="collapsed"
                )
                # Apply filter -- only keep rows whose category is selected
                if selected_categories:
                    df = df[df['AI_Category'].isin(selected_categories)]
            st.markdown('</div>', unsafe_allow_html=True)

            # -- Year filter -----------------------------------------------
            # Extract year from the ticket timestamp for year-level grouping
            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Year</div>', unsafe_allow_html=True)
            if 'tickets_data_issue_datetime' in df.columns:
                # Work on a temporary copy to extract the year column
                # without modifying the main DataFrame prematurely
                df_temp_year = df.copy()
                df_temp_year['year'] = pd.to_datetime(
                    df_temp_year['tickets_data_issue_datetime']
                ).dt.year
                all_years = sorted(df_temp_year['year'].unique().tolist())
                selected_years = st.multiselect(
                    "Select Years",
                    options=all_years,
                    default=all_years,
                    key="excel_year_filter",
                    label_visibility="collapsed"
                )
                # Apply year filter on the main DataFrame
                if selected_years:
                    df['year'] = pd.to_datetime(df['tickets_data_issue_datetime']).dt.year
                    df = df[df['year'].isin(selected_years)]
            st.markdown('</div>', unsafe_allow_html=True)

            # -- Severity filter -------------------------------------------
            st.markdown('<div class="excel-filter-section">', unsafe_allow_html=True)
            st.markdown('<div class="excel-filter-title">Severity</div>', unsafe_allow_html=True)
            if 'tickets_data_severity' in df.columns:
                all_severities = sorted(df['tickets_data_severity'].unique().tolist())
                selected_severities = st.multiselect(
                    "Select Severities",
                    options=all_severities,
                    default=all_severities,
                    key="excel_sev_filter",
                    label_visibility="collapsed"
                )
                if selected_severities:
                    df = df[df['tickets_data_severity'].isin(selected_severities)]
            st.markdown('</div>', unsafe_allow_html=True)

            # -- Clear filters button --------------------------------------
            # Clicking this triggers a full rerun, which resets every
            # multiselect widget back to its default (all options selected),
            # effectively removing all filters.
            if st.button("Clear Filter(s)", key="clear_filters", type="secondary"):
                st.rerun()

        # ---- Metadata section (shown on every page) ----------------------
        st.markdown("---")
        # Show the filesystem path or description of the data source so the
        # user knows which file is driving the dashboard.
        st.markdown(f"**Data Source:**")
        st.caption(str(data_source))
        # Show the number of records *after* filtering so the user can gauge
        # the impact of their filter selections.
        st.markdown(f"**Records:** {len(df):,}")

        # If the pipeline computed financial impact costs, show the sum.
        if 'Financial_Impact' in df.columns:
            total_cost = df['Financial_Impact'].sum()
            st.markdown(f"**Total Cost:** ${total_cost:,.0f}")

        # -- Refresh button ------------------------------------------------
        # Clears all Streamlit cached data (including the load_data cache)
        # and reruns the page so that any new pipeline output is picked up
        # without restarting the entire Streamlit server.
        if st.button("Refresh Data", key="esc_refresh"):
            st.cache_data.clear()
            st.rerun()

        # ---- Settings section: Date-range filter (all pages) -------------
        st.markdown("---")
        st.markdown("### Settings")
        if 'tickets_data_issue_datetime' in df.columns:
            try:
                # Parse the datetime column to determine the min/max dates
                # available in the (already category/year/severity-filtered)
                # data, so the date picker bounds reflect current selections.
                dates = pd.to_datetime(df['tickets_data_issue_datetime'])
                min_date = dates.min().date()
                max_date = dates.max().date()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="esc_date_range",
                )
                # The date_input widget returns a tuple of (start, end) when
                # the user has selected both dates; it may return a single
                # date while the user is mid-selection.  Only apply the
                # filter when both bounds are available.
                if len(date_range) == 2:
                    df = df[
                        (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date >= date_range[0])
                        & (pd.to_datetime(df['tickets_data_issue_datetime']).dt.date <= date_range[1])
                    ]
            except Exception:
                # Silently ignore date parsing errors (e.g. if the column
                # contains non-datetime values) rather than crashing the
                # entire dashboard.
                pass

    # ---- Step 3: Return the filtered DataFrame to the page shim ----------
    return df


# ============================================================================
# CSS CONSTANT -- extracted from streamlit_app.py lines 346-1007
# ============================================================================
#
# This large CSS string was lifted verbatim from the monolith so that every
# page shim gets identical styling without importing the monolith's CSS
# injection helpers.  It covers:
#
#   - Global font (Inter via Google Fonts)
#   - Plotly chart sizing and overflow fixes
#   - Glassmorphism card styles (.glass-card)
#   - Executive summary cards (.exec-card)
#   - Strategic recommendation cards (.strategy-card)  with priority variants
#   - KPI containers (.kpi-container) with critical / warning / success states
#   - Executive KPI large cards (.exec-kpi)
#   - KPI value / label / delta typography
#   - Animated pulse dot status indicators
#   - Main header and executive title typography
#   - Benchmark meter (progress-bar style)
#   - Action item cards with status-colored left borders
#   - Hidden Streamlit default UI elements (hamburger menu, footer, deploy)
#   - Chart container height / overflow guards
#   - Chart title, alert badge, and priority tag styles
#   - Custom tab bar styling
#   - Slider track gradient
#   - Presentation mode slide layout
#   - Confidence badge, impact card styles
#   - Executive table styling
#   - Excel-style dashboard header, filter sections, KPI cards
#   - Excel-style chart cards, legends, progress bars, comparison grids
#   - Excel-style donut, bar chart, product card, scatter, quarter grid
#

_ESCALATION_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* --- Global font override for the Streamlit app shell --- */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* --- Plotly chart minimum height to prevent collapsed / overlapping charts --- */
.stPlotlyChart {
    min-height: 400px !important;
}

.stPlotlyChart > div {
    min-height: 400px !important;
}

/* --- Column gap and padding to prevent side-by-side column overlap --- */
.stColumns {
    gap: 1rem;
}

.stColumn {
    padding: 0 0.5rem;
}

/* --- Executive Glassmorphism cards: frosted-glass effect for card panels --- */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    margin: 10px 0;
}

/* --- Executive Summary Card: dark navy gradient with subtle blue glow --- */
.exec-card {
    background: linear-gradient(145deg, rgba(0, 40, 85, 0.9) 0%, rgba(0, 20, 40, 0.95) 100%);
    border-radius: 20px;
    padding: 32px;
    border: 1px solid rgba(0, 150, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin: 16px 0;
}

/* --- Strategic Recommendation Cards: left-border accent indicates priority --- */
.strategy-card {
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 51, 102, 0.2) 100%);
    border-left: 5px solid #00BFFF;
    border-radius: 0 16px 16px 0;
    padding: 20px 24px;
    margin: 12px 0;
}

/* High-priority variant: red-accented left border */
.strategy-card.high-priority {
    border-left-color: #FF6B6B;
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(139, 0, 0, 0.15) 100%);
}

/* Medium-priority variant: orange-accented left border */
.strategy-card.medium-priority {
    border-left-color: #FFB347;
    background: linear-gradient(135deg, rgba(255, 179, 71, 0.1) 0%, rgba(255, 140, 0, 0.15) 100%);
}

/* --- KPI Cards - Enhanced: interactive hover lift with priority-colored borders --- */
.kpi-container {
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
    border-radius: 16px;
    padding: 24px;
    border-left: 4px solid #0066CC;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

/* Hover effect: float upward with enhanced shadow */
.kpi-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 102, 204, 0.3);
}

/* Critical state: red gradient for SLA breaches, outages, etc. */
.kpi-container.critical {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-left-color: #DC3545;
}

/* Warning state: amber/yellow gradient for approaching thresholds */
.kpi-container.warning {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 140, 0, 0.25) 100%);
    border-left-color: #FFC107;
}

/* Success state: green gradient for healthy metrics */
.kpi-container.success {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
    border-left-color: #28A745;
}

/* --- Executive KPI - Larger: hero-sized KPI cards for top-level metrics --- */
.exec-kpi {
    background: linear-gradient(145deg, rgba(0, 102, 204, 0.2) 0%, rgba(0, 40, 80, 0.4) 100%);
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    border: 2px solid rgba(0, 191, 255, 0.3);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
}

/* Executive KPI value: large gradient text (blue default) */
.exec-kpi-value {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 50%, #004080 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

/* Money variant: green gradient for dollar amounts */
.exec-kpi-value.money {
    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Alert variant: red gradient for critical values */
.exec-kpi-value.alert {
    background: linear-gradient(135deg, #FF6B6B 0%, #DC3545 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Standard KPI value typography */
.kpi-value {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* KPI label: muted uppercase descriptor beneath the value */
.kpi-label {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
}

/* KPI delta: shows period-over-period change */
.kpi-delta {
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 4px;
}

/* Delta direction colors: red = increase (bad for costs), green = decrease */
.delta-up { color: #DC3545; }
.delta-down { color: #28A745; }

/* --- Pulse Indicator: animated dot to show live status --- */
.pulse-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

/* Pulse color variants with matching glow */
.pulse-dot.green { background: #28A745; box-shadow: 0 0 8px #28A745; }
.pulse-dot.yellow { background: #FFC107; box-shadow: 0 0 8px #FFC107; }
.pulse-dot.red { background: #DC3545; box-shadow: 0 0 8px #DC3545; }

/* Pulse animation keyframes: scale + fade cycle */
@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
    100% { opacity: 1; transform: scale(1); }
}

/* --- Main header: gradient text for page titles --- */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

/* Executive title: larger, three-stop gradient for hero headings */
.exec-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #00BFFF 50%, #0066CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    text-align: center;
}

/* Sub-header: muted descriptor text below main headers */
.sub-header {
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* --- Benchmark Meter: horizontal color-coded gauge bar --- */
.benchmark-container {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

/* Gradient bar: green(good) -> yellow(caution) -> red(critical) */
.benchmark-bar {
    height: 24px;
    background: linear-gradient(90deg, #28A745 0%, #FFC107 50%, #DC3545 100%);
    border-radius: 12px;
    position: relative;
    margin: 10px 0;
}

/* White marker line indicating the current value position on the gauge */
.benchmark-marker {
    position: absolute;
    width: 4px;
    height: 32px;
    background: white;
    top: -4px;
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
}

/* --- Action Item Cards: status-colored left border for task tracking --- */
.action-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Completed actions: green border, reduced opacity */
.action-card.completed {
    border-left: 4px solid #28A745;
    opacity: 0.7;
}

/* In-progress actions: blue border */
.action-card.in-progress {
    border-left: 4px solid #0066CC;
}

/* Blocked actions: red border */
.action-card.blocked {
    border-left: 4px solid #DC3545;
}

/* --- Hide default Streamlit UI chrome for a cleaner executive look --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* --- Chart container height constraints to prevent overlap in columns --- */
[data-testid="stPlotlyChart"] {
    min-height: 360px;
    max-height: 450px;
    overflow: visible !important;
}

[data-testid="column"] {
    overflow: visible !important;
}

/* Ensure Plotly charts fill their container width */
.js-plotly-plot {
    width: 100% !important;
}

/* --- Chart title typography --- */
.chart-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #E0E0E0;
    margin-bottom: 12px;
}

/* --- Alert badges: colored pill-shaped labels for status indicators --- */
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-critical { background: #DC3545; color: white; }
.badge-warning { background: #FFC107; color: #212529; }
.badge-success { background: #28A745; color: white; }
.badge-info { background: #0066CC; color: white; }

/* --- Priority Tags: compact inline labels for P1/P2/P3 severity --- */
.priority-p1 { background: linear-gradient(135deg, #DC3545, #8B0000); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.priority-p2 { background: linear-gradient(135deg, #FFC107, #FF8C00); color: #212529; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
.priority-p3 { background: linear-gradient(135deg, #0066CC, #004080); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }

/* --- Custom tab styling: rounded pill-style tabs with active gradient --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.05);
    padding: 4px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0066CC 0%, #004C97 100%);
}

/* --- Slider track gradient (blue-to-cyan) --- */
.stSlider > div > div {
    background: linear-gradient(90deg, #0066CC, #00BFFF);
}

/* --- Metric delta font size fix --- */
[data-testid="stMetricDelta"] {
    font-size: 0.9rem;
}

/* --- Presentation Mode: full-viewport centered slide layout --- */
.presentation-slide {
    min-height: 80vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 40px;
}

/* --- Confidence Score badge: subtle outlined pill --- */
.confidence-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(0, 102, 204, 0.2);
    border: 1px solid rgba(0, 191, 255, 0.3);
}

/* --- Impact Cards: color-coded backgrounds for positive/negative outcomes --- */
.impact-positive {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
    border-color: #28A745;
}

.impact-negative {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-color: #DC3545;
}

/* --- Executive Table: sortable-look dark table with hover highlight --- */
.exec-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
}

.exec-table th {
    background: rgba(0, 102, 204, 0.3);
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 1px;
}

.exec-table td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.exec-table tr:hover {
    background: rgba(0, 102, 204, 0.1);
}

/* ==========================================================================
   EXCEL-STYLE DASHBOARD CSS
   These styles replicate an Excel/Power BI look-and-feel for the Executive
   Dashboard page, including header, filter sections, KPI cards, chart
   containers, legends, progress bars, and grid layouts.
   ========================================================================== */

/* Dashboard header bar: dark navy gradient strip at the page top */
.excel-dashboard-header {
    background: linear-gradient(135deg, #0a2540 0%, #003366 100%);
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Dashboard title: large white text with wide letter-spacing */
.excel-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2px;
    margin: 0;
}

/* Title accent color: red highlight for branding keywords */
.excel-title-accent {
    color: #dc3545;
    font-weight: 800;
}

/* Subtitle: light blue descriptor line below the title */
.excel-subtitle {
    color: #87ceeb;
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}

/* --- Sidebar filter sections: dark navy cards that wrap each filter group --- */
.excel-filter-section {
    background: linear-gradient(180deg, #003366 0%, #002244 100%);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Filter group title: white uppercase label */
.excel-filter-title {
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Clear-filters button (styled via HTML, not the Streamlit button) */
.excel-clear-btn {
    background: #dc3545;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    margin-top: 10px;
}

/* --- Excel-style KPI cards: centered value + label with min-height --- */
.excel-kpi-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Primary KPI variant: blue-accented left border */
.excel-kpi-card.primary {
    background: linear-gradient(145deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
    border-left: 4px solid #0066cc;
}

/* Accent KPI variant: red-accented left border for critical metrics */
.excel-kpi-card.accent {
    background: linear-gradient(145deg, rgba(220, 53, 69, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
    border-left: 4px solid #dc3545;
}

/* KPI value text */
.excel-kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}

/* Large variant for hero KPI values */
.excel-kpi-value.large {
    font-size: 2.5rem;
}

/* Money variant: green for positive financial values */
.excel-kpi-value.money {
    color: #4ade80;
}

/* KPI label: muted uppercase descriptor */
.excel-kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

/* KPI sub-label: even more muted secondary text */
.excel-kpi-sublabel {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}

/* --- Chart card container: subtle bordered box around chart regions --- */
.excel-chart-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.005) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

/* Chart card title: thin divider line underneath */
.excel-chart-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

/* --- Legend container and items for custom chart legends --- */
.excel-legend {
    padding: 10px;
    font-size: 0.75rem;
}

.excel-legend-item {
    display: flex;
    align-items: center;
    margin: 6px 0;
    color: #94a3b8;
}

/* Colored legend dots */
.excel-legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.excel-legend-dot.blue { background: #3b82f6; }
.excel-legend-dot.red { background: #dc3545; }
.excel-legend-dot.green { background: #22c55e; }
.excel-legend-dot.orange { background: #f97316; }

/* --- Progress bar (Excel style): stacked horizontal bar --- */
.excel-progress-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    height: 24px;
    overflow: hidden;
    margin: 8px 0;
}

.excel-progress-bar {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
}

/* Gender-coded progress bar variants (from demographic analysis) */
.excel-progress-bar.male {
    background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
}

.excel-progress-bar.female {
    background: linear-gradient(90deg, #dc3545 0%, #ef4444 100%);
}

/* --- Comparison grid: 2-column layout for side-by-side charts --- */
.excel-comparison-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}

/* --- Donut chart center value overlay --- */
.excel-donut-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}

/* --- Horizontal bar chart row layout --- */
.excel-bar-row {
    display: flex;
    align-items: center;
    margin: 4px 0;
}

.excel-bar-value {
    font-size: 0.7rem;
    color: #94a3b8;
    min-width: 60px;
    text-align: right;
    padding-right: 8px;
}

.excel-bar-name {
    font-size: 0.75rem;
    color: #cbd5e1;
    margin-left: 8px;
}

/* --- Product revenue cards: small icon + value + label tiles --- */
.excel-product-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}

.excel-product-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
}

.excel-product-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: #ffffff;
}

.excel-product-label {
    font-size: 0.7rem;
    color: #64748b;
}

/* --- Scatter plot container (stores analysis) --- */
.excel-scatter-container {
    position: relative;
    height: 200px;
}

/* --- Quarter grid: 4-column layout for Q1-Q4 donut mini-cards --- */
.excel-quarter-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}

.excel-quarter-card {
    text-align: center;
    padding: 8px;
}

.excel-quarter-label {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}
</style>
"""
