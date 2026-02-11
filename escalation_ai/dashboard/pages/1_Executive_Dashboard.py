"""
Escalation AI -- Executive Dashboard (Page Shim)
=================================================

This file is a **page shim**: a thin wrapper (~18 lines of logic) that
plugs the monolith's ``render_excel_dashboard`` function into the unified
Streamlit multi-page application (MPA) without modifying the monolith.

Bridge Pattern
--------------
Instead of refactoring the 9,265-line ``streamlit_app.py`` into proper
Streamlit pages, each page shim follows a three-step protocol:

    1. **Initialize** -- call ``init_escalation_state()`` to set default
       session-state keys (presentation mode, action items, etc.).
    2. **Inject CSS** -- call ``inject_escalation_css()`` to push the
       executive styling ``<style>`` block into the current page.
    3. **Load, filter, render** -- call ``esc_load_and_filter()`` which
       loads the escalation DataFrame (cached), renders the sidebar
       filters, and returns the filtered data.  If data is available,
       pass it to the monolith's render function.

This page passes ``page_name="Executive Dashboard"`` to the bridge,
which causes the full set of Excel-style sidebar filters (Category,
Year, Severity) to be displayed.  The render function called is
``render_excel_dashboard(df)`` from ``streamlit_app.py``.

File location
-------------
``pages/1_Executive_Dashboard.py`` -- the numeric prefix ``1_`` controls
the page ordering in Streamlit's sidebar navigation.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: add the project root (four levels up from this file:
#   pages/ -> dashboard/ -> escalation_ai/ -> project_root/)
# to sys.path so that ``escalation_ai.*`` package imports resolve correctly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the three bridge helpers that every page shim needs:
#   - init_escalation_state : sets session-state defaults (idempotent)
#   - inject_escalation_css : injects the shared CSS <style> block
#   - esc_load_and_filter   : loads data + renders sidebar + returns filtered df
from escalation_ai.dashboard.esc_bridge import (
    init_escalation_state, inject_escalation_css, esc_load_and_filter,
)

# Import the specific render function for this page from the monolith.
# render_excel_dashboard() produces the Executive Dashboard view with
# KPI cards, category breakdowns, severity distributions, and trend charts.
from escalation_ai.dashboard.streamlit_app import render_excel_dashboard

# --- Step 1: Session state initialization (idempotent) ---
init_escalation_state()

# --- Step 2: CSS injection (must run once per page load) ---
inject_escalation_css()

# --- Step 3: Load data, render sidebar filters, and get filtered DataFrame ---
# page_name="Executive Dashboard" enables the full Excel-style filter set.
df = esc_load_and_filter(page_name="Executive Dashboard")

# --- Step 4: Render the page content if data was loaded successfully ---
# esc_load_and_filter returns None when no data file is found; in that case
# the bridge already displayed an error message, so we simply skip rendering.
if df is not None:
    render_excel_dashboard(df)
