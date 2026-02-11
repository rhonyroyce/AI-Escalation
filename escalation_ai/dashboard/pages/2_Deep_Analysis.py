"""
Escalation AI -- Deep Analysis (Page Shim)
============================================

This file is a **page shim**: a thin wrapper that plugs the monolith's
``render_deep_analysis`` function into the unified Streamlit multi-page
application without modifying the monolith.

Bridge Pattern
--------------
Like every other page shim in this directory, it follows the three-step
protocol defined by ``esc_bridge.py``:

    1. ``init_escalation_state()`` -- initialize session-state defaults.
    2. ``inject_escalation_css()`` -- inject executive styling CSS.
    3. ``esc_load_and_filter(page_name)`` -- load data, render sidebar,
       return the filtered DataFrame.

This page passes ``page_name="Deep Analysis"``, which means the sidebar
will **not** show the Excel-style categorical filters (Category, Year,
Severity) -- only the date-range filter and metadata are displayed.  The
render function ``render_deep_analysis(df)`` from the monolith provides
detailed drill-down views including root-cause analysis, engineer
performance, time-series decompositions, and correlation matrices.

File location
-------------
``pages/2_Deep_Analysis.py`` -- the ``2_`` prefix places this page second
in the Streamlit sidebar navigation.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: add the project root to sys.path so ``escalation_ai.*``
# package imports resolve regardless of Streamlit's working directory.
# Four levels up: pages/ -> dashboard/ -> escalation_ai/ -> project_root/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import bridge helpers (shared across all page shims)
from escalation_ai.dashboard.esc_bridge import (
    init_escalation_state, inject_escalation_css, esc_load_and_filter,
)

# Import the monolith's render function for this page.
# render_deep_analysis() produces drill-down charts: root-cause trees,
# engineer quadrants, time-series decompositions, and heatmaps.
from escalation_ai.dashboard.streamlit_app import render_deep_analysis

# --- Step 1: Session state initialization (idempotent) ---
init_escalation_state()

# --- Step 2: CSS injection (must run once per page load) ---
inject_escalation_css()

# --- Step 3: Load data, render sidebar, get filtered DataFrame ---
# page_name="Deep Analysis" shows only the date-range filter (no
# Excel-style categorical filters) to keep the sidebar focused.
df = esc_load_and_filter(page_name="Deep Analysis")

# --- Step 4: Render if data is available ---
if df is not None:
    render_deep_analysis(df)
