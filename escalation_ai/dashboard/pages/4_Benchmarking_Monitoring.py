"""
Escalation AI -- Benchmarking & Monitoring (Page Shim)
========================================================

This file is a **page shim**: a thin wrapper that plugs the monolith's
``render_benchmarking_monitoring`` function into the unified Streamlit
multi-page application without modifying the monolith.

Bridge Pattern
--------------
Follows the standard three-step protocol from ``esc_bridge.py``:

    1. ``init_escalation_state()`` -- initialize session-state defaults.
    2. ``inject_escalation_css()`` -- inject executive styling CSS.
    3. ``esc_load_and_filter(page_name)`` -- load data, render sidebar,
       return the filtered DataFrame.

This page passes ``page_name="Benchmarking & Monitoring"``, which shows
only the date-range filter in the sidebar (no Excel-style categorical
filters).  The render function ``render_benchmarking_monitoring(df)``
from the monolith displays SLA compliance gauges, benchmark comparisons
against industry standards, aging burndown charts, and time-to-resolution
heatmaps.

File location
-------------
``pages/4_Benchmarking_Monitoring.py`` -- the ``4_`` prefix places this
page fourth in the Streamlit sidebar navigation.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: add the project root to sys.path.
# Four levels up: pages/ -> dashboard/ -> escalation_ai/ -> project_root/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import bridge helpers (shared across all page shims)
from escalation_ai.dashboard.esc_bridge import (
    init_escalation_state, inject_escalation_css, esc_load_and_filter,
)

# Import the monolith's render function for this page.
# render_benchmarking_monitoring() displays SLA gauges, benchmark bars,
# aging burndown charts, and time-to-resolution heatmaps.
from escalation_ai.dashboard.streamlit_app import render_benchmarking_monitoring

# --- Step 1: Session state initialization (idempotent) ---
init_escalation_state()

# --- Step 2: CSS injection (must run once per page load) ---
inject_escalation_css()

# --- Step 3: Load data, render sidebar, get filtered DataFrame ---
# page_name="Benchmarking & Monitoring" shows only the date-range filter.
df = esc_load_and_filter(page_name="Benchmarking & Monitoring")

# --- Step 4: Render if data is available ---
if df is not None:
    render_benchmarking_monitoring(df)
