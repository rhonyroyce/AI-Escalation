"""
Escalation AI -- Planning & Actions (Page Shim)
==================================================

This file is a **page shim**: a thin wrapper that plugs the monolith's
``render_planning_actions`` function into the unified Streamlit multi-page
application without modifying the monolith.

Bridge Pattern
--------------
Follows the standard three-step protocol from ``esc_bridge.py``:

    1. ``init_escalation_state()`` -- initialize session-state defaults.
    2. ``inject_escalation_css()`` -- inject executive styling CSS.
    3. ``esc_load_and_filter(page_name)`` -- load data, render sidebar,
       return the filtered DataFrame.

This page passes ``page_name="Planning & Actions"``, which shows only
the date-range filter in the sidebar.  The render function
``render_planning_actions(df)`` from the monolith provides action-item
tracking, priority-based remediation plans, and strategic recommendation
cards.

Session-state dependency
------------------------
This page relies on the ``action_items`` key in ``st.session_state``
(initialized by ``init_escalation_state()``).  User-created action items
persist across page navigations within the same browser session, so a
user can add items here, navigate to another page, and return to find
them intact.

File location
-------------
``pages/5_Planning_Actions.py`` -- the ``5_`` prefix places this page
fifth in the Streamlit sidebar navigation.
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
# render_planning_actions() provides action-item tracking, priority-based
# remediation plans, and strategic recommendation cards.
from escalation_ai.dashboard.streamlit_app import render_planning_actions

# --- Step 1: Session state initialization (idempotent) ---
# Particularly important for this page because it initializes the
# 'action_items' list that render_planning_actions() reads and writes.
init_escalation_state()

# --- Step 2: CSS injection (must run once per page load) ---
inject_escalation_css()

# --- Step 3: Load data, render sidebar, get filtered DataFrame ---
# page_name="Planning & Actions" shows only the date-range filter.
df = esc_load_and_filter(page_name="Planning & Actions")

# --- Step 4: Render if data is available ---
if df is not None:
    render_planning_actions(df)
