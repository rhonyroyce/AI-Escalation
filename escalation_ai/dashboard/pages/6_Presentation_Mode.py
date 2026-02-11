"""
Escalation AI -- Presentation Mode (Page Shim)
=================================================

This file is a **page shim**: a thin wrapper that plugs the monolith's
``render_presentation_mode`` function into the unified Streamlit multi-page
application without modifying the monolith.

Bridge Pattern
--------------
Follows the standard three-step protocol from ``esc_bridge.py``:

    1. ``init_escalation_state()`` -- initialize session-state defaults.
    2. ``inject_escalation_css()`` -- inject executive styling CSS.
    3. ``esc_load_and_filter(page_name)`` -- load data, render sidebar,
       return the filtered DataFrame.

This page passes ``page_name="Presentation Mode"``, which shows only the
date-range filter in the sidebar.  The render function
``render_presentation_mode(df)`` from the monolith transforms the
dashboard data into a navigable slide-deck interface suitable for
executive presentations, with Previous / Next controls and per-slide
chart rendering.

Session-state dependency
------------------------
This page relies on two session-state keys initialized by
``init_escalation_state()``:

- ``presentation_mode`` (bool): toggles the full-screen slide deck view.
- ``current_slide`` (int): tracks which slide is currently displayed.

These are preserved across Streamlit reruns so that clicking Next/Previous
does not reset the slide position.

CSS dependency
--------------
The ``.presentation-slide`` CSS class (in ``_ESCALATION_CSS``) provides
the full-viewport, vertically centered layout that makes each slide fill
the screen.

File location
-------------
``pages/6_Presentation_Mode.py`` -- the ``6_`` prefix places this page
last in the Streamlit sidebar navigation.
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
# render_presentation_mode() builds a navigable slide deck of executive
# summary charts with Previous/Next controls.
from escalation_ai.dashboard.streamlit_app import render_presentation_mode

# --- Step 1: Session state initialization (idempotent) ---
# Particularly important for this page because it initializes
# 'presentation_mode' and 'current_slide' used for slide navigation.
init_escalation_state()

# --- Step 2: CSS injection (must run once per page load) ---
# The .presentation-slide CSS class is essential for full-viewport slides.
inject_escalation_css()

# --- Step 3: Load data, render sidebar, get filtered DataFrame ---
# page_name="Presentation Mode" shows only the date-range filter.
df = esc_load_and_filter(page_name="Presentation Mode")

# --- Step 4: Render if data is available ---
if df is not None:
    render_presentation_mode(df)
