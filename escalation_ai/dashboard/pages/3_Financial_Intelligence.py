"""
Escalation AI -- Financial Intelligence (Page Shim)
=====================================================

This file is a **page shim**: a thin wrapper that plugs the monolith's
``render_financial_analysis`` function into the unified Streamlit multi-page
application without modifying the monolith.

Bridge Pattern
--------------
Follows the standard three-step protocol from ``esc_bridge.py``:

    1. ``init_escalation_state()`` -- initialize session-state defaults.
    2. ``inject_escalation_css()`` -- inject executive styling CSS.
    3. ``esc_load_and_filter(page_name)`` -- load data, render sidebar,
       return the filtered DataFrame.

This page passes ``page_name="Financial Intelligence"``, which disables
the Excel-style categorical filters and shows only the date-range filter
and metadata in the sidebar.  The render function
``render_financial_analysis(df)`` from the monolith displays cost
waterfalls, cost-avoidance projections, ROI calculations, and financial
impact breakdowns by category and severity.

Data dependency
---------------
The ``Financial_Impact`` column in the DataFrame is computed by the
pipeline's ``price_catalog.py`` module during data processing.  If the
pipeline has not been run, this column will be absent and the render
function will show limited output.

File location
-------------
``pages/3_Financial_Intelligence.py`` -- the ``3_`` prefix places this
page third in the Streamlit sidebar navigation.
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
# render_financial_analysis() displays cost waterfalls, ROI projections,
# and financial impact breakdowns by category and severity.
from escalation_ai.dashboard.streamlit_app import render_financial_analysis

# --- Step 1: Session state initialization (idempotent) ---
init_escalation_state()

# --- Step 2: CSS injection (must run once per page load) ---
inject_escalation_css()

# --- Step 3: Load data, render sidebar, get filtered DataFrame ---
# page_name="Financial Intelligence" shows only the date-range filter.
df = esc_load_and_filter(page_name="Financial Intelligence")

from shared_theme import render_page_help
render_page_help("Financial Intelligence", "Cost impact analysis of escalations using the price catalog. Shows financial exposure by category.")

# --- Step 4: Render if data is available ---
if df is not None:
    render_financial_analysis(df)

    # --- Export button ---
    from export_utils import render_export_button
    summary = f"<p>Tickets analyzed: {len(df):,}</p>"
    if 'Financial_Impact' in df.columns:
        total_impact = df['Financial_Impact'].sum()
        avg_impact = df['Financial_Impact'].mean()
        summary += f"<p>Total financial impact: ${total_impact:,.0f}</p>"
        summary += f"<p>Average per ticket: ${avg_impact:,.0f}</p>"
        if 'AI_Category' in df.columns:
            by_cat = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False).head(5)
            rows = ''.join(f'<tr><td>{cat}</td><td>${val:,.0f}</td></tr>' for cat, val in by_cat.items())
            summary += f"<h3>Top Cost Categories</h3><table><tr><th>Category</th><th>Impact</th></tr>{rows}</table>"
    render_export_button("Financial Intelligence", summary)
