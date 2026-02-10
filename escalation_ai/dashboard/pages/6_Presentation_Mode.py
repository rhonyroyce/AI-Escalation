"""Escalation AI - Presentation Mode (page shim)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from escalation_ai.dashboard.esc_bridge import (
    init_escalation_state, inject_escalation_css, esc_load_and_filter,
)
from escalation_ai.dashboard.streamlit_app import render_presentation_mode

init_escalation_state()
inject_escalation_css()

df = esc_load_and_filter(page_name="Presentation Mode")
if df is not None:
    render_presentation_mode(df)
