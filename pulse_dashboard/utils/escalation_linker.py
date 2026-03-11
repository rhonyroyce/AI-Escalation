"""
Pulse Dashboard - Escalation Intelligence Linker
==================================================
Loads Strategic_Report.xlsx and joins escalation data to Pulse projects
via Area <-> ticket_market_name and PM Name <-> engineer_name.
Used by: 6_Project_Details.py for cross-dashboard KPI cards.
"""
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd

REPORT_PATH = Path(__file__).parent.parent.parent / "Strategic_Report.xlsx"

# Mapping from Pulse Area codes to escalation market names
AREA_MARKET_MAP = {
    "SC": "SC", "CR": "CR", "VA": "VA",
    "SOCAL": "SOCAL", "HN": "HN", "NC": "NC",
}


@st.cache_data(ttl=3600)
def load_escalation_data() -> Optional[pd.DataFrame]:
    """Load escalation data from Strategic_Report.xlsx Scored Data sheet."""
    if not REPORT_PATH.exists():
        return None
    try:
        df = pd.read_excel(REPORT_PATH, sheet_name="Scored Data")
        return df
    except Exception:
        return None


def get_project_escalations(
    area: str,
    pm_name: str,
    esc_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Get escalation metrics linked to a Pulse project.

    Joins on Area <-> ticket_market_name and PM Name <-> tickets_data_engineer_name.
    Returns dict with: ticket_count, financial_impact, recurrence_probability,
    friction_score, categories.
    """
    if esc_df is None:
        esc_df = load_escalation_data()
    if esc_df is None or esc_df.empty:
        return {}

    # Match by area (market) OR by PM name (engineer)
    mask = pd.Series(False, index=esc_df.index)
    if area and 'ticket_market_name' in esc_df.columns:
        market = AREA_MARKET_MAP.get(area, area)
        mask = mask | (esc_df['ticket_market_name'].astype(str).str.upper() == market.upper())
    if pm_name and 'tickets_data_engineer_name' in esc_df.columns:
        mask = mask | (esc_df['tickets_data_engineer_name'].astype(str).str.lower() == pm_name.lower())

    matched = esc_df[mask]
    if matched.empty:
        return {}

    result = {
        'ticket_count': len(matched),
    }

    if 'Financial_Impact' in matched.columns:
        result['financial_impact'] = matched['Financial_Impact'].sum()
    if 'AI_Recurrence_Probability' in matched.columns:
        result['recurrence_probability'] = matched['AI_Recurrence_Probability'].mean()
    if 'Strategic_Friction_Score' in matched.columns:
        result['friction_score'] = matched['Strategic_Friction_Score'].mean()
    if 'AI_Category' in matched.columns:
        result['categories'] = matched['AI_Category'].value_counts().head(5).to_dict()

    return result
