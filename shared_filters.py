"""
Shared Filter State — Synchronizes filter context across Pulse and Escalation AI dashboards.
==============================================================================================

When a user selects filters in one dashboard and navigates to the other,
the filter context carries over via ``st.session_state.shared_filter_context``.

Usage::

    from shared_filters import update_context, render_active_filters, get_shared_context

    # Push filter selections (only non-None values are written)
    update_context(regions=['APAC', 'EMEA'], source='pulse')

    # Read current shared context
    ctx = get_shared_context()

    # Show active filter pills in sidebar
    render_active_filters()
"""

import streamlit as st
from datetime import date
from typing import Optional


def get_shared_context() -> dict:
    """Get the current shared filter context, initializing if needed."""
    if 'shared_filter_context' not in st.session_state:
        st.session_state.shared_filter_context = {
            'regions': [],
            'date_start': None,
            'date_end': None,
            'severity': [],
            'market': None,
            'source_dashboard': None,
        }
    return st.session_state.shared_filter_context


def update_context(
    regions: Optional[list] = None,
    date_start: Optional[date] = None,
    date_end: Optional[date] = None,
    severity: Optional[list] = None,
    market: Optional[str] = None,
    source: Optional[str] = None,
):
    """Update the shared filter context. Only non-None values are written."""
    ctx = get_shared_context()
    if regions is not None:
        ctx['regions'] = regions
    if date_start is not None:
        ctx['date_start'] = date_start
    if date_end is not None:
        ctx['date_end'] = date_end
    if severity is not None:
        ctx['severity'] = severity
    if market is not None:
        ctx['market'] = market
    if source is not None:
        ctx['source_dashboard'] = source


def render_active_filters():
    """Show active filter pills in sidebar."""
    ctx = get_shared_context()
    pills = []
    if ctx.get('regions'):
        pills.extend([f"Region: {r}" for r in ctx['regions']])
    if ctx.get('severity'):
        pills.extend([f"Severity: {s}" for s in ctx['severity']])
    if ctx.get('market'):
        pills.append(f"Market: {ctx['market']}")
    if ctx.get('date_start') and ctx.get('date_end'):
        pills.append(f"{ctx['date_start']} \u2192 {ctx['date_end']}")

    if pills:
        source = ctx.get('source_dashboard')
        source_label = f' <span style="color:#64748b;font-size:0.6rem;">(from {source})</span>' if source else ''
        pill_html = ' '.join([
            f'<span style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);'
            f'border-radius:12px;padding:2px 10px;font-size:0.7rem;color:#93c5fd;'
            f'margin:2px;display:inline-block;">{p}</span>'
            for p in pills
        ])
        st.sidebar.markdown(
            f'<div style="margin:8px 0;">'
            f'<div style="color:#64748b;font-size:0.65rem;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:4px;">Active Filters{source_label}</div>'
            f'{pill_html}</div>',
            unsafe_allow_html=True
        )
