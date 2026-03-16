"""
Shared Theme Module — Single Source of Truth for Cross-Dashboard Styling
========================================================================
Both Pulse and Escalation AI dashboards import from here to ensure
visual consistency across the unified platform.
"""
import streamlit as st

# ── Color Palette (Pulse-aligned Tailwind colors) ──────────────────
COLORS = {
    'bg_primary': '#0a0f1a',
    'bg_card': '#111827',
    'bg_card_hover': '#1f2937',
    'border_subtle': '#1e3a5f',
    'text_primary': '#e2e8f0',
    'text_secondary': '#94a3b8',
    'text_muted': '#64748b',
    'accent_blue': '#3b82f6',
    'accent_cyan': '#06b6d4',
    'status_red': '#ef4444',
    'status_amber': '#f59e0b',
    'status_green': '#22c55e',
    'status_dark_green': '#059669',
    'chart_blue': '#3b82f6',
    'chart_red': '#ef4444',
    'chart_green': '#22c55e',
    'chart_amber': '#f59e0b',
    'chart_purple': '#8b5cf6',
    'chart_cyan': '#06b6d4',
    'chart_pink': '#ec4899',
    'chart_orange': '#f97316',
}

FONT_FAMILY = "'DM Sans', 'Inter', 'Segoe UI', sans-serif"


def inject_shared_css():
    """Inject the shared platform CSS that normalizes both dashboards."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {{
        font-family: {FONT_FAMILY};
    }}

    /* Shared data freshness badge */
    .data-freshness {{
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 8px 0;
        text-align: center;
    }}
    .data-freshness .label {{
        color: {COLORS['text_secondary']};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .data-freshness .value {{
        color: {COLORS['text_primary']};
        font-weight: 600;
    }}

    /* Shared KPI card pattern */
    .shared-kpi {{
        background: linear-gradient(145deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid {COLORS['accent_blue']};
        text-align: center;
    }}
    .shared-kpi .value {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {COLORS['accent_blue']} 0%, {COLORS['accent_cyan']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .shared-kpi .label {{
        font-size: 0.8rem;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }}

    /* Responsive breakpoints */
    @media (max-width: 1024px) {{
        .stApp {{ font-size: 0.9rem; }}
        .shared-kpi .value {{ font-size: 2rem; }}
        .exec-kpi-value {{ font-size: 3rem; }}
        .main-header {{ font-size: 2rem; }}
    }}

    @media (max-width: 768px) {{
        .stApp {{ font-size: 0.85rem; }}
        .shared-kpi .value {{ font-size: 1.5rem; }}
        .shared-kpi {{ padding: 16px; }}
        .stPlotlyChart {{ min-height: 300px !important; }}
        [data-testid="stPlotlyChart"] {{
            min-height: 280px;
            max-height: 350px;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


def render_breadcrumb(dashboard: str, page: str, context: str = None):
    """Render a breadcrumb trail at the top of the page."""
    parts = [f'<span style="color:#64748b;">CSE Platform</span>']
    parts.append(f'<span style="color:#64748b;"> › </span>')
    parts.append(f'<span style="color:#94a3b8;">{dashboard}</span>')
    parts.append(f'<span style="color:#64748b;"> › </span>')
    parts.append(f'<span style="color:#e2e8f0;font-weight:600;">{page}</span>')
    if context:
        parts.append(f'<span style="color:#64748b;"> › </span>')
        parts.append(f'<span style="color:#3b82f6;">{context}</span>')

    st.markdown(
        f'<div style="font-size:0.75rem;padding:8px 0 16px 0;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:16px;">'
        f'{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_page_help(title: str, description: str, tips: list[str] = None):
    """Render a collapsible help section at the top of a page."""
    with st.expander(f"\u2139\ufe0f About this page: {title}", expanded=False):
        st.markdown(
            f"<p style='color:#94a3b8;font-size:0.9rem;'>{description}</p>",
            unsafe_allow_html=True,
        )
        if tips:
            for tip in tips:
                st.markdown(f"- {tip}")
