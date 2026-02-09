"""
Pulse Dashboard - Styles & Theme Configuration

Single source of truth for:
- 4-tier scoring thresholds and colors
- McKinsey color palette
- Plotly chart theme
- Dark theme CSS
- Dimension metadata
"""

import streamlit as st

# ============================================================================
# SCORING DIMENSIONS
# ============================================================================
SCORE_DIMENSIONS = ['Design', 'IX', 'PAG', 'RF Opt', 'Field', 'CSAT', 'PM Performance', 'Potential']

DIMENSION_LABELS = {
    'LOB': {0: 'Escalation', 1: 'Complaint/Concern', 2: 'BAU / NA', 3: 'Appreciation'},
    'CSAT': {0: 'Escalation', 1: 'Complaint', 2: 'Mixed', 3: 'Positive'},
    'PM Performance': {0: 'Escalation', 1: 'Issues', 2: 'On-time / Good Quality', 3: 'Exceptional'},
    'Potential': {0: 'Declining / At Risk', 1: 'Stagnant', 2: 'Moderate Opportunity', 3: 'Strong Future ROI'},
}

# ============================================================================
# 4-TIER PULSE STATUS (User-confirmed thresholds)
# ============================================================================
STATUS_CONFIG = {
    'Red':        {'color': '#ef4444', 'label': 'Critical',    'range': '1–13',  'min': 1,  'max': 13.999},
    'Yellow':     {'color': '#f59e0b', 'label': 'At Risk',     'range': '14–15', 'min': 14, 'max': 15.999},
    'Green':      {'color': '#22c55e', 'label': 'On Track',    'range': '16–19', 'min': 16, 'max': 19.999},
    'Dark Green': {'color': '#059669', 'label': 'Exceptional', 'range': '20–24', 'min': 20, 'max': 24},
}

STATUS_ORDER = ['Red', 'Yellow', 'Green', 'Dark Green']

CONTINUOUS_COLOR_SCALE = ['#ef4444', '#f59e0b', '#22c55e', '#059669']
COLOR_MIDPOINT = 16

DISCRETE_COLOR_MAP = {s: c['color'] for s, c in STATUS_CONFIG.items()}


def get_pulse_status(score: float) -> str:
    if score >= 20:
        return 'Dark Green'
    if score >= 16:
        return 'Green'
    if score >= 14:
        return 'Yellow'
    return 'Red'


def get_pulse_color(score: float) -> str:
    return STATUS_CONFIG[get_pulse_status(score)]['color']


def pulse_css_class(score) -> str:
    """CSS class for pulse score cell in HTML tables."""
    import pandas as pd
    if pd.isna(score):
        return ""
    s = float(score)
    if s >= 20:
        return "pulse-darkgreen"
    if s >= 16:
        return "pulse-green"
    if s >= 14:
        return "pulse-yellow"
    return "pulse-red"


def heat_css_class(score) -> str:
    """CSS class for weekly heatmap cell."""
    import pandas as pd
    if pd.isna(score):
        return ""
    s = float(score)
    if s >= 20:
        return "heat-darkgreen"
    if s >= 17:
        return "heat-green"
    if s >= 16:
        return "heat-lime"
    if s >= 14:
        return "heat-yellow"
    return "heat-red"


def score_css_class(score) -> str:
    """CSS class for individual dimension score cell (0-3)."""
    import pandas as pd
    if pd.isna(score):
        return ""
    s = float(score)
    if s >= 2.5:
        return "score-high"
    if s >= 2.0:
        return "score-mid"
    if s >= 1.5:
        return "score-low"
    return "score-critical"


# ============================================================================
# MCKINSEY COLOR PALETTE
# ============================================================================
MCKINSEY_COLORS = {
    'primary_blue': '#004165',
    'secondary_blue': '#0077B6',
    'accent_teal': '#00A5A8',
    'positive': '#00843D',
    'negative': '#E31B23',
    'warning': '#F2A900',
    'neutral': '#6D6E71',
    'light_gray': '#D0D0CE',
    'modern_blue': '#0066CC',
    'bright_blue': '#00BFFF',
}

DIMENSION_COLORS = {
    'Design':          '#1B5E7B',
    'IX':              '#7B2D8E',
    'PAG':             '#00BCD4',
    'RF Opt':          '#8D6E63',
    'Field':           '#FFC107',
    'CSAT':            '#26A69A',
    'PM Performance':  '#9E9E9E',
    'Potential':       '#E040FB',
}

REGION_LINE_COLORS = {
    'Central': '#60A5FA',
    'NE':      '#34D399',
    'South':   '#FBBF24',
    'West':    '#F472B6',
}


# ============================================================================
# PLOTLY THEME
# ============================================================================
def get_plotly_theme() -> dict:
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#E0E0E0'),
        margin=dict(l=40, r=40, t=50, b=40),
    )


# Default axis styling — apply with fig.update_xaxes() / fig.update_yaxes()
AXIS_STYLE = dict(gridcolor='#1e293b', zerolinecolor='#1e293b')


# ============================================================================
# CSS INJECTION
# ============================================================================
def inject_css():
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* Plotly chart min height */
    .stPlotlyChart { min-height: 400px !important; }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        margin: 10px 0;
    }

    /* ── KPI Container ── */
    .kpi-container {
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid #0066CC;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 102, 204, 0.3);
    }
    .kpi-container.critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
        border-left-color: #ef4444;
    }
    .kpi-container.warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(255, 140, 0, 0.25) 100%);
        border-left-color: #f59e0b;
    }
    .kpi-container.success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
        border-left-color: #22c55e;
    }
    .kpi-container.exceptional {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.15) 0%, rgba(0, 80, 60, 0.25) 100%);
        border-left-color: #059669;
    }

    /* ── KPI Typography ── */
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-value.red { background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .kpi-value.yellow { background: linear-gradient(135deg, #f59e0b, #d97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .kpi-value.green { background: linear-gradient(135deg, #22c55e, #16a34a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .kpi-value.dark-green { background: linear-gradient(135deg, #059669, #047857); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }
    .kpi-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-positive { color: #22c55e; }
    .delta-negative { color: #ef4444; }

    /* ── SCR Boxes (Situation / Complications / Resolution) ── */
    .scr-situation {
        background: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .scr-complication {
        background: #1c1917;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .scr-resolution {
        background: #052e16;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .scr-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }

    /* ── Insight Callout ── */
    .insight-callout {
        background: linear-gradient(135deg, #1e3a5f, #0c4a6e);
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;
        margin: 16px 0;
    }

    /* ── Pulse Dot Indicator ── */
    .pulse-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-anim 2s infinite;
    }
    .pulse-dot.red { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
    .pulse-dot.yellow { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
    .pulse-dot.green { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
    .pulse-dot.dark-green { background: #059669; box-shadow: 0 0 8px #059669; }
    @keyframes pulse-anim {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* ── Headers ── */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    /* ── Executive Card ── */
    .exec-card {
        background: linear-gradient(145deg, rgba(0, 40, 85, 0.9) 0%, rgba(0, 20, 40, 0.95) 100%);
        border-radius: 20px;
        padding: 32px;
        border: 1px solid rgba(0, 150, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 16px 0;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-critical { background: #ef4444; color: white; }
    .badge-warning { background: #f59e0b; color: #212529; }
    .badge-success { background: #22c55e; color: white; }
    .badge-exceptional { background: #059669; color: white; }
    .badge-info { background: #0066CC; color: white; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1929 0%, #001e3c 100%);
    }

    /* ── Tabs ── */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066CC 0%, #004C97 100%) !important;
        color: white !important;
        border-radius: 8px 8px 0 0;
    }

    /* ── Recommendation Table ── */
    .rec-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
    }
    .rec-table th {
        background: rgba(0, 102, 204, 0.3);
        padding: 10px 12px;
        text-align: left;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #334155;
    }
    .rec-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #1e293b;
        font-size: 0.9rem;
    }

    /* ── Section Titles ── */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin: 0.8rem 0 0.4rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #1e3a5f;
    }

    /* ── Matrix Table (Region/Area hierarchy) ── */
    .matrix-container {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        overflow: hidden;
    }
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;
    }
    .matrix-table th {
        background: #162a4a;
        color: #94a3b8;
        font-weight: 600;
        padding: 0.5rem 0.4rem;
        text-align: center;
        border-bottom: 2px solid #2563eb;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .matrix-table td {
        padding: 0.35rem 0.4rem;
        border-bottom: 1px solid #1e293b;
        text-align: center;
        color: #cbd5e1;
    }
    .matrix-table tr:hover td { background: rgba(37, 99, 235, 0.1); }
    .matrix-table .region-row td { background: #111d32; font-weight: 600; }
    .matrix-table .area-row td { background: transparent; padding-left: 1.5rem; }
    .matrix-table .total-row td { background: #1a365d; font-weight: 700; border-top: 2px solid #2563eb; }
    .matrix-table .region-name { text-align: left !important; font-weight: 600; color: #e2e8f0; }
    .matrix-table .area-name { text-align: left !important; color: #94a3b8; }

    /* ── Heatmap Scroll Container ── */
    .heatmap-scroll-container {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        overflow-x: auto;
        max-width: 100%;
    }
    .heatmap-scroll-container .heatmap-table th:first-child,
    .heatmap-scroll-container .heatmap-table td:first-child {
        position: sticky;
        left: 0;
        z-index: 1;
        background: #0d1526;
    }
    .heatmap-scroll-container .heatmap-table .total-row td:first-child {
        background: #1a365d;
    }
    .heatmap-scroll-container .heatmap-table th:first-child {
        z-index: 2;
        background: #162a4a;
    }

    /* ── Score Cell Colors (dimension 0-3) ── */
    .score-cell {
        font-weight: 600;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        min-width: 35px;
        display: inline-block;
    }
    .score-high     { background: rgba(34, 197, 94, 0.2); color: #86efac; }
    .score-mid      { background: rgba(59, 130, 246, 0.15); color: #93c5fd; }
    .score-low      { background: rgba(251, 191, 36, 0.2); color: #fcd34d; }
    .score-critical { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }

    /* ── Pulse Score Cell Colors (0-24) ── */
    .pulse-red       { background: #7f1d1d; color: #fca5a5; }
    .pulse-yellow    { background: #78350f; color: #fcd34d; }
    .pulse-green     { background: #1e4d3a; color: #6ee7b7; }
    .pulse-darkgreen { background: #065f46; color: #6ee7b7; }

    /* ── Weekly Heatmap Table ── */
    .heatmap-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;
    }
    .heatmap-table th {
        background: #162a4a;
        color: #94a3b8;
        font-weight: 600;
        padding: 0.4rem 0.3rem;
        text-align: center;
        font-size: 0.7rem;
        border-bottom: 2px solid #2563eb;
    }
    .heatmap-table td {
        padding: 0.3rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.7rem;
        border-bottom: 1px solid #1e293b;
    }
    .heatmap-table .region-col {
        text-align: left;
        padding-left: 0.5rem;
        color: #e2e8f0;
        min-width: 60px;
    }
    .heatmap-cell { border-radius: 2px; padding: 0.2rem 0.3rem; }
    .heat-darkgreen { background: #065f46; color: #bbf7d0; }
    .heat-green     { background: #166534; color: #bbf7d0; }
    .heat-lime      { background: #3f6212; color: #d9f99d; }
    .heat-yellow    { background: #854d0e; color: #fef08a; }
    .heat-red       { background: #991b1b; color: #fecaca; }

    /* ── Ratings Legend ── */
    .legend-table {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        overflow: hidden;
        font-size: 0.7rem;
        min-width: 140px;
    }
    .legend-table th {
        background: #162a4a;
        color: #94a3b8;
        padding: 0.4rem 0.5rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 1px solid #2563eb;
    }
    .legend-table td {
        padding: 0.3rem 0.5rem;
        border-bottom: 1px solid #1e293b;
        color: #cbd5e1;
    }
    .rating-badge {
        display: inline-block;
        width: 18px; height: 18px;
        border-radius: 3px;
        text-align: center;
        line-height: 18px;
        font-weight: 700;
        font-size: 0.65rem;
        margin-right: 0.4rem;
    }
    .rating-0 { background: #dc2626; color: white; }
    .rating-1 { background: #f97316; color: white; }
    .rating-2 { background: #3b82f6; color: white; }
    .rating-3 { background: #22c55e; color: white; }

    /* ── Notes Box ── */
    .notes-box {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        font-size: 0.75rem;
        color: #94a3b8;
        line-height: 1.4;
    }
    .notes-title { font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }

    /* ── Drill-Down Panel ── */
    .drilldown-panel {
        background: linear-gradient(180deg, #0f1d32 0%, #0d1526 100%);
        border: 1px solid #2563eb;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .drilldown-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e3a5f;
    }
    .drilldown-badge {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .drilldown-context { color: #64748b; font-size: 0.8rem; }

    /* ── Hide default streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0.5rem 1rem 1rem 1rem; max-width: 100%; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1526; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""", unsafe_allow_html=True)
