"""
Pulse Dashboard - Styles & Theme Configuration
===============================================

This module is the SINGLE SOURCE OF TRUTH for every visual and scoring
constant used by the Pulse Dashboard.  Nothing in the rest of the
application hard-codes colors, thresholds, or CSS -- it all comes from here.

Scoring System Overview
-----------------------
The Pulse Score measures account health across **8 dimensions**, each scored
on a **0-3 integer scale**:

    Dimension        What it captures
    ---------------  -----------------------------------------------
    Design           Quality / maturity of the network design
    IX               Interconnect / integration health
    PAG              Proactive Account Governance
    RF Opt           RF optimisation & tuning status
    Field            Field engineering execution
    CSAT             Customer satisfaction sentiment
    PM Performance   Project management delivery
    Potential        Future revenue / relationship opportunity

Because there are 8 dimensions at 0-3 each, the **Total Pulse Score**
ranges from **0 to 24**.

Status Thresholds (4-tier traffic light)
----------------------------------------
    Status       Score Range   Hex Color   Meaning
    -----------  -----------   ---------   --------------------
    Red          1 - 13        #ef4444     Critical -- immediate action needed
    Yellow       14 - 15       #f59e0b     At Risk  -- trending poorly
    Green        16 - 19       #22c55e     On Track -- healthy
    Dark Green   20 - 24       #059669     Exceptional -- best-in-class

These thresholds were **confirmed by the business stakeholders** and are
encoded in ``STATUS_CONFIG``.

Color Philosophy
----------------
All colors follow a **McKinsey-inspired dark theme** palette:
- Deep navy / slate backgrounds for readability on dark mode dashboards
- Saturated accent colors (blue, teal, green) for data ink
- Status colors that map directly to the 4-tier pulse rating

Module Contents at a Glance
----------------------------
- ``SCORE_DIMENSIONS`` / ``DIMENSION_LABELS`` -- dimension metadata
- ``STATUS_CONFIG`` / ``STATUS_ORDER`` -- 4-tier status thresholds & colors
- ``get_pulse_status()`` / ``get_pulse_color()`` -- score -> status/color helpers
- ``pulse_css_class()`` / ``heat_css_class()`` / ``score_css_class()`` -- CSS class helpers
- ``MCKINSEY_COLORS`` / ``DIMENSION_COLORS`` / ``REGION_LINE_COLORS`` -- palettes
- ``get_plotly_theme()`` / ``AXIS_STYLE`` -- Plotly chart theming
- ``inject_css()`` -- injects the full dark-theme stylesheet into Streamlit
"""

import streamlit as st

# ============================================================================
# SCORING DIMENSIONS
# ============================================================================

# Ordered list of the 8 scoring dimensions.  This ordering is used when
# rendering radar charts, score tables, and anywhere dimension columns
# need to appear in a consistent sequence.
SCORE_DIMENSIONS = ['Design', 'IX', 'PAG', 'RF Opt', 'Field', 'CSAT', 'PM Performance', 'Potential']

# Human-readable labels for each integer score (0-3) within selected
# dimensions.  Only dimensions whose labels are NOT self-evident get
# entries here.  For example "Design" doesn't need labels because its
# 0-3 meaning is implied by context, but "LOB" (Line of Business) and
# "CSAT" benefit from explicit textual descriptions at each level.
#
# Structure:  { dimension_name: { score_int: display_label, ... }, ... }
DIMENSION_LABELS = {
    # LOB = Line of Business context -- classifies the nature of the engagement
    'LOB': {
        0: 'Escalation',          # Score 0: active escalation in progress
        1: 'Complaint/Concern',   # Score 1: complaint or concern raised
        2: 'BAU / NA',            # Score 2: business-as-usual or not applicable
        3: 'Appreciation',        # Score 3: positive feedback / appreciation
    },
    # CSAT = Customer Satisfaction sentiment captured from surveys / interactions
    'CSAT': {
        0: 'Escalation',          # Score 0: escalation-level dissatisfaction
        1: 'Complaint',           # Score 1: explicit complaint
        2: 'Mixed',               # Score 2: mixed / neutral feedback
        3: 'Positive',            # Score 3: clearly positive sentiment
    },
    # PM Performance = Project Management delivery quality
    'PM Performance': {
        0: 'Escalation',          # Score 0: PM issues causing escalation
        1: 'Issues',              # Score 1: notable PM issues present
        2: 'On-time / Good Quality',  # Score 2: meeting expectations
        3: 'Exceptional',         # Score 3: above-and-beyond delivery
    },
    # Potential = Forward-looking revenue / relationship trajectory
    'Potential': {
        0: 'Declining / At Risk', # Score 0: account is declining or at risk of churn
        1: 'Stagnant',            # Score 1: no growth trajectory
        2: 'Moderate Opportunity', # Score 2: some upside potential
        3: 'Strong Future ROI',   # Score 3: high-growth opportunity
    },
}

# ============================================================================
# 4-TIER PULSE STATUS (User-confirmed thresholds)
# ============================================================================

# Master configuration for the 4-tier pulse status system.
# Each key is a status name; each value carries:
#   - color : hex color used in charts, badges, and CSS
#   - label : human-friendly display label
#   - range : text representation of the score band (for tooltips / legends)
#   - min   : inclusive lower bound of the band
#   - max   : exclusive upper bound (except Dark Green which includes 24)
#
# The fractional max values (e.g. 13.999) ensure that scores are bucketed
# correctly when compared with >= / < operators; in practice, scores are
# integers so 13.999 is functionally equivalent to "< 14".
STATUS_CONFIG = {
    'Red':        {'color': '#ef4444', 'label': 'Critical',    'range': '1–13',  'min': 1,  'max': 13.999},
    'Yellow':     {'color': '#f59e0b', 'label': 'At Risk',     'range': '14–15', 'min': 14, 'max': 15.999},
    'Green':      {'color': '#22c55e', 'label': 'On Track',    'range': '16–19', 'min': 16, 'max': 19.999},
    'Dark Green': {'color': '#059669', 'label': 'Exceptional', 'range': '20–24', 'min': 20, 'max': 24},
}

# Canonical ordering of statuses from worst to best.
# Used by charts and legends that need a deterministic severity ordering.
STATUS_ORDER = ['Red', 'Yellow', 'Green', 'Dark Green']

# A 4-stop linear color scale for Plotly continuous color maps.
# Progresses from Red (worst) -> Yellow -> Green -> Dark Green (best).
CONTINUOUS_COLOR_SCALE = ['#ef4444', '#f59e0b', '#22c55e', '#059669']

# The midpoint score (16) that separates the "below average" colors from
# the "above average" colors on diverging continuous scales.  A score of
# 16 is the threshold between Yellow (At Risk) and Green (On Track).
COLOR_MIDPOINT = 16

# Convenience dict that maps status name -> hex color string directly,
# derived from STATUS_CONFIG to avoid duplication.
# Example: {'Red': '#ef4444', 'Yellow': '#f59e0b', ...}
DISCRETE_COLOR_MAP = {s: c['color'] for s, c in STATUS_CONFIG.items()}


def get_pulse_status(score: float) -> str:
    """Return the 4-tier status name for a given Total Pulse Score.

    The thresholds (confirmed by stakeholders) are:
        >= 20  ->  'Dark Green'  (Exceptional)
        >= 16  ->  'Green'       (On Track)
        >= 14  ->  'Yellow'      (At Risk)
        <  14  ->  'Red'         (Critical)

    Parameters
    ----------
    score : float
        Total Pulse Score in the range 0-24 (sum of eight 0-3 dimensions).

    Returns
    -------
    str
        One of 'Dark Green', 'Green', 'Yellow', or 'Red'.
    """
    if score >= 20:
        return 'Dark Green'
    if score >= 16:
        return 'Green'
    if score >= 14:
        return 'Yellow'
    return 'Red'


def get_pulse_color(score: float) -> str:
    """Return the hex color string for a given Total Pulse Score.

    This is a convenience wrapper that chains ``get_pulse_status()`` into
    a ``STATUS_CONFIG`` color lookup.

    Parameters
    ----------
    score : float
        Total Pulse Score (0-24).

    Returns
    -------
    str
        Hex color string, e.g. '#22c55e'.
    """
    return STATUS_CONFIG[get_pulse_status(score)]['color']


def pulse_css_class(score) -> str:
    """Return a CSS class name for styling a Total Pulse Score cell in HTML tables.

    The returned class maps to one of the ``.pulse-*`` rules defined in
    ``inject_css()``.  These provide background + text color combinations
    tuned for readability on the dark theme.

    Thresholds mirror ``get_pulse_status()`` exactly:
        >= 20  ->  'pulse-darkgreen'
        >= 16  ->  'pulse-green'
        >= 14  ->  'pulse-yellow'
        <  14  ->  'pulse-red'

    Parameters
    ----------
    score : numeric or NaN
        Total Pulse Score (0-24).  NaN / None returns an empty string so
        that missing data renders without any special styling.

    Returns
    -------
    str
        CSS class name (without leading dot), or '' for NaN/missing values.
    """
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
    """Return a CSS class name for a weekly heatmap cell.

    The heatmap uses a **5-tier** color scheme (finer granularity than the
    4-tier pulse status) to give more visual differentiation when scores
    are displayed week-over-week:

        >= 20  ->  'heat-darkgreen'   (deep green -- exceptional)
        >= 17  ->  'heat-green'       (green -- solid)
        >= 16  ->  'heat-lime'        (lime / olive -- just above threshold)
        >= 14  ->  'heat-yellow'      (amber -- at risk)
        <  14  ->  'heat-red'         (red -- critical)

    The extra "lime" band at 16-16.999 highlights accounts that are *just
    barely* On Track, providing an early-warning visual cue.

    Parameters
    ----------
    score : numeric or NaN
        Total Pulse Score (0-24).

    Returns
    -------
    str
        CSS class name, or '' for NaN/missing values.
    """
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
    """Return a CSS class name for an individual dimension score cell (0-3 scale).

    Unlike ``pulse_css_class`` (which operates on the 0-24 total), this
    function is used for **single-dimension** cells where each score is
    between 0 and 3.  The 4 bands are:

        >= 2.5  ->  'score-high'      (green tint -- strong)
        >= 2.0  ->  'score-mid'       (blue tint -- acceptable)
        >= 1.5  ->  'score-low'       (yellow tint -- needs attention)
        <  1.5  ->  'score-critical'  (red tint -- failing)

    Parameters
    ----------
    score : numeric or NaN
        Individual dimension score (0-3).

    Returns
    -------
    str
        CSS class name, or '' for NaN/missing values.
    """
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

# Core brand / data-visualization palette inspired by McKinsey's visual
# identity guidelines.  These colors are used throughout charts, badges,
# and UI accents to maintain a cohesive professional aesthetic.
MCKINSEY_COLORS = {
    'primary_blue':   '#004165',  # Deep navy -- primary brand anchor
    'secondary_blue': '#0077B6',  # Medium blue -- secondary accents
    'accent_teal':    '#00A5A8',  # Teal -- complementary accent
    'positive':       '#00843D',  # Forest green -- positive indicators
    'negative':       '#E31B23',  # Red -- negative indicators / alerts
    'warning':        '#F2A900',  # Amber -- warning state
    'neutral':        '#6D6E71',  # Mid-gray -- neutral / inactive text
    'light_gray':     '#D0D0CE',  # Light gray -- borders & dividers
    'modern_blue':    '#0066CC',  # Bright modern blue -- buttons & links
    'bright_blue':    '#00BFFF',  # Vivid sky blue -- highlights & gradients
}

# Per-dimension colors used in radar charts, stacked bars, and anywhere
# individual dimensions need visually distinct representation.
# Each dimension receives a unique hue so they are easy to distinguish
# even without labels (important for small multiples / sparklines).
DIMENSION_COLORS = {
    'Design':          '#1B5E7B',  # Steel blue
    'IX':              '#7B2D8E',  # Purple
    'PAG':             '#00BCD4',  # Cyan
    'RF Opt':          '#8D6E63',  # Brown
    'Field':           '#FFC107',  # Amber / gold
    'CSAT':            '#26A69A',  # Teal green
    'PM Performance':  '#9E9E9E',  # Neutral gray
    'Potential':       '#E040FB',  # Magenta / pink-purple
}

# Line colors for regional trend charts.  Each of the 4 geographic regions
# gets a distinct, high-contrast color to support overlay line charts.
REGION_LINE_COLORS = {
    'Central': '#60A5FA',  # Light blue
    'NE':      '#34D399',  # Mint green
    'South':   '#FBBF24',  # Gold / amber
    'West':    '#F472B6',  # Pink
}


# ============================================================================
# PLOTLY THEME
# ============================================================================

def get_plotly_theme() -> dict:
    """Return a base Plotly layout configuration for the dark dashboard theme.

    This dict is designed to be unpacked into ``fig.update_layout(**get_plotly_theme())``
    and provides:
    - Transparent paper and plot backgrounds (the dark CSS background shows through)
    - Inter font family with light gray (#E0E0E0) text
    - Compact margins suitable for Streamlit column layouts

    Returns
    -------
    dict
        Plotly layout keyword arguments.
    """
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',   # Transparent -- lets CSS background show
        plot_bgcolor='rgba(0,0,0,0)',    # Transparent plot area
        font=dict(family='Inter', color='#E0E0E0'),  # Light gray text with Inter font
        margin=dict(l=40, r=40, t=50, b=40),          # Compact margins (px)
    )


# Default styling for Plotly axes -- provides subtle grid lines that
# blend with the dark theme.  Apply with:
#   fig.update_xaxes(**AXIS_STYLE)
#   fig.update_yaxes(**AXIS_STYLE)
AXIS_STYLE = dict(
    gridcolor='#1e293b',      # Dark slate grid lines (barely visible)
    zerolinecolor='#1e293b',  # Zero-line matches grid for visual consistency
)


# ============================================================================
# CSS INJECTION
# ============================================================================

def inject_css():
    """Inject the complete dark-theme CSS stylesheet into the Streamlit app.

    This function should be called **once** at the top of the main Streamlit
    page (before any other content is rendered).  It uses
    ``st.markdown(..., unsafe_allow_html=True)`` to embed a ``<style>`` block
    that defines every CSS class referenced by the dashboard's HTML fragments.

    The stylesheet covers:
    - **Typography**: Google Fonts "Inter" import, font weights 300-800
    - **Glass Card**: Frosted-glass container with ``backdrop-filter: blur``
    - **KPI Containers**: Gradient cards with left-border status color and
      hover lift animation; variants for critical / warning / success / exceptional
    - **KPI Typography**: Large gradient-text values with status-colored variants,
      uppercase labels, and delta (change) indicators
    - **SCR Boxes**: Situation / Complication / Resolution storytelling framework
      with color-coded left borders (blue / red / green)
    - **Insight Callout**: Highlighted tip / insight box with teal accent
    - **Pulse Dot**: Animated circular indicator with pulsing glow animation
    - **Headers**: Gradient-text main header
    - **Executive Card**: Premium dark card with subtle blue border glow
    - **Badges**: Pill-shaped status badges (critical / warning / success / exceptional / info)
    - **Sidebar**: Deep navy gradient background
    - **Tabs**: Active tab with blue gradient background
    - **Recommendation Table**: Styled ``<table>`` for action-item lists
    - **Section Title**: Subtle labeled divider
    - **Matrix Table**: Region / Area hierarchical table with row-type styling
      (region-row, area-row, total-row)
    - **Heatmap Scroll Container**: Horizontally scrollable wrapper with sticky
      first column for account names
    - **Score Cells**: Per-dimension (0-3) color classes (high / mid / low / critical)
    - **Pulse Score Cells**: Total score (0-24) color classes (red / yellow / green / darkgreen)
    - **Weekly Heatmap Table**: Compact table with 5-tier color cells
    - **Legend Table**: Compact legend showing the 0-3 rating badge colors
    - **Notes Box**: Muted annotation container
    - **Drill-Down Panel**: Expandable detail panel with blue/purple accent
    - **Streamlit Chrome**: Hides default Streamlit header, footer, and menu
    - **Custom Scrollbar**: Thin dark scrollbar matching the theme

    Returns
    -------
    None
        Side effect: injects HTML/CSS into the Streamlit page.
    """
    st.markdown("""
<style>
    /* ================================================================
       FONT IMPORT
       Load Google's "Inter" variable font with weights 300-800.
       Inter is a highly legible UI typeface designed for screens.
       ================================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Apply Inter to the entire Streamlit application container */
    .stApp { font-family: 'Inter', sans-serif; }

    /* Ensure Plotly charts have a minimum height so small datasets
       don't collapse into unreadably tiny visualizations */
    .stPlotlyChart { min-height: 400px !important; }

    /* ================================================================
       GLASS CARD
       A frosted-glass container used to group related content.
       Uses backdrop-filter blur for a modern translucent effect
       over the dark background.
       ================================================================ */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);       /* Nearly transparent white */
        backdrop-filter: blur(10px);                   /* Frosted glass blur */
        border-radius: 16px;                           /* Rounded corners */
        border: 1px solid rgba(255, 255, 255, 0.1);   /* Subtle white border */
        padding: 24px;
        margin: 10px 0;
    }

    /* ================================================================
       KPI CONTAINER
       Primary card for displaying Key Performance Indicators.
       Features a gradient background, left accent border, and a
       hover animation that lifts the card with a glowing shadow.
       ================================================================ */
    .kpi-container {
        background: linear-gradient(135deg, rgba(0, 102, 204, 0.15) 0%, rgba(0, 51, 102, 0.25) 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid #0066CC;     /* Default blue accent border */
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;  /* Smooth hover animation */
    }
    /* Hover state: lift card 4px and add a blue glow */
    .kpi-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 102, 204, 0.3);
    }
    /* Status variant: Critical (Red) -- red gradient + red border */
    .kpi-container.critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(139, 0, 0, 0.25) 100%);
        border-left-color: #ef4444;
    }
    /* Status variant: Warning (Yellow) -- amber gradient + amber border */
    .kpi-container.warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(255, 140, 0, 0.25) 100%);
        border-left-color: #f59e0b;
    }
    /* Status variant: Success (Green) -- green gradient + green border */
    .kpi-container.success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(0, 100, 0, 0.25) 100%);
        border-left-color: #22c55e;
    }
    /* Status variant: Exceptional (Dark Green) -- deep teal gradient + teal border */
    .kpi-container.exceptional {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.15) 0%, rgba(0, 80, 60, 0.25) 100%);
        border-left-color: #059669;
    }

    /* ================================================================
       KPI TYPOGRAPHY
       Styles for the numeric value, label, and delta (change) text
       inside KPI containers.
       ================================================================ */

    /* The large numeric KPI value -- uses gradient text via
       background-clip trick (paint gradient, clip to text shape) */
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);  /* Default blue gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Status-colored KPI value variants.  Each overrides the gradient
       to match the corresponding pulse status color. */
    .kpi-value.red { background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .kpi-value.yellow { background: linear-gradient(135deg, #f59e0b, #d97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .kpi-value.green { background: linear-gradient(135deg, #22c55e, #16a34a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .kpi-value.dark-green { background: linear-gradient(135deg, #059669, #047857); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* KPI label -- small, uppercase, muted text below the value */
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;              /* Slate gray -- low emphasis */
        text-transform: uppercase;
        letter-spacing: 1.5px;       /* Wide tracking for readability at small size */
        margin-top: 8px;
    }
    /* KPI delta -- shows week-over-week or period-over-period change */
    .kpi-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 4px;
    }
    /* Delta direction colors: green for improvement, red for decline */
    .delta-positive { color: #22c55e; }
    .delta-negative { color: #ef4444; }

    /* ================================================================
       SCR BOXES (Situation / Complication / Resolution)
       Used for the McKinsey SCR storytelling framework in executive
       summaries.  Each box has a distinct left-border color:
         - Situation:    blue   (#3b82f6) -- neutral context
         - Complication: red    (#ef4444) -- problem statement
         - Resolution:   green  (#22c55e) -- recommended action
       ================================================================ */
    .scr-situation {
        background: #0f172a;                 /* Deep navy background */
        border-left: 4px solid #3b82f6;      /* Blue accent -- informational */
        padding: 1rem;
        border-radius: 0 8px 8px 0;         /* Rounded right corners only */
        margin-bottom: 8px;
    }
    .scr-complication {
        background: #1c1917;                 /* Dark warm gray background */
        border-left: 4px solid #ef4444;      /* Red accent -- problem */
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .scr-resolution {
        background: #052e16;                 /* Dark green background */
        border-left: 4px solid #22c55e;      /* Green accent -- solution */
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    /* SCR section title (e.g., "SITUATION", "COMPLICATION", "RESOLUTION") */
    .scr-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;           /* Extra-wide tracking for label feel */
        margin-bottom: 6px;
    }

    /* ================================================================
       INSIGHT CALLOUT
       A highlighted callout box for surfacing AI-generated insights
       or analyst commentary.  Uses a teal/sky-blue accent to
       distinguish it from SCR content.
       ================================================================ */
    .insight-callout {
        background: linear-gradient(135deg, #1e3a5f, #0c4a6e);  /* Blue-to-teal gradient */
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;   /* Sky blue accent */
        margin: 16px 0;
    }

    /* ================================================================
       PULSE DOT INDICATOR
       A small animated circle that visually represents pulse status.
       Used inline next to account names in tables and headers.
       The dot has a colored glow (box-shadow) and a gentle
       scale/opacity animation on a 2-second loop.
       ================================================================ */
    .pulse-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;          /* Perfect circle */
        margin-right: 8px;
        animation: pulse-anim 2s infinite;  /* Continuous breathing animation */
    }
    /* Status-colored dot variants with matching glow */
    .pulse-dot.red { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
    .pulse-dot.yellow { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
    .pulse-dot.green { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
    .pulse-dot.dark-green { background: #059669; box-shadow: 0 0 8px #059669; }
    /* Keyframe animation: gentle pulse effect (scale up + fade out + scale back) */
    @keyframes pulse-anim {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* ================================================================
       HEADERS
       Main page header with gradient text (blue -> sky blue).
       Uses the same background-clip gradient text technique as KPI values.
       ================================================================ */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00BFFF 0%, #0066CC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    /* ================================================================
       EXECUTIVE CARD
       Premium container for executive-level summaries and reports.
       Deep navy gradient with a subtle blue border and heavy shadow
       to create visual hierarchy above regular glass cards.
       ================================================================ */
    .exec-card {
        background: linear-gradient(145deg, rgba(0, 40, 85, 0.9) 0%, rgba(0, 20, 40, 0.95) 100%);
        border-radius: 20px;
        padding: 32px;
        border: 1px solid rgba(0, 150, 255, 0.2);    /* Faint blue border */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);   /* Deep shadow for depth */
        margin: 16px 0;
    }

    /* ================================================================
       BADGES
       Pill-shaped inline labels for tagging status, priority, or
       category.  Variants match the 4-tier pulse status plus a
       general "info" blue badge.
       ================================================================ */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;          /* Fully rounded pill shape */
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-critical { background: #ef4444; color: white; }       /* Red -- critical */
    .badge-warning { background: #f59e0b; color: #212529; }      /* Amber -- warning (dark text for contrast) */
    .badge-success { background: #22c55e; color: white; }        /* Green -- success */
    .badge-exceptional { background: #059669; color: white; }    /* Dark green -- exceptional */
    .badge-info { background: #0066CC; color: white; }           /* Blue -- informational */

    /* ================================================================
       SIDEBAR
       Override Streamlit's default sidebar background with a deep
       navy-to-dark-navy vertical gradient that matches the overall
       dark theme.
       ================================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1929 0%, #001e3c 100%);
    }

    /* ================================================================
       TABS
       Style the active/selected Streamlit tab with a blue gradient
       background and white text.  Rounded top corners create a
       "folder tab" appearance.
       ================================================================ */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066CC 0%, #004C97 100%) !important;
        color: white !important;
        border-radius: 8px 8px 0 0;
    }

    /* ================================================================
       RECOMMENDATION TABLE
       A clean, minimal table used for listing recommended actions.
       Header row has a blue-tinted background with uppercase text;
       data rows have subtle bottom borders.
       ================================================================ */
    .rec-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
    }
    .rec-table th {
        background: rgba(0, 102, 204, 0.3);     /* Translucent blue header */
        padding: 10px 12px;
        text-align: left;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #334155;        /* Slate divider under header */
    }
    .rec-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #1e293b;        /* Subtle row separator */
        font-size: 0.9rem;
    }

    /* ================================================================
       SECTION TITLES
       Small, muted headings used to label subsections within cards
       and panels.  Include a bottom border for visual separation.
       ================================================================ */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;                          /* Slate gray */
        margin: 0.8rem 0 0.4rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #1e3a5f;        /* Dark blue underline */
    }

    /* ================================================================
       MATRIX TABLE (Region / Area Hierarchy)
       A structured table that displays pulse scores organized by
       Region -> Area hierarchy.  Features:
       - Dark container with rounded corners
       - Uppercase column headers with blue underline
       - Three row types with distinct styling:
           .region-row  -- darker background, bold text (parent row)
           .area-row    -- indented, default background (child row)
           .total-row   -- navy background, bold, top border (summary)
       - Hover highlight on all rows
       - Left-aligned name columns, center-aligned score columns
       ================================================================ */
    .matrix-container {
        background: #0d1526;                     /* Deep dark navy */
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        overflow: hidden;                        /* Clip content to rounded corners */
    }
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;                     /* Compact text for dense data */
    }
    .matrix-table th {
        background: #162a4a;                     /* Slightly lighter navy header */
        color: #94a3b8;
        font-weight: 600;
        padding: 0.5rem 0.4rem;
        text-align: center;
        border-bottom: 2px solid #2563eb;        /* Blue accent underline */
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .matrix-table td {
        padding: 0.35rem 0.4rem;
        border-bottom: 1px solid #1e293b;
        text-align: center;
        color: #cbd5e1;                          /* Light slate text */
    }
    /* Row hover: subtle blue highlight */
    .matrix-table tr:hover td { background: rgba(37, 99, 235, 0.1); }
    /* Region parent row: slightly darker bg + bold text */
    .matrix-table .region-row td { background: #111d32; font-weight: 600; }
    /* Area child row: transparent bg + left indent to show hierarchy */
    .matrix-table .area-row td { background: transparent; padding-left: 1.5rem; }
    /* Total / summary row: navy bg + bold + top border accent */
    .matrix-table .total-row td { background: #1a365d; font-weight: 700; border-top: 2px solid #2563eb; }
    /* Name column overrides: left-align with appropriate text color */
    .matrix-table .region-name { text-align: left !important; font-weight: 600; color: #e2e8f0; }
    .matrix-table .area-name { text-align: left !important; color: #94a3b8; }

    /* ================================================================
       HEATMAP SCROLL CONTAINER
       A horizontally scrollable wrapper for the weekly pulse heatmap.
       When the heatmap has many week columns, this container allows
       horizontal scrolling while keeping the first column (account /
       region name) sticky / frozen so it remains visible.
       ================================================================ */
    .heatmap-scroll-container {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        overflow-x: auto;              /* Enable horizontal scrolling */
        max-width: 100%;
    }
    /* Sticky first column: freeze the name column on horizontal scroll.
       z-index:1 keeps it above scrolling data cells. */
    .heatmap-scroll-container .heatmap-table th:first-child,
    .heatmap-scroll-container .heatmap-table td:first-child {
        position: sticky;
        left: 0;
        z-index: 1;
        background: #0d1526;           /* Must match container bg to avoid see-through */
    }
    /* Total row's first cell needs a different bg to match .total-row */
    .heatmap-scroll-container .heatmap-table .total-row td:first-child {
        background: #1a365d;
    }
    /* Header first cell gets higher z-index so it stays above both
       sticky data cells and scrolling header cells */
    .heatmap-scroll-container .heatmap-table th:first-child {
        z-index: 2;
        background: #162a4a;           /* Match header row bg */
    }

    /* ================================================================
       SCORE CELL COLORS (Individual Dimension, 0-3 Scale)
       Background + text color combinations for individual dimension
       scores.  Each class corresponds to a threshold band:
         .score-high     (>= 2.5) -- green tint, strong performance
         .score-mid      (>= 2.0) -- blue tint, acceptable
         .score-low      (>= 1.5) -- yellow tint, needs attention
         .score-critical (< 1.5)  -- red tint, failing
       ================================================================ */
    .score-cell {
        font-weight: 600;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        min-width: 35px;               /* Prevent cells from collapsing on narrow values */
        display: inline-block;
    }
    .score-high     { background: rgba(34, 197, 94, 0.2); color: #86efac; }   /* Green tint */
    .score-mid      { background: rgba(59, 130, 246, 0.15); color: #93c5fd; } /* Blue tint */
    .score-low      { background: rgba(251, 191, 36, 0.2); color: #fcd34d; }  /* Yellow tint */
    .score-critical { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }   /* Red tint */

    /* ================================================================
       PULSE SCORE CELL COLORS (Total Score, 0-24 Scale)
       Background + text color for total pulse score cells in tables.
       Maps directly to the 4-tier status system:
         .pulse-red       (< 14)   -- dark red bg, light red text
         .pulse-yellow    (14-15)  -- dark amber bg, gold text
         .pulse-green     (16-19)  -- dark green bg, mint text
         .pulse-darkgreen (20-24)  -- deep green bg, mint text
       ================================================================ */
    .pulse-red       { background: #7f1d1d; color: #fca5a5; }
    .pulse-yellow    { background: #78350f; color: #fcd34d; }
    .pulse-green     { background: #1e4d3a; color: #6ee7b7; }
    .pulse-darkgreen { background: #065f46; color: #6ee7b7; }

    /* ================================================================
       WEEKLY HEATMAP TABLE
       Compact table for displaying pulse scores across multiple weeks.
       Designed for high data density with small font sizes and tight
       padding.  Works inside .heatmap-scroll-container for horizontal
       scrolling support.
       ================================================================ */
    .heatmap-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;
    }
    .heatmap-table th {
        background: #162a4a;                   /* Navy header */
        color: #94a3b8;
        font-weight: 600;
        padding: 0.4rem 0.3rem;
        text-align: center;
        font-size: 0.7rem;
        border-bottom: 2px solid #2563eb;      /* Blue accent underline */
    }
    .heatmap-table td {
        padding: 0.3rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.7rem;
        border-bottom: 1px solid #1e293b;
    }
    /* Left-aligned region/account name column */
    .heatmap-table .region-col {
        text-align: left;
        padding-left: 0.5rem;
        color: #e2e8f0;
        min-width: 60px;               /* Prevent name column from shrinking too small */
    }
    /* Base heatmap cell styling (border-radius for softer look) */
    .heatmap-cell { border-radius: 2px; padding: 0.2rem 0.3rem; }

    /* 5-tier heatmap color classes.  Note: the heatmap uses 5 tiers
       (vs. the standard 4-tier pulse status) to provide finer visual
       granularity when comparing scores week-over-week.
         .heat-darkgreen (>= 20)  -- deepest green, exceptional
         .heat-green     (>= 17)  -- solid green
         .heat-lime      (>= 16)  -- olive/lime, just above On Track threshold
         .heat-yellow    (>= 14)  -- amber, at risk
         .heat-red       (< 14)   -- red, critical
    */
    .heat-darkgreen { background: #065f46; color: #bbf7d0; }
    .heat-green     { background: #166534; color: #bbf7d0; }
    .heat-lime      { background: #3f6212; color: #d9f99d; }
    .heat-yellow    { background: #854d0e; color: #fef08a; }
    .heat-red       { background: #991b1b; color: #fecaca; }

    /* ================================================================
       RATINGS LEGEND
       A small reference table showing what each dimension score (0-3)
       means.  Typically rendered in the sidebar or as a popover.
       ================================================================ */
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
    /* Rating badge: small square with rounded corners showing the
       numeric score (0, 1, 2, or 3) in a color-coded box */
    .rating-badge {
        display: inline-block;
        width: 18px; height: 18px;
        border-radius: 3px;
        text-align: center;
        line-height: 18px;            /* Vertically center the number */
        font-weight: 700;
        font-size: 0.65rem;
        margin-right: 0.4rem;
    }
    /* Per-score rating badge colors:
       0 = red (critical), 1 = orange (low), 2 = blue (mid), 3 = green (high) */
    .rating-0 { background: #dc2626; color: white; }
    .rating-1 { background: #f97316; color: white; }
    .rating-2 { background: #3b82f6; color: white; }
    .rating-3 { background: #22c55e; color: white; }

    /* ================================================================
       NOTES BOX
       A muted container for free-text analyst notes or commentary.
       Uses a dark background and gray text to keep it visually
       subordinate to primary data content.
       ================================================================ */
    .notes-box {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        font-size: 0.75rem;
        color: #94a3b8;                /* Muted slate text */
        line-height: 1.4;
    }
    /* Notes title: slightly brighter text for the heading */
    .notes-title { font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }

    /* ================================================================
       DRILL-DOWN PANEL
       An expandable detail panel shown when a user clicks into a
       specific account or region.  Features:
       - Gradient background (dark navy)
       - Blue border for visual prominence
       - Header with a blue-to-purple gradient badge
       ================================================================ */
    .drilldown-panel {
        background: linear-gradient(180deg, #0f1d32 0%, #0d1526 100%);
        border: 1px solid #2563eb;                   /* Blue border */
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    /* Drill-down header: flex row with badge + context text */
    .drilldown-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e3a5f;            /* Separator line */
    }
    /* Badge inside drill-down header: blue-to-purple gradient pill */
    .drilldown-badge {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    /* Context text next to the badge (e.g., "Region: Central > Area: NYC") */
    .drilldown-context { color: #64748b; font-size: 0.8rem; }

    /* ================================================================
       STREAMLIT CHROME CLEANUP
       Hide default Streamlit UI elements (hamburger menu, footer,
       header) for a cleaner, more app-like appearance.
       Also expand the content area to use the full viewport width
       with minimal padding.
       ================================================================ */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0.5rem 1rem 1rem 1rem; max-width: 100%; }

    /* ================================================================
       CUSTOM SCROLLBAR
       Override the default browser scrollbar with a thin, dark-themed
       scrollbar that blends with the dashboard aesthetic.
       Only affects WebKit browsers (Chrome, Edge, Safari).
       ================================================================ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }              /* Thin scrollbar */
    ::-webkit-scrollbar-track { background: #0d1526; }            /* Dark track */
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }  /* Slate thumb */
    ::-webkit-scrollbar-thumb:hover { background: #475569; }      /* Lighter on hover */
</style>
""", unsafe_allow_html=True)
