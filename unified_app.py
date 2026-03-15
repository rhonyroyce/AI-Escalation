"""
CSE Intelligence Platform -- Unified Dashboard (Single Entry Point)
===================================================================

Architecture Overview
---------------------
This file is the **sole Streamlit entry point** for the entire CSE Intelligence
Platform.  It stitches together two previously independent dashboards into a
single multi-page application using Streamlit's ``st.navigation`` API
(introduced in Streamlit >= 1.36):

1. **Project Pulse** -- a telecom project-health tracker whose pages live
   under ``pulse_dashboard/``.  It reads a weekly Excel scorecard
   (``ProjectPulse.xlsx``) and optionally calls a local Ollama LLM for
   AI-generated executive summaries and semantic search.

2. **Escalation AI** -- an ML-based escalation analysis suite whose pages
   live under ``escalation_ai/dashboard/pages/``.  It consumes the output of
   a scikit-learn / cuML pipeline (run via ``run.py``) and renders financial,
   benchmarking, and planning dashboards.

The two dashboard "groups" appear as collapsible sections in the sidebar
navigation.  All pages share a single ``st.session_state`` dict, which is why
we initialise every key centrally here rather than inside each page file.

Launch methods
--------------
    streamlit run unified_app.py --server.port 8501   # direct
    python run.py                                      # pipeline then dashboard
    python run.py --dashboard-only                     # skip pipeline
    python run.py --port 8502

Execution flow (per browser session)
-------------------------------------
1. ``st.set_page_config`` -- must be the very first Streamlit call.
2. CSS injection -- both Pulse and Escalation custom themes are applied.
3. Session-state initialisation -- sensible defaults for every key used by
   either dashboard so that pages never encounter a missing key.
4. Data auto-load -- ``ProjectPulse.xlsx`` is loaded once into
   ``st.session_state.df`` so every Pulse page can read it without re-parsing.
5. AI cache hydration -- if ``run.py`` pre-generated Ollama insights, they are
   deserialised from ``.cache/ai_insights.pkl`` into session state.
6. Navigation registration -- ``st.navigation`` is called with two groups,
   and ``pg.run()`` renders whichever page the user selected.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the repository root so we can build absolute import paths.
# ``__file__`` is ``unified_app.py`` at the repo root, so its parent *is*
# the project root.
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent

# ---------------------------------------------------------------------------
# Prepend both the project root and the ``pulse_dashboard/`` directory to
# ``sys.path``.  The project root is needed so that ``import escalation_ai``
# works.  The ``pulse_dashboard/`` directory is added because its internal
# modules use *relative-style* imports like ``from utils.styles import ...``
# that assume the package root is directly on ``sys.path``.
# We use ``insert(0, ...)`` so these paths take precedence over any
# site-packages that might shadow them.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'pulse_dashboard'))

import streamlit as st

# ============================================================================
# SINGLE PAGE CONFIG (must be the very first Streamlit call)
# ============================================================================
# Streamlit enforces a rule that ``set_page_config`` is called exactly once
# and before any other ``st.*`` rendering call.  Because this file is the
# single entry point, we own that call.  Individual page files must NOT call
# ``set_page_config`` again -- doing so would raise a ``StreamlitAPIException``.
# ``layout="wide"`` gives dashboards full horizontal space; ``initial_sidebar_state``
# ensures the nav is visible on first load.
# ============================================================================
st.set_page_config(
    page_title="CSE Intelligence Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS -- inject both Pulse and Escalation custom stylesheets
# ============================================================================
# Each dashboard was originally a standalone app with its own dark-theme CSS.
# We inject both here so that regardless of which page the user navigates to,
# the correct styles are already present in the DOM.  Injecting CSS is
# idempotent (Streamlit deduplicates identical ``st.markdown`` blocks), so
# having both loaded simultaneously causes no conflicts as long as their
# selectors are scoped properly.
# ============================================================================
from pulse_dashboard.utils.styles import inject_css as inject_pulse_css

inject_pulse_css()  # Pulse dark theme, metric cards, table styles

from escalation_ai.dashboard.esc_bridge import inject_escalation_css

inject_escalation_css()  # Escalation AI-specific styles (charts, KPI cards)

from shared_theme import inject_shared_css

inject_shared_css()  # Shared platform CSS (font normalization, KPI cards, badges)

# ============================================================================
# AUTO-LOAD PULSE DATA
# ============================================================================
# ``load_pulse_data`` reads ``ProjectPulse.xlsx``, cleans non-breaking spaces,
# derives columns (Pulse_Status, Year_Week, Effort), and returns a DataFrame.
# ``get_default_file_path`` resolves the Excel file at the repo root.
# ============================================================================
from pulse_dashboard.utils.data_loader import load_pulse_data, get_default_file_path, render_pulse_freshness

# ---------------------------------------------------------------------------
# Initialise every Pulse-related session-state key with a safe default.
#
# WHY: Streamlit reruns the entire script on every user interaction (widget
# click, slider drag, etc.).  ``st.session_state`` is the only object that
# survives between reruns.  By setting defaults here (guarded by ``if key
# not in``), we guarantee that downstream pages can always reference these
# keys without a ``KeyError``, even on the very first rerun before any page
# logic has executed.
#
# Key descriptions:
#   df              -- the full (unfiltered) Pulse DataFrame
#   filtered_df     -- the currently-filtered view (region/week selection)
#   selected_year   -- active year filter
#   selected_week   -- active week filter
#   selected_regions-- list of region strings the user picked in the sidebar
#   pulse_target    -- "target" threshold for Total Score (default 17.0)
#   pulse_stretch   -- "stretch" threshold for Total Score (default 19.0)
#   green_pct_target-- % of projects that should be Green to meet the goal
#   max_red_target  -- maximum acceptable number of Red projects
#   embeddings_index-- FAISS / numpy index for semantic search over comments
#   selected_project-- project name the user drilled into
#   ollama_available-- tri-state: None = not checked, True/False = result
#   selected_drill  -- which drill-down view the user is exploring
# ---------------------------------------------------------------------------
PULSE_DEFAULTS = {
    'df': None,
    'filtered_df': None,
    'selected_year': None,
    'selected_week': None,
    'selected_regions': [],
    'pulse_target': 17.0,
    'pulse_stretch': 19.0,
    'green_pct_target': 80,
    'max_red_target': 3,
    'embeddings_index': None,
    'selected_project': None,
    'ollama_available': None,
    'selected_drill': None,
    'esc_data_available': None,
    'cross_nav_context': None,
}
for key, default in PULSE_DEFAULTS.items():
    # Guard: only set a default if the key is truly absent.  This preserves
    # any value that a previous rerun (or another page) already wrote.
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Eagerly load the Pulse Excel data on the very first run so that all Pulse
# pages have ``st.session_state.df`` ready immediately, with no spinner or
# "upload a file" prompt.
#
# Guard: ``df is None`` ensures we only read the Excel file once per browser
# session.  On subsequent Streamlit reruns the DataFrame is already in
# session state, so we skip the (relatively expensive) Excel parse entirely.
# ---------------------------------------------------------------------------
if st.session_state.df is None:
    default_path = get_default_file_path()  # -> Path or None
    if default_path:
        try:
            st.session_state.df = load_pulse_data(str(default_path))
        except FileNotFoundError:
            st.error("📁 **ProjectPulse.xlsx not found.** Place the file in the project root directory.")
        except Exception as e:
            st.error(f"📁 **Error loading ProjectPulse.xlsx:** {type(e).__name__}: {e}")
    else:
        st.warning(
            "📁 **No Pulse data file found.** "
            "Place `ProjectPulse.xlsx` in the project root to enable the Pulse dashboard."
        )

# Show Pulse data freshness badge in the sidebar
render_pulse_freshness(st.session_state.df)

# Check if escalation data is available for cross-dashboard features
if st.session_state.get('esc_data_available') is None:
    esc_path = project_root / 'Strategic_Report.xlsx'
    st.session_state.esc_data_available = esc_path.exists()

# ============================================================================
# LOAD AI CACHE -- pre-generated by ``run.py`` before Streamlit starts
# ============================================================================
# When launched via ``python run.py``, the orchestrator runs the Escalation AI
# pipeline and *also* calls Ollama to generate executive summaries, issue
# classifications, and a FAISS-style embeddings index for the Pulse dashboard.
# Those results are serialised to ``.cache/ai_insights.pkl`` so the dashboard
# can hydrate instantly without waiting for LLM inference.
#
# Guard: we use ``ollama_available is None`` as the sentinel.  If it is still
# ``None``, it means neither the cache nor a previous rerun has set it yet,
# so we need to attempt loading.  Once set to ``True`` or ``False``, we skip
# this block on every subsequent rerun -- avoiding repeated disk I/O.
# ============================================================================
if st.session_state.get('ollama_available') is None:
    import json as _json

    # Security: JSON replaces pickle to prevent arbitrary code execution
    cache_file = project_root / '.cache' / 'ai_insights.json'

    if not cache_file.exists():
        import logging as _logging
        _logging.getLogger(__name__).info(
            "AI insights cache not found at %s. AI features will use live inference or be disabled.",
            cache_file
        )

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                _cache = _json.load(f)

            # Restore numpy arrays from JSON lists (embeddings index)
            if 'embeddings_index' in _cache and isinstance(_cache['embeddings_index'], dict):
                import numpy as np
                emb = _cache['embeddings_index']
                if 'embeddings' in emb and isinstance(emb['embeddings'], list):
                    emb['embeddings'] = np.array(emb['embeddings'])

            # Selectively hydrate session state from the cache dict.
            # We only overwrite keys that are still ``None`` (i.e., not yet
            # populated by any other mechanism) to respect any runtime changes
            # the user may have triggered.
            for key in ('ollama_available', 'ai_exec_summary', 'ai_issue_categories',
                        'ai_issue_texts', 'embeddings_index'):
                if key in _cache and st.session_state.get(key) is None:
                    st.session_state[key] = _cache[key]
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"Cache load failed: {e}")

    # If no cache file was found (e.g. ``streamlit run unified_app.py``
    # launched directly without ``run.py``), we still need to know whether
    # Ollama is reachable so the AI Insights page can show a "generate now"
    # button or a "server unavailable" message.  ``check_ollama()`` pings
    # the local Ollama HTTP API with a short timeout -- it does NOT trigger
    # any generation.
    if st.session_state.get('ollama_available') is None:
        try:
            from pulse_dashboard.utils.pulse_insights import check_ollama
            st.session_state.ollama_available = check_ollama()
        except Exception:
            # Import failure (missing dependency) or network error --
            # treat as "Ollama not available".
            st.session_state.ollama_available = False

# ============================================================================
# UNIFIED COLOR OVERRIDES
# ============================================================================
# Normalize the Escalation AI color palette to match Pulse's Tailwind-based
# colors so both halves of the dashboard look like one product.
# ============================================================================
st.markdown("""
<style>
/* ── Sidebar expander nav styling ─────────────────────────────── */
[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: linear-gradient(135deg, #0a1929 0%, #001e3c 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.92rem;
    color: #e2e8f0;
}
[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
    border-color: #3b82f6;
}
[data-testid="stSidebar"] .streamlit-expanderContent {
    padding: 4px 0 4px 4px;
}

/* ── Escalation color overrides (align to Pulse palette) ─── */
/* Override Escalation's Bootstrap-style reds/greens/yellows */
.badge-critical, .strategy-card.high-priority { border-color: #ef4444 !important; }
.badge-warning, .strategy-card.medium-priority { border-color: #f59e0b !important; }
.badge-success { border-color: #22c55e !important; }
.impact-negative { color: #ef4444 !important; }
.impact-positive { color: #22c55e !important; }

/* ── Platform header in sidebar ────────────────────────────── */
.platform-header {
    text-align: center;
    padding: 8px 0 12px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
}
.platform-header h3 {
    margin: 0;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
}
.platform-header p {
    margin: 2px 0 0 0;
    color: #64748b;
    font-size: 0.75rem;
}

/* ── Make sidebar collapse/expand button highly visible ────── */
[data-testid="stSidebar"][aria-expanded="false"] ~ button[kind="header"],
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    background: #3b82f6 !important;
    color: white !important;
    border-radius: 0 8px 8px 0 !important;
    width: 36px !important;
    height: 36px !important;
    opacity: 1 !important;
    z-index: 999 !important;
}
[data-testid="collapsedControl"] button {
    background: #3b82f6 !important;
    color: white !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 8px !important;
}
[data-testid="collapsedControl"] svg {
    fill: white !important;
    stroke: white !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# NAVIGATION -- register pages with hidden default nav, custom sidebar
# ============================================================================
# We hide Streamlit's default navigation and render a custom sidebar with
# collapsible <details> sections so both dashboard groups appear compact
# and visually unified, with a +/- toggle for each group.
# ============================================================================

# -- Group 1: Project Pulse pages ------------------------------------------
pulse_pages = [
    st.Page(
        str(project_root / "pulse_dashboard" / "app.py"),
        title="Pulse Home",
        icon="🏠",
        default=True,
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "1_Executive_Summary.py"),
        title="Executive Summary",
        icon="📊",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "2_Drill_Down.py"),
        title="Drill Down",
        icon="🔍",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "3_Trends.py"),
        title="Trends",
        icon="📈",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "4_Prioritization.py"),
        title="Prioritization",
        icon="🎯",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "5_AI_Insights.py"),
        title="AI Insights",
        icon="🤖",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "6_Project_Details.py"),
        title="Project Details",
        icon="📋",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "7_Rankings.py"),
        title="Rankings",
        icon="🏅",
    ),
    st.Page(
        str(project_root / "pulse_dashboard" / "pages" / "8_Comparison.py"),
        title="Comparison",
        icon="🔄",
    ),
]

# -- Group 2: Escalation AI pages ------------------------------------------
esc_pages = [
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "1_Executive_Dashboard.py"),
        title="Executive Dashboard",
        icon="📊",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "2_Deep_Analysis.py"),
        title="Deep Analysis",
        icon="📈",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "3_Financial_Intelligence.py"),
        title="Financial Intelligence",
        icon="💰",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "4_Benchmarking_Monitoring.py"),
        title="Benchmarking & Monitoring",
        icon="🏆",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "5_Planning_Actions.py"),
        title="Planning & Actions",
        icon="🎯",
    ),
    st.Page(
        str(project_root / "escalation_ai" / "dashboard" / "pages" / "6_Presentation_Mode.py"),
        title="Presentation Mode",
        icon="📽️",
    ),
]

# Register all pages with Streamlit but hide the default navigation.
# We build our own sidebar nav below with collapsible HTML sections.
all_pages = pulse_pages + esc_pages
pg = st.navigation(all_pages, position="hidden")

# ── Custom sidebar navigation ─────────────────────────────────────────────
# Use native st.expander() so the collapsible toggle works correctly.
# st.page_link() already renders the page icon, so we only pass the title.
with st.sidebar:
    st.markdown("""
    <div class="platform-header">
        <h3>CSE Intelligence Platform</h3>
        <p>Unified Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("**+ Project Pulse**", expanded=True):
        for page in pulse_pages:
            st.page_link(page, label=page.title, icon=page.icon)

    with st.expander("**+ Escalation AI**", expanded=True):
        for page in esc_pages:
            st.page_link(page, label=page.title, icon=page.icon)

    st.markdown("---")

pg.run()
