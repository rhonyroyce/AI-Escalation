"""
CSE Intelligence Platform — Home / Landing Page
=================================================
Orientation page showing data health, quick stats, and navigation.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent


def render_landing():
    # Header
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px 0;">
        <h1 style="font-size:2.5rem;font-weight:800;
            background:linear-gradient(135deg,#3b82f6,#06b6d4);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            margin:0;">CSE Intelligence Platform</h1>
        <p style="color:#94a3b8;font-size:1.1rem;margin-top:8px;">
            Unified Operations Intelligence — Project Health × Escalation Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Data health cards
    col1, col2, col3 = st.columns(3)

    with col1:
        pulse_ok = st.session_state.get('df') is not None
        count = len(st.session_state.df) if pulse_ok else 0
        st.markdown(f"""
        <div style="background:{'rgba(34,197,94,0.1)' if pulse_ok else 'rgba(239,68,68,0.1)'};
            border:1px solid {'rgba(34,197,94,0.3)' if pulse_ok else 'rgba(239,68,68,0.3)'};
            border-radius:12px;padding:20px;text-align:center;">
            <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">
                Pulse Data</div>
            <div style="font-size:2rem;font-weight:700;color:{'#22c55e' if pulse_ok else '#ef4444'};
                margin:8px 0;">{'✓ Loaded' if pulse_ok else '✗ Missing'}</div>
            <div style="font-size:0.85rem;color:#64748b;">{count:,} records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        esc_ok = st.session_state.get('esc_data_available', False)
        st.markdown(f"""
        <div style="background:{'rgba(34,197,94,0.1)' if esc_ok else 'rgba(239,68,68,0.1)'};
            border:1px solid {'rgba(34,197,94,0.3)' if esc_ok else 'rgba(239,68,68,0.3)'};
            border-radius:12px;padding:20px;text-align:center;">
            <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">
                Escalation Data</div>
            <div style="font-size:2rem;font-weight:700;color:{'#22c55e' if esc_ok else '#ef4444'};
                margin:8px 0;">{'✓ Available' if esc_ok else '✗ Run Pipeline'}</div>
            <div style="font-size:0.85rem;color:#64748b;">
                {'Strategic_Report.xlsx found' if esc_ok else 'python run.py --no-gui'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        ai_ok = st.session_state.get('ollama_available', False)
        st.markdown(f"""
        <div style="background:{'rgba(34,197,94,0.1)' if ai_ok else 'rgba(245,158,11,0.1)'};
            border:1px solid {'rgba(34,197,94,0.3)' if ai_ok else 'rgba(245,158,11,0.3)'};
            border-radius:12px;padding:20px;text-align:center;">
            <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">
                AI Engine</div>
            <div style="font-size:2rem;font-weight:700;color:{'#22c55e' if ai_ok else '#f59e0b'};
                margin:8px 0;">{'✓ Online' if ai_ok else '○ Offline'}</div>
            <div style="font-size:0.85rem;color:#64748b;">
                {'Ollama connected' if ai_ok else 'Cached mode — ollama serve'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick navigation
    st.markdown("### Quick Start")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        st.markdown("""
        <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);
            border-radius:12px;padding:20px;">
            <h4 style="color:#3b82f6;margin:0 0 8px 0;">📊 Project Pulse</h4>
            <p style="color:#94a3b8;font-size:0.85rem;margin:0;">
                Weekly health ratings across 8 dimensions for all projects.
                Start with the Executive Summary for the leadership view.</p>
        </div>
        """, unsafe_allow_html=True)
    with nav_col2:
        st.markdown("""
        <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);
            border-radius:12px;padding:20px;">
            <h4 style="color:#8b5cf6;margin:0 0 8px 0;">🔍 Escalation AI</h4>
            <p style="color:#94a3b8;font-size:0.85rem;margin:0;">
                ML-powered escalation analysis with severity scoring,
                recurrence prediction, and similar-ticket matching.</p>
        </div>
        """, unsafe_allow_html=True)

    # Last pipeline run timestamp
    cache_file = project_root / '.cache' / 'ai_insights.json'
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        st.caption(f"Last pipeline run: {mod_time.strftime('%B %d, %Y at %I:%M %p')}")


render_landing()
