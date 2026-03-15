"""
Ollama Health Check — cached resource that checks LLM server availability.
"""
import streamlit as st
import urllib.request
import json


@st.cache_resource(ttl=300)  # Re-check every 5 minutes
def check_ollama_health() -> dict:
    """Check Ollama server health. Returns dict with status info."""
    try:
        req = urllib.request.Request('http://localhost:11434/api/tags', method='GET')
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m['name'] for m in data.get('models', [])]
            return {'available': True, 'models': models, 'error': None}
    except Exception as e:
        return {'available': False, 'models': [], 'error': str(e)}


def render_ollama_status():
    """Show Ollama connection status in sidebar."""
    health = check_ollama_health()
    if health['available']:
        st.sidebar.markdown(
            '<div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);'
            'border-radius:8px;padding:6px 10px;margin:4px 0;text-align:center;font-size:0.75rem;">'
            '<span style="color:#22c55e;">● AI Engine Online</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
            'border-radius:8px;padding:6px 10px;margin:4px 0;text-align:center;font-size:0.75rem;">'
            '<span style="color:#ef4444;">● AI Engine Offline — Using cached results</span></div>',
            unsafe_allow_html=True
        )


def ai_feature_guard(feature_name: str) -> bool:
    """Check if an AI feature can run. Shows warning if not. Returns True if available."""
    health = check_ollama_health()
    if not health['available']:
        st.info(
            f"ℹ️ **{feature_name}** requires the Ollama AI server. "
            f"Showing cached/static results. Start Ollama with: `ollama serve`"
        )
        return False
    return True
