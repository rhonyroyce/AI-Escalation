"""
Export Utilities — Generate HTML snapshots of dashboard pages.
Uses self-contained HTML with branded styling for offline viewing.
"""
import io
import streamlit as st
from datetime import datetime


def export_page_to_html(title: str, content_html: str) -> bytes:
    """Generate a self-contained HTML report from page content."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title} — CSE Intelligence Platform</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0a0f1a; color: #e2e8f0; padding: 40px; }}
        .header {{ text-align: center; padding: 20px 0; border-bottom: 2px solid #1e3a5f; margin-bottom: 30px; }}
        .header h1 {{ font-size: 1.8rem; color: #3b82f6; }}
        .header p {{ color: #64748b; font-size: 0.85rem; }}
        .content {{ max-width: 1000px; margin: 0 auto; }}
        .metric {{ display: inline-block; background: linear-gradient(145deg, rgba(0,102,204,0.15), rgba(0,51,102,0.25)); border-radius: 12px; padding: 16px 24px; margin: 8px; border-left: 3px solid #3b82f6; text-align: center; min-width: 160px; }}
        .metric .value {{ font-size: 1.6rem; font-weight: 700; color: #3b82f6; }}
        .metric .label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th {{ background: #1e3a5f; color: #e2e8f0; padding: 10px 12px; text-align: left; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #1e3a5f; font-size: 0.85rem; }}
        .footer {{ text-align: center; color: #64748b; font-size: 0.7rem; margin-top: 40px; padding-top: 20px; border-top: 1px solid #1e3a5f; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>CSE Intelligence Platform — Exported {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    <div class="content">{content_html}</div>
    <div class="footer">CSE Intelligence Platform — Confidential — Internal Use Only</div>
</body>
</html>"""
    return html.encode('utf-8')


def render_export_button(page_title: str, content_html: str = ""):
    """Render an export button in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        if content_html:
            html_bytes = export_page_to_html(page_title, content_html)
            st.download_button(
                label="Export Page as HTML",
                data=html_bytes,
                file_name=f"{page_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                key=f"export_{page_title}",
            )
        else:
            st.caption("Export not available for this page")
