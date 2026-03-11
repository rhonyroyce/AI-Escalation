"""
Pulse Dashboard - AI Output Parsers
=====================================
Parse structured LLM output (SCORE/LEVEL/FACTORS/RECOMMENDATION)
into dicts and render as styled HTML cards.
Used by: 5_AI_Insights.py (Risk Scoring + Action Items tabs).
"""
import re
from typing import Optional


def parse_risk_assessment(raw: str) -> Optional[dict]:
    """Parse SCORE/LEVEL/FACTORS/RECOMMENDATION from LLM text.

    Returns dict with keys: score, level, factors, recommendation.
    Returns None if the output cannot be parsed.
    """
    if not raw or not raw.strip():
        return None

    result = {}

    # Score (0-10)
    score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if score_match:
        result['score'] = float(score_match.group(1))
    else:
        return None  # Score is required

    # Level
    level_match = re.search(r'LEVEL:\s*(Low|Medium|High|Critical)', raw, re.IGNORECASE)
    result['level'] = level_match.group(1).title() if level_match else _infer_level(result['score'])

    # Factors (bullet list)
    factors_match = re.search(r'FACTORS:\s*\n((?:\s*-\s*.+\n?)+)', raw, re.IGNORECASE)
    if factors_match:
        factors_text = factors_match.group(1)
        result['factors'] = [
            line.strip().lstrip('- ').strip()
            for line in factors_text.strip().split('\n')
            if line.strip().startswith('-')
        ]
    else:
        result['factors'] = []

    # Recommendation
    rec_match = re.search(r'RECOMMENDATION:\s*(.+?)(?:\n\n|\Z)', raw, re.IGNORECASE | re.DOTALL)
    result['recommendation'] = rec_match.group(1).strip() if rec_match else ""

    return result


def _infer_level(score: float) -> str:
    """Infer risk level from numeric score if not parsed."""
    if score <= 2:
        return "Low"
    elif score <= 5:
        return "Medium"
    elif score <= 7:
        return "High"
    return "Critical"


def _md_to_html(text: str) -> str:
    """Convert markdown bold/italic to HTML and escape residual markdown."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    return text


def render_risk_card_html(parsed: dict, project: str) -> str:
    """Render a parsed risk assessment as a styled HTML card."""
    score = parsed['score']
    level = parsed['level']
    factors = [_md_to_html(f) for f in parsed.get('factors', [])]
    recommendation = _md_to_html(parsed.get('recommendation', ''))

    # Color mapping
    level_colors = {
        'Low': '#22c55e',
        'Medium': '#f59e0b',
        'High': '#ef4444',
        'Critical': '#dc2626',
    }
    color = level_colors.get(level, '#94a3b8')
    pct = min(score / 10 * 100, 100)

    # Factor bullets
    factor_html = ""
    if factors:
        bullets = "".join(f"<li>{f}</li>" for f in factors)
        factor_html = f'<ul style="margin: 8px 0; padding-left: 20px; color: #e2e8f0;">{bullets}</ul>'

    # Recommendation callout — kept on ONE line to prevent CommonMark from
    # exiting the HTML block and escaping the tag as raw text.
    rec_html = ""
    if recommendation:
        rec_html = (
            f'<div style="margin-top:12px;padding:10px;background:rgba(59,130,246,0.1);'
            f'border-left:3px solid #3b82f6;border-radius:4px;">'
            f'<b style="color:#93c5fd;">Recommendation:</b> '
            f'<span style="color:#e2e8f0;">{recommendation}</span>'
            f'</div>'
        )

    # Entire card on one unbroken HTML block — no blank lines inside so
    # Streamlit's markdown parser never terminates the HTML block early.
    return (
        f'<div class="glass-card" style="padding:16px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        f'<b style="color:#E0E0E0;font-size:1.05rem;">{project}</b>'
        f'<span class="badge" style="background:{color};color:white;padding:6px 14px;font-size:0.95rem;">'
        f'{level} ({score:.0f}/10)</span>'
        f'</div>'
        f'<div style="background:#1e293b;border-radius:6px;height:12px;overflow:hidden;margin-bottom:10px;">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:6px;"></div>'
        f'</div>'
        f'<b style="color:#94a3b8;">Risk Factors:</b>'
        f'{factor_html}'
        f'{rec_html}'
        f'</div>'
    )


def parse_action_items(raw: str) -> list[dict]:
    """Parse ACTION/OWNER/STATUS triples from LLM output.

    Returns a list of dicts with keys: action, owner, status.
    """
    if not raw or "NO_ACTIONS" in raw:
        return []

    items = []
    # Split by ACTION: markers
    blocks = re.split(r'(?:^|\n)\s*-?\s*ACTION:\s*', raw, flags=re.IGNORECASE)

    for block in blocks:
        if not block.strip():
            continue
        item = {}
        # First line or up to OWNER: is the action text
        action_match = re.match(r'(.+?)(?:\n\s*-?\s*OWNER:|\Z)', block, re.DOTALL | re.IGNORECASE)
        item['action'] = action_match.group(1).strip() if action_match else block.strip()

        owner_match = re.search(r'OWNER:\s*(.+?)(?:\n|\Z)', block, re.IGNORECASE)
        item['owner'] = owner_match.group(1).strip() if owner_match else "Unassigned"

        status_match = re.search(r'STATUS:\s*(.+?)(?:\n|\Z)', block, re.IGNORECASE)
        item['status'] = status_match.group(1).strip() if status_match else "pending"

        if item['action']:
            items.append(item)

    return items
