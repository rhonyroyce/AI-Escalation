"""Deep Analysis — Lessons Learned tab.

Renders the 6-pillar scorecard, learning grades, completion rates,
recurrence vs lessons scatter, at-risk categories, and lesson effectiveness.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRADE_COLORS = {'A': '#22c55e', 'B': '#3b82f6', 'C': '#f97316', 'D': '#ef4444', 'F': '#dc2626'}


# ---------------------------------------------------------------------------
# Pure figure builders
# ---------------------------------------------------------------------------

def scorecard_bar(grades_data: dict) -> go.Figure | None:
    """Horizontal bar of learning effectiveness score per category."""
    if not grades_data:
        return None

    sorted_grades = sorted(grades_data.items(), key=lambda x: x[1]['score'], reverse=True)
    categories = [g[0] for g in sorted_grades]
    scores = [g[1]['score'] for g in sorted_grades]
    grades = [g[1]['grade'] for g in sorted_grades]

    hover_texts = []
    for cat, data in sorted_grades:
        recurrence_score = max(0, 100 - data['recurrence_rate'])
        hover_texts.append(
            f"<b>{cat}</b><br><br>"
            f"<b>Total Score:</b> {data['score']:.1f}/100<br>"
            f"<b>Grade:</b> {data['grade']}<br><br>"
            f"<b>Score Breakdown:</b><br>"
            f"• Recurrence (35%): {recurrence_score:.1f} pts<br>"
            f"  └ Rate: {data['recurrence_rate']:.1f}%<br>"
            f"• Lesson Completion (30%): {data['lesson_completion']:.1f} pts<br>"
            f"• Consistency (25%): {data['consistency']:.1f} pts<br>"
            f"• Documentation Bonus: {'+10' if data.get('lessons_documented', 0) > 0 else '0'} pts<br><br>"
            f"<b>Tickets:</b> {data['ticket_count']}"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=scores,
        orientation='h',
        marker_color=[_GRADE_COLORS[g] for g in grades],
        text=[f"{g} ({s:.0f})" for g, s in zip(grades, scores)],
        textposition='outside',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    ))
    fig.update_layout(
        title=dict(text='Learning Effectiveness Score by Category', font=dict(size=14, color='#e2e8f0')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=max(300, len(categories) * 35),
        margin=dict(l=10, r=80, t=40, b=10),
        xaxis=dict(range=[0, 110], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8'), title='Score'),
        yaxis=dict(tickfont=dict(color='#e2e8f0', size=10))
    )
    return fig


def lesson_effectiveness_bar(lesson_effectiveness: list[dict]) -> go.Figure | None:
    """Horizontal bar of lesson effectiveness % per category."""
    if not lesson_effectiveness:
        return None

    eff_df = pd.DataFrame(lesson_effectiveness).sort_values('effectiveness', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=eff_df['category'],
        x=eff_df['effectiveness'],
        orientation='h',
        marker_color=[
            '#22c55e' if e >= 50 else '#f97316' if e >= 0 else '#ef4444'
            for e in eff_df['effectiveness']
        ],
        text=[f"{e:.0f}%" for e in eff_df['effectiveness']],
        textposition='outside',
        hovertemplate=(
            '<b>%{y}</b><br>Effectiveness: %{x:.1f}%<br>'
            'Recurrence WITH lessons: %{customdata[0]:.1f}%<br>'
            'Recurrence WITHOUT lessons: %{customdata[1]:.1f}%<extra></extra>'
        ),
        customdata=list(zip(eff_df['recurrence_with_lesson'], eff_df['recurrence_without_lesson']))
    ))
    fig.update_layout(
        title='Lesson Effectiveness by Category<br><sub>Negative = lessons not preventing recurrence</sub>',
        xaxis_title='Effectiveness %',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=max(300, len(eff_df) * 35),
        margin=dict(l=10, r=80, t=60, b=30)
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_lesson_effectiveness(df: pd.DataFrame) -> list[dict]:
    """Compute per-category lesson effectiveness metrics."""
    lesson_col = None
    for col in ['tickets_data_lessons_learned_title', 'tickets_data_lessons_learned_preventive_actions', 'Lesson_Title']:
        if col in df.columns:
            lesson_col = col
            break

    if lesson_col is None or 'Similar_Ticket_Count' not in df.columns or 'tickets_data_issue_datetime' not in df.columns:
        return []

    df_a = df.copy()
    df_a['Has_Lesson'] = df_a[lesson_col].notna() & (df_a[lesson_col].astype(str).str.strip() != '')
    df_a['Has_Similar'] = df_a['Similar_Ticket_Count'] > 0
    df_a['Issue_Date'] = pd.to_datetime(df_a['tickets_data_issue_datetime'], errors='coerce')

    results = []
    for cat in df_a['AI_Category'].dropna().unique():
        cat_df = df_a[df_a['AI_Category'] == cat]
        if len(cat_df) < 3:
            continue

        total = len(cat_df)
        with_lessons = cat_df['Has_Lesson'].sum()
        recurring_with_lessons = ((cat_df['Has_Similar']) & (cat_df['Has_Lesson'])).sum()
        recurring_without_lessons = ((cat_df['Has_Similar']) & (~cat_df['Has_Lesson'])).sum()

        recurrence_with = (recurring_with_lessons / with_lessons * 100) if with_lessons > 0 else 0
        without_lessons = total - with_lessons
        recurrence_without = (recurring_without_lessons / without_lessons * 100) if without_lessons > 0 else 0

        if recurrence_without > 0:
            effectiveness = ((recurrence_without - recurrence_with) / recurrence_without) * 100
        else:
            effectiveness = 0 if recurrence_with > 0 else 100

        results.append({
            'category': cat,
            'total_tickets': total,
            'with_lessons': with_lessons,
            'recurring_with_lessons': recurring_with_lessons,
            'recurrence_with_lesson': recurrence_with,
            'recurrence_without_lesson': recurrence_without,
            'effectiveness': effectiveness,
            'lesson_coverage': (with_lessons / total * 100) if total > 0 else 0,
        })

    results.sort(key=lambda x: x['effectiveness'])
    return results


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_tab(df: pd.DataFrame) -> None:
    """Render the Lessons Learned tab inside an already-active st.tabs context."""
    from escalation_ai.dashboard.streamlit_app import (
        _calculate_learning_grades, chart_recurrence_vs_lessons,
    )

    st.markdown("#### 📚 Learning Effectiveness Analysis")
    st.markdown("*Analyzing how well lessons are being learned and applied to prevent recurrence*")

    grades_data = _calculate_learning_grades(df)

    if grades_data:
        _render_grades_section(df, grades_data)
    else:
        _render_fallback(df)


def _render_grades_section(df: pd.DataFrame, grades_data: dict) -> None:
    """Main grades-available rendering path."""
    from escalation_ai.dashboard.streamlit_app import chart_recurrence_vs_lessons

    # Row 1: Recurrence vs Lessons + Grade Summary
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_recurrence = chart_recurrence_vs_lessons(df)
        if fig_recurrence:
            with st.spinner("Generating visualization..."):
                st.plotly_chart(fig_recurrence, use_container_width=True)
        else:
            st.info("Insufficient data for recurrence analysis")

    with col2:
        _render_grade_summary(grades_data)

    # Scoring criteria expander
    _render_scoring_criteria()

    # Row 2: Category scorecard
    st.markdown("#### 🎯 Category Learning Scorecard")
    fig = scorecard_bar(grades_data)
    if fig:
        with st.spinner("Generating visualization..."):
            st.plotly_chart(fig, use_container_width=True)

    # Row 3: At-risk categories
    _render_at_risk(grades_data)

    # Row 4: Lesson effectiveness
    _render_lesson_effectiveness(df)


def _render_grade_summary(grades_data: dict) -> None:
    """Grade distribution cards."""
    grades_summary = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for cat_data in grades_data.values():
        grades_summary[cat_data['grade']] += 1

    st.markdown("""
    <div style="background: rgba(139, 92, 246, 0.1); border-radius: 12px; padding: 20px; border: 1px solid rgba(139, 92, 246, 0.3);">
        <div style="color: #c4b5fd; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px;">📊 Learning Grade Distribution</div>
    """, unsafe_allow_html=True)

    for grade, count in grades_summary.items():
        if count > 0:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0;">
                <span style="background: {_GRADE_COLORS[grade]}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 700;">Grade {grade}</span>
                <span style="color: #e2e8f0; font-size: 1.2rem; font-weight: 600;">{count} categories</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    avg_recurrence = sum(d['recurrence_rate'] for d in grades_data.values()) / len(grades_data)
    st.markdown(f"""
    <div style="background: rgba(239, 68, 68, 0.1); border-radius: 8px; padding: 15px; margin-top: 15px; border: 1px solid rgba(239, 68, 68, 0.3);">
        <div style="color: #fca5a5; font-size: 0.8rem;">Average Recurrence Rate</div>
        <div style="color: #ef4444; font-size: 2rem; font-weight: 700;">{avg_recurrence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


def _render_scoring_criteria() -> None:
    """Expander explaining how the score is calculated."""
    with st.expander("📋 **How is the Learning Effectiveness Score Calculated?**", expanded=False):
        st.markdown("##### 📊 Score Components (Total: 100 points)")
        score_cols = st.columns(2)
        with score_cols[0]:
            st.success("🔄 **Recurrence Score** (35% weight)")
            st.caption("Formula: 100 - Recurrence Rate")
            st.caption("*Lower recurrence = higher score*")
            st.warning("⚙️ **Resolution Consistency** (25% weight)")
            st.caption("% of tickets with consistent resolution")
            st.caption("*More consistency = higher score*")
        with score_cols[1]:
            st.info("📝 **Lesson Completion** (30% weight)")
            st.caption("% of lessons marked complete/done/closed")
            st.caption("*Higher completion = higher score*")
            st.markdown("✅ **Documentation Bonus** (+10 points)")
            st.caption("Awarded if any lessons are documented")
            st.caption("*for the category*")

        st.markdown("---")
        st.markdown("##### 🎓 Grade Thresholds")
        grade_cols = st.columns(5)
        with grade_cols[0]:
            st.markdown("🟢 **A**: ≥80")
        with grade_cols[1]:
            st.markdown("🔵 **B**: 65-79")
        with grade_cols[2]:
            st.markdown("🟠 **C**: 50-64")
        with grade_cols[3]:
            st.markdown("🔴 **D**: 35-49")
        with grade_cols[4]:
            st.markdown("⛔ **F**: <35")


def _render_at_risk(grades_data: dict) -> None:
    """Categories needing attention section."""
    st.markdown("#### ⚠️ Categories Needing Attention")

    at_risk = [
        (cat, data)
        for cat, data in grades_data.items()
        if data['grade'] in ['D', 'F'] or data['recurrence_rate'] > 40
    ]

    if not at_risk:
        st.success("✅ All categories are performing well on learning effectiveness!")
        return

    for cat, data in sorted(at_risk, key=lambda x: x[1]['score'])[:5]:
        risk_color = '#ef4444' if data['grade'] == 'F' else '#f97316'
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {risk_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #e2e8f0; font-weight: 600; font-size: 1.1rem;">{cat}</span>
                <span style="background: {risk_color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 700;">Grade {data['grade']}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; font-size: 0.85rem;">
                <div>
                    <div style="color: #94a3b8;">Recurrence Rate</div>
                    <div style="color: #ef4444; font-weight: 600;">{data['recurrence_rate']:.1f}%</div>
                </div>
                <div>
                    <div style="color: #94a3b8;">Lesson Completion</div>
                    <div style="color: {'#22c55e' if data['lesson_completion'] > 50 else '#f97316'}; font-weight: 600;">{data['lesson_completion']:.1f}%</div>
                </div>
                <div>
                    <div style="color: #94a3b8;">Tickets</div>
                    <div style="color: #3b82f6; font-weight: 600;">{data['ticket_count']}</div>
                </div>
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                💡 <b>Action:</b> {'Urgent - Mandate lesson documentation and review root causes' if data['recurrence_rate'] > 50 else 'Review and improve lesson application process'}
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_lesson_effectiveness(df: pd.DataFrame) -> None:
    """Row 4: Similarity-based lessons-not-learned analysis."""
    st.markdown("#### 🔄 Lessons Not Learned - Recurrence Despite Documentation")
    st.markdown("*Identifying cases where similar issues keep appearing despite having documented lessons*")

    lesson_effectiveness = _compute_lesson_effectiveness(df)

    if not lesson_effectiveness:
        if 'Similar_Ticket_Count' in df.columns:
            st.info("Lesson documentation column not found. Add lessons_learned data to enable effectiveness analysis.")
        else:
            st.info("Similar ticket analysis required for lessons effectiveness tracking. Run the similarity analysis pipeline.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = lesson_effectiveness_bar(lesson_effectiveness)
        if fig:
            with st.spinner("Generating visualization..."):
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_effectiveness = np.mean([e['effectiveness'] for e in lesson_effectiveness])
        total_recurring = sum(e['recurring_with_lessons'] for e in lesson_effectiveness)
        total_with = sum(e['with_lessons'] for e in lesson_effectiveness)

        st.metric("Avg Lesson Effectiveness", f"{avg_effectiveness:.1f}%",
                  delta="Good" if avg_effectiveness > 30 else "Needs Improvement",
                  delta_color="normal" if avg_effectiveness > 30 else "inverse")
        st.metric("Issues Recurring Despite Lessons", f"{total_recurring}",
                  delta=f"of {total_with} with lessons", delta_color="inverse")

        st.markdown("##### ⚠️ Lessons Not Working")
        for item in lesson_effectiveness[:3]:
            if item['effectiveness'] < 30 and item['recurring_with_lessons'] > 0:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border-radius: 6px; padding: 10px; margin: 5px 0; border-left: 3px solid #ef4444;">
                    <div style="color: #fca5a5; font-weight: 600;">{item['category']}</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">{item['recurring_with_lessons']} issues recurred despite having lessons</div>
                </div>
                """, unsafe_allow_html=True)

    with st.expander("📊 **Detailed Lesson Effectiveness Data**"):
        eff_display = pd.DataFrame(lesson_effectiveness).rename(columns={
            'category': 'Category',
            'total_tickets': 'Total Tickets',
            'with_lessons': 'With Lessons',
            'recurring_with_lessons': 'Recurring (With Lessons)',
            'recurrence_with_lesson': 'Recurrence % (With)',
            'recurrence_without_lesson': 'Recurrence % (Without)',
            'effectiveness': 'Effectiveness %',
            'lesson_coverage': 'Lesson Coverage %'
        }).round(1)
        st.dataframe(eff_display, use_container_width=True, hide_index=True)


def _render_fallback(df: pd.DataFrame) -> None:
    """Fallback when no grades data is available."""
    lessons_col = 'tickets_data_lessons_learned_preventive_actions'
    if lessons_col not in df.columns:
        st.info("No lessons learned data available")
        return

    df_lessons = df[df[lessons_col].notna() & (df[lessons_col].astype(str) != '')]
    st.markdown(f"**{len(df_lessons)}** records have documented lessons learned")

    if len(df_lessons) > 0:
        st.markdown("#### 📚 Recent Lessons")
        for _, row in df_lessons.head(5).iterrows():
            lesson_text = str(row[lessons_col])[:200]
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 12px; margin: 8px 0; border-left: 3px solid #8b5cf6;">
                <div style="color: #c4b5fd; font-weight: 600;">{row.get('AI_Category', 'Unknown')}</div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 5px;">{lesson_text}...</div>
            </div>
            """, unsafe_allow_html=True)
