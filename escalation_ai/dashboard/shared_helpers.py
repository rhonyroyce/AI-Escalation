"""
Shared UI helper functions for the Escalation AI dashboard.

Extracted from streamlit_app.py to be shared across multiple dashboard modules
(analytics_view, planning_view, report_generator_view, main_controller).
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def create_plotly_theme():
    """Get consistent Plotly theme settings."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#E0E0E0'),
        margin=dict(l=40, r=40, t=50, b=40),
    )


# Chart descriptions explaining what each visualization signifies
CHART_DESCRIPTIONS = {
    'trend_timeline': {
        'title': 'Escalation Trend (7-Day Moving Average)',
        'description': 'Tracks escalation volume and friction scores over time using a 7-day rolling average to smooth out daily fluctuations.',
        'what_it_shows': 'Shows patterns in escalation frequency and intensity. Rising trends indicate growing issues; falling trends suggest improvements.',
        'how_to_read': 'Blue area = friction intensity, Orange line = ticket count. Watch for correlation between volume spikes and friction increases.',
    },
    'severity_distribution': {
        'title': 'Severity Distribution',
        'description': 'Breakdown of escalations by severity level (Critical, Major, Minor).',
        'what_it_shows': 'Helps prioritize resources and identify if too many tickets are being marked Critical.',
        'how_to_read': 'Healthy distribution: <15% Critical, 30-40% Major, 50-60% Minor. High Critical % suggests either severe issues or over-escalation.',
    },
    'friction_by_category': {
        'title': 'Strategic Friction by Category',
        'description': 'Total friction score accumulated by each issue category.',
        'what_it_shows': 'Identifies which categories cause the most organizational friction and should be prioritized for process improvements.',
        'how_to_read': 'Longer bars = more cumulative friction. Focus on top categories for maximum impact on reducing escalation burden.',
    },
    'recurrence_risk': {
        'title': 'Average Recurrence Risk',
        'description': 'AI-predicted probability that similar issues will recur based on historical patterns.',
        'what_it_shows': 'High recurrence indicates systemic issues that need root cause fixes, not just symptom treatment.',
        'how_to_read': 'Green (<30%) = Well-managed. Yellow (30-60%) = Needs attention. Red (>60%) = Systemic problem requiring intervention.',
    },
    'category_sunburst': {
        'title': 'Category & Sub-Category Sunburst',
        'description': 'Hierarchical view of categories and their sub-categories showing ticket distribution.',
        'what_it_shows': 'Reveals the composition of each category and helps identify specific sub-categories driving volume.',
        'how_to_read': 'Click on segments to drill down. Larger segments = more tickets. Inner ring = categories, outer = sub-categories.',
    },
    'engineer_performance': {
        'title': 'Engineer Performance Matrix',
        'description': 'Compares engineers by friction score and ticket volume handled.',
        'what_it_shows': 'Identifies high performers (low friction, high volume) and those needing support (high friction).',
        'how_to_read': 'Ideal: Lower-left quadrant (low friction, efficient). Investigate high-friction engineers for training needs.',
    },
    'resolution_distribution': {
        'title': 'Resolution Time Distribution',
        'description': 'Histogram of predicted resolution times across all tickets.',
        'what_it_shows': 'Helps set realistic SLA targets and identify if resolution times follow expected patterns.',
        'how_to_read': 'Narrow distribution = consistent performance. Long tail = some tickets take disproportionately long.',
    },
    'pareto_analysis': {
        'title': 'Pareto Analysis (80/20 Rule)',
        'description': 'Shows which categories account for 80% of the total friction.',
        'what_it_shows': 'Classic Pareto principle - focus on the vital few categories that drive most of the impact.',
        'how_to_read': 'Bars = category friction. Line = cumulative %. Categories before line crosses 80% are your priority focus.',
    },
    'benchmark_gauge': {
        'title': 'Competitive Benchmark',
        'description': 'Compares your metrics against industry standards.',
        'what_it_shows': 'Positions your performance relative to best-in-class, industry average, and laggard benchmarks.',
        'how_to_read': 'Green zone = Best-in-class. Yellow = Industry average. Red = Below average. Needle shows your position.',
    },
    'risk_heatmap': {
        'title': 'Risk Heatmap',
        'description': 'Matrix showing risk levels across categories and severity.',
        'what_it_shows': 'Identifies dangerous combinations of category and severity that need immediate attention.',
        'how_to_read': 'Darker colors = higher risk. Focus on dark red cells for critical risk areas.',
    },
    # Similarity Search Charts
    'similarity_count': {
        'title': 'Similar Ticket Count Distribution',
        'description': 'Shows how many similar historical tickets were found for each current issue.',
        'what_it_shows': 'Zero matches may indicate new issue types. High counts suggest well-documented problem patterns.',
        'how_to_read': 'Bars show frequency. Zero-match tickets need manual review. High-match tickets have good historical data for predictions.',
    },
    'resolution_consistency': {
        'title': 'Resolution Consistency Analysis',
        'description': 'Compares how current tickets are being resolved vs how similar historical tickets were resolved.',
        'what_it_shows': 'Inconsistent resolutions suggest either evolving best practices or engineers deviating from proven solutions.',
        'how_to_read': 'Green = consistent with history. Red = different approach being taken. High inconsistency warrants investigation.',
    },
    'similarity_score': {
        'title': 'Similarity Score Distribution',
        'description': 'Quality of best matches found for each ticket (0 = no match, 1 = identical).',
        'what_it_shows': 'Higher scores mean more confident predictions. Low scores suggest unique or poorly documented issues.',
        'how_to_read': 'Green zone (>0.7) = high confidence. Yellow (0.5-0.7) = moderate. Red (<0.5) = low confidence.',
    },
    'inconsistent_resolution': {
        'title': 'Inconsistent Resolutions by Category',
        'description': 'Categories where current resolutions differ from similar historical cases.',
        'what_it_shows': 'Identifies categories with resolution approach inconsistency - may indicate training gaps or process changes.',
        'how_to_read': 'Longer bars = more inconsistent cases. Focus on top categories for standardization opportunities.',
    },
    'similarity_effectiveness': {
        'title': 'Similarity Search Effectiveness',
        'description': 'Heatmap showing average similar ticket matches by category and origin.',
        'what_it_shows': 'Green areas have good historical coverage. Red areas lack similar historical data.',
        'how_to_read': 'Higher values = better knowledge base coverage. Low values need historical data enrichment.',
    },
}


def get_chart_insight(chart_key: str, df: pd.DataFrame) -> dict:
    """
    Generate data-driven insights for a specific chart based on current data.

    Returns dict with: description, what_it_shows, how_to_read, current_insight
    """
    base_info = CHART_DESCRIPTIONS.get(chart_key, {
        'title': 'Chart',
        'description': 'Visualization of escalation data.',
        'what_it_shows': 'Data patterns and trends.',
        'how_to_read': 'Analyze the visual patterns.',
    })

    # Generate data-driven insight based on chart type
    current_insight = ""

    try:
        if chart_key == 'trend_timeline':
            if 'tickets_data_issue_datetime' in df.columns:
                df_temp = df.copy()
                df_temp['date'] = pd.to_datetime(df_temp['tickets_data_issue_datetime']).dt.date
                daily = df_temp.groupby('date').size()
                if len(daily) >= 7:
                    recent_avg = daily.tail(7).mean()
                    older_avg = daily.head(7).mean() if len(daily) >= 14 else recent_avg
                    trend = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"
                    current_insight = f"📊 Current trend is **{trend}**. Recent 7-day avg: {recent_avg:.1f} tickets/day vs earlier: {older_avg:.1f}"

        elif chart_key == 'severity_distribution':
            if 'tickets_data_severity' in df.columns:
                severity = df['tickets_data_severity'].value_counts(normalize=True) * 100
                critical_pct = severity.get('Critical', 0)
                if critical_pct > 25:
                    current_insight = f"⚠️ **High Alert**: {critical_pct:.0f}% Critical tickets - review escalation criteria or investigate systemic issues."
                elif critical_pct > 15:
                    current_insight = f"🟡 **Elevated**: {critical_pct:.0f}% Critical tickets - slightly above healthy threshold of 15%."
                else:
                    current_insight = f"✅ **Healthy**: {critical_pct:.0f}% Critical tickets - within normal range."

        elif chart_key == 'friction_by_category':
            if 'AI_Category' in df.columns and 'Strategic_Friction_Score' in df.columns:
                friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
                top_cat = friction.index[0]
                top_pct = (friction.iloc[0] / friction.sum()) * 100
                current_insight = f"🎯 **{top_cat}** drives {top_pct:.0f}% of total friction. Prioritize this category for maximum impact."

        elif chart_key == 'recurrence_risk':
            if 'AI_Recurrence_Probability' in df.columns:
                avg_risk = df['AI_Recurrence_Probability'].mean() * 100
                if avg_risk > 60:
                    current_insight = f"🔴 **Critical**: {avg_risk:.0f}% avg recurrence risk indicates systemic issues requiring root cause analysis."
                elif avg_risk > 30:
                    current_insight = f"🟡 **Elevated**: {avg_risk:.0f}% avg recurrence risk - some patterns need investigation."
                else:
                    current_insight = f"✅ **Good**: {avg_risk:.0f}% avg recurrence risk - issues are generally being resolved effectively."

        elif chart_key == 'engineer_performance':
            if 'Engineer' in df.columns and 'Strategic_Friction_Score' in df.columns:
                eng_stats = df.groupby('Engineer').agg({
                    'Strategic_Friction_Score': 'mean',
                    'AI_Category': 'count'
                }).rename(columns={'AI_Category': 'count'})
                top_performer = eng_stats[eng_stats['count'] >= 5].nsmallest(1, 'Strategic_Friction_Score')
                if not top_performer.empty:
                    current_insight = f"⭐ **Top Performer**: {top_performer.index[0]} - lowest friction with {top_performer['count'].iloc[0]} tickets."

        elif chart_key == 'category_sunburst':
            if 'AI_Category' in df.columns:
                cat_counts = df['AI_Category'].value_counts()
                top_3 = cat_counts.head(3)
                top_3_pct = (top_3.sum() / len(df)) * 100
                current_insight = f"📈 Top 3 categories ({', '.join(top_3.index[:3])}) account for **{top_3_pct:.0f}%** of all tickets."

        elif chart_key == 'pareto_analysis':
            if 'AI_Category' in df.columns and 'Strategic_Friction_Score' in df.columns:
                friction = df.groupby('AI_Category')['Strategic_Friction_Score'].sum().sort_values(ascending=False)
                cumsum = friction.cumsum() / friction.sum() * 100
                cats_for_80 = (cumsum <= 80).sum() + 1
                current_insight = f"📊 **{cats_for_80} categories** drive 80% of total friction. Focus improvement efforts here for maximum impact."

        elif chart_key == 'risk_heatmap':
            if 'AI_Category' in df.columns and 'tickets_data_severity' in df.columns:
                high_risk = df[(df['tickets_data_severity'] == 'Critical')]
                if len(high_risk) > 0:
                    top_critical_cat = high_risk['AI_Category'].value_counts().idxmax()
                    critical_count = high_risk['AI_Category'].value_counts().iloc[0]
                    current_insight = f"🔴 **{top_critical_cat}** has the most Critical tickets ({critical_count}). This is your highest risk area."

        elif chart_key == 'resolution_distribution':
            if 'Predicted_Resolution_Days' in df.columns:
                avg_days = df['Predicted_Resolution_Days'].mean()
                median_days = df['Predicted_Resolution_Days'].median()
                if avg_days > median_days * 1.5:
                    current_insight = f"⚠️ Mean ({avg_days:.1f}d) >> Median ({median_days:.1f}d) - some tickets take disproportionately long. Investigate outliers."
                else:
                    current_insight = f"✅ Resolution times are consistent. Avg: {avg_days:.1f} days, Median: {median_days:.1f} days."

    except Exception:
        pass  # Silently fail and return base info only

    return {
        **base_info,
        'current_insight': current_insight
    }


def render_spectacular_header(title: str, subtitle: str, icon: str = "📊"):
    """Render a spectacular gradient header with timestamp."""
    current_time = datetime.now().strftime("%b %d, %Y at %I:%M %p")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0a1628 0%, #1a365d 50%, #0a1628 100%);
                padding: 25px 35px; border-radius: 16px; margin-bottom: 20px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="font-size: 2rem; font-weight: 800; margin: 0;
                           background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #3b82f6 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {icon} {title}
                </h1>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 8px 0 0 0; letter-spacing: 1px;">
                    {subtitle}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Last Updated</div>
                <div style="color: #60a5fa; font-size: 1rem; font-weight: 600; margin-top: 4px;">{current_time}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chart_with_insight(chart_key: str, chart_fig, df: pd.DataFrame, container=None):
    """
    Render a chart with an expandable insight tooltip.

    Args:
        chart_key: Key from CHART_DESCRIPTIONS
        chart_fig: Plotly figure object
        df: DataFrame for generating insights
        container: Optional streamlit container (defaults to st)
    """
    if container is None:
        container = st

    insight = get_chart_insight(chart_key, df)

    # Add chart info popover in top-right corner style
    with container.container():
        # Header row with title and info button
        title_col, info_col = st.columns([10, 1])

        with info_col:
            with st.popover("ℹ️"):
                st.markdown(f"### {insight.get('title', 'Chart')}")
                st.markdown(f"*{insight.get('description', '')}*")
                st.divider()
                st.markdown(f"**📊 What it shows:**")
                st.markdown(insight.get('what_it_shows', ''))
                st.markdown(f"**📖 How to read:**")
                st.markdown(insight.get('how_to_read', ''))
                if insight.get('current_insight'):
                    st.divider()
                    st.markdown(f"**💡 Current Data Insight:**")
                    st.markdown(insight['current_insight'])

        # Render the chart
        st.plotly_chart(chart_fig, use_container_width=True)
