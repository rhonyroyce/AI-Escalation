"""
Analytics view for the Escalation AI dashboard.

Extracted from streamlit_app.py. Provides render_analytics() which renders
a legacy/alternative analytics page with 6 tabs: Categories, Engineers,
Distributions, Financial, Similarity, and Lessons Learned.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from escalation_ai.dashboard.shared_helpers import (
    create_plotly_theme,
    render_spectacular_header,
    render_chart_with_insight,
)


def _render_categories_tab(df):
    """Render the Categories tab with 4 sub-tabs: Overview, Drill-Down, Treemap, Details."""
    from escalation_ai.dashboard.streamlit_app import (
        chart_category_sunburst, chart_friction_by_category,
    )
    from escalation_ai.dashboard.advanced_plotly_charts import (
        chart_subcategory_breakdown, chart_subcategory_comparison_table,
        chart_category_treemap,
    )

    cat_tabs = st.tabs(["📊 Overview", "🔍 Drill-Down", "📈 Treemap", "📋 Details"])

    with cat_tabs[0]:
        render_chart_with_insight('category_sunburst', chart_category_sunburst(df), df)
        render_chart_with_insight('friction_by_category', chart_friction_by_category(df), df)

    with cat_tabs[1]:
        st.markdown("### Sub-Category Drill-Down")
        st.markdown("Select a category to view detailed sub-category breakdown")

        categories = ['All Categories'] + sorted(df['AI_Category'].unique().tolist())
        selected_cat = st.selectbox("Select Category", categories, key="cat_drilldown_select")

        if selected_cat == 'All Categories':
            st.plotly_chart(chart_subcategory_breakdown(df, None), use_container_width=True)
        else:
            st.plotly_chart(chart_subcategory_breakdown(df, selected_cat), use_container_width=True)

        st.markdown("### Sub-Category Comparison")
        comparison_df = chart_subcategory_comparison_table(df)
        if not comparison_df.empty:
            if selected_cat != 'All Categories':
                comparison_df = comparison_df[comparison_df['Category'] == selected_cat]
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with cat_tabs[2]:
        st.markdown("### Category Treemap")
        st.markdown("*Click on categories to drill down into sub-categories*")
        st.plotly_chart(chart_category_treemap(df), use_container_width=True)

    with cat_tabs[3]:
        _render_category_details(df)


def _render_category_details(df):
    """Render detailed category statistics sub-tab."""
    st.markdown("### Category Statistics")

    if 'AI_Sub_Category' in df.columns:
        agg_dict = {'AI_Sub_Category': 'count'}
        col_names = ['Ticket Count']

        if 'AI_Confidence' in df.columns:
            agg_dict['AI_Confidence'] = 'mean'
            col_names.append('Avg Confidence')

        if 'Strategic_Friction_Score' in df.columns:
            agg_dict['Strategic_Friction_Score'] = 'sum'
            col_names.append('Total Friction')

        cat_stats = df.groupby('AI_Category').agg(agg_dict).round(2)
        cat_stats.columns = col_names
        cat_stats = cat_stats.sort_values('Ticket Count', ascending=False)

        if 'Financial_Impact' in df.columns:
            fin_stats = df.groupby('AI_Category')['Financial_Impact'].sum()
            cat_stats['Total Impact'] = fin_stats
            cat_stats['Total Impact'] = cat_stats['Total Impact'].apply(lambda x: f"${x:,.0f}")

        st.dataframe(cat_stats, use_container_width=True)

        st.markdown("### Sub-Category Distribution")
        for cat in df['AI_Category'].unique():
            with st.expander(f"📁 {cat}"):
                cat_df = df[df['AI_Category'] == cat]
                sub_counts = cat_df['AI_Sub_Category'].value_counts()

                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = go.Figure(go.Bar(
                        x=sub_counts.values,
                        y=sub_counts.index,
                        orientation='h',
                        marker_color='#0066CC'
                    ))
                    fig.update_layout(
                        **{
                            **create_plotly_theme(),
                            'margin': dict(l=10, r=10, t=10, b=10),
                        },
                        height=200,
                        yaxis_title='',
                        xaxis_title='Count'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.metric("Total Tickets", len(cat_df))
                    if 'Financial_Impact' in df.columns:
                        st.metric("Total Impact", f"${cat_df['Financial_Impact'].sum():,.0f}")
    else:
        st.info("Sub-category data not available. Run classification with the updated system.")


def _render_engineers_tab(df):
    """Render the Engineers tab with performance chart."""
    from escalation_ai.dashboard.streamlit_app import chart_engineer_performance

    fig = chart_engineer_performance(df)
    if fig:
        render_chart_with_insight('engineer_performance', fig, df)
    else:
        st.info("Engineer data not available.")


def _render_distributions_tab(df):
    """Render the Distributions tab with resolution and recurrence charts."""
    from escalation_ai.dashboard.streamlit_app import (
        chart_resolution_distribution, chart_recurrence_risk,
    )

    col1, col2 = st.columns(2)
    with col1:
        render_chart_with_insight('resolution_distribution', chart_resolution_distribution(df), df)
    with col2:
        render_chart_with_insight('recurrence_risk', chart_recurrence_risk(df), df)


def _render_financial_tab(df):
    """Render the Financial tab with category costs and drill-down."""
    from escalation_ai.dashboard.advanced_plotly_charts import (
        chart_category_financial_drilldown,
    )

    if 'Financial_Impact' not in df.columns:
        st.info("Financial impact data not available.")
        return

    fin_tabs = st.tabs(["📊 By Category", "🔍 Sub-Category Drill-Down", "📋 Summary Table"])

    with fin_tabs[0]:
        _render_financial_by_category(df)

    with fin_tabs[1]:
        _render_financial_drilldown(df)

    with fin_tabs[2]:
        _render_financial_summary_table(df)


def _render_financial_by_category(df):
    """Render financial impact by category bar chart and metrics."""
    fin_by_cat = df.groupby('AI_Category')['Financial_Impact'].sum().sort_values(ascending=False)

    fig = go.Figure(go.Bar(
        x=fin_by_cat.index,
        y=fin_by_cat.values,
        marker=dict(
            color=fin_by_cat.values,
            colorscale='Reds'
        ),
        text=[f"${v/1000:.0f}K" for v in fin_by_cat.values],
        textposition='outside'
    ))

    theme = create_plotly_theme()
    theme.pop('margin', None)

    fig.update_layout(
        **theme,
        title='Financial Impact by Category',
        xaxis_tickangle=-45,
        height=420,
        margin=dict(l=40, r=60, t=80, b=100)
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Impact", f"${df['Financial_Impact'].sum():,.0f}")
    with col2:
        st.metric("Average per Ticket", f"${df['Financial_Impact'].mean():,.0f}")
    with col3:
        st.metric("Max Single Ticket", f"${df['Financial_Impact'].max():,.0f}")


def _render_financial_drilldown(df):
    """Render sub-category financial drill-down."""
    from escalation_ai.dashboard.advanced_plotly_charts import (
        chart_category_financial_drilldown,
    )

    st.markdown("### Financial Impact by Sub-Category")
    st.markdown("*Click on categories in the chart to drill down*")

    st.plotly_chart(chart_category_financial_drilldown(df), use_container_width=True)

    if 'AI_Sub_Category' in df.columns:
        st.markdown("### Sub-Category Cost Breakdown")

        categories = ['All Categories'] + sorted(df['AI_Category'].unique().tolist())
        selected_cat = st.selectbox("Select Category for Details", categories, key="fin_cat_select")

        if selected_cat == 'All Categories':
            subcat_fin = df.groupby('AI_Sub_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])
        else:
            subcat_fin = df[df['AI_Category'] == selected_cat].groupby('AI_Sub_Category')['Financial_Impact'].agg(['sum', 'mean', 'count'])

        subcat_fin.columns = ['Total', 'Average', 'Count']
        subcat_fin = subcat_fin.sort_values('Total', ascending=False)

        fig = go.Figure(go.Bar(
            x=subcat_fin['Total'],
            y=subcat_fin.index,
            orientation='h',
            marker=dict(
                color=subcat_fin['Total'],
                colorscale='Reds'
            ),
            text=[f"${v:,.0f}" for v in subcat_fin['Total']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Total: $%{x:,.0f}<extra></extra>'
        ))

        theme = create_plotly_theme()
        theme.pop('margin', None)

        fig.update_layout(
            **theme,
            title=f'Financial Impact: {selected_cat}',
            height=max(300, len(subcat_fin) * 35),
            margin=dict(l=200, r=80, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)


def _render_financial_summary_table(df):
    """Render financial summary table."""
    st.markdown("### Financial Summary Table")

    if 'AI_Sub_Category' in df.columns:
        summary = df.groupby(['AI_Category', 'AI_Sub_Category']).agg({
            'Financial_Impact': ['sum', 'mean', 'count']
        }).round(2)
        summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
        summary = summary.reset_index()
        summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f"${x:,.0f}")
        summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f"${x:,.0f}")
        summary = summary.sort_values(['AI_Category', 'Ticket Count'], ascending=[True, False])
    else:
        summary = df.groupby('AI_Category').agg({
            'Financial_Impact': ['sum', 'mean', 'count']
        }).round(2)
        summary.columns = ['Total Impact', 'Avg Impact', 'Ticket Count']
        summary = summary.reset_index()
        summary['Total Impact'] = summary['Total Impact'].apply(lambda x: f"${x:,.0f}")
        summary['Avg Impact'] = summary['Avg Impact'].apply(lambda x: f"${x:,.0f}")

    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_similarity_tab(df):
    """Render the Similarity Search Analysis tab."""
    from escalation_ai.dashboard.streamlit_app import (
        chart_similarity_count_distribution, chart_similarity_score_distribution,
        chart_resolution_consistency, chart_inconsistent_by_category,
        chart_similarity_effectiveness_heatmap, chart_expected_vs_predicted_resolution,
    )

    st.markdown("### 🔗 Similarity Search Analysis")
    st.markdown("*Insights from comparing tickets to historical patterns*")

    has_similarity = any(col in df.columns for col in ['Similar_Ticket_Count', 'Best_Match_Similarity', 'Resolution_Consistency'])

    if not has_similarity:
        st.info("🔍 **Similarity search data not available.**\n\nRun the analysis with similarity search enabled to populate this section.")
        return

    sim_tabs = st.tabs(["📊 Overview", "📈 Score Analysis", "⚖️ Consistency", "🔥 Heatmap"])

    with sim_tabs[0]:
        _render_similarity_overview(df, chart_similarity_count_distribution)

    with sim_tabs[1]:
        _render_similarity_scores(df, chart_similarity_score_distribution, chart_expected_vs_predicted_resolution)

    with sim_tabs[2]:
        _render_similarity_consistency(df, chart_resolution_consistency, chart_inconsistent_by_category)

    with sim_tabs[3]:
        _render_similarity_heatmap(df, chart_similarity_effectiveness_heatmap)


def _render_similarity_overview(df, chart_similarity_count_distribution):
    """Render similarity overview sub-tab with count distribution."""
    col1, col2, col3 = st.columns(3)

    if 'Similar_Ticket_Count' in df.columns:
        counts = df['Similar_Ticket_Count'].dropna()
        with col1:
            avg_similar = counts.mean()
            st.metric("Avg Similar Tickets", f"{avg_similar:.1f}")
        with col2:
            zero_matches = (counts == 0).sum()
            st.metric("No Matches Found", f"{zero_matches}", delta=f"{zero_matches/len(counts)*100:.0f}%", delta_color="inverse")
        with col3:
            high_matches = (counts >= 5).sum()
            st.metric("Good Coverage (5+)", f"{high_matches}", delta=f"{high_matches/len(counts)*100:.0f}%")

    st.markdown("---")

    fig = chart_similarity_count_distribution(df)
    if fig:
        render_chart_with_insight('similarity_count', fig, df)
    else:
        st.info("Similar ticket count data not available.")


def _render_similarity_scores(df, chart_similarity_score_distribution, chart_expected_vs_predicted_resolution):
    """Render similarity score analysis sub-tab."""
    st.markdown("### Similarity Score Analysis")
    st.markdown("*How confident are we in the similar ticket matches?*")

    if 'Best_Match_Similarity' in df.columns:
        scores = df['Best_Match_Similarity'].dropna()
        scores = scores[scores > 0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Score", f"{scores.mean():.2f}")
        with col2:
            high_conf = (scores >= 0.7).sum()
            st.metric("High Confidence", f"{high_conf}", delta=f"{high_conf/len(scores)*100:.0f}%")
        with col3:
            low_conf = (scores < 0.5).sum()
            st.metric("Low Confidence", f"{low_conf}", delta=f"{low_conf/len(scores)*100:.0f}%", delta_color="inverse")

        st.markdown("---")
        fig = chart_similarity_score_distribution(df)
        if fig:
            render_chart_with_insight('similarity_score', fig, df)

    fig = chart_expected_vs_predicted_resolution(df)
    if fig:
        st.markdown("### Expected vs AI Predicted Resolution")
        st.plotly_chart(fig, use_container_width=True)


def _render_similarity_consistency(df, chart_resolution_consistency, chart_inconsistent_by_category):
    """Render resolution consistency sub-tab."""
    st.markdown("### Resolution Consistency")
    st.markdown("*Are we resolving similar issues the same way?*")

    col1, col2 = st.columns(2)

    with col1:
        fig = chart_resolution_consistency(df)
        if fig:
            render_chart_with_insight('resolution_consistency', fig, df)
        else:
            st.info("Resolution consistency data not available.")

    with col2:
        fig = chart_inconsistent_by_category(df)
        if fig:
            render_chart_with_insight('inconsistent_resolution', fig, df)
        else:
            st.info("Inconsistent resolution data not available.")

    if 'Inconsistent_Resolution' in df.columns:
        inconsistent = df[df['Inconsistent_Resolution'] == True]
        if len(inconsistent) > 0:
            st.markdown("### Tickets with Inconsistent Resolutions")
            st.markdown(f"*{len(inconsistent)} tickets resolved differently than similar historical cases*")

            display_cols = ['Identity', 'AI_Category', 'AI_Sub_Category', 'Similar_Ticket_Count', 'Best_Match_Similarity']
            display_cols = [c for c in display_cols if c in inconsistent.columns]

            if display_cols:
                st.dataframe(
                    inconsistent[display_cols].head(20),
                    use_container_width=True,
                    hide_index=True
                )


def _render_similarity_heatmap(df, chart_similarity_effectiveness_heatmap):
    """Render similarity effectiveness heatmap sub-tab."""
    st.markdown("### Similarity Search Effectiveness")
    st.markdown("*Where do we have good historical coverage?*")

    fig = chart_similarity_effectiveness_heatmap(df)
    if fig:
        render_chart_with_insight('similarity_effectiveness', fig, df)
    else:
        st.info("Need both category and origin data for effectiveness heatmap.")

    if 'Similar_Ticket_Count' in df.columns and 'AI_Category' in df.columns:
        st.markdown("### Coverage by Category")
        coverage = df.groupby('AI_Category').agg({
            'Similar_Ticket_Count': ['mean', 'sum'],
            'AI_Category': 'count'
        })
        coverage.columns = ['Avg Matches', 'Total Matches', 'Ticket Count']
        coverage['Coverage Score'] = (coverage['Avg Matches'] * 20).clip(0, 100).round(0).astype(int)
        coverage = coverage.sort_values('Coverage Score', ascending=False)

        st.dataframe(coverage, use_container_width=True)


def _render_lessons_tab(df):
    """Render the Lessons Learned Effectiveness tab with 6-pillar scorecard."""
    from escalation_ai.dashboard.streamlit_app import (
        get_comprehensive_scorecard, chart_scorecard_radar,
        chart_scorecard_comparison, chart_learning_grades,
        chart_recurrence_vs_lessons, chart_learning_heatmap,
        _calculate_learning_grades,
    )

    st.markdown("### 📚 Lessons Learned Effectiveness")
    st.markdown("*Comprehensive 6-pillar assessment of organizational learning*")

    scorecard = get_comprehensive_scorecard(df)

    if not scorecard or not scorecard.category_scorecards:
        _render_lessons_simple_fallback(df, chart_learning_grades, _calculate_learning_grades)
        return

    summary_df = scorecard.get_summary_df()
    at_risk = scorecard.get_at_risk_categories()

    _render_lessons_summary_metrics(summary_df, scorecard)

    st.markdown("---")

    lesson_tabs = st.tabs([
        "🎯 Scorecard Overview",
        "📊 Category Rankings",
        "🔬 Pillar Deep-Dive",
        "📈 Trends & Patterns",
        "⚠️ At-Risk Categories",
        "💡 AI Recommendations"
    ])

    with lesson_tabs[0]:
        _render_lessons_scorecard_overview(df, scorecard, chart_scorecard_radar)

    with lesson_tabs[1]:
        _render_lessons_category_rankings(df, summary_df, chart_learning_grades)

    with lesson_tabs[2]:
        _render_lessons_pillar_deepdive(df, scorecard, chart_scorecard_radar, chart_scorecard_comparison)

    with lesson_tabs[3]:
        _render_lessons_trends(df, chart_recurrence_vs_lessons, chart_learning_heatmap)

    with lesson_tabs[4]:
        _render_lessons_at_risk(df, at_risk, chart_scorecard_radar)

    with lesson_tabs[5]:
        _render_lessons_ai_recommendations(scorecard)


def _render_lessons_simple_fallback(df, chart_learning_grades, _calculate_learning_grades):
    """Render simple learning grades when comprehensive scorecard is unavailable."""
    grades_data = _calculate_learning_grades(df)
    if not grades_data:
        st.info("🔍 **Lessons learned data not available.**\n\nThis analysis requires:\n- `AI_Category` column\n- `tickets_data_lessons_learned_title` or similar column\n- Ideally `AI_Recurrence_Probability` from similarity analysis")
    else:
        avg_score = np.mean([d['score'] for d in grades_data.values()])
        st.metric("Average Score (Simple)", f"{avg_score:.0f}/100")
        fig = chart_learning_grades(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="learning_grades_simple")


def _render_lessons_summary_metrics(summary_df, scorecard):
    """Render summary metrics row for lessons learned."""
    total_cats = len(scorecard.category_scorecards)
    avg_score = summary_df['Overall Score'].mean()

    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for grade in summary_df['Grade']:
        base_grade = grade[0] if grade else 'F'
        if base_grade in grade_counts:
            grade_counts[base_grade] += 1

    at_risk_count = grade_counts['D'] + grade_counts['F']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Score", f"{avg_score:.0f}/100")
    with col2:
        st.metric("Categories Analyzed", total_cats)
    with col3:
        st.metric("At-Risk (D/F)", at_risk_count,
                 delta=f"{at_risk_count/total_cats*100:.0f}%" if total_cats > 0 else "0%",
                 delta_color="inverse")
    with col4:
        excellent = grade_counts['A'] + grade_counts['B']
        st.metric("Excellent (A/B)", excellent,
                 delta=f"{excellent/total_cats*100:.0f}%" if total_cats > 0 else "0%")


def _render_lessons_scorecard_overview(df, scorecard, chart_scorecard_radar):
    """Render scorecard overview with radar chart and pillar averages."""
    st.markdown("### Organization Learning Effectiveness Scorecard")
    st.markdown("""
    *The 6-pillar scorecard evaluates learning effectiveness across:*
    - **Learning Velocity**: Improvement trends over time
    - **Impact Management**: Handling of high-severity recurring issues
    - **Knowledge Quality**: Quality of documented lessons (AI-assessed)
    - **Process Maturity**: Consistency and documentation completeness
    - **Knowledge Transfer**: Cross-team learning and knowledge sharing
    - **Outcome Effectiveness**: Actual results and improvements
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = chart_scorecard_radar(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="radar_org_overview")

    with col2:
        st.markdown("### Pillar Averages")
        pillar_avgs = []
        for pillar_key in ['learning_velocity', 'impact_management', 'knowledge_quality',
                          'process_maturity', 'knowledge_transfer', 'outcome_effectiveness']:
            scores = [sc.pillars[pillar_key].score for sc in scorecard.category_scorecards.values()]
            avg = np.mean(scores)
            trend = 'improving' if avg > 60 else ('needs work' if avg < 40 else 'stable')
            pillar_avgs.append({
                'Pillar': pillar_key.replace('_', ' ').title(),
                'Avg Score': f"{avg:.0f}",
                'Status': '✅' if avg >= 70 else ('⚠️' if avg >= 50 else '🔴'),
                'Trend': trend
            })
        st.dataframe(pd.DataFrame(pillar_avgs), use_container_width=True, hide_index=True)


def _render_lessons_category_rankings(df, summary_df, chart_learning_grades):
    """Render category rankings tab."""
    st.markdown("### Category Rankings")
    st.markdown("*All categories ranked by overall learning effectiveness score*")

    fig = chart_learning_grades(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="learning_grades_ranked")

    st.markdown("### Detailed Scores")
    display_df = summary_df.copy()
    display_df['Overall Score'] = display_df['Overall Score'].round(1)
    for col in display_df.columns:
        if col not in ['Category', 'Rank', 'Grade', 'Overall Score']:
            display_df[col] = display_df[col].round(1)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_lessons_pillar_deepdive(df, scorecard, chart_scorecard_radar, chart_scorecard_comparison):
    """Render pillar deep-dive analysis tab."""
    st.markdown("### Pillar Deep-Dive Analysis")
    st.markdown("*Select a category to see detailed pillar breakdown*")

    selected_cat = st.selectbox(
        "Select Category",
        options=['Organization Average'] + list(scorecard.category_scorecards.keys()),
        key="pillar_drilldown_cat"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if selected_cat == 'Organization Average':
            fig = chart_scorecard_radar(df)
        else:
            fig = chart_scorecard_radar(df, selected_cat)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"radar_pillar_{selected_cat}")

    with col2:
        if selected_cat != 'Organization Average' and selected_cat in scorecard.category_scorecards:
            cat_sc = scorecard.category_scorecards[selected_cat]

            st.markdown(f"### {selected_cat}")
            st.markdown(f"**Grade: {cat_sc.overall_grade}** | Score: {cat_sc.overall_score:.0f}/100 | Rank: #{cat_sc.rank}")

            if cat_sc.strengths:
                st.markdown("**Strengths:** " + ", ".join(cat_sc.strengths))
            if cat_sc.weaknesses:
                st.markdown("**Weaknesses:** " + ", ".join(cat_sc.weaknesses))

            st.markdown("#### Pillar Details")
            for name, pillar in cat_sc.pillars.items():
                status = '✅' if pillar.score >= 70 else ('⚠️' if pillar.score >= 50 else '🔴')
                with st.expander(f"{status} {name.replace('_', ' ').title()}: {pillar.score:.0f}"):
                    for sub_name, sub_score in pillar.sub_scores.items():
                        st.write(f"  • {sub_name.replace('_', ' ').title()}: {sub_score:.0f}")
                    if pillar.insights:
                        st.markdown("**Insights:**")
                        for insight in pillar.insights:
                            st.write(f"  {insight}")

    st.markdown("---")
    st.markdown("### Compare Categories")
    compare_cats = st.multiselect(
        "Select categories to compare (max 5)",
        options=list(scorecard.category_scorecards.keys()),
        default=list(scorecard.category_scorecards.keys())[:3],
        max_selections=5,
        key="compare_cats"
    )
    if compare_cats:
        fig = chart_scorecard_comparison(df, compare_cats)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="scorecard_comparison")


def _render_lessons_trends(df, chart_recurrence_vs_lessons, chart_learning_heatmap):
    """Render trends & patterns tab."""
    st.markdown("### Trends & Patterns")

    fig = chart_recurrence_vs_lessons(df)
    if fig:
        st.markdown("#### Recurrence vs Lesson Completion")
        render_chart_with_insight('recurrence_lessons', fig, df)

    fig = chart_learning_heatmap(df)
    if fig:
        st.markdown("#### Learning Effectiveness by Category & LOB")
        st.plotly_chart(fig, use_container_width=True, key="learning_heatmap")


def _render_lessons_at_risk(df, at_risk, chart_scorecard_radar):
    """Render at-risk categories tab."""
    st.markdown("### ⚠️ At-Risk Categories")
    st.markdown("*Categories scoring below C- require immediate attention*")

    if at_risk:
        for cat_sc in at_risk[:10]:
            with st.expander(f"🔴 {cat_sc.category} - Grade {cat_sc.overall_grade} ({cat_sc.overall_score:.0f})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    fig = chart_scorecard_radar(df, cat_sc.category)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"radar_atrisk_{cat_sc.category}")
                with col2:
                    st.markdown("**Weaknesses:**")
                    for w in cat_sc.weaknesses:
                        st.write(f"  • {w}")

                    st.markdown("**Key Recommendations:**")
                    for rec in cat_sc.recommendations[:5]:
                        st.write(f"  • {rec}")

                    worst_pillars = sorted(
                        cat_sc.pillars.items(),
                        key=lambda x: x[1].score
                    )[:2]
                    st.markdown("**Focus Areas:**")
                    for name, pillar in worst_pillars:
                        st.write(f"  🎯 {name.replace('_', ' ').title()}: {pillar.score:.0f}")
    else:
        st.success("✅ No categories with grades below C-! Organization is learning effectively.")


def _render_lessons_ai_recommendations(scorecard):
    """Render AI recommendations tab."""
    st.markdown("### 💡 AI-Powered Recommendations")

    top_recs = scorecard.get_top_recommendations(10)

    if st.button("🤖 Generate AI Executive Summary", key="gen_ai_summary"):
        with st.spinner("Generating AI analysis..."):
            summary = scorecard.generate_ai_summary(use_ollama=True)
            st.markdown("#### Executive Summary")
            st.markdown(summary)
    else:
        st.markdown("*Click button above for AI-generated executive summary*")

    st.markdown("---")
    st.markdown("### Priority Recommendations")

    if top_recs:
        for rec in top_recs:
            priority = "🔴 HIGH" if rec['grade'] in ['F', 'D-', 'D'] else "🟡 MEDIUM"
            st.markdown(f"**{priority}** | {rec['category']} ({rec['grade']})")
            st.write(f"  → {rec['recommendation']}")
            st.markdown("")
    else:
        st.success("✅ All categories performing well!")


def render_analytics(df):
    """Render a legacy/alternative analytics page with 6 tabs.

    Tabs:
    1. Categories: Sunburst overview, sub-category drill-down, treemap,
       detailed comparison table (all from advanced_plotly_charts)
    2. Engineers: Performance bar chart with insights
    3. Distributions: Resolution time histogram, recurrence gauge
    4. Financial: Category cost comparison, financial drilldown charts
    5. Similarity: Full similarity search analysis (if data available)
    6. Lessons Learned: Comprehensive 6-pillar scorecard with radar charts,
       learning grades, completion rates, recurrence correlation, heatmap,
       at-risk categories, and AI recommendations

    This is a more detailed alternative to render_deep_analysis().

    Args:
        df: Processed DataFrame with all standard columns.
    """
    render_spectacular_header("Analytics", "Deep dive into escalation patterns and performance", "📈")

    tabs = st.tabs(["🎯 Categories", "👥 Engineers", "📊 Distributions", "💰 Financial", "🔗 Similarity", "📚 Lessons Learned"])

    with tabs[0]:
        _render_categories_tab(df)

    with tabs[1]:
        _render_engineers_tab(df)

    with tabs[2]:
        _render_distributions_tab(df)

    with tabs[3]:
        _render_financial_tab(df)

    with tabs[4]:
        _render_similarity_tab(df)

    with tabs[5]:
        _render_lessons_tab(df)
