"""Deep Analysis — Similarity tab.

Renders similarity search analysis: overview metrics, score distribution,
resolution consistency, and effectiveness heatmap.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Pure figure builders
# ---------------------------------------------------------------------------

def count_distribution(df: pd.DataFrame) -> go.Figure | None:
    """Histogram of Similar_Ticket_Count values."""
    if 'Similar_Ticket_Count' not in df.columns:
        return None
    fig = go.Figure(data=[go.Histogram(
        x=df['Similar_Ticket_Count'].dropna(),
        nbinsx=20,
        marker_color='#8b5cf6'
    )])
    fig.update_layout(
        title="Similar Ticket Count Distribution",
        xaxis_title="Number of Similar Tickets",
        yaxis_title="Frequency",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    return fig


def score_distribution(df: pd.DataFrame) -> go.Figure | None:
    """Histogram of similarity scores."""
    score_col = _score_col(df)
    if score_col is None:
        return None
    scores = df[score_col].dropna()
    scores = scores[scores > 0]
    if len(scores) == 0:
        return None

    fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20, marker_color='#06b6d4')])
    fig.update_layout(
        title="Similarity Score Distribution",
        xaxis_title="Similarity Score",
        yaxis_title="Frequency",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    return fig


def score_by_category(df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar of average similarity score per category."""
    score_col = _score_col(df)
    if score_col is None or 'AI_Category' not in df.columns:
        return None

    cat_scores = df.groupby('AI_Category')[score_col].agg(['mean', 'std', 'count'])
    cat_scores = cat_scores[cat_scores['count'] >= 3].sort_values('mean', ascending=True)
    if cat_scores.empty:
        return None

    fig = go.Figure(data=[go.Bar(
        y=cat_scores.index,
        x=cat_scores['mean'],
        orientation='h',
        marker_color=['#22c55e' if x >= 0.7 else '#f97316' if x >= 0.5 else '#ef4444' for x in cat_scores['mean']],
        error_x=dict(type='data', array=cat_scores['std'], visible=True),
        text=[f"{x:.2f}" for x in cat_scores['mean']],
        textposition='outside'
    )])
    fig.update_layout(
        title="Average Similarity Score by Category",
        xaxis_title="Average Similarity Score",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=max(300, len(cat_scores) * 35),
        margin=dict(l=10, r=80)
    )
    return fig


def consistency_pie(df: pd.DataFrame) -> go.Figure | None:
    """Donut chart of resolution consistency distribution."""
    if 'Resolution_Consistency' not in df.columns:
        return None
    counts = df['Resolution_Consistency'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=['#22c55e' if 'Consistent' in str(l) else '#ef4444' for l in counts.index],
        hole=0.4
    )])
    fig.update_layout(
        title="Resolution Consistency Distribution",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    return fig


def consistency_by_category(df: pd.DataFrame) -> go.Figure | None:
    """Horizontal bar of consistency rate per category."""
    if 'AI_Category' not in df.columns or 'Resolution_Consistency' not in df.columns:
        return None
    cat_cons = df.groupby('AI_Category')['Resolution_Consistency'].apply(
        lambda x: (x.str.contains('Consistent', case=False, na=False)).mean() * 100
    ).sort_values(ascending=True)
    if cat_cons.empty:
        return None

    fig = go.Figure(data=[go.Bar(
        y=cat_cons.index,
        x=cat_cons.values,
        orientation='h',
        marker_color=['#22c55e' if x >= 70 else '#f97316' if x >= 50 else '#ef4444' for x in cat_cons.values],
        text=[f"{x:.0f}%" for x in cat_cons.values],
        textposition='outside'
    )])
    fig.update_layout(
        title="Consistency Rate by Category",
        xaxis_title="Consistency %",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=max(300, len(cat_cons) * 35),
        margin=dict(l=10, r=60),
        xaxis=dict(range=[0, 110])
    )
    return fig


def effectiveness_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Heatmap of average similar ticket count by category and origin."""
    if 'AI_Category' not in df.columns or 'Similar_Ticket_Count' not in df.columns:
        return None

    origin_col = None
    for col in ['AI_Origin', 'Origin', 'Source']:
        if col in df.columns:
            origin_col = col
            break
    if origin_col is None:
        return None

    heatmap_data = df.pivot_table(
        values='Similar_Ticket_Count', index='AI_Category',
        columns=origin_col, aggfunc='mean'
    ).fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        text=np.round(heatmap_data.values, 1),
        texttemplate="%{text}",
        hovertemplate='Category: %{y}<br>Origin: %{x}<br>Avg Matches: %{z:.1f}<extra></extra>'
    ))
    fig.update_layout(
        title="Average Similar Ticket Count by Category & Origin",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=max(400, len(heatmap_data) * 30)
    )
    return fig


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_tab(df: pd.DataFrame) -> None:
    """Render the Similarity tab inside an already-active st.tabs context."""
    st.markdown("#### 🔗 Similarity Search Analysis")
    st.markdown("*Insights from comparing tickets to historical patterns*")

    has_similarity = any(
        col in df.columns
        for col in ['Similar_Ticket_Count', 'Best_Match_Similarity', 'Similarity_Score', 'Resolution_Consistency']
    )

    if not has_similarity:
        st.info("🔍 **Similarity search data not available.**\n\nRun the analysis with similarity search enabled to populate this section.")
        return

    sim_tabs = st.tabs(["📊 Overview", "📈 Score Analysis", "⚖️ Consistency", "🔥 Heatmap"])

    # --- Overview ---
    with sim_tabs[0]:
        if 'Similar_Ticket_Count' in df.columns:
            counts = df['Similar_Ticket_Count'].dropna()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Similar Tickets", f"{counts.mean():.1f}")
            with col2:
                zero_matches = (counts == 0).sum()
                st.metric("No Matches Found", f"{zero_matches}",
                          delta=f"{zero_matches/len(counts)*100:.0f}%" if len(counts) > 0 else "0%",
                          delta_color="inverse")
            with col3:
                high_matches = (counts >= 5).sum()
                st.metric("Good Coverage (5+)", f"{high_matches}",
                          delta=f"{high_matches/len(counts)*100:.0f}%" if len(counts) > 0 else "0%")
            st.markdown("---")

        fig = count_distribution(df)
        if fig:
            with st.spinner("Generating visualization..."):
                st.plotly_chart(fig, use_container_width=True)

    # --- Score Analysis ---
    with sim_tabs[1]:
        st.markdown("##### Similarity Score Analysis")
        st.markdown("*How confident are we in the similar ticket matches?*")
        score_col = _score_col(df)

        if score_col:
            scores = df[score_col].dropna()
            scores = scores[scores > 0]
            if len(scores) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Score", f"{scores.mean():.2f}")
                with col2:
                    high_conf = (scores >= 0.7).sum()
                    st.metric("High Confidence (≥0.7)", f"{high_conf}", delta=f"{high_conf/len(scores)*100:.0f}%")
                with col3:
                    low_conf = (scores < 0.5).sum()
                    st.metric("Low Confidence (<0.5)", f"{low_conf}",
                              delta=f"{low_conf/len(scores)*100:.0f}%", delta_color="inverse")
                st.markdown("---")

                fig = score_distribution(df)
                if fig:
                    with st.spinner("Generating visualization..."):
                        st.plotly_chart(fig, use_container_width=True)

                fig = score_by_category(df)
                if fig:
                    st.markdown("##### Similarity Scores by Category")
                    with st.spinner("Generating visualization..."):
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Similarity score data not available. Run the analysis pipeline to generate Best_Match_Similarity.")

    # --- Consistency ---
    with sim_tabs[2]:
        st.markdown("##### Resolution Consistency")
        st.markdown("*Are we resolving similar issues the same way?*")

        if 'Resolution_Consistency' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = consistency_pie(df)
                if fig:
                    with st.spinner("Generating visualization..."):
                        st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = consistency_by_category(df)
                if fig:
                    with st.spinner("Generating visualization..."):
                        st.plotly_chart(fig, use_container_width=True)

            # Inconsistent tickets table
            inconsistent = df[df['Resolution_Consistency'].str.contains('Inconsistent|Different', case=False, na=False)]
            if len(inconsistent) > 0:
                st.markdown("##### ⚠️ Tickets with Inconsistent Resolutions")
                st.markdown(f"*{len(inconsistent)} tickets resolved differently than similar historical cases*")

                display_cols = ['Identity', 'AI_Category', 'Similar_Ticket_Count']
                if 'AI_Sub_Category' in df.columns:
                    display_cols.insert(2, 'AI_Sub_Category')
                if 'Best_Match_Similarity' in df.columns:
                    display_cols.append('Best_Match_Similarity')
                display_cols = [c for c in display_cols if c in inconsistent.columns]
                if display_cols:
                    st.dataframe(inconsistent[display_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Resolution consistency data not available. Run similarity analysis with consistency checking enabled.")

    # --- Heatmap ---
    with sim_tabs[3]:
        st.markdown("##### Similarity Search Effectiveness")
        st.markdown("*Where do we have good historical coverage?*")

        if 'AI_Category' in df.columns and 'Similar_Ticket_Count' in df.columns:
            fig = effectiveness_heatmap(df)
            if fig:
                with st.spinner("Generating visualization..."):
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Origin column not found for heatmap. Showing coverage table instead.")

            # Coverage summary table
            st.markdown("##### Coverage by Category")
            coverage = df.groupby('AI_Category').agg({
                'Similar_Ticket_Count': ['mean', 'sum', 'count']
            })
            coverage.columns = ['Avg Matches', 'Total Matches', 'Ticket Count']
            coverage['Coverage Score'] = (coverage['Avg Matches'] * 20).clip(0, 100).round(0).astype(int)
            coverage = coverage.sort_values('Coverage Score', ascending=False)
            st.dataframe(
                coverage.style.background_gradient(subset=['Coverage Score'], cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.info("Need category and similar ticket count data for effectiveness analysis.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_col(df: pd.DataFrame) -> str | None:
    if 'Best_Match_Similarity' in df.columns:
        return 'Best_Match_Similarity'
    if 'Similarity_Score' in df.columns:
        return 'Similarity_Score'
    return None
