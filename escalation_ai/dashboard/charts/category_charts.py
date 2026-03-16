"""Deep Analysis — Categories tab.

Renders cross-category sub-category performance (top/bottom), category
selector with sunburst, severity breakdown, and single-category drill-down.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Pure figure builders
# ---------------------------------------------------------------------------

def subcategory_performance(df: pd.DataFrame, view_mode: str) -> go.Figure | None:
    """Horizontal bar of top/bottom sub-categories by the selected metric."""
    sub_cat_col = _sub_cat_col(df)
    if sub_cat_col is None or 'AI_Category' not in df.columns:
        return None

    all_sub_data = df.groupby([sub_cat_col, 'AI_Category']).agg({
        'Financial_Impact': 'sum',
        'AI_Recurrence_Risk': 'mean' if 'AI_Recurrence_Risk' in df.columns else 'count',
        'Predicted_Resolution_Days': 'mean' if 'Predicted_Resolution_Days' in df.columns else 'count'
    }).reset_index()
    all_sub_data.columns = ['SubCategory', 'Category', 'Cost', 'Recurrence', 'Resolution']
    all_sub_data['Count'] = df.groupby([sub_cat_col, 'AI_Category']).size().values

    if view_mode == "💰 Highest Cost":
        display_data = all_sub_data.nlargest(10, 'Cost')
        metric_col, color_scale, format_str = 'Cost', 'Reds', '${:,.0f}'
    elif view_mode == "💚 Lowest Cost":
        display_data = all_sub_data[all_sub_data['Count'] >= 2].nsmallest(10, 'Cost')
        metric_col, color_scale, format_str = 'Cost', 'Greens', '${:,.0f}'
    elif view_mode == "🔴 Highest Recurrence":
        display_data = all_sub_data[all_sub_data['Count'] >= 2].nlargest(10, 'Recurrence')
        metric_col, color_scale, format_str = 'Recurrence', 'Reds', '{:.1%}'
    else:
        display_data = all_sub_data[all_sub_data['Count'] >= 2].nlargest(10, 'Resolution')
        metric_col, color_scale, format_str = 'Resolution', 'Oranges', '{:.1f}d'

    display_data = display_data.sort_values(metric_col, ascending=True)

    labels = [
        f"{row['SubCategory'][:25]}... ({row['Category'][:15]})"
        if len(row['SubCategory']) > 25
        else f"{row['SubCategory']} ({row['Category'][:15]})"
        for _, row in display_data.iterrows()
    ]

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=display_data[metric_col],
        orientation='h',
        marker=dict(
            color=display_data[metric_col],
            colorscale=color_scale,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[
            format_str.format(v) if metric_col != 'Recurrence' else f'{v*100:.1f}%'
            for v in display_data[metric_col]
        ],
        textposition='outside',
        textfont=dict(size=11, color='#e2e8f0'),
        hovertemplate='<b>%{y}</b><br>Value: %{x}<br>Tickets: %{customdata}<extra></extra>',
        customdata=display_data['Count']
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=max(300, len(display_data) * 40),
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color='#e2e8f0')),
        showlegend=False
    )
    return fig


def severity_by_category(df: pd.DataFrame) -> go.Figure | None:
    """Stacked bar of severity distribution per category."""
    if 'AI_Category' not in df.columns or 'tickets_data_severity' not in df.columns:
        return None

    sev_by_cat = df.groupby(['AI_Category', 'tickets_data_severity']).size().reset_index(name='Count')
    severity_colors = {'Critical': '#ef4444', 'Major': '#f97316', 'Minor': '#22c55e'}

    fig = go.Figure()
    for severity in ['Critical', 'Major', 'Minor']:
        sev_data = sev_by_cat[sev_by_cat['tickets_data_severity'] == severity]
        if len(sev_data) > 0:
            fig.add_trace(go.Bar(
                name=severity,
                x=sev_data['AI_Category'],
                y=sev_data['Count'],
                marker_color=severity_colors.get(severity, '#3b82f6'),
                hovertemplate=f'<b>%{{x}}</b><br>{severity}: %{{y}}<extra></extra>'
            ))

    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(t=10, b=80, l=50, r=20),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color='#94a3b8'), showgrid=False),
        yaxis=dict(tickfont=dict(size=10, color='#94a3b8'), gridcolor='rgba(255,255,255,0.1)', title='Count'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(color='#94a3b8'))
    )
    return fig


def subcategory_drilldown(df: pd.DataFrame, selected_cat: str) -> go.Figure | None:
    """Horizontal bar of sub-categories within a single selected category."""
    sub_cat_col = _sub_cat_col(df)
    if sub_cat_col is None:
        return None

    cat_df = df[df['AI_Category'] == selected_cat]
    sub_data = cat_df.groupby(sub_cat_col).agg({
        'Financial_Impact': 'sum',
        'AI_Category': 'count'
    }).rename(columns={'AI_Category': 'Count'}).sort_values('Financial_Impact', ascending=True)

    if sub_data.empty:
        return None

    max_cost = sub_data['Financial_Impact'].max()
    colors = [
        f'rgba({int(59 + 180 * (v/max_cost))}, {int(130 - 62 * (v/max_cost))}, {int(246 - 178 * (v/max_cost))}, 0.9)'
        for v in sub_data['Financial_Impact']
    ]

    fig = go.Figure(data=[go.Bar(
        y=sub_data.index,
        x=sub_data['Financial_Impact'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.3)', width=1)),
        text=[f'${v:,.0f} ({c} tickets)' for v, c in zip(sub_data['Financial_Impact'], sub_data['Count'])],
        textposition='outside',
        textfont=dict(size=11, color='#e2e8f0'),
        hovertemplate='<b>%{y}</b><br>Cost: $%{x:,.0f}<extra></extra>'
    )])
    fig.update_layout(
        title=dict(text=f"Sub-Categories in {selected_cat}", font=dict(size=14, color='#e2e8f0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=max(250, len(sub_data) * 40),
        margin=dict(l=10, r=120, t=40, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                   tickfont=dict(size=10, color='#94a3b8'), tickformat='$,.0f'),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color='#e2e8f0')),
        showlegend=False
    )
    return fig


# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_tab(df: pd.DataFrame) -> None:
    """Render the Categories tab inside an already-active st.tabs context."""
    from escalation_ai.dashboard.streamlit_app import chart_category_sunburst

    sub_cat_col = _sub_cat_col(df)

    # ROW 1: Cross-category sub-category analysis
    st.markdown("#### 🏆 Top & Bottom Performing Sub-Categories (All Categories)")
    if sub_cat_col and 'AI_Category' in df.columns:
        view_mode = st.radio(
            "View by:",
            ["💰 Highest Cost", "💚 Lowest Cost", "🔴 Highest Recurrence", "⏱️ Slowest Resolution"],
            horizontal=True,
            key="subcategory_view_mode"
        )
        fig = subcategory_performance(df, view_mode)
        if fig:
            with st.spinner("Generating visualization..."):
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ROW 2: Sunburst + Severity Breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Category & Sub-Category Drill-Down")
        with st.spinner("Generating visualization..."):
            st.plotly_chart(chart_category_sunburst(df), use_container_width=True)
    with col2:
        st.markdown("#### 📊 Severity Breakdown by Category")
        fig = severity_by_category(df)
        if fig:
            with st.spinner("Generating visualization..."):
                st.plotly_chart(fig, use_container_width=True, key="sev_by_cat_deep")

    st.markdown("---")

    # ROW 3: Single category drill-down
    st.markdown("#### 🔍 Single Category Drill-Down")
    if 'AI_Category' in df.columns:
        categories = sorted(df['AI_Category'].unique().tolist())
        selected_cat = st.selectbox("Select a category to explore:", categories, key="deep_cat_select")

        cat_df = df[df['AI_Category'] == selected_cat]
        cat_cost = cat_df['Financial_Impact'].sum() if 'Financial_Impact' in cat_df.columns else 0
        cat_count = len(cat_df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tickets", f"{cat_count:,}")
        m2.metric("Total Cost", f"${cat_cost:,.0f}")
        m3.metric("Avg Cost", f"${cat_cost/cat_count:,.0f}" if cat_count > 0 else "$0")
        m4.metric("% of Total", f"{cat_count/len(df)*100:.1f}%")

        if sub_cat_col:
            fig = subcategory_drilldown(df, selected_cat)
            if fig:
                with st.spinner("Generating visualization..."):
                    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sub_cat_col(df: pd.DataFrame) -> str | None:
    if 'AI_Sub_Category' in df.columns:
        return 'AI_Sub_Category'
    if 'AI_SubCategory' in df.columns:
        return 'AI_SubCategory'
    return None
