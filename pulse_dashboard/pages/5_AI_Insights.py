"""
Pulse Dashboard - AI Insights (Ollama-Powered)

5 tabs:
1. Executive Summary Generator
2. Issue Categorization
3. Risk Scoring
4. Semantic Search
5. Action Item Extraction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from utils.sidebar import render_sidebar
from utils.styles import inject_css
from utils.pulse_insights import (
    check_ollama, ollama_generate, build_embeddings_index,
    semantic_search, CHAT_MODEL, EMBED_MODEL,
)

inject_css()

filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

df = st.session_state.df

st.markdown('<p class="main-header">AI Insights</p>', unsafe_allow_html=True)

# ============================================================================
# OLLAMA CHECK
# ============================================================================
ollama_ok = check_ollama()
if not ollama_ok:
    st.error("Ollama is not running. AI features are unavailable.")
    st.code(f"ollama serve\nollama pull {CHAT_MODEL}\nollama pull {EMBED_MODEL}", language="bash")
    st.info("Start Ollama and refresh this page to enable AI features.")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary", "Issue Categories", "Risk Scoring",
    "Semantic Search", "Action Items",
])

# ── Tab 1: Executive Summary Generator ──
with tab1:
    st.markdown("### AI Executive Summary")
    st.markdown("Analyze pain points and generate a structured executive briefing.")

    pain_points = filtered_df['Pain Points'].dropna()
    st.markdown(f"*{len(pain_points)} pain points available from {len(filtered_df)} entries*")

    if st.button("Generate Executive Summary", disabled=not ollama_ok, key="gen_exec"):
        if pain_points.empty:
            st.warning("No pain points to analyze.")
        else:
            texts = pain_points.head(20).tolist()
            combined = "\n".join([f"- {t[:300]}" for t in texts])

            prompt = f"""Analyze these project pain points from a telecom portfolio and provide:
1. A 2-3 sentence executive summary
2. Top 5 recurring themes
3. Most urgent issue requiring immediate attention

Pain Points:
{combined}

Format your response exactly as:
SUMMARY: <summary>
THEMES:
1. <theme>: <description>
2. <theme>: <description>
3. <theme>: <description>
4. <theme>: <description>
5. <theme>: <description>
URGENT: <urgent issue>"""

            with st.spinner("Generating AI summary..."):
                response = ollama_generate(prompt)

            if response:
                st.markdown("---")
                st.markdown(response)
            else:
                st.error("AI generation failed. Check Ollama status.")

# ── Tab 2: Issue Categorization ──
with tab2:
    st.markdown("### Issue Categorization")
    st.markdown("Classify pain points into operational categories.")

    CATEGORIES = [
        "Resource/Staffing", "Timeline/Delays", "Technical/Engineering",
        "Vendor/Partner", "Communication", "Process/Workflow",
        "Customer Satisfaction", "Budget/Commercial", "Equipment/Tools",
        "Scope Change", "Other"
    ]

    pain_points_cat = filtered_df['Pain Points'].dropna()
    st.markdown(f"*{len(pain_points_cat)} pain points to classify*")

    if st.button("Classify Issues", disabled=not ollama_ok, key="classify"):
        if pain_points_cat.empty:
            st.warning("No pain points to classify.")
        else:
            categories_result = []
            progress = st.progress(0)
            texts = pain_points_cat.head(30).tolist()

            for i, text in enumerate(texts):
                prompt = f"""Categorize this telecom project issue into ONE of: {', '.join(CATEGORIES)}
Issue: {text[:300]}
Respond with ONLY the category name, nothing else."""

                result = ollama_generate(prompt, temperature=0.1, timeout=30)
                if result:
                    # Match to closest category
                    result_clean = result.strip()
                    matched = next((c for c in CATEGORIES if c.lower() in result_clean.lower()), 'Other')
                    categories_result.append(matched)
                else:
                    categories_result.append('Other')
                progress.progress((i + 1) / len(texts))

            progress.empty()

            # Show distribution
            cat_series = pd.Series(categories_result)
            cat_counts = cat_series.value_counts()

            import plotly.express as px
            from utils.styles import get_plotly_theme

            fig = px.bar(
                x=cat_counts.index, y=cat_counts.values,
                title='Issue Category Distribution',
                labels={'x': 'Category', 'y': 'Count'},
                color=cat_counts.values,
                color_continuous_scale='Blues',
            )
            fig.update_layout(**get_plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Risk Scoring ──
with tab3:
    st.markdown("### Project Risk Assessment")
    st.markdown("AI-powered risk scoring for individual projects.")

    projects = sorted(filtered_df['Project'].dropna().unique())
    selected_project = st.selectbox("Select Project", projects, key="risk_project")

    if st.button("Assess Risk", disabled=not ollama_ok, key="risk_assess"):
        proj_rows = filtered_df[filtered_df['Project'] == selected_project]
        if proj_rows.empty:
            st.warning("No data for selected project.")
        else:
            # Combine all text fields for the project
            text_parts = []
            for _, row in proj_rows.iterrows():
                for col in ['Comments', 'Pain Points', 'Resolution Plan']:
                    val = row.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        text_parts.append(f"{col}: {str(val)[:300]}")

            if not text_parts:
                st.info("No text data available for risk assessment.")
            else:
                combined = "\n".join(text_parts[:10])
                avg_score = proj_rows['Total Score'].mean()

                prompt = f"""Analyze this telecom project and rate escalation risk 0-10.

Project: {selected_project}
Average Pulse Score: {avg_score:.1f}/24

Recent Data:
{combined}

Provide your assessment as:
SCORE: <0-10>
LEVEL: <Low/Medium/High/Critical>
FACTORS:
- <factor 1>
- <factor 2>
- <factor 3>
RECOMMENDATION: <one sentence recommendation>"""

                with st.spinner("Assessing risk..."):
                    response = ollama_generate(prompt)

                if response:
                    st.markdown("---")
                    st.markdown(response)
                else:
                    st.error("Risk assessment failed.")

# ── Tab 4: Semantic Search ──
with tab4:
    st.markdown("### Semantic Search")
    st.markdown("Search project data using natural language.")

    query = st.text_input("Search query", placeholder="e.g., design approval delays")

    # Build index button
    if st.button("Build/Refresh Search Index", disabled=not ollama_ok, key="build_index"):
        with st.spinner("Building embeddings index (this may take a few minutes)..."):
            index = build_embeddings_index(df, columns=['Comments', 'Pain Points', 'Resolution Plan'])
            st.session_state.embeddings_index = index
            st.success(f"Index built: {len(index['texts'])} documents embedded.")

    if query and st.button("Search", disabled=not ollama_ok, key="search"):
        index = st.session_state.get('embeddings_index')
        if index is None or len(index.get('texts', [])) == 0:
            st.warning("Build the search index first.")
        else:
            with st.spinner("Searching..."):
                results = semantic_search(query, index, top_k=10)

            if results:
                for r in results:
                    sim_pct = r['similarity'] * 100
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 12px 16px;">
                        <div style="display: flex; justify-content: space-between;">
                            <b style="color: #E0E0E0;">{r['project']}</b>
                            <span class="badge badge-info">{sim_pct:.0f}% match</span>
                        </div>
                        <p style="color: #94a3b8; font-size: 0.8rem;">{r['region']} / {r['area']} | {r['column']} | Score: {r['score']}</p>
                        <p style="color: #e2e8f0; font-size: 0.9rem;">{r['text'][:300]}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No results found.")

# ── Tab 5: Action Item Extraction ──
with tab5:
    st.markdown("### Action Item Extraction")
    st.markdown("Extract action items from resolution plans.")

    projects_action = sorted(filtered_df['Project'].dropna().unique())
    selected_proj_action = st.selectbox("Select Project", projects_action, key="action_project")

    if st.button("Extract Actions", disabled=not ollama_ok, key="extract_actions"):
        proj_rows = filtered_df[filtered_df['Project'] == selected_proj_action]
        resolutions = proj_rows['Resolution Plan'].dropna().tolist()

        if not resolutions:
            st.info("No resolution plans found for this project.")
        else:
            combined = "\n\n".join([r[:500] for r in resolutions[:5]])

            prompt = f"""Extract specific action items from these resolution plans for project "{selected_proj_action}":

{combined}

List each action as:
- ACTION: <specific action>
- OWNER: <who, if mentioned>
- STATUS: <pending/in-progress/blocked>

If no clear actions can be extracted, respond "NO_ACTIONS"."""

            with st.spinner("Extracting actions..."):
                response = ollama_generate(prompt)

            if response and "NO_ACTIONS" not in response:
                st.markdown("---")
                st.markdown(response)
            elif response:
                st.info("No clear action items could be extracted from the resolution plans.")
            else:
                st.error("Action extraction failed.")
