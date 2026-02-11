"""
Pulse Dashboard - Page 5: AI Insights (Ollama-Powered)

This page provides AI-driven analysis of project escalation data using a locally
hosted Ollama LLM server.  It is organised into five tabs:

1. **Executive Summary** -- A pre-generated (on app startup) natural-language
   summary of portfolio-wide pain points, recurring themes, and the most urgent
   issue.  The summary is stored in ``st.session_state['ai_exec_summary']`` so
   that it persists across Streamlit re-runs and page switches.  A "Regenerate"
   button is available when Ollama is live.

2. **Issue Categorization** -- Each pain-point text is sent (one-at-a-time) to
   the LLM, which classifies it into one of 11 predefined telecom issue
   categories (e.g. Resource/Staffing, Timeline/Delays).  Results are cached in
   ``st.session_state['ai_issue_categories']`` (list of category strings) and
   ``st.session_state['ai_issue_texts']`` (the original texts that were
   classified).  A horizontal bar chart shows the distribution of categories.

3. **Risk Scoring** -- On-demand, per-project risk assessment.  The user picks a
   project from a dropdown; the page assembles its text fields (Comments, Pain
   Points, Resolution Plan) together with the project's average Pulse score and
   asks the LLM to rate the escalation risk on a 0-10 scale.

4. **Semantic Search** -- Natural-language search over project text data using
   vector embeddings.  An embeddings index (``st.session_state['embeddings_index']``)
   is pre-built on app startup by calling ``build_embeddings_index()`` from the
   ``pulse_insights`` utility.  The index stores text chunks, their embedding
   vectors, and associated metadata (project, region, area, column, score).
   At query time, ``semantic_search()`` computes the cosine similarity between
   the query embedding and all stored embeddings, returning the top-k most
   similar documents.

5. **Action Item Extraction** -- On-demand extraction of structured action items
   from a selected project's Resolution Plan text.  The LLM is asked to produce
   ACTION / OWNER / STATUS triples.

**AI Cache Mechanism**
All expensive AI outputs are stored in ``st.session_state`` so they survive
Streamlit reruns:
- ``ai_exec_summary``    -- string: the executive summary text
- ``ai_issue_categories`` -- list[str]: per-issue category labels
- ``ai_issue_texts``      -- list[str]: the original issue texts that were classified
- ``embeddings_index``    -- dict with keys 'texts', 'embeddings', 'metadata':
                             the vector search index

When cached data exists, it is displayed immediately and a "Regenerate" /
"Reclassify" / "Refresh Index" button allows the user to overwrite the cache.
When no cache is present AND Ollama is offline, a helpful message explains how
to start the server.

**Scoring Context**
The Pulse scoring system assigns each project a score across 8 dimensions
(Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential), each rated
0-3, yielding a Total Score of 0-24.  The Total Score is referenced in the
Risk Scoring tab when contextualising the LLM prompt.
"""

# ---------------------------------------------------------------------------
# Path setup -- allow imports from the parent ``pulse_dashboard`` package.
# This is needed because Streamlit pages live in a ``pages/`` subdirectory
# and cannot resolve sibling packages (``utils/``) without this sys.path hack.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

# Sidebar renders the global Region / Area / Year / Week filters and returns
# the filtered DataFrame.  ``inject_css`` pushes the dark-theme stylesheet
# into the Streamlit page.
from utils.sidebar import render_sidebar
from utils.styles import inject_css

# AI utility functions wrapping the Ollama REST API:
# - check_ollama()          -- ping the Ollama server
# - ollama_generate()       -- send a prompt and receive generated text
# - build_embeddings_index() -- embed text chunks and build a search index
# - semantic_search()       -- query the index with a natural-language string
# CHAT_MODEL / EMBED_MODEL  -- model identifiers (e.g. "qwen3:14b" / "qwen2:1.5b")
from utils.pulse_insights import (
    check_ollama, ollama_generate, build_embeddings_index,
    semantic_search, CHAT_MODEL, EMBED_MODEL,
)

# ---------------------------------------------------------------------------
# Page initialisation
# ---------------------------------------------------------------------------

# Inject the shared dark-theme CSS into this page.
inject_css()

# render_sidebar() applies the user's sidebar filters (Region, Area, Year,
# Week) and returns the filtered DataFrame.  If the DataFrame is empty or None
# (e.g. no data file loaded), we bail out early.
filtered_df = render_sidebar()
if filtered_df is None or filtered_df.empty:
    st.warning("No data loaded or no data matches filters.")
    st.stop()

# Keep a reference to the *unfiltered* full DataFrame.  This is used later
# when we need cross-filter context (e.g. semantic search indexes all data).
df = st.session_state.df

# Page title rendered as styled HTML using the ``main-header`` CSS class.
st.markdown('<p class="main-header">AI Insights</p>', unsafe_allow_html=True)

# ============================================================================
# OLLAMA STATUS CHECK
# ============================================================================
# The main app (Home page) checks Ollama availability on startup and stores
# the boolean in session_state['ollama_available'].  We read that cached flag
# here rather than re-pinging the server on every page navigation.
ollama_ok = st.session_state.get('ollama_available', False)
if not ollama_ok:
    # When Ollama is unreachable, display a warning and show setup instructions
    # inside a collapsible expander so the page is not cluttered.
    st.warning(
        "Ollama is not running — AI features are unavailable. "
        "Pre-computed results (if any) are shown below."
    )
    with st.expander("How to enable Ollama"):
        # Show the shell commands the user needs to run to start the LLM server
        # and download the required chat + embedding models.
        st.code(
            f"ollama serve\nollama pull {CHAT_MODEL}\nollama pull {EMBED_MODEL}",
            language="bash",
        )
else:
    st.success("Ollama is connected — AI features are active.")

# ============================================================================
# TABS -- Five horizontal tabs, one per AI feature
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary", "Issue Categories", "Risk Scoring",
    "Semantic Search", "Action Items",
])

# ── Tab 1: Executive Summary ───────────────────────────────────────────────
# This tab shows a natural-language summary of the portfolio's pain points.
# The summary is generated by the LLM from up to 20 pain-point texts and is
# cached in ``st.session_state['ai_exec_summary']``.
with tab1:
    st.markdown("### AI Executive Summary")

    # Attempt to load the cached summary from session_state.  If the main app
    # already generated it at startup, this will be a non-None string.
    cached_summary = st.session_state.get('ai_exec_summary')

    if cached_summary:
        # ── Display the cached summary ──
        st.markdown("---")
        st.markdown(cached_summary)
        st.caption("Generated automatically on startup.")

        # Offer a "Regenerate" button only when Ollama is reachable.
        if ollama_ok and st.button("Regenerate Summary", key="regen_exec"):
            # Gather pain-point texts from the *filtered* DataFrame (respects
            # the user's current Region/Area/Week selections).
            pain_points = filtered_df['Pain Points'].dropna()
            if not pain_points.empty:
                # Take at most 20 pain points to keep the prompt within the
                # model's context window.  Each text is truncated to 300 chars
                # to avoid excessively long prompts.
                texts = pain_points.head(20).tolist()
                combined = "\n".join([f"- {t[:300]}" for t in texts])

                # The prompt instructs the LLM to return a structured response:
                # SUMMARY, THEMES (top 5), and URGENT (most critical issue).
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
                with st.spinner("Regenerating summary..."):
                    response = ollama_generate(prompt)
                if response:
                    # Overwrite the cached summary and trigger a Streamlit
                    # rerun so the page re-renders with the new text.
                    st.session_state.ai_exec_summary = response
                    st.rerun()
                else:
                    st.error("Regeneration failed.")

    elif ollama_ok:
        # ── No cached summary, but Ollama is available: offer to generate ──
        pain_points = filtered_df['Pain Points'].dropna()
        st.markdown(f"*{len(pain_points)} pain points available from {len(filtered_df)} entries*")
        st.info("Executive summary was not pre-generated. Click below to generate now.")
        if st.button("Generate Executive Summary", key="gen_exec"):
            if pain_points.empty:
                st.warning("No pain points to analyze.")
            else:
                # Same prompt construction as the regenerate path above.
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
                    # Store in session_state so subsequent reruns show the
                    # cached version instead of asking the user again.
                    st.session_state.ai_exec_summary = response
                    st.rerun()
                else:
                    st.error("AI generation failed. Check Ollama status.")
    else:
        # ── No cached summary AND Ollama is offline ──
        st.info("Ollama is not available. Start Ollama and restart the app to auto-generate the executive summary.")

# ── Tab 2: Issue Categorization ────────────────────────────────────────────
# Each pain-point is individually classified into one of 11 predefined
# categories by the LLM.  Results are cached in two parallel session_state
# lists: 'ai_issue_categories' (the labels) and 'ai_issue_texts' (the
# original texts).  A bar chart shows the distribution.
with tab2:
    st.markdown("### Issue Categorization")

    # The 11 predefined telecom issue categories.  The LLM is asked to pick
    # exactly one of these for each pain-point.  If its output does not match
    # any category (fuzzy substring match), the default "Other" is used.
    CATEGORIES = [
        "Resource/Staffing", "Timeline/Delays", "Technical/Engineering",
        "Vendor/Partner", "Communication", "Process/Workflow",
        "Customer Satisfaction", "Budget/Commercial", "Equipment/Tools",
        "Scope Change", "Other"
    ]

    # Check for cached classification results.
    cached_cats = st.session_state.get('ai_issue_categories')
    cached_texts = st.session_state.get('ai_issue_texts')

    if cached_cats and cached_texts:
        # ── Display cached classification results ──
        import plotly.express as px
        from utils.styles import get_plotly_theme

        # Count the frequency of each category and display as a bar chart.
        cat_series = pd.Series(cached_cats)
        cat_counts = cat_series.value_counts()

        fig = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            title=f'Issue Category Distribution ({len(cached_cats)} issues classified)',
            labels={'x': 'Category', 'y': 'Count'},
            color=cat_counts.values,
            color_continuous_scale='Blues',  # Darker bars = higher count
        )
        fig.update_layout(**get_plotly_theme())
        st.plotly_chart(fig, use_container_width=True)

        # Show a detail table mapping each (truncated) issue text to its
        # assigned category.
        detail_df = pd.DataFrame({
            'Issue': [t[:120] for t in cached_texts],  # Truncate for readability
            'Category': cached_cats,
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        st.caption("Classified automatically on startup.")

        # Offer a "Reclassify" button (only when Ollama is live) to redo the
        # classification using the current filtered data.
        if ollama_ok and st.button("Reclassify Issues", key="regen_cats"):
            pain_points_cat = filtered_df['Pain Points'].dropna()
            if not pain_points_cat.empty:
                categories_result = []
                texts = pain_points_cat.head(30).tolist()  # Cap at 30 issues
                progress = st.progress(0)  # Visual progress bar
                for i, text in enumerate(texts):
                    # Each issue is sent individually so the LLM can focus on
                    # a single classification.  Temperature 0.1 for determinism.
                    p = f"Categorize this telecom project issue into ONE of: {', '.join(CATEGORIES)}\nIssue: {text[:300]}\nRespond with ONLY the category name, nothing else."
                    result = ollama_generate(p, temperature=0.1, timeout=30)
                    if result:
                        # Fuzzy matching: find the first predefined category
                        # whose name appears (case-insensitive) in the LLM's
                        # response.  Falls back to "Other" if nothing matches.
                        matched = next((c for c in CATEGORIES if c.lower() in result.strip().lower()), 'Other')
                        categories_result.append(matched)
                    else:
                        categories_result.append('Other')
                    # Update the progress bar after each classification.
                    progress.progress((i + 1) / len(texts))
                progress.empty()  # Remove the progress bar when done
                # Store results in session_state and rerun to display them.
                st.session_state.ai_issue_categories = categories_result
                st.session_state.ai_issue_texts = texts
                st.rerun()

    elif ollama_ok:
        # ── No cached results, but Ollama is available ──
        pain_points_cat = filtered_df['Pain Points'].dropna()
        st.markdown(f"*{len(pain_points_cat)} pain points to classify*")
        st.info("Issue categories were not pre-generated. Click below to classify now.")
        if st.button("Classify Issues", key="classify"):
            if pain_points_cat.empty:
                st.warning("No pain points to classify.")
            else:
                # Same classification loop as the reclassify path above.
                categories_result = []
                progress = st.progress(0)
                texts = pain_points_cat.head(30).tolist()
                for i, text in enumerate(texts):
                    prompt = f"Categorize this telecom project issue into ONE of: {', '.join(CATEGORIES)}\nIssue: {text[:300]}\nRespond with ONLY the category name, nothing else."
                    result = ollama_generate(prompt, temperature=0.1, timeout=30)
                    if result:
                        matched = next((c for c in CATEGORIES if c.lower() in result.strip().lower()), 'Other')
                        categories_result.append(matched)
                    else:
                        categories_result.append('Other')
                    progress.progress((i + 1) / len(texts))
                progress.empty()
                st.session_state.ai_issue_categories = categories_result
                st.session_state.ai_issue_texts = texts
                st.rerun()
    else:
        # ── No cache AND Ollama is offline ──
        st.info("Ollama is not available. Start Ollama and restart the app to auto-classify issues.")

# ── Tab 3: Risk Scoring ────────────────────────────────────────────────────
# On-demand risk assessment for a single project.  The LLM receives the
# project's text fields and its average Pulse Total Score (0-24), and returns
# a structured risk rating on a 0-10 scale.
with tab3:
    st.markdown("### Project Risk Assessment")
    st.markdown("AI-powered risk scoring for individual projects.")

    # Populate the project dropdown from the filtered data.
    projects = sorted(filtered_df['Project'].dropna().unique())
    selected_project = st.selectbox("Select Project", projects, key="risk_project")

    # The "Assess Risk" button is disabled when Ollama is offline.
    if st.button("Assess Risk", disabled=not ollama_ok, key="risk_assess"):
        # Get all rows for the selected project within the current filters.
        proj_rows = filtered_df[filtered_df['Project'] == selected_project]
        if proj_rows.empty:
            st.warning("No data for selected project.")
        else:
            # Assemble text fragments from the three key narrative columns.
            # We check for NaN values because these text columns may be sparse.
            text_parts = []
            for _, row in proj_rows.iterrows():
                for col in ['Comments', 'Pain Points', 'Resolution Plan']:
                    val = row.get(col)
                    # Guard against both None and float NaN (pandas NaN).
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        text_parts.append(f"{col}: {str(val)[:300]}")

            if not text_parts:
                st.info("No text data available for risk assessment.")
            else:
                # Limit to 10 text fragments to keep the prompt manageable.
                combined = "\n".join(text_parts[:10])
                # Compute the project's average Total Score (0-24 scale) to
                # give the LLM quantitative context alongside the text.
                avg_score = proj_rows['Total Score'].mean()

                # Structured prompt requesting SCORE (0-10), LEVEL (categorical),
                # FACTORS (bullet list), and RECOMMENDATION.
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

# ── Tab 4: Semantic Search ─────────────────────────────────────────────────
# Vector-based search over project text data using Ollama embeddings.
# The embeddings index is a dict stored in st.session_state['embeddings_index']
# with keys:
#   'texts'      -- list[str]: the original text chunks
#   'embeddings' -- np.ndarray: shape (n_docs, embed_dim), float32
#   'metadata'   -- list[dict]: each dict has 'project', 'region', 'area',
#                   'column', 'score' for the source row
# At query time, the query string is embedded and compared to all stored
# embeddings via cosine similarity to find the most relevant matches.
with tab4:
    st.markdown("### Semantic Search")
    st.markdown("Search project data using natural language.")

    # Check whether the embeddings index has been built (typically on startup).
    index = st.session_state.get('embeddings_index')
    if index and len(index.get('texts', [])) > 0:
        st.success(f"Search index ready: {len(index['texts'])} documents embedded.")
    else:
        st.info("Search index not yet built.")

    # Text input for the user's natural-language query.
    query = st.text_input("Search query", placeholder="e.g., design approval delays")

    # Two side-by-side buttons: one to build/refresh the index, one to search.
    col_a, col_b = st.columns(2)
    with col_a:
        # Build/Refresh: re-embeds the Comments, Pain Points, and Resolution
        # Plan columns from the *full* (unfiltered) DataFrame so that all
        # projects are searchable regardless of the current sidebar filters.
        if st.button("Build/Refresh Index", disabled=not ollama_ok, key="build_index"):
            with st.spinner("Building embeddings index (this may take a few minutes)..."):
                new_index = build_embeddings_index(df, columns=['Comments', 'Pain Points', 'Resolution Plan'])
                st.session_state.embeddings_index = new_index
                st.success(f"Index built: {len(new_index['texts'])} documents embedded.")

    with col_b:
        search_clicked = st.button("Search", disabled=not ollama_ok, key="search")

    # Execute the search when the user clicks "Search" and has entered a query.
    if query and search_clicked:
        # Re-fetch the index in case it was just built above in the same run.
        index = st.session_state.get('embeddings_index')
        if index is None or len(index.get('texts', [])) == 0:
            st.warning("Build the search index first.")
        else:
            with st.spinner("Searching..."):
                # semantic_search returns a list of dicts sorted by descending
                # cosine similarity, each containing: 'text', 'project',
                # 'region', 'area', 'column', 'score', 'similarity'.
                results = semantic_search(query, index, top_k=10)

            if results:
                for r in results:
                    # Convert similarity (0-1 float) to a human-friendly percentage.
                    sim_pct = r['similarity'] * 100
                    # Render each result as a styled glass-card using the
                    # dashboard's dark-theme CSS classes.
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

# ── Tab 5: Action Item Extraction ──────────────────────────────────────────
# On-demand extraction of structured action items from a project's
# Resolution Plan text.  The LLM parses free-text plans into structured
# ACTION / OWNER / STATUS triples.
with tab5:
    st.markdown("### Action Item Extraction")
    st.markdown("Extract action items from resolution plans.")

    # Separate project list for this tab (uses its own selectbox key).
    projects_action = sorted(filtered_df['Project'].dropna().unique())
    selected_proj_action = st.selectbox("Select Project", projects_action, key="action_project")

    if st.button("Extract Actions", disabled=not ollama_ok, key="extract_actions"):
        # Get the Resolution Plan entries for the selected project.
        proj_rows = filtered_df[filtered_df['Project'] == selected_proj_action]
        resolutions = proj_rows['Resolution Plan'].dropna().tolist()

        if not resolutions:
            st.info("No resolution plans found for this project.")
        else:
            # Concatenate up to 5 resolution plans (each truncated to 500 chars)
            # into a single context block for the LLM.
            combined = "\n\n".join([r[:500] for r in resolutions[:5]])

            # The prompt asks for structured action items.  "NO_ACTIONS" is a
            # sentinel the LLM should return if nothing actionable is found.
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
