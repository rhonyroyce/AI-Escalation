# CLAUDE CODE PROMPT: McKinsey-Grade Pulse Dashboard with AI Insights

## PROJECT OVERVIEW

Build a production-ready Streamlit dashboard for telecommunications project portfolio management. The app analyzes weekly "Pulse" scores across projects, regions, and performance dimensions to surface insights, identify risks, and recommend actions.

**Key Principle:** This is a McKinsey-grade executive dashboard. Every chart must answer "So what?" and "Now what?" — not just display data.

---

## TECHNICAL REQUIREMENTS

### Stack
- **Framework:** Streamlit (multi-page app structure)
- **Data:** Excel file with "Project Pulse" tab (user selects file via file_uploader)
- **Charts:** Plotly (interactive, dark theme compatible)
- **AI Backend:** Ollama (local) with Qwen3:14b for chat, Qwen2:1.5b for embeddings
- **Styling:** Dark theme (#0a0f1a background), McKinsey color palette for charts

### File Handling
```python
# User must select the Excel file
uploaded_file = st.file_uploader("Upload Pulse Tracker", type=['xlsx', 'xls'])

if uploaded_file:
    # MUST read specifically from "Project Pulse" tab
    df = pd.read_excel(uploaded_file, sheet_name='Project Pulse')
```

---

## DATA SCHEMA (Project Pulse Tab)

| Column | Type | Description |
|--------|------|-------------|
| Wk | int | Week number (1-52+) |
| Region | str | Geographic region (South, West, North, etc.) |
| Area | str | Sub-region/market area |
| Project | str | Project name/identifier |
| PM | str | Project Manager name |
| Design | float | Score 0-3 |
| IX | float | Score 0-3 (Integration) |
| PAG | float | Score 0-3 |
| RF Opt | float | Score 0-3 (RF Optimization) |
| Field | float | Score 0-3 |
| CSAT | float | Score 0-3 (Customer Satisfaction) |
| PM Performance | float | Score 0-3 |
| Potential | float | Score 0-3 |
| Total Score | float | Sum of dimension scores (0-24, but typically 0-21) |
| Comments | str | Free text - project updates |
| Pain Points | str | Free text - issues and blockers |
| Resolution Plan | str | Free text - planned remediation |

### Scoring Thresholds
- **Green:** Total Score ≥ 17
- **Yellow:** Total Score 14-16
- **Red:** Total Score < 14

### Targets (configurable in sidebar)
- Pulse Score Target: 17.0
- Pulse Score Stretch: 19.0
- Green Projects %: 80%
- Max Red Projects: 3

---

## APP STRUCTURE (Multi-Page)

```
pulse_dashboard/
├── app.py                          # Main entry with file uploader
├── pages/
│   ├── 1_📊_Executive_Summary.py   # McKinsey SCR format
│   ├── 2_🔍_Drill_Down.py          # Sunburst, Treemap, Sankey
│   ├── 3_📈_Trends.py              # Time series with forecast
│   ├── 4_🎯_Prioritization.py      # 2x2 matrix, Pareto
│   ├── 5_🤖_AI_Insights.py         # Ollama-powered analysis
│   └── 6_📋_Project_Details.py     # Filterable data table
├── utils/
│   ├── data_loader.py              # Excel loading, caching
│   ├── mckinsey_charts.py          # All chart functions
│   ├── pulse_insights.py           # Ollama integration
│   └── styles.py                   # Theme and CSS
└── requirements.txt
```

---

## PAGE 1: EXECUTIVE SUMMARY (McKinsey SCR Format)

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│ HEADER: Week X | Portfolio Status: [ON TRACK/AT RISK/OFF TRACK]    │
│         Pulse Score: XX.X vs 17.0 target (+/-X.X)                  │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────┬──────────────────┐
│ SITUATION        │ COMPLICATIONS    │ RESOLUTION       │
│ (Blue border)    │ (Red border)     │ (Green border)   │
│ Where are we?    │ What's wrong?    │ What to do?      │
└──────────────────┴──────────────────┴──────────────────┘

┌────────┬────────┬────────┬────────┬────────┐
│ Pulse  │ Green  │ Yellow │ Red    │ Total  │
│ Score  │ Proj   │ Proj   │ Proj   │ Proj   │
│ XX.X   │ XX     │ XX     │ XX     │ XX     │
│ ▲/▼ WoW│ vs 80% │        │ ≤3 tgt │        │
└────────┴────────┴────────┴────────┴────────┘

┌─────────────────────────────┬─────────────────────────────┐
│ VARIANCE CHART              │ TREND + FORECAST            │
│ Bullet: Actual vs Target    │ Line with projection        │
└─────────────────────────────┴─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 💡 KEY INSIGHT: [Auto-generated insight about worst dimension]      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┬─────────────────────────────┐
│ WATERFALL                   │ PARETO                      │
│ Score decomposition         │ 80/20 problem areas         │
└─────────────────────────────┴─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ RECOMMENDATIONS TABLE                                               │
│ Priority | Category | Action | Rationale | Owner | Timeline         │
└─────────────────────────────────────────────────────────────────────┘
```

### Auto-Generated Content Logic

**Situation:** 
```python
f"Week {latest_week}: Portfolio pulse at {avg_pulse:.1f} ({variance:+.1f} vs target of {target})"
```

**Complications (add if true):**
- Red projects > 3: "{n} projects in red status (>{threshold} threshold)"
- Green % < 80%: "Only {pct}% green vs {target}% target"
- WoW decline: "Pulse declined {delta:.2f} pts WoW"
- Region decline > 0.5: "{region} region down {delta:.1f} pts"

**Resolution (auto-recommend):**
- Worst dimensions: "Focus on {dim1} (avg {score1:.2f}) and {dim2} (avg {score2:.2f})"
- Red projects: "Escalate: {project1}, {project2}, {project3}"
- Worst region: "Deploy support to {region} region"

---

## PAGE 2: DRILL-DOWN VISUALIZATIONS

### Sidebar Filters
- Week selector (single or range)
- Region multiselect
- Area multiselect
- Pulse Status filter (Green/Yellow/Red)

### Chart Tabs

**Tab 1: Sunburst**
```python
fig = px.sunburst(
    df, 
    path=['Region', 'Area', 'Project'],
    color='Total Score',
    color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
    color_continuous_midpoint=16
)
```
- Click to drill: Region → Area → Project
- Color = pulse health

**Tab 2: Treemap**
```python
# Aggregate first
agg = df.groupby(['Region', 'Area']).agg(
    Projects=('Project', 'count'),
    AvgPulse=('Total Score', 'mean')
).reset_index()

fig = px.treemap(
    agg,
    path=['Region', 'Area'],
    values='Projects',      # Size = project count
    color='AvgPulse',       # Color = health
    color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e']
)
```

**Tab 3: Sankey (Score Flow)**
```python
# Show: Dimension → Score Level → Pulse Status
# Nodes: [Design, IX, PAG, RF Opt, Field, CSAT, PM Performance, Potential] 
#        + [Critical (0-1), At Risk (2), Healthy (3)]
#        + [Red (<14), Yellow (14-16), Green (17+)]

# Melt score columns, categorize, count flows
# Build go.Sankey with proper node colors and link flows
```
- Shows which dimensions are dragging scores down
- Visualizes flow from individual scores → overall status

**Tab 4: Icicle**
```python
fig = px.icicle(
    df,
    path=['Region', 'Area', 'Pulse Status'],
    color='Total Score',
    color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e']
)
```

### Click Interactions
When user clicks a chart element, show detail panel below with:
- Filtered project list
- Key metrics for selection
- Pain Points and Comments for selected projects

---

## PAGE 3: TRENDS

### Time Series Chart
```python
weekly = df.groupby('Wk').agg({
    'Total Score': 'mean',
    'Project': 'count'
}).reset_index()

# Main line
fig.add_trace(go.Scatter(x=weekly['Wk'], y=weekly['Total Score'], name='Actual'))

# Target line
fig.add_hline(y=17, line_dash='dash', annotation_text='Target')

# Forecast (linear regression on last 8 weeks)
slope, intercept = np.polyfit(recent_x, recent_y, 1)
forecast_weeks = [last_week + i for i in range(1, 5)]
forecast_values = [slope * w + intercept for w in forecast_weeks]

fig.add_trace(go.Scatter(
    x=forecast_weeks, y=forecast_values,
    mode='lines', line=dict(dash='dash'),
    name='Forecast'
))
```

### Sparklines by Region
Small multiples showing trend for each region

### Week-over-Week Changes Table
| Region | This Week | Last Week | Δ | Trend |
|--------|-----------|-----------|---|-------|

---

## PAGE 4: PRIORITIZATION

### 2x2 Impact-Effort Matrix
```python
# Impact = gap to target (17 - Total Score), clipped at 0
# Effort = count of dimensions < 2

# Quadrants:
# - Quick Wins: High Impact, Low Effort (prioritize)
# - Major Projects: High Impact, High Effort (plan carefully)
# - Fill-ins: Low Impact, Low Effort (if time permits)
# - Deprioritize: Low Impact, High Effort (skip)

fig = px.scatter(
    df,
    x='Effort', y='Impact',
    color='Quadrant',
    hover_data=['Project', 'Region', 'Total Score']
)

# Add quadrant lines and labels
```

### Quick Wins Panel
Show top 5 projects in "Quick Wins" quadrant with:
- Project name
- Current score
- Improvement potential
- Primary dimension to fix

### Pareto Chart
```python
# Count at-risk projects by Area
issues = df[df['Total Score'] < 17].groupby('Area').size().sort_values(ascending=False)
cumulative_pct = issues.cumsum() / issues.sum() * 100

# Bar + Line combo
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=issues.index, y=issues.values), secondary_y=False)
fig.add_trace(go.Scatter(x=issues.index, y=cumulative_pct, mode='lines+markers'), secondary_y=True)
fig.add_hline(y=80, secondary_y=True, line_dash='dash')
```

---

## PAGE 5: AI INSIGHTS (Ollama Integration)

### Configuration
```python
OLLAMA_CONFIG = {
    'base_url': 'http://localhost:11434',
    'chat_model': 'qwen3:14b',
    'embed_model': 'qwen2:1.5b',
    'temperature': 0.3,
    'timeout': 120
}
```

### Connection Check
```python
def check_ollama():
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

# Show status indicator
if not check_ollama():
    st.error("⚠️ Ollama not running. Start with: `ollama serve`")
    st.code("ollama pull qwen3:14b\nollama pull qwen2:1.5b", language="bash")
    return
```

### Tab 1: Executive Summary Generator
```python
# Combine top N pain points
texts = df['Pain Points'].dropna().head(20).tolist()
combined = "\n".join([f"- {t[:300]}" for t in texts])

prompt = f"""Analyze these project pain points and provide:
1. A 2-3 sentence executive summary
2. Top 5 recurring themes
3. Most urgent issue requiring immediate attention

Pain Points:
{combined}

Format:
SUMMARY: <summary>
THEMES:
1. <theme>: <description>
...
URGENT: <urgent issue>
"""

response = ollama_generate(prompt)
```

### Tab 2: Issue Categorization
```python
CATEGORIES = [
    "Resource/Staffing", "Timeline/Delays", "Technical/Engineering",
    "Vendor/Partner", "Communication", "Process/Workflow",
    "Customer Satisfaction", "Budget/Commercial", "Equipment/Tools",
    "Scope Change", "Other"
]

# Classify each pain point
prompt = f"""Categorize this issue into ONE of: {', '.join(CATEGORIES)}
Issue: {text}
Respond with ONLY the category name."""

# Aggregate and show pie/bar chart of distribution
```

### Tab 3: Risk Scoring
```python
# For selected project, analyze all text fields
texts = []
for col in ['Comments', 'Pain Points', 'Resolution Plan']:
    if pd.notna(row[col]):
        texts.append(f"{col}: {row[col]}")

prompt = f"""Analyze this project and rate escalation risk 0-10.

{chr(10).join(texts)}

Provide:
SCORE: <0-10>
LEVEL: <Low/Medium/High/Critical>
FACTORS:
- <factor 1>
- <factor 2>
- <factor 3>
"""
```

### Tab 4: Semantic Search
```python
# Build embeddings index
def build_index(df, columns=['Comments', 'Pain Points']):
    texts, metadata = [], []
    for col in columns:
        for idx, row in df.iterrows():
            if pd.notna(row[col]):
                texts.append(str(row[col])[:500])
                metadata.append({
                    'index': idx, 'column': col,
                    'project': row['Project'], 'region': row['Region']
                })
    
    embeddings = [ollama_embed(t) for t in texts]
    return {'embeddings': np.array(embeddings), 'metadata': metadata, 'texts': texts}

# Search
def semantic_search(query, index, top_k=5):
    query_emb = ollama_embed(query)
    similarities = cosine_similarity(index['embeddings'], query_emb)
    top_idx = np.argsort(similarities)[::-1][:top_k]
    return [{'similarity': similarities[i], **index['metadata'][i]} for i in top_idx]
```

### Tab 5: Action Item Extraction
```python
prompt = f"""Extract specific action items from this resolution plan:

{resolution_text}

List each as:
- ACTION: <specific action>
- OWNER: <who, if mentioned>
- STATUS: <pending/in-progress/blocked>

If no clear actions, respond "NO_ACTIONS"
"""
```

---

## PAGE 6: PROJECT DETAILS

### Filterable Data Table
```python
# Column configuration
st.dataframe(
    filtered_df,
    column_config={
        'Total Score': st.column_config.ProgressColumn(
            min_value=0, max_value=24, format="%.1f"
        ),
        'Project': st.column_config.TextColumn(width='large'),
        'Comments': st.column_config.TextColumn(width='large'),
    },
    use_container_width=True,
    hide_index=True
)
```

### Project Deep Dive
On row select, show expanded view with:
- All dimension scores as radar chart
- Historical trend for this project
- Full text of Comments, Pain Points, Resolution Plan
- AI-generated risk assessment (if Ollama available)

---

## STYLING

### Dark Theme CSS
```python
st.markdown("""
<style>
    .stApp { background-color: #0a0f1a; }
    .stMarkdown { color: #e2e8f0; }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .kpi-delta-positive { color: #22c55e; }
    .kpi-delta-negative { color: #ef4444; }
    
    /* SCR Boxes */
    .scr-situation {
        background: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .scr-complication {
        background: #1c1917;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .scr-resolution {
        background: #052e16;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    /* Insight Callout */
    .insight-callout {
        background: linear-gradient(135deg, #1e3a5f, #0c4a6e);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;
    }
</style>
""", unsafe_allow_html=True)
```

### McKinsey Chart Colors
```python
MCKINSEY_COLORS = {
    'primary_blue': '#004165',
    'secondary_blue': '#0077B6',
    'accent_teal': '#00A5A8',
    'positive': '#00843D',
    'negative': '#E31B23',
    'warning': '#F2A900',
    'neutral': '#6D6E71',
    'light_gray': '#D0D0CE',
}

# For Plotly dark theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Inter'),
    xaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
    yaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
)
```

---

## SIDEBAR (Global)

```python
with st.sidebar:
    st.image("logo.png", width=150)  # Optional
    st.title("Pulse Dashboard")
    
    # File uploader (only on main page, then store in session_state)
    if 'df' not in st.session_state:
        uploaded = st.file_uploader("Upload Pulse Tracker", type=['xlsx'])
        if uploaded:
            st.session_state.df = pd.read_excel(uploaded, sheet_name='Project Pulse')
            st.rerun()
    
    if 'df' in st.session_state:
        df = st.session_state.df
        
        st.markdown("---")
        st.markdown("### Filters")
        
        # Week filter
        weeks = sorted(df['Wk'].dropna().unique())
        selected_week = st.selectbox("Week", weeks, index=len(weeks)-1)
        
        # Region filter
        regions = df['Region'].dropna().unique()
        selected_regions = st.multiselect("Regions", regions, default=list(regions))
        
        # Status filter
        status_filter = st.multiselect(
            "Pulse Status",
            ['Green (17+)', 'Yellow (14-16)', 'Red (<14)'],
            default=['Green (17+)', 'Yellow (14-16)', 'Red (<14)']
        )
        
        st.markdown("---")
        st.markdown("### Targets")
        pulse_target = st.number_input("Pulse Target", value=17.0, step=0.5)
        green_pct_target = st.slider("Green % Target", 0, 100, 80)
        
        st.markdown("---")
        st.markdown("### Ollama Status")
        if check_ollama():
            st.success("✓ Connected")
        else:
            st.error("✗ Not running")
```

---

## SESSION STATE MANAGEMENT

```python
# Initialize
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embeddings_index' not in st.session_state:
    st.session_state.embeddings_index = None
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None

# Cache expensive operations
@st.cache_data
def load_and_process_data(file):
    df = pd.read_excel(file, sheet_name='Project Pulse')
    # Data cleaning...
    return df

@st.cache_resource
def build_embeddings_index(df):
    # Build once, reuse
    return create_index(df)
```

---

## REQUIREMENTS.TXT

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
openpyxl>=3.1.0
requests>=2.31.0
scikit-learn>=1.3.0
```

---

## IMPORTANT IMPLEMENTATION NOTES

1. **File Upload Flow:**
   - Main app.py shows file uploader if no data in session_state
   - Once uploaded, data persists across pages via session_state
   - "Project Pulse" tab is REQUIRED - show error if not found

2. **Ollama Graceful Degradation:**
   - AI Insights page works independently
   - If Ollama not running, show setup instructions but don't break app
   - Cache embeddings index in session_state to avoid rebuilding

3. **Chart Interactions:**
   - Use `st.plotly_chart(fig, on_select="rerun")` for click handling
   - Store selection in session_state
   - Show detail panel based on selection

4. **Performance:**
   - Use `@st.cache_data` for data loading
   - Limit AI operations to user-triggered actions (buttons)
   - Paginate large tables

5. **Error Handling:**
   - Check for "Project Pulse" tab existence
   - Handle missing columns gracefully
   - Validate numeric columns before calculations

---

## START BUILDING

Begin with:
1. `app.py` - Main entry with file uploader
2. `utils/data_loader.py` - Load and validate Excel
3. `pages/1_📊_Executive_Summary.py` - Core McKinsey dashboard
4. Add remaining pages incrementally

Test with the provided Pulse_Tracker.xlsx file.
