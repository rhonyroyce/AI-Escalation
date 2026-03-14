# CLAUDE.md

## Project Overview

Escalation AI is an end-to-end ML pipeline and executive dashboard that transforms raw IT escalation ticket data into actionable strategic intelligence. It ingests Excel/CSV ticket exports, classifies each ticket using a hybrid keyword + LLM approach via a local Ollama server, computes a McKinsey-style Strategic Friction Score (base severity x cascading multipliers), detects repeat offenders through FAISS-powered recidivism analysis, predicts resolution times with Random Forest models, and generates a multi-sheet Excel report with embedded charts. A unified Streamlit dashboard (combining Escalation AI and Project Pulse views) lets operations leaders and executives explore results interactively. The system runs entirely on-premises with optional GPU acceleration.

## Architecture

### 7-Phase Pipeline (`escalation_ai/pipeline/orchestrator.py`)

```text
[Excel/CSV] → load_data() → self.df (raw DataFrame)
  → Phase 1: AI Classification        — adds AI_Category, AI_Confidence, AI_Sub_Category
  → Phase 2: Strategic Friction Scoring — composite risk score per ticket
  → Phase 3: Recidivism/Learning       — FAISS nearest-neighbor search for repeat issues
  → Phase 4: Recurrence Prediction     — Random Forest probability of recurrence
  → Phase 5: Similar Ticket Analysis   — GPU-accelerated embedding search for resolution strategies
  → Phase 6: Resolution Time Prediction — Random Forest regressor for time-to-resolve
  → Phase 7: Executive Summary & Report — LLM-generated summary + multi-sheet Excel output
```

### 3-Stage Execution (`run.py`)

1. **ML Pipeline** — runs all 7 phases, outputs `Strategic_Report.xlsx`
2. **AI Cache Pre-Generation** — pre-computes executive summary, classifications, and embeddings index to `.cache/ai_insights.json`
3. **Dashboard Launch** — spawns Streamlit subprocess running `unified_app.py`

### Dashboard Architecture

- `unified_app.py` serves both Pulse + Escalation AI via `st.navigation()` with grouped pages
- Pulse Dashboard: `pulse_dashboard/` (7 pages, utilities, McKinsey-grade charts)
- Escalation AI Dashboard: `escalation_ai/dashboard/streamlit_app.py` (9,260-line monolith, refactoring in progress)
- Page shims in `escalation_ai/dashboard/pages/` call render functions via `esc_bridge.py`
- Only ONE `st.set_page_config()` allowed per app — other modules guard with `_STANDALONE = __name__ == "__main__"`

## Tech Stack

- **Python 3.10+** — core language
- **Ollama** — local LLM server (qwen3:14b for generation, qwen2:1.5b for embeddings)
- **scikit-learn** — Random Forest models for recurrence and resolution prediction
- **FAISS (faiss-cpu)** — vector similarity search for recidivism and similar ticket analysis
- **pandas / numpy** — tabular data manipulation
- **Streamlit** — interactive dashboard framework
- **Plotly / matplotlib / seaborn** — visualization
- **openpyxl** — Excel report generation
- **tqdm** — progress bars
- **NVIDIA GPU + CUDA 12.8+** — optional, auto-detected for accelerated inference

## Entry Points

- `python run.py` — canonical entry (pipeline + dashboard)
- `python run.py --dashboard-only` — skip pipeline, launch dashboard only
- `python run.py --no-gui` — run pipeline without dashboard
- `python run.py --health-check` — verify Ollama server and dependencies
- `python run.py --log-level DEBUG --log-file custom.log` — logging configuration
- `streamlit run unified_app.py` — dashboard only (direct)
- `python main.py` — DEPRECATED, routes to modular package

## Key Files

- `escalation_ai/pipeline/orchestrator.py` — pipeline coordinator (all 7 phases)
- `escalation_ai/core/config.py` — all constants, weights, thresholds, column mappings (single source of truth)
- `escalation_ai/core/ai_engine.py` — Ollama LLM interface (embeddings + text generation)
- `escalation_ai/core/logging_config.py` — structured JSON logging with PhaseTimer context manager
- `escalation_ai/core/gpu_utils.py` — GPU detection and VRAM-based model tier selection
- `escalation_ai/predictors/` — ML models (recurrence, resolution, similar tickets, vector store)
- `escalation_ai/predictors/vector_store.py` — FAISS-backed TicketVectorStore
- `escalation_ai/scoring/` — Strategic Friction Score computation
- `escalation_ai/classification/` — hybrid keyword + embedding classifier
- `escalation_ai/dashboard/` — Streamlit dashboard pages
- `pulse_dashboard/` — Project Pulse dashboard (7 pages)
- `unified_app.py` — unified dashboard entry point (both dashboards)
- `run.py` — CLI entry point (pipeline + cache + dashboard)

## Data Sources

- **Escalation tickets**: Excel/CSV input → `Strategic_Report.xlsx` (pipeline output, `Scored Data` sheet)
- **Pulse**: `ProjectPulse.xlsx` → `Project Pulse` sheet (2,074 rows, 55 projects, 4 regions)
- **Financial**: `price_catalog.xlsx` (single source of truth for cost calculations)

## Active Branches

- **main**: Stable release branch
- **pulse**: Cross-dashboard features, pipeline metadata, structured logging, FAISS integration (active development)

## Deleted Branches (merged into main)

- `strategic` — Escalation AI + unified features (merged 2026-03-13)
- `newembed` — Embedding model improvements (merged 2026-03-13)
- `newinsights` — Financial metrics and executive slides (merged 2026-03-13)
- `rtx5070` — CUDA 12.9 / Blackwell GPU support (merged 2026-03-13)

## Remote-Only Branches

- `origin/iter2` — iteration 2 (no local tracking branch)
- `origin/rapids` — RAPIDS GPU acceleration (no local tracking branch)
- `origin/streamlit` — Streamlit experiments (no local tracking branch)

## Known Issues

- `EscalationAI0126.py` is a deprecated monolith — do not modify, will be removed
- `streamlit_app.py` is ~9,260 lines — refactoring in progress (4 largest functions extracted)
- Test coverage is <10% — see `tests/` for existing suites (classification, scoring, financial, predictors, pipeline integration)

## Development Patterns

- All config in `escalation_ai/core/config.py` — never hardcode magic numbers
- Mock Ollama in tests (no real LLM server needed)
- Use joblib (not pickle) for ML model serialization (security: prevents arbitrary code execution)
- Error handling: always use specific exception types, never bare `except:`
- Use explicit imports from config.py, not wildcard imports
- Structured logging via `escalation_ai/core/logging_config.py` — use `logger.info()` with extra fields, not `print()`
- CI/CD: GitHub Actions runs lint, typecheck, test, and integration jobs (`.github/workflows/ci.yml`)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_classification.py -v
pytest tests/test_scoring.py -v
pytest tests/test_financial.py -v
pytest tests/test_predictors.py -v
pytest tests/test_pipeline_integration.py -v
pytest tests/test_vector_store.py -v
```

## Running

```bash
# Full pipeline + dashboard
python run.py

# Dashboard only
python run.py --dashboard-only

# Pipeline only (no GUI)
python run.py --no-gui

# Health check
python run.py --health-check

# Custom input/output
python run.py --file input.xlsx --output report.xlsx

# Debug logging
python run.py --log-level DEBUG --log-file debug.log
```
