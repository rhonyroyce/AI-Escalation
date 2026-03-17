"""
Pipeline Orchestrator - Main execution flow for Escalation AI.

This module is the central coordinator for the entire Escalation AI analysis
pipeline.  It wires together every subsystem -- data loading, AI classification,
strategic scoring, recidivism detection, ML-based prediction, financial analysis,
and executive report generation -- into a single sequential workflow.

Architecture overview
---------------------
The pipeline is modelled as seven discrete phases, each of which reads the
shared ``pandas.DataFrame`` produced by the previous phase and enriches it
with additional columns:

  Phase 0  (implicit) : Load & validate raw ticket data from Excel / CSV.
  Phase 1  : AI Classification       -- assigns ``AI_Category``, ``AI_Confidence``,
              and ``AI_Sub_Category`` via the hybrid keyword + embedding classifier.
  Phase 2  : Strategic Friction Scoring -- computes a composite
              ``Strategic_Friction_Score`` per ticket.
  Phase 3  : Recidivism / Learning Analysis -- embeds every ticket and uses
              FAISS nearest-neighbor search to flag repeat / similar issues.
  Phase 4  : Recurrence Prediction   -- trains a Random Forest on engineered
              features to predict the probability that a ticket will recur.
  Phase 5  : Similar Ticket Analysis -- finds the closest resolved tickets to
              suggest resolution strategies.
  Phase 6  : Resolution Time Prediction -- trains a Random Forest regressor to
              estimate how long a ticket will take to resolve.
  Phase 7  : Executive Summary & Report -- feeds aggregated statistics into an
              LLM prompt (via Ollama) and writes the final Excel report.

Data flow
---------
::

    [Excel / CSV]
         |
         v
    load_data() --> self.df  (raw DataFrame)
         |
         v
    prepare_text() --> adds 'Combined_Text' column (cleaned concatenation
                       of summary + category fields)
         |
         v
    run_classification() --> adds 'AI_Category', 'AI_Confidence',
                             'AI_Sub_Category'
         |
         v
    run_scoring() --> adds 'Strategic_Friction_Score' and sub-components
         |
         v
    run_recidivism_analysis() --> adds 'Learning_Status', 'Recidivism_Score',
                                  'Similar_Historical_Issue', 'embedding'
         |
         v
    run_recurrence_prediction() --> adds 'AI_Recurrence_Probability',
                                    'AI_Recurrence_Risk_Tier'
         |
         v
    run_similar_ticket_analysis() --> adds similarity match columns
         |
         v
    run_resolution_time_prediction() --> adds 'AI_Predicted_Resolution_Hours'
         |
         v
    generate_executive_summary() --> computes financial metrics and invokes
                                     the LLM to produce a McKinsey-style report
         |
         v
    generate_report() --> writes everything to a multi-sheet Excel workbook
                          with embedded charts

Key external dependencies
-------------------------
- **Ollama** (local LLM server) for embedding generation and text synthesis.
- **pandas / numpy** for tabular data manipulation.
- **tqdm** for user-facing progress bars.
- **tkinter** for GUI file-picker dialogs and message boxes.
- **xlwings / pywin32** (optional, Windows-only) for refreshing Excel API /
  Power Query connections before reading the workbook.

Coordinates all 7 phases of the analysis pipeline:
1. Data Loading & Validation
2. AI Classification
3. Strategic Friction Scoring
4. Recidivism/Learning Analysis
5. Similar Ticket Analysis
6. Resolution Time Prediction
7. Report Generation
"""
from __future__ import annotations


import os
import sys
import logging
import requests
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Internal imports -- each maps to a distinct subsystem of the pipeline.
# ---------------------------------------------------------------------------
from ..core.config import (
    OLLAMA_BASE_URL, EMBED_MODEL, GEN_MODEL,       # Ollama server & model names
    COL_SUMMARY, COL_CATEGORY, COL_DATETIME, COL_TYPE,  # Column name constants
    COL_ORIGIN, COL_SEVERITY,
    TIMEOUT_GPU_QUERY,                              # Timeout for model listing
)
from ..core.logging_config import PhaseTimer
from ..core.ai_engine import OllamaBrain               # Embedding + generation wrapper
from ..core.utils import clean_text                     # Text normalisation helper
from ..classification import classify_rows              # Phase 1 classifier
from ..scoring import calculate_strategic_friction       # Phase 2 scorer
from ..feedback import FeedbackLearning, ResolutionFeedbackLearning, PriceCatalog
from ..predictors import (
    apply_recurrence_predictions,          # Phase 4 -- Random-Forest recurrence model
    apply_similar_ticket_analysis,         # Phase 5 -- k-NN similarity search
    apply_resolution_time_prediction       # Phase 6 -- Random-Forest regression
)
from ..financial import (
    calculate_financial_metrics,           # Core dollar-value calculations
    calculate_roi_metrics,                 # ROI opportunity identification
    calculate_cost_avoidance,              # Cost-avoidance projections
    calculate_efficiency_metrics,          # Efficiency KPIs
    calculate_financial_forecasts,         # 30/90-day and annual projections
    generate_financial_insights            # Narrative insight generation
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons for feedback / pricing systems.
# These are lazily initialised via their respective ``get_*`` helpers below so
# that the expensive I/O (loading Excel-based feedback & pricing data) only
# happens once per process lifetime.
# ---------------------------------------------------------------------------
feedback_learner = None
resolution_feedback_learner = None
price_catalog = None


# ============================================================================
# EXCEL API REFRESH UTILITY
# ============================================================================
# On Windows, the input workbook may contain live Power Query / API
# connections (e.g. pulling fresh data from a ticketing system REST API).
# The helpers below attempt to open the file in a background Excel COM
# instance, trigger a RefreshAll(), and save -- so that the pipeline always
# operates on the freshest data.
# ============================================================================

def is_excel_available() -> bool:
    """Check if Microsoft Excel is available on this system.

    Returns ``True`` only on Windows when either **xlwings** or **pywin32**
    (``win32com.client``) is importable, meaning we can automate Excel via COM.
    """
    import platform
    if platform.system() != 'Windows':
        return False
    try:
        import win32com.client
        return True
    except ImportError:
        try:
            import xlwings
            return True
        except ImportError:
            return False


def refresh_excel_connections(file_path: str, timeout_seconds: int = 120) -> tuple[bool, str]:
    """
    Refresh all data connections (APIs, Power Query) in an Excel file.

    This is a **Windows-only** operation that automates Excel in the background
    to pull fresh data from any configured OLE DB / ODBC / Power Query
    connections.  Two COM automation libraries are tried in order:

    1. **xlwings** (preferred -- more Pythonic API, better error handling).
    2. **pywin32** (``win32com.client``) as a fallback.

    The function is intentionally defensive: every COM call is wrapped in
    try/except because the connection state can be unpredictable (e.g. VPN
    down, API key expired, etc.).

    Args:
        file_path: Absolute path to the ``.xlsx`` / ``.xls`` workbook.
        timeout_seconds: Maximum wall-clock time (in seconds) to wait for
            all background queries to finish refreshing.

    Returns:
        Tuple of ``(success: bool, message: str)`` describing the outcome.
    """
    import platform
    import time

    if platform.system() != 'Windows':
        return False, "Excel refresh only available on Windows"

    print_status("refresh", "Checking for API connections in Excel...", "🔄")

    # ------------------------------------------------------------------
    # Strategy 1: xlwings  (higher-level, more reliable)
    # ------------------------------------------------------------------
    try:
        import xlwings as xw

        # Open Excel in the background -- no UI, no alert popups
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False
        app.screen_updating = False

        try:
            # Open the workbook
            wb = app.books.open(file_path)

            # Detect whether the workbook actually has data connections.
            # Connections can be OLE DB links, Power Query (M) queries, etc.
            has_connections = False
            connection_count = 0

            try:
                # .api accesses the underlying COM object (Excel.Workbook)
                connection_count = wb.api.Connections.Count
                if connection_count > 0:
                    has_connections = True
            except Exception as e:
                logger.debug(f"Handled: {e}")

            try:
                # Power Query objects are exposed separately via Queries
                query_count = wb.api.Queries.Count
                if query_count > 0:
                    has_connections = True
                    connection_count += query_count
            except Exception as e:
                logger.debug(f"Handled: {e}")

            if not has_connections:
                wb.close()
                app.quit()
                return False, "No API/Power Query connections found - using file as-is"

            print_status("refresh", f"Found {connection_count} data connection(s) - refreshing...", "📡")

            # Trigger a refresh of every connection in the workbook
            wb.api.RefreshAll()

            # Poll until all OLE DB connections report "not refreshing" or
            # until we exceed the caller-specified timeout.
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                try:
                    still_refreshing = False
                    for conn in wb.api.Connections:
                        try:
                            if conn.OLEDBConnection.Refreshing:
                                still_refreshing = True
                                break
                        except Exception as e:
                            logger.debug(f"Handled: {e}")
                except Exception as e:
                    logger.debug(f"Handled: {e}")

                if not still_refreshing:
                    break
                time.sleep(1)

            # Persist the refreshed data back to disk
            wb.save()
            wb.close()
            app.quit()

            return True, f"Successfully refreshed {connection_count} data connection(s)"

        except Exception as e:
            try:
                wb.close()
            except Exception as e2:
                logger.debug(f"Handled during cleanup: {e2}")
            app.quit()
            raise e

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"xlwings refresh failed: {e}, trying win32com...")

    # ------------------------------------------------------------------
    # Strategy 2: pywin32 (``win32com.client``)  -- lower-level fallback
    # ------------------------------------------------------------------
    try:
        import win32com.client
        import pythoncom

        # COM must be initialised on the calling thread
        pythoncom.CoInitialize()

        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        try:
            wb = excel.Workbooks.Open(file_path)

            # Check for connections
            connection_count = wb.Connections.Count

            if connection_count == 0:
                wb.Close(SaveChanges=False)
                excel.Quit()
                pythoncom.CoUninitialize()
                return False, "No API/Power Query connections found - using file as-is"

            print_status("refresh", f"Found {connection_count} data connection(s) - refreshing...", "📡")

            # Refresh all connections
            wb.RefreshAll()

            # Initial grace period for background queries to start
            time.sleep(5)

            # Wait for all asynchronous queries to complete
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                try:
                    excel.CalculateUntilAsyncQueriesDone()
                    break
                except Exception as e:
                    logger.debug(f"Async query not done yet: {e}")
                    time.sleep(2)

            # Save and close
            wb.Save()
            wb.Close(SaveChanges=True)
            excel.Quit()
            pythoncom.CoUninitialize()

            return True, f"Successfully refreshed {connection_count} data connection(s)"

        except Exception as e:
            try:
                wb.Close(SaveChanges=False)
            except Exception as e2:
                logger.debug(f"Handled during cleanup: {e2}")
            excel.Quit()
            pythoncom.CoUninitialize()
            return False, f"Error refreshing Excel: {str(e)}"

    except ImportError:
        return False, "Neither xlwings nor pywin32 installed (pip install xlwings pywin32)"
    except Exception as e:
        return False, f"Error: {str(e)}"


# ============================================================================
# CONSOLE OUTPUT HELPERS
# ============================================================================

def print_banner(text: str, char: str = "=", width: int = 60) -> None:
    """Log a formatted banner for visual phase delimiting.

    In interactive mode (TTY), prints the banner to stdout for visual
    feedback.  Always logs the banner text at INFO level.
    """
    if sys.stdout.isatty():
        print()
        print(char * width)
        print(f"  {text}")
        print(char * width)
    else:
        logger.info(text)


def print_status(phase: str, message: str, icon: str = "→") -> None:
    """Log a status message with phase context.

    In interactive mode (TTY), prints to stdout with the icon for visual
    feedback.  Always logs at INFO level with the phase as extra data.
    """
    if sys.stdout.isatty():
        print(f"  {icon} {message}")
        sys.stdout.flush()
    else:
        logger.info(message, extra={"phase": phase})


# ============================================================================
# SINGLETON ACCESSORS FOR FEEDBACK / PRICING SUBSYSTEMS
# ============================================================================
# These follow the "lazy singleton" pattern: the expensive object is only
# instantiated on first access and then reused for the rest of the process.
# ============================================================================

def get_feedback_learner() -> FeedbackLearning:
    """Get or create the global feedback learner instance.

    The ``FeedbackLearning`` object loads a human-curated Excel file of
    classification corrections.  When present, these corrections shift the
    embedding centroids used by the classifier, implementing a lightweight
    human-in-the-loop learning cycle.
    """
    global feedback_learner
    if feedback_learner is None:
        feedback_learner = FeedbackLearning()
    return feedback_learner


def get_resolution_feedback_learner() -> ResolutionFeedbackLearning:
    """Get or create the global resolution feedback learner instance.

    The ``ResolutionFeedbackLearning`` object stores human-provided estimates
    for how long tickets *should* take to resolve.  These ground-truth labels
    improve the resolution-time prediction model over successive runs.
    """
    global resolution_feedback_learner
    if resolution_feedback_learner is None:
        resolution_feedback_learner = ResolutionFeedbackLearning()
    return resolution_feedback_learner


def get_price_catalog() -> PriceCatalog:
    """Get or create the global price catalog instance.

    The ``PriceCatalog`` maps ticket categories and severities to dollar-cost
    estimates (labour hours, hourly rates, SLA penalty amounts, etc.) used by
    the financial metrics module.
    """
    global price_catalog
    if price_catalog is None:
        price_catalog = PriceCatalog()
    return price_catalog


# ============================================================================
# OLLAMA SERVER & MODEL HEALTH CHECKS
# ============================================================================

def check_ollama_server() -> bool:
    """Check if Ollama server is running.

    Sends a lightweight GET to the ``/api/tags`` endpoint.  If the server is
    unreachable a tkinter error dialog is shown to guide the operator.

    Returns:
        ``True`` if the server responds with HTTP 200, ``False`` otherwise.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=TIMEOUT_GPU_QUERY)
        if response.status_code == 200:
            print_status("init", "Ollama server is running", "✅")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"[Ollama] Server check failed: {e}")

    print_status("init", "Ollama server NOT running!", "❌")
    messagebox.showerror(
        "Ollama Not Running",
        "Ollama server is not running.\n\n"
        "Please start Ollama with: ollama serve\n"
        f"Expected at: {OLLAMA_BASE_URL}"
    )
    return False


def check_models(ai: OllamaBrain) -> bool:
    """Verify that required AI models are available.

    Performs a quick round-trip embedding of the word ``"test"`` to confirm
    the configured embedding model is loaded and returning vectors of the
    expected dimensionality.

    Args:
        ai: An ``OllamaBrain`` instance used to issue the test embedding.

    Returns:
        ``True`` if the embedding model responds with a non-empty vector.
    """
    try:
        # Validate the embedding model by requesting a single test vector
        print_status("init", f"Testing embedding model: {EMBED_MODEL}...", "🔄")
        test_embed = ai.get_embedding("test")
        if test_embed is None or len(test_embed) == 0:
            print_status("init", f"Embedding model '{EMBED_MODEL}' not available!", "❌")
            messagebox.showerror(
                "Model Error",
                f"Embedding model '{EMBED_MODEL}' is not available.\n\n"
                f"Please install with: ollama pull {EMBED_MODEL}"
            )
            return False

        print_status("init", f"Embedding model verified: {EMBED_MODEL} ({len(test_embed)} dims)", "✅")
        return True

    except Exception as e:
        logger.error(f"[Ollama] Model check failed: {e}")
        messagebox.showerror("Model Error", f"Model verification failed: {e}")
        return False


# ============================================================================
# DATA QUALITY GATE
# ============================================================================

def validate_data_quality(df: pd.DataFrame) -> bool:
    """
    Validate that the dataframe has enough usable data.

    This is a lightweight sanity check run immediately after file loading.
    It ensures we have at least one row and at least one column that could
    serve as input text for the classifier.

    Returns:
        ``True`` if the DataFrame passes all checks, ``False`` otherwise.
    """
    if df is None or len(df) == 0:
        return False

    # Check for at least one recognised text column (or any column at all)
    text_cols = [COL_SUMMARY, COL_CATEGORY]
    has_text = any(col in df.columns for col in text_cols) or len(df.columns) > 0

    if not has_text:
        return False

    # Minimum row threshold (currently just 1 -- any non-empty file)
    if len(df) < 1:
        return False

    return True


# ============================================================================
# PHASE 3: RECIDIVISM / LEARNING ANALYSIS
# ============================================================================

def audit_learning(df: pd.DataFrame, ai: OllamaBrain, show_progress: bool = True) -> pd.DataFrame:
    """
    Enhanced recidivism analysis using embedding-based similarity.

    This function implements **Phase 3** of the pipeline.  Its goal is to
    identify tickets that are semantically similar to *other* tickets in the
    same dataset, which signals that the organisation may be encountering the
    same problem repeatedly (a "repeat offense").

    Algorithm
    ---------
    1. For every ticket, request an embedding vector from the Ollama embedding
       model (via ``ai.get_embedding``).
    2. Build a FAISS index of all valid embeddings for O(n log n) nearest
       neighbor search (falls back to O(n^2) pairwise scan if FAISS is
       unavailable).
    3. For each ticket, find its nearest neighbor and label it with a
       recidivism status:

       - >= 0.85  :  ``REPEAT OFFENSE``  -- near-duplicate, very likely the
         same root cause.
       - >= 0.75  :  ``POSSIBLE REPEAT`` -- similar enough to warrant
         investigation.
       - >= 0.65  :  ``Monitored``       -- on the radar but not alarming.
       - <  0.65  :  ``New Issue``        -- no strong similarity detected.

    Columns added to ``df``
    -----------------------
    - ``Learning_Status``          -- human-readable recidivism label.
    - ``Recidivism_Score``         -- raw max cosine similarity [0, 1].
    - ``Similar_Historical_Issue`` -- first 100 chars of the most-similar
      ticket's summary (for quick human review).
    - ``embedding``                -- the raw embedding vector (used
      downstream by Phase 5 similar-ticket search).

    Args:
        df: DataFrame with a ``Combined_Text`` column (produced by
            ``prepare_text``).
        ai: ``OllamaBrain`` instance for generating embeddings.
        show_progress: Whether to render tqdm progress bars.

    Returns:
        The same DataFrame with the four new columns described above.
    """
    from ..predictors.vector_store import FAISS_AVAILABLE, TicketVectorStore

    print_status("Phase 3", "Calculating embeddings for similarity analysis...", "🧠")

    # Initialise output columns with safe defaults
    df['Learning_Status'] = 'New'
    df['Recidivism_Score'] = 0.0
    df['Similar_Historical_Issue'] = ''

    # ------------------------------------------------------------------
    # Step 1: Embed every ticket individually.
    # NOTE: This uses single-ticket embedding calls (not batched) because
    # the progress bar provides per-ticket feedback.  For very large
    # datasets the batched path in ``get_embeddings_batch`` would be faster
    # but would lose the granular progress reporting.
    # ------------------------------------------------------------------
    texts = df['Combined_Text'].tolist()
    embeddings = []

    iterator = tqdm(texts, desc="  Embedding tickets", unit="ticket",
                   disable=not show_progress, ncols=80,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for text in iterator:
        if pd.isna(text) or str(text).strip() == '':
            # Missing / empty text -> no embedding available
            embeddings.append(None)
        else:
            emb = ai.get_embedding(str(text))
            embeddings.append(emb)

    # Store raw embeddings on the DataFrame for downstream reuse (Phase 5)
    df['embedding'] = embeddings

    # ------------------------------------------------------------------
    # Step 2: Find nearest neighbor for each ticket.
    # Uses FAISS index for O(n log n) search when available, otherwise
    # falls back to brute-force O(n^2) pairwise cosine similarity.
    # ------------------------------------------------------------------
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]

    if FAISS_AVAILABLE and len(valid_indices) >= 2:
        print_status("Phase 3", "Building FAISS index for similarity search...", "🔍")

        valid_matrix = np.array(
            [embeddings[i] for i in valid_indices], dtype=np.float32
        )
        dimension = valid_matrix.shape[1]
        valid_ids = [str(i) for i in valid_indices]

        store = TicketVectorStore(dimension=dimension)
        store.add_embeddings(valid_matrix, valid_ids)

        # Cache the index for reuse
        cache_path = ".cache/ticket_vectors.faiss"
        try:
            store.save(cache_path)
        except Exception as e:
            logger.warning(f"Could not cache FAISS index: {e}")

        print_status("Phase 3", f"Searching {len(valid_indices)} tickets via FAISS...", "🔍")

        # Build a reverse map: str(index) -> position in valid_indices
        id_to_pos = {str(i): pos for pos, i in enumerate(valid_indices)}

        iterator = tqdm(range(len(valid_indices)), desc="  Finding similar tickets",
                       unit="ticket", disable=not show_progress, ncols=80,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for pos in iterator:
            i = valid_indices[pos]
            neighbor_id, max_similarity = store.find_nearest_neighbor(pos)

            if neighbor_id is not None:
                most_similar_idx = int(neighbor_id)
            else:
                most_similar_idx = -1

            # Write the max similarity score for this ticket
            df.at[df.index[i], 'Recidivism_Score'] = max_similarity

            # Label based on similarity thresholds
            if max_similarity >= 0.85:
                df.at[df.index[i], 'Learning_Status'] = '🔴 REPEAT OFFENSE'
                if most_similar_idx >= 0:
                    similar_text = str(df.iloc[most_similar_idx].get(COL_SUMMARY, ''))[:100]
                    df.at[df.index[i], 'Similar_Historical_Issue'] = similar_text
            elif max_similarity >= 0.75:
                df.at[df.index[i], 'Learning_Status'] = '🟡 POSSIBLE REPEAT'
            elif max_similarity >= 0.65:
                df.at[df.index[i], 'Learning_Status'] = '🟢 Monitored'
            else:
                df.at[df.index[i], 'Learning_Status'] = '🆕 New Issue'

    else:
        # Fallback: O(n^2) brute-force pairwise cosine similarity
        if not FAISS_AVAILABLE:
            import warnings
            warnings.warn(
                "FAISS not installed — using O(n^2) pairwise similarity. "
                "Install faiss-cpu for better performance.",
                stacklevel=2,
            )

        print_status("Phase 3", "Computing similarity matrix (brute-force)...", "🔍")
        valid_embeddings = [(i, embeddings[i]) for i in valid_indices]

        iterator = tqdm(valid_embeddings, desc="  Finding similar tickets", unit="ticket",
                       disable=not show_progress, ncols=80,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for i, emb_i in iterator:
            max_similarity = 0.0
            most_similar_idx = -1

            for j, emb_j in valid_embeddings:
                if i == j:
                    continue

                dot = np.dot(emb_i, emb_j)
                norm_i = np.linalg.norm(emb_i)
                norm_j = np.linalg.norm(emb_j)

                if norm_i > 0 and norm_j > 0:
                    similarity = dot / (norm_i * norm_j)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_idx = j

            df.at[df.index[i], 'Recidivism_Score'] = max_similarity

            if max_similarity >= 0.85:
                df.at[df.index[i], 'Learning_Status'] = '🔴 REPEAT OFFENSE'
                if most_similar_idx >= 0:
                    similar_text = str(df.iloc[most_similar_idx].get(COL_SUMMARY, ''))[:100]
                    df.at[df.index[i], 'Similar_Historical_Issue'] = similar_text
            elif max_similarity >= 0.75:
                df.at[df.index[i], 'Learning_Status'] = '🟡 POSSIBLE REPEAT'
            elif max_similarity >= 0.65:
                df.at[df.index[i], 'Learning_Status'] = '🟢 Monitored'
            else:
                df.at[df.index[i], 'Learning_Status'] = '🆕 New Issue'

    # Log a concise summary of recidivism findings
    repeat_count = (df['Learning_Status'].str.contains('REPEAT', na=False)).sum()
    possible_count = (df['Learning_Status'].str.contains('POSSIBLE', na=False)).sum()

    print_status("Phase 3", f"Found {repeat_count} repeat offenses, {possible_count} possible repeats", "📊")

    return df


# ============================================================================
# PIPELINE ORCHESTRATOR CLASS
# ============================================================================

class EscalationPipeline:
    """
    Main pipeline orchestrator for Escalation AI.

    This class is the "conductor" that coordinates all seven analysis phases
    in the correct order.  It owns:

    - ``self.ai``  -- the ``OllamaBrain`` instance used for embeddings and
      LLM generation across all phases.
    - ``self.df``  -- the *mutable* working DataFrame that each phase
      enriches with new columns.
    - ``self.df_raw`` -- an untouched copy of the original loaded data,
      preserved so the report generator can include the raw sheet.
    - ``self.feedback_learner`` / ``self.resolution_feedback_learner`` --
      human-in-the-loop correction systems.
    - ``self.price_catalog`` -- category-to-cost lookup for financial
      calculations.

    Typical usage
    -------------
    ::

        pipe = EscalationPipeline()
        pipe.initialize()            # Connect to Ollama, load feedback data
        pipe.load_data("input.xlsx") # Read & validate Excel
        pipe.run_all_phases()        # Phase 1-6
        summary = pipe.generate_executive_summary()  # Phase 7
    """

    def __init__(self) -> None:
        """Initialise all instance attributes to ``None`` / safe defaults.

        No expensive work happens here -- actual initialisation is deferred
        to :meth:`initialize` so that errors can be reported gracefully via
        tkinter dialogs.
        """
        self.ai = None                          # OllamaBrain instance
        self.df = None                          # Working DataFrame (enriched by phases)
        self.df_raw = None                      # Immutable copy of the original data
        self.file_path = None                   # Path to the input file
        self.output_path = None                 # Path to the output report
        self.feedback_learner = None            # Classification correction feedback
        self.resolution_feedback_learner = None # Resolution time correction feedback
        self.price_catalog = None               # Category -> cost lookup
        self.show_progress = True               # Controls tqdm progress bars

    def initialize(self) -> bool:
        """Initialize the pipeline components.

        Performs four sequential checks / setup steps:

        1. Verify that the Ollama server is reachable.
        2. Instantiate the ``OllamaBrain`` and confirm the embedding model
           returns valid vectors.
        3. Load the classification feedback file (human corrections from
           previous runs).
        4. Load the resolution feedback file and the price catalog.

        Returns:
            ``True`` if all steps succeed, ``False`` if any step fails (in
            which case a tkinter error dialog will already have been shown).
        """
        print_banner("INITIALIZING ESCALATION AI", "=")

        # Step 1: Ensure the Ollama inference server is up
        if not check_ollama_server():
            return False

        # Step 2: Create the AI engine and log which models are configured
        self.ai = OllamaBrain()
        print_status("init", f"Embedding model: {self.ai.embed_model}", "🧠")
        print_status("init", f"Generation model: {self.ai.gen_model}", "🤖")

        # Step 3: Validate that the embedding model actually works
        if not check_models(self.ai):
            return False

        # Step 4: Load the feedback / pricing subsystems.
        # ``load_feedback(self.ai)`` needs the AI instance because it may
        # re-embed user-corrected examples to shift the classification
        # centroids (human-in-the-loop learning).
        print_status("init", "Loading feedback learning system...", "📚")
        self.feedback_learner = get_feedback_learner()
        self.feedback_learner.load_feedback(self.ai)

        # Resolution feedback uses simpler statistical features (no embeddings)
        self.resolution_feedback_learner = get_resolution_feedback_learner()
        self.resolution_feedback_learner.load_feedback()

        # Price catalog maps (category, severity) -> dollar cost
        self.price_catalog = get_price_catalog()
        self.price_catalog.load_catalog()
        print_status("init", "Initialization complete!", "✅")

        return True

    def load_data(self, file_path: Optional[str] = None) -> bool:
        """Load data from file, refreshing API connections if available.

        If ``file_path`` is ``None`` a tkinter file-picker dialog is shown.
        On Windows, before reading the workbook, the function attempts to
        refresh any live data connections (Power Query / API) via
        :func:`refresh_excel_connections`.

        The loader tries Excel first (``pd.read_excel``), preferring a sheet
        whose name contains ``"raw"`` (case-insensitive).  If the Excel read
        fails it falls back to ``pd.read_csv``.

        Args:
            file_path: Optional explicit path; if omitted a GUI picker opens.

        Returns:
            ``True`` on success, ``False`` if the user cancelled the picker
            or the data failed validation.
        """
        print_banner("LOADING DATA", "-")

        # Show a file-picker dialog if no path was provided
        if file_path is None:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select Input Excel File",
                filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
            )
            if not file_path:
                return False

        self.file_path = file_path
        print_status("load", f"File: {os.path.basename(file_path)}", "📁")

        # ---- Windows-only: refresh Power Query / API connections ----
        if file_path.lower().endswith(('.xlsx', '.xls')):
            if is_excel_available():
                print_status("load", "Excel detected - checking for API connections...", "🔍")
                success, msg = refresh_excel_connections(file_path)
                if success:
                    print_status("load", msg, "✅")
                else:
                    print_status("load", msg, "ℹ️")
            else:
                print_status("load", "Excel not available - loading file as-is", "ℹ️")

        # ---- Read the file into a pandas DataFrame ----
        try:
            xls = pd.ExcelFile(file_path)
            # Prefer a sheet named "raw" (e.g. "Raw Data") if one exists
            sheet = next((s for s in xls.sheet_names if 'raw' in str(s).lower()), xls.sheet_names[0])
            print_status("load", f"Sheet: {sheet}", "📄")
            self.df = pd.read_excel(file_path, sheet_name=sheet)
            self.df_raw = self.df.copy()  # Preserve the untouched original
        except Exception as e:
            logger.warning(f"Excel read failed, trying CSV: {e}")
            self.df = pd.read_csv(file_path, engine='python')
            self.df_raw = self.df.copy()

        # ---- Data quality gate ----
        if not validate_data_quality(self.df):
            messagebox.showerror("Data Error", "The selected file contains no usable data.")
            return False

        print_status("load", f"Loaded {len(self.df):,} tickets with {len(self.df.columns)} columns", "✅")
        return True

    def prepare_text(self) -> None:
        """Prepare the ``Combined_Text`` column used by all downstream phases.

        Concatenates the summary and category columns (if present) with a
        dash separator, then applies :func:`clean_text` to normalise
        whitespace, remove special characters, etc.

        If neither recognised text column is found the first column of the
        DataFrame is used as a fallback.
        """
        print_status("prep", "Preparing text for analysis...", "📝")

        # Build a list of recognised text columns that actually exist in df
        text_cols = [COL_SUMMARY, COL_CATEGORY]
        actual_cols = [c for c in self.df.columns if c.strip().lower() in [t.lower() for t in text_cols]]

        if actual_cols:
            # Join non-null values from all matching columns with " - "
            self.df['Combined_Text'] = self.df[actual_cols].apply(
                lambda x: ' - '.join(x.dropna().astype(str)), axis=1
            )
        else:
            # Fallback: use whatever the first column is
            self.df['Combined_Text'] = self.df.iloc[:, 0].astype(str)

        # Normalise: lowercase, strip whitespace, remove noise characters
        self.df['Combined_Text'] = self.df['Combined_Text'].apply(clean_text)
        print_status("prep", "Text preparation complete", "✅")

    # ======================================================================
    # INDIVIDUAL PHASE RUNNERS
    # Each method wraps the corresponding subsystem call with banner/status
    # output so the operator can see progress in real time.
    # ======================================================================

    def run_classification(self) -> None:
        """Phase 1: AI Classification.

        Delegates to :func:`classify_rows` which runs the three-tier
        hybrid classifier (regex -> keyword -> embedding) and adds
        ``AI_Category``, ``AI_Confidence``, and ``AI_Sub_Category`` columns.
        """
        print_banner("PHASE 1: AI CLASSIFICATION", "─")
        print_status("Phase 1", f"Classifying {len(self.df):,} tickets using {EMBED_MODEL}...", "🏷️")
        self.df = classify_rows(self.df, self.ai, show_progress=self.show_progress)

        # Report how many distinct categories were assigned
        if 'AI_Category' in self.df.columns:
            n_categories = self.df['AI_Category'].nunique()
            print_status("Phase 1", f"Classified into {n_categories} categories", "✅")

    def run_scoring(self) -> None:
        """Phase 2: Strategic Friction Scoring.

        Delegates to :func:`calculate_strategic_friction` which computes a
        weighted composite score per ticket based on severity, origin,
        recurrence, age, and other factors.  Adds the
        ``Strategic_Friction_Score`` column.
        """
        print_banner("PHASE 2: STRATEGIC FRICTION SCORING", "─")
        print_status("Phase 2", "Calculating friction scores...", "📊")
        self.df = calculate_strategic_friction(self.df)

        if 'Strategic_Friction_Score' in self.df.columns:
            total_friction = self.df['Strategic_Friction_Score'].sum()
            avg_friction = self.df['Strategic_Friction_Score'].mean()
            print_status("Phase 2", f"Total friction: {total_friction:,.0f} | Avg: {avg_friction:.1f}", "✅")

    def run_recidivism_analysis(self) -> None:
        """Phase 3: Recidivism & Learning Analysis.

        Delegates to :func:`audit_learning` (defined above) which embeds
        every ticket and uses FAISS nearest-neighbor search (or O(n^2)
        pairwise fallback) to detect repeat offenses.  Adds
        ``Learning_Status``, ``Recidivism_Score``,
        ``Similar_Historical_Issue``, and ``embedding`` columns.
        """
        print_banner("PHASE 3: RECIDIVISM ANALYSIS", "─")
        self.df = audit_learning(self.df, self.ai, show_progress=self.show_progress)

    def run_recurrence_prediction(self) -> None:
        """Phase 4: ML-based Recurrence Prediction.

        Delegates to :func:`apply_recurrence_predictions` which engineers
        features from the enriched DataFrame and trains a **Random Forest
        Classifier** (GPU-accelerated via cuML when available) to predict the
        probability that each ticket will recur.

        Adds ``AI_Recurrence_Probability`` and ``AI_Recurrence_Risk_Tier``
        columns.
        """
        print_banner("PHASE 4: RECURRENCE PREDICTION", "─")
        print_status("Phase 4", "Training recurrence model...", "🔮")
        self.df = apply_recurrence_predictions(self.df)

        if 'AI_Recurrence_Probability' in self.df.columns:
            high_risk = (self.df['AI_Recurrence_Probability'] > 0.7).sum()
            print_status("Phase 4", f"Identified {high_risk} high-risk tickets (>70% recurrence)", "✅")

    def run_similar_ticket_analysis(self) -> None:
        """Phase 5: Similar Ticket Analysis.

        Delegates to :func:`apply_similar_ticket_analysis` which uses the
        embeddings computed in Phase 3 and a **k-Nearest-Neighbours** search
        (GPU-accelerated via cuML NearestNeighbors when available) to find
        the most similar *resolved* tickets.  This provides resolution
        strategy suggestions for open tickets.
        """
        print_banner("PHASE 5: SIMILAR TICKET ANALYSIS", "─")
        print_status("Phase 5", "Finding similar resolved tickets...", "🔍")
        self.df = apply_similar_ticket_analysis(self.df, self.ai)
        print_status("Phase 5", "Similar ticket analysis complete", "✅")

    def run_resolution_time_prediction(self) -> None:
        """Phase 6: Resolution Time Prediction.

        Delegates to :func:`apply_resolution_time_prediction` which trains a
        **Random Forest Regressor** (GPU-accelerated via cuML when available)
        on tickets that have known resolution times and then predicts the
        expected resolution hours for all tickets.

        Adds ``Predicted_Resolution_Days`` column.
        """
        print_banner("PHASE 6: RESOLUTION TIME PREDICTION", "─")
        print_status("Phase 6", "Training resolution time model...", "⏱️")
        self.df = apply_resolution_time_prediction(self.df)

        if 'Predicted_Resolution_Days' in self.df.columns:
            avg_pred = self.df['Predicted_Resolution_Days'].mean()
            print_status("Phase 6", f"Average predicted resolution: {avg_pred:.1f} days", "✅")

    def generate_executive_summary(self) -> str:
        """Phase 7: Generate an AI-written executive summary.

        This is the final analytical phase.  It:

        1. Computes comprehensive **financial metrics** (total cost, revenue
           at risk, cost avoidance potential, ROI opportunities, forecasts).
        2. Assembles a structured text context containing all key statistics.
        3. Feeds that context into the LLM (via ``OllamaBrain.generate_synthesis``)
           which produces a McKinsey-style executive summary structured around
           the Pyramid Principle, MECE, and Pareto analysis.

        The computed financial metric objects are also stored as instance
        attributes (``self.financial_metrics``, ``self.roi_metrics``, etc.)
        so the Streamlit dashboard can access them without recomputation.

        Returns:
            A string containing the LLM-generated executive summary (or a
            fallback auto-generated summary if the LLM is unavailable).
        """
        print_banner("PHASE 7: EXECUTIVE SUMMARY", "─")
        print_status("Phase 7", f"Generating insights with {GEN_MODEL}...", "✍️")

        # Extract pre-requisite values from earlier phases
        total_friction = self.df['Strategic_Friction_Score'].sum() if 'Strategic_Friction_Score' in self.df.columns else 0
        total_tickets = len(self.df)

        # ------------------------------------------------------------------
        # Compute the full suite of financial metrics.
        # Each function operates on the enriched DataFrame and returns either
        # a dataclass or a dict.
        # ------------------------------------------------------------------
        financial_metrics = calculate_financial_metrics(self.df)
        roi_metrics = calculate_roi_metrics(self.df)
        cost_avoidance = calculate_cost_avoidance(self.df)
        efficiency_metrics = calculate_efficiency_metrics(self.df)
        forecasts = calculate_financial_forecasts(self.df)
        insights = generate_financial_insights(self.df)

        # Persist metrics on the instance for dashboard access
        self.financial_metrics = financial_metrics
        self.roi_metrics = roi_metrics
        self.cost_avoidance = cost_avoidance
        self.efficiency_metrics = efficiency_metrics
        self.financial_forecasts = forecasts
        self.financial_insights = insights

        # ------------------------------------------------------------------
        # Legacy scalar metrics -- kept for backward compatibility with
        # older report templates and chart generators.
        # ------------------------------------------------------------------
        total_financial_impact = financial_metrics.total_cost
        avg_cost_per_ticket = financial_metrics.avg_cost_per_ticket
        max_single_ticket_cost = self.df['Financial_Impact'].max() if 'Financial_Impact' in self.df.columns else 0
        revenue_at_risk = financial_metrics.revenue_at_risk
        # Rule-of-thumb split: 65% of cost is labour, 35% is opportunity cost
        labor_cost = total_financial_impact * 0.65
        opportunity_cost = financial_metrics.opportunity_cost
        high_cost_tickets = financial_metrics.high_cost_tickets_count
        high_cost_total = self.df['Financial_Impact'].quantile(0.9) * high_cost_tickets if 'Financial_Impact' in self.df.columns else 0
        recurrence_exposure = financial_metrics.recurrence_exposure

        # ------------------------------------------------------------------
        # Build the structured text context that will be injected into the
        # LLM prompt.  The prompt (in ai_engine.py) instructs the model to
        # use ONLY numbers present in this context -- no fabrication.
        # ------------------------------------------------------------------
        context_lines = [
            "=" * 60,
            "ESCALATION REPORT STATISTICAL SUMMARY",
            "=" * 60,
            "",
            f"Total Tickets Analyzed: {total_tickets}",
            f"Total Weighted Friction Score: {total_friction:,.0f}",
        ]

        # Severity distribution breakdown
        if 'Severity_Norm' in self.df.columns:
            severity_counts = self.df['Severity_Norm'].value_counts()
            context_lines.append(f"\nSeverity Distribution:")
            for sev, count in severity_counts.items():
                context_lines.append(f"  - {sev}: {count} tickets")

        # Top AI-assigned categories by volume
        if 'AI_Category' in self.df.columns:
            cat_counts = self.df['AI_Category'].value_counts().head(5)
            context_lines.append(f"\nTop Categories by Ticket Count:")
            for cat, count in cat_counts.items():
                pct = (count / total_tickets * 100)
                context_lines.append(f"  - {cat}: {count} tickets ({pct:.1f}%)")

        # ------------------------------------------------------------------
        # FINANCIAL IMPACT SECTION -- the richest part of the context, giving
        # the LLM every dollar figure it needs for its analysis.
        # ------------------------------------------------------------------
        context_lines.extend([
            "",
            "=" * 60,
            "FINANCIAL IMPACT METRICS",
            "=" * 60,
            "",
            f"Total Direct Financial Impact: ${total_financial_impact:,.2f}",
            f"Average Cost per Escalation: ${avg_cost_per_ticket:,.2f}",
            f"Median Cost per Escalation: ${financial_metrics.median_cost:,.2f}",
            f"Highest Single Ticket Cost: ${max_single_ticket_cost:,.2f}",
            f"Revenue at Risk (downstream business impact): ${revenue_at_risk:,.2f}",
            f"Labor Cost Component (65%): ${labor_cost:,.2f}",
            f"Opportunity Cost Component (35%): ${opportunity_cost:,.2f}",
            f"Customer Impact Cost (external issues): ${financial_metrics.customer_impact_cost:,.2f}",
            f"SLA Penalty Exposure: ${financial_metrics.sla_penalty_exposure:,.2f}",
            "",
            f"High-Cost Tickets (top 10%): {high_cost_tickets} tickets = ${high_cost_total:,.2f}",
            f"Cost Concentration: {financial_metrics.cost_concentration_ratio*100:.0f}% of costs from top 20% tickets",
            f"Recurrence Risk Exposure: ${recurrence_exposure:,.2f}",
            "",
            "ROI & COST OPTIMIZATION",
            f"Preventable Cost (process improvements): ${financial_metrics.preventable_cost:,.2f}",
            f"Recurring Issue Cost (root cause fixes): ${financial_metrics.recurring_issue_cost:,.2f}",
            f"Total Cost Avoidance Potential: ${cost_avoidance['total_avoidance']:,.2f}",
            f"ROI Opportunity (from prevention): ${financial_metrics.roi_opportunity:,.2f}",
            f"Cost Efficiency Score: {financial_metrics.cost_efficiency_score:.0f}/100",
        ])

        # Top ROI opportunities (investment -> savings with payback period)
        if roi_metrics['top_opportunities']:
            context_lines.append("\nTop ROI Opportunities:")
            for opp in roi_metrics['top_opportunities'][:3]:
                context_lines.append(
                    f"  - {opp['category']}: Invest ${opp['investment_required']:,.0f} → "
                    f"Save ${opp['annual_savings']:,.0f}/year (ROI: {opp['roi_percentage']:.0f}%, "
                    f"Payback: {opp['payback_months']:.1f} months)"
                )

        # Financial trend forecasts (only if a non-stable trend is detected)
        if forecasts['trend'] != 'stable':
            context_lines.extend([
                "",
                "FINANCIAL FORECAST",
                f"Trend: {forecasts['trend'].upper()} (confidence: {forecasts['confidence']})",
                f"30-Day Projection: ${financial_metrics.cost_forecast_30d:,.2f}",
                f"90-Day Projection: ${financial_metrics.cost_forecast_90d:,.2f}",
                f"Annual Projection: ${forecasts['annual_projection']:,.2f}",
            ])

        # Top auto-generated insights (priority-ranked)
        if insights:
            context_lines.append("\nKEY FINANCIAL INSIGHTS:")
            for insight in insights[:3]:
                context_lines.append(
                    f"  [{insight['priority'].upper()}] {insight['title']}: "
                    f"{insight['description']} | {insight['recommendation']}"
                )

        # Financial impact broken down by AI_Category
        if 'Financial_Impact' in self.df.columns and 'AI_Category' in self.df.columns:
            fin_by_cat = self.df.groupby('AI_Category')['Financial_Impact'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False).head(5)
            context_lines.append("\nFinancial Impact by Category:")
            for cat in fin_by_cat.index:
                total = fin_by_cat.loc[cat, 'sum']
                avg = fin_by_cat.loc[cat, 'mean']
                count = int(fin_by_cat.loc[cat, 'count'])
                pct = (total / total_financial_impact * 100) if total_financial_impact > 0 else 0
                context_lines.append(f"  - {cat}: ${total:,.2f} total ({pct:.1f}%), ${avg:,.2f} avg, {count} tickets")

        # Financial impact broken down by normalised severity level
        if 'Financial_Impact' in self.df.columns and 'Severity_Norm' in self.df.columns:
            fin_by_sev = self.df.groupby('Severity_Norm')['Financial_Impact'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
            context_lines.append("\nFinancial Impact by Severity:")
            for sev in fin_by_sev.index:
                total = fin_by_sev.loc[sev, 'sum']
                avg = fin_by_sev.loc[sev, 'mean']
                pct = (total / total_financial_impact * 100) if total_financial_impact > 0 else 0
                context_lines.append(f"  - {sev}: ${total:,.2f} ({pct:.1f}%), ${avg:,.2f} avg per ticket")

        # Categories with the highest average cost per ticket (cost hotspots)
        if 'Financial_Impact' in self.df.columns and 'AI_Category' in self.df.columns:
            cost_efficiency = self.df.groupby('AI_Category')['Financial_Impact'].mean().sort_values(ascending=False).head(3)
            context_lines.append("\nHighest Avg Cost per Ticket (categories to watch):")
            for cat, avg in cost_efficiency.items():
                context_lines.append(f"  - {cat}: ${avg:,.2f} per ticket")

        # Combine all lines into a single string for the LLM prompt
        context = "\n".join(context_lines)

        # Send the context to the LLM for executive summary generation
        summary = self.ai.generate_synthesis(context)
        print_status("Phase 7", "Executive summary generated", "✅")
        return summary

    # ======================================================================
    # FULL PIPELINE EXECUTION
    # ======================================================================

    def run_all_phases(self) -> pd.DataFrame:
        """Run all pipeline phases in sequence.

        This is the main entry point after :meth:`initialize` and
        :meth:`load_data` have succeeded.  It calls phases 1-6 in order,
        then saves the current classifications to the feedback file for
        human review (enabling the human-in-the-loop correction cycle).

        Returns:
            The enriched ``pandas.DataFrame`` with all computed columns.
        """
        total_phases = 6

        print_banner(f"RUNNING ANALYSIS PIPELINE ({len(self.df):,} tickets)", "═")
        logger.info(
            "Starting analysis pipeline",
            extra={"ticket_count": len(self.df)},
        )

        self.prepare_text()

        with PhaseTimer("AI Classification", logger, phase=1):
            self.run_classification()

        with PhaseTimer("Strategic Friction Scoring", logger, phase=2):
            self.run_scoring()

        with PhaseTimer("Recidivism Analysis", logger, phase=3):
            self.run_recidivism_analysis()

        with PhaseTimer("Recurrence Prediction", logger, phase=4):
            self.run_recurrence_prediction()

        with PhaseTimer("Similar Ticket Analysis", logger, phase=5):
            self.run_similar_ticket_analysis()

        with PhaseTimer("Resolution Time Prediction", logger, phase=6):
            self.run_resolution_time_prediction()

        # Persist the AI-assigned labels so a human can review and correct
        # them before the next pipeline run.  Corrections are loaded by
        # ``FeedbackLearning`` on the next ``initialize()`` call.
        self._save_feedback_for_review()

        print_banner("ALL PHASES COMPLETE", "═")
        logger.info("All %d phases complete", total_phases)

        return self.df

    def _save_feedback_for_review(self) -> None:
        """Save current classifications and resolution predictions to feedback files for human review.

        Two separate feedback files are written:

        1. **Classification feedback** (``classification_feedback.xlsx``) --
           contains each ticket's ``AI_Category`` so a human can correct
           misclassifications.  Corrections shift the anchor centroids on the
           next run.

        2. **Resolution feedback** (``resolution_feedback.xlsx``) -- contains
           each ticket's predicted resolution time so a human can provide the
           actual time.  Actuals improve the resolution-time model on the
           next run.

        Both files are written to the same directory as the input file.
        """
        try:
            # Determine output directory (use file's directory or current directory)
            if self.file_path:
                output_dir = os.path.dirname(self.file_path)
            else:
                output_dir = os.getcwd()

            # Save classification feedback
            print_status("feedback", "Updating classification feedback file for human review...", "📝")
            feedback_path = self.feedback_learner.save_for_review(self.df, output_dir)
            print_status("feedback", f"Feedback file updated: {os.path.basename(feedback_path)}", "✅")

            # Save resolution time feedback
            print_status("feedback", "Updating resolution feedback file for human review...", "📝")
            resolution_path = self.resolution_feedback_learner.save_for_review(self.df, output_dir)
            print_status("feedback", f"Resolution feedback updated: {os.path.basename(resolution_path)}", "✅")
        except Exception as e:
            logger.warning(f"Could not save feedback file: {e}")
            print_status("feedback", f"Warning: Could not save feedback file: {e}", "⚠️")

    def get_results(self) -> pd.DataFrame:
        """Get the processed dataframe.

        Returns:
            The enriched ``pandas.DataFrame`` containing all original columns
            plus every column added by phases 1-6.
        """
        return self.df


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

def main_pipeline() -> None:
    """
    Main entry point for the Escalation AI pipeline.

    Designed to be called directly from ``__main__`` or from a GUI button.
    The function orchestrates the full end-to-end flow:

    1. Shows a tkinter file-selection dialog for the input workbook.
    2. Initialises the AI engine and feedback systems.
    3. Runs all six analysis phases.
    4. Generates the Phase 7 executive summary via the LLM.
    5. Shows a "save-as" dialog for the output report path.
    6. Calls the report generator to produce a multi-sheet Excel workbook
       with embedded charts.

    All exceptions are caught at the top level, logged, and surfaced to the
    user via tkinter message boxes.
    """
    # Create (and immediately hide) a root tkinter window -- required for
    # file dialogs and message boxes to work without a full GUI app.
    root = tk.Tk()
    root.withdraw()

    try:
        pipeline = EscalationPipeline()

        # ---- Initialise: Ollama server, models, feedback, pricing ----
        if not pipeline.initialize():
            return

        # ---- Load and validate input data ----
        if not pipeline.load_data():
            return

        # ---- Run the six analysis phases ----
        df = pipeline.run_all_phases()

        # ---- Phase 7: Executive summary via LLM ----
        exec_summary = pipeline.generate_executive_summary()

        # ---- Prompt user for output file location ----
        print_banner("SAVING REPORT", "-")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile="Strategic_Report.xlsx"
        )
        if not save_path:
            return

        # ---- Generate the final Excel report with charts ----
        print_status("save", "Generating Excel report with charts...", "📊")
        from ..reports import generate_report
        generate_report(df, save_path, exec_summary, pipeline.df_raw)

        print_status("save", f"Saved to: {save_path}", "✅")
        print_banner("ANALYSIS COMPLETE! 🎉", "═")

        messagebox.showinfo(
            "Analysis Complete",
            f"Report generated successfully!\n\nSaved to: {save_path}"
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        messagebox.showerror("Error", f"Analysis failed: {e}")
    finally:
        # Clean up the hidden tkinter root window
        try:
            root.destroy()
        except Exception as e:
            logger.debug(f"Handled: {e}")


if __name__ == "__main__":
    main_pipeline()
