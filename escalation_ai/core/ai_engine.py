"""
AI Engine module - Handles Ollama-based embeddings and text generation.

This module provides the ``OllamaBrain`` class, which is the single interface
through which the entire Escalation AI pipeline communicates with the **Ollama**
local LLM server.  It exposes two core capabilities:

1. **Embedding generation** (``get_embedding`` / ``get_embeddings_batch``) --
   Converts free-text ticket descriptions into dense numeric vectors (typically
   768 or 1024 dimensions, depending on the configured model).  These vectors
   are used by:
     - The hybrid classifier (Phase 1) for cosine-similarity-based category
       assignment.
     - The recidivism detector (Phase 3) for pairwise similarity scoring.
     - The similar-ticket finder (Phase 5) for k-NN search.

2. **Text generation / synthesis** (``generate_synthesis``) --
   Sends a structured statistical context to the generative LLM and receives
   back a McKinsey-style executive summary.  Used in Phase 7.

Architecture notes
------------------
- Communication with Ollama uses its REST API (``/api/embed`` for embeddings,
  ``/api/generate`` for text generation).  No Python SDK is required.
- The embedding model and generation model are independently configurable via
  ``core.config`` (``EMBED_MODEL`` and ``GEN_MODEL``).  This allows using a
  lightweight embedding model (e.g. ``nomic-embed-text``) alongside a more
  capable generation model (e.g. ``deepseek-r1``).
- Batch embedding automatically selects the optimal batch size based on GPU
  VRAM (via ``gpu_utils.get_optimal_embedding_batch_size``).

Data flow within the pipeline
-----------------------------
::

    [Pipeline Phase]                 [OllamaBrain method]
    ─────────────────               ─────────────────────
    Phase 1 (classify)     ──>  get_embeddings_batch()  ──>  cosine sim vs anchors
    Phase 3 (recidivism)   ──>  get_embedding()          ──>  pairwise similarity
    Phase 5 (similar)      ──>  (reuses Phase 3 embeddings)
    Phase 7 (summary)      ──>  generate_synthesis()     ──>  LLM text output

Error handling strategy
-----------------------
All API calls are wrapped in try/except.  On failure, embeddings return a
zero-vector (which will result in low similarity scores and an "Unclassified"
label) and generation returns a fallback auto-generated summary.  This ensures
the pipeline never crashes due to a transient Ollama issue.
"""

import re
import requests
import logging
import numpy as np
import pandas as pd
from typing import List
import tkinter.messagebox as messagebox

from escalation_ai.core.config import OLLAMA_BASE_URL, EMBED_MODEL, GEN_MODEL
from escalation_ai.core.gpu_utils import get_optimal_embedding_batch_size

logger = logging.getLogger(__name__)


class OllamaBrain:
    """Handles both Embedding (Left Brain) and Generation (Right Brain).

    This class encapsulates all interaction with the Ollama inference server.
    It maintains a small amount of state:

    - ``embed_model`` / ``gen_model`` -- the model identifiers loaded from
      config.
    - ``_embed_dim`` -- cached embedding dimensionality (lazily discovered on
      the first call to :meth:`get_dim`).

    Thread safety: instances are **not** thread-safe.  The pipeline is
    single-threaded so this is acceptable.
    """

    def __init__(self):
        """Initialise model names from configuration.

        No network calls are made here -- model availability is verified
        separately by :func:`check_models`.
        """
        self.embed_model = EMBED_MODEL   # e.g. "nomic-embed-text"
        self.gen_model = GEN_MODEL       # e.g. "deepseek-r1:14b"
        self._embed_dim = None           # Cached embedding vector length

    # ==================================================================
    # EMBEDDING METHODS
    # ==================================================================

    def get_embedding(self, text):
        """Get a dense embedding vector for a single string.

        Calls the Ollama ``/api/embed`` endpoint with a single input string
        and returns the resulting vector as a 1-D numpy array.

        The Ollama API may return the vector under either the ``"embedding"``
        key (older API) or the ``"embeddings"`` key (newer batch API).  Both
        are handled transparently.

        Args:
            text: The input string to embed.  If ``NaN`` or empty, a zero
                vector of the correct dimensionality is returned immediately
                (no API call).

        Returns:
            A 1-D ``numpy.ndarray`` of shape ``(embedding_dim,)``.  On any
            error a zero vector is returned so downstream cosine-similarity
            calculations produce a score of 0 rather than crashing.
        """
        if pd.isna(text) or text == "":
            return np.zeros(self.get_dim())
        try:
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": self.embed_model, "input": str(text)},
                timeout=30
            )
            if res.status_code == 200:
                # Handle both old ("embedding") and new ("embeddings") response schemas
                vec = res.json().get('embedding') or res.json().get('embeddings', [[]])[0]
                return np.array(vec)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
        # Fallback: zero vector -- produces 0 cosine similarity with any anchor
        return np.zeros(self.get_dim())

    def get_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[np.ndarray]:
        """Get embedding vectors for multiple strings in batched API calls.

        This is the high-throughput path used during Phase 1 classification,
        where hundreds or thousands of tickets need to be embedded.  Texts
        are sent to Ollama in batches to balance throughput against memory
        usage and API timeout constraints.

        Batch size selection
        --------------------
        If ``batch_size`` is ``None``, the optimal size is auto-detected from
        the GPU VRAM via :func:`gpu_utils.get_optimal_embedding_batch_size`:

        - 24 GB+ VRAM (e.g. RTX 5090, A100):  100 items per batch
        - 16-24 GB VRAM (e.g. RTX 5080):       50 items per batch
        - 12-16 GB VRAM (e.g. RTX 5070 Ti):    20 items per batch
        -  8-12 GB VRAM (e.g. RTX 5070):        10 items per batch
        - < 8 GB or CPU-only:                    5 items per batch

        Empty / NaN handling
        --------------------
        Empty texts are *not* sent to the API.  Their positions are
        pre-filled with zero vectors so the returned list has a 1:1
        correspondence with the input ``texts`` list.

        Args:
            texts: List of strings to embed.
            batch_size: Items per batch.  If ``None``, auto-detected from GPU
                VRAM.

        Returns:
            List of ``numpy.ndarray`` with the same length as ``texts``.
            Each element is a 1-D array of shape ``(embedding_dim,)``.
        """
        # Auto-detect optimal batch size based on GPU VRAM
        if batch_size is None:
            batch_size = get_optimal_embedding_batch_size()
            logger.info(f"Auto-detected embedding batch size: {batch_size}")
        if not texts:
            return []

        # Separate valid (non-empty) texts from blanks while preserving
        # original indices so we can reassemble the results in order.
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if pd.isna(text) or text == "":
                continue
            valid_indices.append(i)
            valid_texts.append(str(text))

        # Pre-fill the result list with zero vectors for every position
        result = [np.zeros(self.get_dim()) for _ in texts]

        if not valid_texts:
            return result

        # ------------------------------------------------------------------
        # Process valid texts in batches to avoid API timeouts.
        # Each batch is a single POST to /api/embed with a list of strings.
        # ------------------------------------------------------------------
        all_embeddings = []
        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]

            try:
                res = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": self.embed_model, "input": batch_texts},
                    timeout=600  # 10 minutes per batch for slower GPUs
                )
                if res.status_code == 200:
                    # The batch API returns {"embeddings": [[...], [...], ...]}
                    batch_embeddings = res.json().get('embeddings', [])
                    all_embeddings.extend(batch_embeddings)
                else:
                    # Non-200 response -- fill this batch with zero vectors
                    logger.warning(f"Batch embedding returned status {res.status_code}")
                    all_embeddings.extend([np.zeros(self.get_dim()).tolist() for _ in batch_texts])
            except Exception as e:
                # Network error or timeout -- fill this batch with zero vectors
                logger.warning(f"Batch embedding failed: {e}")
                all_embeddings.extend([np.zeros(self.get_dim()).tolist() for _ in batch_texts])

        # Map the collected embeddings back to their original positions
        for idx, vec in zip(valid_indices, all_embeddings):
            result[idx] = np.array(vec)

        return result

    def get_dim(self):
        """Get the embedding dimensionality by issuing a test embedding.

        The result is cached after the first call so subsequent invocations
        are free.

        Returns:
            Integer dimensionality of the configured embedding model (e.g. 768).
        """
        if self._embed_dim:
            return self._embed_dim
        # Issue a lightweight test embedding to discover the dimension
        v = self.get_embedding("test")
        self._embed_dim = len(v)
        return self._embed_dim

    # ==================================================================
    # TEXT GENERATION METHODS
    # ==================================================================

    def _strip_thinking_tags(self, text):
        """Remove ``<think>...</think>`` blocks from LLM output.

        Some reasoning-oriented models (e.g. DeepSeek-R1) emit their
        chain-of-thought inside ``<think>`` tags.  This internal reasoning
        is not suitable for the executive summary, so we strip it out,
        keeping only the final answer.

        Args:
            text: Raw LLM output string.

        Returns:
            The cleaned string with all ``<think>`` blocks removed.
        """
        if not text:
            return text
        # re.DOTALL makes '.' match newlines so multi-line think blocks are removed
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def generate_synthesis(self, context_text):
        """Use the LLM to write a comprehensive executive summary.

        This method constructs a detailed prompt following **McKinsey
        consulting frameworks** (Pyramid Principle, MECE, Pareto 80/20,
        Impact-Effort Matrix) and sends it to the Ollama generation model.

        Prompt design
        -------------
        The prompt is structured in seven sections:

        1. **The Bottom Line** -- Pyramid Principle answer-first paragraph.
        2. **Situation Overview** -- high-level context bullets.
        3. **Key Findings** -- MECE-structured findings across four pillars.
        4. **80/20 Analysis** -- Pareto identification of vital-few drivers.
        5. **Prioritized Recommendations** -- Impact-Effort matrix actions.
        6. **Risk Assessment** -- RAG (Red/Amber/Green) traffic lights.
        7. **Executive Callout** -- one memorable CEO-level statement.

        The ``CRITICAL RULES`` section in the prompt explicitly instructs the
        LLM to use **only** numbers from the provided ``context_text`` and
        never fabricate statistics.

        Generation parameters
        ---------------------
        - ``num_predict: 12000`` -- generous token budget for a 600-800 word
          output.
        - ``num_ctx: 32768`` -- large context window to accommodate the full
          statistical summary.
        - ``temperature: 0.5`` -- moderate creativity; biased toward factual
          accuracy for financial data.
        - ``timeout: 600s`` -- 10-minute ceiling; the generation model may
          need to be loaded into VRAM on first call.

        Args:
            context_text: The structured statistical summary string assembled
                by ``EscalationPipeline.generate_executive_summary()``.

        Returns:
            The LLM-generated executive summary as a string.  If the LLM is
            unavailable or times out, :meth:`_generate_fallback_summary` is
            called instead.
        """
        prompt = f"""You are a McKinsey Principal Consultant specializing in Telecom Operations. Write an executive-ready analysis using consulting best practices.

CRITICAL RULES:
1. ONLY use numbers, percentages, dollar amounts from the DATA CONTEXT below
2. DO NOT fabricate statistics - every claim must be traceable to the data
3. Apply MECE (Mutually Exclusive, Collectively Exhaustive) structure
4. Use PYRAMID PRINCIPLE: Lead with the answer, then support
5. Every finding needs a "SO WHAT?" implication

DATA CONTEXT:
{context_text}

═══════════════════════════════════════════════════════════════
PRODUCE THIS STRUCTURED OUTPUT:
═══════════════════════════════════════════════════════════════

SECTION 1 - THE BOTTOM LINE (Pyramid Principle - Answer First)
Write 2-3 sentences with THE KEY MESSAGE. State:
- Total financial exposure ($X)
- The #1 problem causing it
- The recommended action
Example format: "Analysis reveals $X in financial exposure driven primarily by [top category]. Immediate focus on [action] can reduce costs by X%."

SECTION 2 - SITUATION OVERVIEW
In 3-4 bullet points, provide context:
- Total tickets analyzed and time period scope
- Total financial impact and revenue at risk
- Critical vs Major vs Minor distribution
- Overall health assessment (Critical/At Risk/Stable)

SECTION 3 - KEY FINDINGS (MECE Structure)
Organize findings into 4 MUTUALLY EXCLUSIVE categories. For each, include the data AND the "SO WHAT":

A) PROCESS GAPS (scheduling, workflow, compliance issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

B) KNOWLEDGE GAPS (documentation, training, expertise issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

C) SYSTEM/TOOL ISSUES (configuration, data, technical issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

D) COMMUNICATION FAILURES (response, handoff, coordination issues)
   - Data: [specific numbers from context]
   - So What: [business implication]

SECTION 4 - 80/20 ANALYSIS (Pareto Principle)
Identify the vital few driving majority of impact:
- Which 2-3 categories drive 80% of financial impact? (use actual $ from data)
- Which 2-3 root causes drive 80% of ticket volume?
- Where should resources be concentrated for maximum ROI?

SECTION 5 - PRIORITIZED RECOMMENDATIONS (Impact-Effort Matrix)
List 4-5 actions categorized by:

QUICK WINS (High Impact, Low Effort) - Do First:
1. [Action] | Impact: $X savings | Timeline: Week 1-2

MAJOR PROJECTS (High Impact, High Effort) - Plan For:
2. [Action] | Impact: $X savings | Timeline: Month 2-3

FILL-INS (Low Impact, Low Effort) - If Time Permits:
3. [Action] | Impact: $X savings | Timeline: Ongoing

For each, cite specific financial benefit from the data.

SECTION 6 - RISK ASSESSMENT (RAG Status)
Provide traffic light status for each area:
- Financial Exposure: [RED/AMBER/GREEN] - [reason with $ figure]
- Recurrence Risk: [RED/AMBER/GREEN] - [reason with %]
- Process Maturity: [RED/AMBER/GREEN] - [reason]
- Resolution Capability: [RED/AMBER/GREEN] - [reason]

SECTION 7 - EXECUTIVE CALLOUT
One powerful statement (2 sentences max) that a CEO should remember:
"[Key insight with financial stake]. [Recommended action with expected outcome]."

FORMATTING:
- Use **SECTION X - TITLE** for headers
- Blank line between sections
- Use - for bullets, numbers for ordered lists
- Include $ figures throughout
- 600-800 words total
- Be direct and actionable, not academic"""
        try:
            logger.info(f"  Requesting AI synthesis from {self.gen_model}...")
            res = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.gen_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 12000,  # Generous token budget for full summary
                        "num_ctx": 32768,      # Large context window for stats + prompt
                        "temperature": 0.5,    # Moderate temp -- factual bias for financials
                    }
                },
                timeout=600  # 10-minute ceiling for comprehensive financial analysis
            )
            if res.status_code == 200:
                raw_response = res.json()['response'].strip()
                # Strip chain-of-thought tags from reasoning models (e.g. DeepSeek-R1)
                cleaned = self._strip_thinking_tags(raw_response)
                if cleaned:
                    logger.info(f"  ✓ AI synthesis complete ({len(cleaned)} chars)")
                    return cleaned
                else:
                    logger.warning(f"  AI returned empty response after cleaning")
                    return self._generate_fallback_summary(context_text)
            else:
                logger.error(f"  AI request failed with status {res.status_code}: {res.text[:200]}")
                return self._generate_fallback_summary(context_text)
        except requests.exceptions.Timeout:
            logger.error(f"AI Synthesis timed out after 480s - model may be loading. Try again or use smaller model.")
            return self._generate_fallback_summary(context_text)
        except Exception as e:
            logger.error(f"AI Synthesis Failed: {e}")
            return self._generate_fallback_summary(context_text)

    def _generate_fallback_summary(self, context_text):
        """Generate a basic summary when the LLM is unavailable.

        Parses a few key metrics from the context string using regex and
        assembles a minimal auto-generated summary.  This ensures the report
        always contains *some* executive text, even when Ollama is down.

        Args:
            context_text: The same structured statistical summary that would
                have been sent to the LLM.

        Returns:
            A short plaintext summary prefixed with ``[AUTO-GENERATED SUMMARY]``
            to make it clear this was not AI-written.
        """
        logger.warning("Using fallback summary generation (AI unavailable)")

        # Extract key metrics from the context string via regex
        total_match = re.search(r'Total Weighted Friction Score: ([\d,]+)', context_text)
        external_match = re.search(r'External.*?: ([\d.]+)%', context_text)
        repeat_match = re.search(r'Confirmed Repeat Offenses: (\d+)', context_text)

        total = total_match.group(1) if total_match else "N/A"
        external = external_match.group(1) if external_match else "N/A"
        repeats = repeat_match.group(1) if repeat_match else "0"

        return (
            f"[AUTO-GENERATED SUMMARY - AI Unavailable]\n\n"
            f"This report analyzed escalation data with a total weighted friction score of {total}. "
            f"External-facing issues account for {external}% of total risk. "
            f"There are {repeats} confirmed repeat offenses indicating potential organizational learning gaps. "
            f"Please review the detailed data and charts below for full analysis."
        )

    # ==================================================================
    # RESOURCE MANAGEMENT
    # ==================================================================

    def unload(self):
        """Unload both models from Ollama to free GPU VRAM.

        Sends a ``keep_alive: 0`` request for each model, which instructs
        Ollama to evict the model from memory immediately rather than
        keeping it warm for the default keep-alive duration.

        This is typically called at the end of the pipeline or when
        switching to a different set of models.
        """
        for m in [self.embed_model, self.gen_model]:
            try:
                requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": m, "keep_alive": 0})
            except Exception as e:
                logger.warning(f"Failed to unload model {m}: {e}")


# ============================================================================
# MODULE-LEVEL HEALTH-CHECK FUNCTIONS
# ============================================================================
# These are used by the pipeline orchestrator during initialisation to verify
# that the Ollama server is up and that the required models are available.
# ============================================================================

def check_models(ai):
    """Quick probe to ensure required AI models exist and are working.

    Performs two checks:

    1. **Embedding model** (blocking) -- a test embedding is requested.  If
       it returns a zero vector or raises an exception the pipeline cannot
       proceed, so ``False`` is returned and a tkinter error dialog is shown.

    2. **Generation model** (non-blocking) -- a minimal generation request is
       sent.  If it fails a warning is logged but the pipeline is allowed to
       continue (the executive summary will fall back to auto-generation).

    Args:
        ai: An ``OllamaBrain`` instance.

    Returns:
        ``True`` if the embedding model is operational, ``False`` otherwise.
    """
    logger.info("Checking AI model availability...")

    # ---- Check embedding model (required) ----
    try:
        test_vec = ai.get_embedding("test connection")
        if len(test_vec) == 0 or np.all(test_vec == 0):
            raise ValueError("Embedding returned zero vector")
        logger.info(f"✓ Embedding model '{EMBED_MODEL}' is active (dim={len(test_vec)})")
    except Exception as e:
        logger.error(f"Embedding model check failed: {e}")
        messagebox.showerror(
            "Model Not Found",
            f"Embedding model '{EMBED_MODEL}' is not available.\n\n"
            f"Please run:\n  ollama pull {EMBED_MODEL}\n\n"
            f"Error: {e}"
        )
        return False

    # ---- Check generation model (optional -- warn only) ----
    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": GEN_MODEL, "prompt": "test", "stream": False},
            timeout=10
        )
        if res.status_code == 200:
            logger.info(f"✓ Generation model '{GEN_MODEL}' is active")
        else:
            logger.warning(f"Generation model '{GEN_MODEL}' may not be available (status: {res.status_code})")
    except Exception as e:
        logger.warning(f"Generation model '{GEN_MODEL}' check failed: {e}. Executive summary may not work.")

    return True


def check_ollama_server():
    """Check if Ollama server is running.

    Sends a simple GET to the Ollama root URL.  If the server is unreachable
    a tkinter error dialog guides the user to start it.

    Returns:
        ``True`` if the server responds, ``False`` otherwise.
    """
    try:
        requests.get(f"{OLLAMA_BASE_URL}/", timeout=3)
        return True
    except Exception as e:
        logger.error(f"Ollama server not reachable: {e}")
        messagebox.showerror(
            "Ollama Not Running",
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}\n\n"
            "Please ensure Ollama is running:\n  ollama serve"
        )
        return False
