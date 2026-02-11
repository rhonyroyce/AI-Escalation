"""
Pulse Dashboard - Ollama AI Integration
========================================

This module provides a self-contained client for the local Ollama LLM server,
enabling AI-powered features throughout the Pulse Dashboard.  It is designed
with **graceful degradation** as a core principle: every function returns a
sensible fallback (None, empty list, zero vector) when Ollama is unavailable,
so the rest of the dashboard continues to work without AI features.

Capabilities
------------
1. **Text Generation** (`ollama_generate`):
   Sends a prompt to a large chat model (qwen3:14b) and returns the response.
   Used for generating narrative insights, executive summaries, and
   recommendations from structured project data.

2. **Single Embedding** (`ollama_embed`):
   Converts a text string into a dense vector representation using a smaller,
   fast embedding model (qwen2:1.5b).  Used as a building block for semantic
   search.

3. **Batch Embedding** (`ollama_embed_batch`):
   Embeds multiple texts efficiently by sending them in batches to avoid
   per-request overhead.  Used when building the search index from all
   project comments and pain points.

4. **Embedding Index** (`build_embeddings_index`):
   Constructs an in-memory search index from a DataFrame's text columns.
   Each text snippet is paired with metadata (project name, region, PM, score)
   so search results can be displayed with full context.

5. **Semantic Search** (`semantic_search`):
   Given a natural-language query, finds the most similar text snippets in
   the embedding index using cosine similarity.  This lets users ask questions
   like "which projects have network issues?" and get relevant results even
   if the exact words don't match.

Model Selection
---------------
- **Chat model** (qwen3:14b): A 14-billion parameter model that produces
  high-quality narrative text.  The `/think` reasoning mode is used internally
  by the model, so `strip_thinking_tags()` removes the `<think>` blocks from
  the output before returning it to the user.

- **Embedding model** (qwen2:1.5b): A small, fast model optimised for
  generating dense vector embeddings.  Its 384-dimensional output is compact
  enough for real-time similarity computation in-memory without needing a
  vector database.

Server Configuration
--------------------
Ollama runs locally on the default port 11434.  No authentication is required.
The base URL is configurable via the `OLLAMA_BASE_URL` constant.  All HTTP
calls include explicit timeouts to prevent the dashboard from hanging if
Ollama becomes unresponsive.

Error Handling Strategy
-----------------------
Every function wraps its HTTP call in a try/except block:
- Network errors, timeouts, and non-200 responses all fall through to the
  same fallback path.
- No exceptions are propagated to the caller -- the dashboard never crashes
  due to an AI service issue.
- Callers should check for None / empty results to decide whether to show
  AI-generated content or a "AI unavailable" placeholder.
"""

import re
import requests
import numpy as np
from typing import List, Optional

# ── Ollama Server Configuration ──────────────────────────────────────────────
# Base URL for the local Ollama REST API.  Ollama's default port is 11434.
# All endpoints are relative to this base: /api/generate, /api/embed, /api/tags.
OLLAMA_BASE_URL = 'http://localhost:11434'

# CHAT_MODEL: the large language model used for text generation (narrative
# insights, summaries, recommendations).  qwen3:14b provides good quality
# reasoning and structured output at a practical speed for interactive use.
CHAT_MODEL = 'qwen3:14b'

# EMBED_MODEL: the smaller model used exclusively for generating embedding
# vectors.  qwen2:1.5b is fast enough for real-time batch embedding and
# produces 384-dimensional vectors suitable for cosine similarity search.
EMBED_MODEL = 'qwen2:1.5b'


def check_ollama() -> bool:
    """Check if the Ollama server is running and reachable.

    Makes a lightweight GET request to the /api/tags endpoint, which lists
    available models.  We don't actually need the model list -- we just
    care whether the server responds with HTTP 200.

    This is called by the sidebar to show a status indicator, and can also
    be used as a gate before attempting expensive generation/embedding calls.

    Returns:
        True if Ollama is reachable and responding, False otherwise.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        # Any exception (ConnectionError, Timeout, etc.) means Ollama is down.
        return False


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    The qwen3 model family uses a "thinking" mode where it wraps its internal
    chain-of-thought reasoning in <think>...</think> XML tags.  This reasoning
    is useful for the model's accuracy but should not be shown to the end user.
    We strip these blocks entirely using a non-greedy regex with DOTALL flag
    (so the dot matches newlines within the thinking block).

    Args:
        text: Raw model output that may contain <think> blocks.

    Returns:
        Cleaned text with all thinking blocks removed and whitespace trimmed.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def ollama_generate(prompt: str, temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
    """Generate text using the Ollama chat model.

    Sends a prompt to the /api/generate endpoint with streaming disabled
    (we wait for the complete response).  The response is cleaned by removing
    any <think> tags before being returned.

    Design choices:
    - `stream: False` simplifies the code -- we get the full response in one
      JSON payload instead of handling server-sent events.
    - `num_predict: 4096` caps the output length to prevent runaway generation
      on open-ended prompts.  4096 tokens is enough for detailed summaries.
    - `temperature: 0.3` (default) produces focused, deterministic output
      suitable for data-driven insights.  Callers can raise this for more
      creative/varied responses.
    - `timeout: 120` seconds is generous because large models on CPU can be
      slow.  The 14B model may take 30-90 seconds for a long generation.

    Args:
        prompt: The full prompt text to send to the model.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
                     Default 0.3 keeps outputs factual and consistent.
        timeout: Maximum seconds to wait for a response before giving up.

    Returns:
        The generated text with thinking tags stripped, or None if generation
        failed for any reason (server down, timeout, non-200 response).
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,          # Wait for complete response
                "options": {
                    "num_predict": 4096,  # Max tokens to generate
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            # Extract the 'response' field from the JSON payload.
            # This contains the model's generated text.
            text = resp.json().get('response', '')
            # Remove <think>...</think> reasoning blocks before returning.
            return strip_thinking_tags(text)
        # Non-200 status code (e.g. 404 model not found, 500 internal error).
        return None
    except requests.exceptions.Timeout:
        # The model took too long -- return None so the caller can show a
        # "generation timed out" message instead of crashing.
        return None
    except Exception:
        # Catch-all for connection errors, JSON decode errors, etc.
        return None


def ollama_embed(text: str) -> np.ndarray:
    """Get an embedding vector for a single text string.

    Calls the Ollama /api/embed endpoint with the embedding model.  The
    response JSON structure varies slightly between Ollama versions:
    - Newer versions return {"embedding": [...]}, a single vector.
    - Older versions return {"embeddings": [[...]]}, a list of vectors.
    We handle both formats by checking for 'embedding' first, then falling
    back to the first element of 'embeddings'.

    The fallback zero vector has 384 dimensions, which matches the output
    dimensionality of qwen2:1.5b.  Using a zero vector as the fallback
    ensures that downstream code (which expects a fixed-size array) never
    crashes, and the zero vector will have zero cosine similarity with all
    other vectors, so failed embeddings naturally sort to the bottom of
    search results.

    Args:
        text: The text to embed.  Will be cast to str as a safety measure.

    Returns:
        numpy array of shape (384,) containing the embedding vector,
        or a zero vector of the same shape if embedding failed.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": str(text)},
            timeout=30,  # Embeddings are fast; 30s is very generous
        )
        if resp.status_code == 200:
            data = resp.json()
            # Handle both response formats: single vector or list-of-vectors.
            vec = data.get('embedding') or data.get('embeddings', [[]])[0]
            return np.array(vec)
    except Exception:
        # Connection error, timeout, JSON error -- all handled the same way.
        pass
    # Return a zero vector as a safe fallback.  384 is the embedding dimension
    # for qwen2:1.5b.  A zero vector has zero cosine similarity with everything,
    # so it won't pollute search results.
    return np.zeros(384)  # Default dim for small models


def ollama_embed_batch(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """Embed multiple texts in batches for efficiency.

    Rather than calling ollama_embed() once per text (which incurs HTTP
    overhead for each call), this function sends texts in batches of
    `batch_size` to the /api/embed endpoint.  Ollama supports batch input
    natively by accepting a list of strings in the "input" field.

    The batch_size of 32 is a pragmatic default that balances:
    - Memory usage on the Ollama server (each text is tokenized and processed)
    - HTTP payload size (keeping requests reasonable)
    - Throughput (fewer round trips = faster total time)

    If any batch fails (server error, timeout), the failed texts get zero
    vectors so the overall result list always has the same length as the
    input list.  This is critical because the caller uses positional
    correspondence between the input texts and output vectors.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to send per HTTP request.  Default 32.

    Returns:
        List of numpy arrays, one per input text.  Each array has shape (384,).
        Failed embeddings are represented as zero vectors.
    """
    results = []
    # Process texts in chunks of batch_size.
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": batch},  # Batch input: list of strings
                timeout=120,  # Longer timeout for large batches
            )
            if resp.status_code == 200:
                # 'embeddings' is a list of vectors, one per input text.
                embeddings = resp.json().get('embeddings', [])
                results.extend([np.array(e) for e in embeddings])
            else:
                # Server returned an error -- fill in zero vectors for this batch
                # so the output length still matches the input length.
                results.extend([np.zeros(384) for _ in batch])
        except Exception:
            # Network/timeout error -- same fallback as above.
            results.extend([np.zeros(384) for _ in batch])
    return results


def build_embeddings_index(df, columns=None) -> dict:
    """Build an in-memory embeddings index for semantic search.

    This function iterates over specified text columns in the DataFrame,
    extracts non-empty text snippets, embeds them in batch, and packages
    everything into a dictionary structure that `semantic_search()` can
    query against.

    Why not use a vector database (Pinecone, Weaviate, etc.)?
    - The dataset is small enough (hundreds to low thousands of rows) that
      in-memory numpy cosine similarity is instantaneous.
    - No external infrastructure dependency beyond Ollama.
    - The index is rebuilt on demand (when filters change), so persistence
      isn't needed.

    Text preprocessing:
    - Texts shorter than 5 characters are skipped (they're likely noise:
      "N/A", "OK", "-", etc.).
    - Texts are truncated to 500 characters before embedding.  This ensures
      consistent embedding quality (very long texts can degrade embedding
      accuracy) and keeps batch sizes manageable.

    Metadata:
    Each text snippet is paired with metadata from its source row so that
    search results can display context (project name, region, PM, score)
    without needing to look up the original DataFrame.

    Args:
        df: DataFrame containing the text columns to index.  Expected to
            have columns like 'Project', 'Region', 'Area', 'PM Name',
            'Total Score' for metadata extraction.
        columns: List of column names to extract text from.
                 Defaults to ['Comments', 'Pain Points'] -- the two richest
                 free-text columns in the ProjectPulse workbook.

    Returns:
        Dictionary with three keys:
        - 'embeddings': numpy array of shape (N, 384) where N is the number
          of valid text snippets found.
        - 'metadata': list of N dicts, each containing {index, column,
          project, region, area, pm, score} for the corresponding text.
        - 'texts': list of N strings (the actual text that was embedded).

        Returns empty structures if no valid texts are found.
    """
    if columns is None:
        # Default to the two primary narrative columns.  These contain the
        # most searchable content: PM comments about project status and
        # descriptions of pain points / blockers.
        columns = ['Comments', 'Pain Points']

    texts = []     # Parallel list: the text snippets to embed
    metadata = []  # Parallel list: metadata dicts for each snippet

    for col in columns:
        # Skip columns that don't exist in this particular DataFrame
        # (some workbook versions may not have all columns).
        if col not in df.columns:
            continue
        for idx, row in df.iterrows():
            val = row.get(col)

            # Skip missing values.  Check for None explicitly and for NaN
            # (which is a float, so we use isinstance + np.isnan).
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            text = str(val).strip()

            # Skip very short texts -- they contain no meaningful semantic
            # content (e.g. "OK", "N/A", "--", "TBD").
            if len(text) < 5:
                continue

            # Truncate to 500 characters.  Embedding models work best on
            # moderate-length text.  Very long text dilutes the semantic
            # signal and increases compute cost without proportional benefit.
            texts.append(text[:500])

            # Store metadata alongside the text so search results can be
            # presented with full context.  Using .get() with defaults to
            # handle any missing columns gracefully.
            metadata.append({
                'index': idx,                          # DataFrame row index (for cross-referencing)
                'column': col,                         # Which column this text came from
                'project': row.get('Project', ''),     # Project name
                'region': row.get('Region', ''),       # Geographic region
                'area': row.get('Area', ''),           # Sub-area within region
                'pm': row.get('PM Name', ''),          # Project manager name
                'score': row.get('Total Score', 0),    # Pulse score (0-24)
            })

    # If no valid texts were found (empty DataFrame, all values NaN, etc.),
    # return an empty index structure so callers don't need special-case logic.
    if not texts:
        return {'embeddings': np.array([]), 'metadata': [], 'texts': []}

    # Embed all texts in batch for efficiency (see ollama_embed_batch docs).
    embeddings = ollama_embed_batch(texts)

    # Package everything into the index structure.
    # The three lists are positionally aligned: embeddings[i] is the vector
    # for texts[i], which has metadata in metadata[i].
    return {
        'embeddings': np.array(embeddings),  # Shape: (N, 384)
        'metadata': metadata,                # Length: N
        'texts': texts,                      # Length: N
    }


def semantic_search(query: str, index: dict, top_k: int = 5) -> list:
    """Search the embeddings index for texts similar to the query.

    This implements a simple but effective semantic search pipeline:
    1. Embed the user's query using the same embedding model.
    2. Compute cosine similarity between the query vector and all indexed vectors.
    3. Return the top_k most similar results with their metadata.

    Why cosine similarity?
    - It measures the angle between vectors, ignoring magnitude.  This means
      texts of different lengths are compared fairly.
    - It's the standard similarity metric for text embeddings.
    - scikit-learn's cosine_similarity is fast for small datasets (< 10K vectors).

    Edge cases handled:
    - Empty index: returns [] immediately.
    - Failed query embedding (zero vector): returns [] because a zero vector
      would have zero similarity with everything, producing meaningless results.

    Args:
        query: Natural language search query (e.g. "network coverage issues").
        index: The embeddings index dict from `build_embeddings_index()`.
        top_k: Number of top results to return.  Default 5.

    Returns:
        List of up to top_k result dicts, each containing:
        - 'similarity': float cosine similarity score (0.0 to 1.0)
        - 'text': the matched text snippet
        - Plus all metadata fields (index, column, project, region, area, pm, score)

        Results are sorted by descending similarity.
    """
    # Guard: empty index means there's nothing to search.
    if index['embeddings'].size == 0:
        return []

    # Embed the query using the same model that produced the index embeddings.
    # This is essential -- cosine similarity is only meaningful when both vectors
    # come from the same embedding space.
    query_emb = ollama_embed(query)

    # If embedding failed (returned zero vector), bail out.  A zero vector has
    # zero cosine similarity with everything, so results would be meaningless.
    if query_emb.sum() == 0:
        return []

    # Lazy import of sklearn to avoid importing it at module load time.
    # This keeps the module lightweight for code paths that don't use search.
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute cosine similarity between the query and ALL indexed embeddings.
    # cosine_similarity expects 2D arrays, so we wrap query_emb in a list.
    # Result shape: (1, N) -- we take [0] to get a 1D array of N similarities.
    similarities = cosine_similarity([query_emb], index['embeddings'])[0]

    # Get indices of the top_k highest similarities.
    # np.argsort returns indices in ascending order, so [::-1] reverses to
    # descending, and [:top_k] takes just the top results.
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Build result dicts combining similarity score, text, and metadata.
    results = []
    for i in top_indices:
        results.append({
            'similarity': float(similarities[i]),  # Cast to Python float for JSON serialization
            'text': index['texts'][i],             # The matched text snippet
            **index['metadata'][i],                # Unpack all metadata fields
        })
    return results
