"""
AI Classification Engine for escalation categorization.

This module implements **Phase 1** of the Escalation AI pipeline: automatically
categorising each escalation ticket into one of eight predefined issue types
optimised for telecom operations.

Classification approach: Three-tier hybrid
------------------------------------------
The classifier uses a **three-tier cascade** that balances speed, confidence,
and semantic understanding.  Each ticket is evaluated by the tiers in order;
the first tier that produces a confident result "claims" the ticket:

::

    Tier 1: Regex pattern matching
        |  Fastest.  Highest confidence (0.95).
        |  Matches explicit phrases like "SLA breach" or "config change".
        v
    Tier 2: Keyword / phrase scoring
        |  Fast.  High confidence (0.50-0.85, scaled by hit count).
        |  Counts weighted keyword occurrences per category.
        |  Cross-validated against embedding similarity (>= 0.25 threshold).
        v
    Tier 3: Embedding similarity (semantic)
        |  Slowest.  Variable confidence (depends on similarity score).
        |  Computes cosine similarity between the ticket's embedding vector
        |  and precomputed **category anchor centroids**.
        v
    Final: Apply minimum confidence threshold or mark "Unclassified".

Category taxonomy (8 categories)
--------------------------------
The eight categories are defined in ``core.config.ANCHORS`` and refined by
``core.config.CATEGORY_KEYWORDS``.  Typical categories for telecom:

- Scheduling & Resource Management
- Documentation & Knowledge Management
- Configuration & Data Quality
- Communication & Response Management
- Process & Compliance Issues
- Training & Skill Development
- Tool & System Issues
- Vendor & Third-Party Management

Each category has an **anchor centroid** -- the mean embedding of all its
representative phrases.  During classification, a ticket is assigned to the
category whose centroid is closest (highest cosine similarity) to the
ticket's own embedding.

Human-in-the-loop feedback
--------------------------
On initialisation, the module checks for user corrections in the feedback
file (``classification_feedback.xlsx``).  If corrections exist, the anchor
centroids are shifted to incorporate the human-provided labels.  This allows
the classifier to improve over successive pipeline runs without retraining.

GPU acceleration
----------------
When NVIDIA GPU + RAPIDS cuML are available (detected at module load via
``gpu_utils.is_gpu_available``), cosine similarity computations are
dispatched to the GPU via CuPy.  Otherwise, they fall back to NumPy on CPU.

Uses hybrid keyword + embedding approach for classification.

8-category system optimized for telecom escalation analysis.
GPU-accelerated with RAPIDS cuML when available.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from ..core.config import (
    ANCHORS, CATEGORY_KEYWORDS, SUB_CATEGORIES, MIN_CLASSIFICATION_CONFIDENCE, USE_GPU
)
from ..core.gpu_utils import (
    batch_cosine_similarity_gpu,   # Vectorised similarity: query vs matrix
    cosine_similarity_gpu,         # Pairwise similarity: vector vs vector
    is_gpu_available               # GPU + cuML availability check
)
from ..feedback.feedback_learning import get_feedback_learner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU availability flag -- evaluated once at module import time.
# If USE_GPU is True in config AND cuML/cuDF/GPU are all present, similarity
# computations will be dispatched to the GPU.
# ---------------------------------------------------------------------------
_USE_GPU = USE_GPU and is_gpu_available()
if _USE_GPU:
    logger.info("[Classification] GPU acceleration enabled")

# ---------------------------------------------------------------------------
# Module-level caches.
# ``_anchor_centroids_cache`` holds the per-category mean embedding vector.
# ``_COMPILED_PATTERNS`` holds pre-compiled regex objects for Tier 1.
# Both are populated lazily on first use and persist for the process lifetime.
# ---------------------------------------------------------------------------
_anchor_centroids_cache: Dict[str, np.ndarray] = {}
_COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {}


# ============================================================================
# REGEX PATTERN COMPILATION (Tier 1 support)
# ============================================================================

def _compile_patterns():
    """Compile regex patterns from ``CATEGORY_KEYWORDS`` for faster matching.

    Each category in the config may define a ``"patterns"`` list of regex
    strings (e.g. ``r"SLA\\s+breach"``).  This function compiles them once
    into ``re.Pattern`` objects with the ``re.IGNORECASE`` flag for efficient
    reuse across all tickets.

    Returns:
        Dict mapping category name to a list of compiled ``re.Pattern`` objects.
    """
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS:
        return _COMPILED_PATTERNS  # Already compiled -- return cached

    for category, keywords in CATEGORY_KEYWORDS.items():
        patterns = keywords.get("patterns", [])
        _COMPILED_PATTERNS[category] = [
            re.compile(p, re.IGNORECASE) for p in patterns
        ]

    return _COMPILED_PATTERNS


# ============================================================================
# ANCHOR CENTROID COMPUTATION
# ============================================================================

def get_anchor_centroids(ai) -> Dict[str, np.ndarray]:
    """Get or compute anchor centroids with caching and feedback adjustment.

    Anchor centroids are the heart of the embedding-based (Tier 3)
    classification.  Each category is represented by a single "centre-of-mass"
    vector computed as the **arithmetic mean** of the embedding vectors of all
    its representative phrases (from ``ANCHORS`` + ``CATEGORY_KEYWORDS``).

    Computation steps
    -----------------
    1. Collect all anchor phrases across all 8 categories.
    2. Batch-embed them in a single ``ai.get_embeddings_batch()`` call for
       efficiency.
    3. Group the resulting vectors by category and compute the element-wise
       mean to obtain each centroid.
    4. If the feedback learner contains human corrections, apply
       ``adjust_centroids()`` to shift the centroids toward the corrected
       labels.

    Caching
    -------
    The result is stored in the module-level ``_anchor_centroids_cache`` dict.
    Subsequent calls return immediately from cache.  Call
    :func:`clear_centroid_cache` to force recomputation.

    Args:
        ai: An ``OllamaBrain`` instance for embedding generation.

    Returns:
        Dict mapping each category name to its centroid ``numpy.ndarray``.
    """
    global _anchor_centroids_cache

    if _anchor_centroids_cache:
        logger.info("Using cached anchor centroids...")
        return _anchor_centroids_cache

    logger.info("Computing anchor centroids for 8-category system (will be cached)...")

    # ------------------------------------------------------------------
    # Step 1: Gather all representative phrases across all categories.
    # We combine both ``ANCHORS`` (simple word lists) and
    # ``CATEGORY_KEYWORDS`` (richer phrase/pattern lists) to build a
    # comprehensive set of anchor phrases.
    # ------------------------------------------------------------------
    all_phrases = []
    phrase_to_category = []

    for category in ANCHORS.keys():
        # Prefer the richer keyword set if available; fall back to ANCHORS
        if category in CATEGORY_KEYWORDS:
            kw_data = CATEGORY_KEYWORDS[category]
            phrases = (
                kw_data.get("primary", []) +
                kw_data.get("phrases", [])
            )
        else:
            phrases = ANCHORS[category]

        for phrase in phrases:
            all_phrases.append(phrase)
            phrase_to_category.append(category)

    # ------------------------------------------------------------------
    # Step 2: Batch-embed all anchor phrases in one API call.
    # This is much faster than embedding them one at a time (especially
    # on GPU, where batching amortises the kernel launch overhead).
    # ------------------------------------------------------------------
    logger.info(f"  Computing embeddings for {len(all_phrases)} anchor phrases...")
    all_vecs = ai.get_embeddings_batch(all_phrases)

    # ------------------------------------------------------------------
    # Step 3: Group vectors by category and compute centroids.
    # centroid_k = mean(v_i) for all phrases i belonging to category k.
    # ------------------------------------------------------------------
    category_vecs: Dict[str, List[np.ndarray]] = {cat: [] for cat in ANCHORS.keys()}
    for phrase, category, vec in zip(all_phrases, phrase_to_category, all_vecs):
        if vec is not None:
            category_vecs[category].append(vec)

    for category, vecs in category_vecs.items():
        if vecs:
            # Element-wise mean across all phrase vectors for this category
            _anchor_centroids_cache[category] = np.mean(vecs, axis=0)
        else:
            logger.warning(f"No embeddings for category: {category}")

    logger.info(f"Computed centroids for {len(_anchor_centroids_cache)} categories")

    # ------------------------------------------------------------------
    # Step 4: Apply human feedback adjustments.
    # If the user has corrected any classifications in the feedback file,
    # the feedback learner shifts centroids toward the corrected labels.
    # This is the human-in-the-loop learning mechanism.
    # ------------------------------------------------------------------
    feedback_learner = get_feedback_learner()
    if feedback_learner.stats['loaded'] > 0 or feedback_learner.stats['custom_categories'] > 0:
        logger.info("Applying user feedback adjustments to centroids...")
        adjusted = feedback_learner.adjust_centroids(_anchor_centroids_cache)
        _anchor_centroids_cache.clear()
        _anchor_centroids_cache.update(adjusted)
        logger.info(f"Final category count: {len(_anchor_centroids_cache)}")

    return _anchor_centroids_cache


# ============================================================================
# TIER 1: REGEX PATTERN CLASSIFICATION
# ============================================================================

def pattern_classify(text: str) -> Optional[Tuple[str, float]]:
    """
    Check if text matches any regex patterns for high-confidence classification.

    This is the **fastest** classification tier.  It scans the input text
    against precompiled regex patterns defined in ``CATEGORY_KEYWORDS``.
    A single regex hit is sufficient for classification because patterns are
    designed to be highly specific (e.g. ``"power\\s+outage"``).

    Args:
        text: The combined ticket text to classify.

    Returns:
        ``(category, 0.95)`` if a pattern matches, or ``None`` if no pattern
        matched.  The fixed confidence of 0.95 reflects high trust in
        hand-crafted regex rules.
    """
    if not text:
        return None

    text_str = str(text)
    patterns = _compile_patterns()

    for category, compiled_patterns in patterns.items():
        for pattern in compiled_patterns:
            if pattern.search(text_str):
                return (category, 0.95)  # High confidence for pattern match

    return None


# ============================================================================
# TIER 2: KEYWORD / PHRASE SCORING
# ============================================================================

def keyword_classify(text: str) -> Optional[Tuple[str, float]]:
    """
    Check if text contains category-specific phrases or keywords.

    Uses the extended ``CATEGORY_KEYWORDS`` taxonomy for matching.  This tier
    is more flexible than regex (Tier 1) but less semantic than embeddings
    (Tier 3).

    Scoring algorithm
    -----------------
    For each category, a weighted score is computed:

    - Each **primary keyword** hit adds +1 to the score.
    - Each **phrase** hit adds +3 (phrases are multi-word and more specific,
      so they receive higher weight).

    A minimum score of **5** is required (e.g. one phrase + two keywords) to
    prevent false positives from single-word coincidences.  The confidence is
    scaled linearly from 0.50 up to a maximum of 0.85.

    Args:
        text: The combined ticket text to classify.

    Returns:
        ``(category, confidence)`` if a strong keyword match is found, or
        ``None`` if no category exceeds the threshold.
    """
    if not text:
        return None

    text_lower = str(text).lower()
    best_match = None
    best_score = 0

    for category, kw_data in CATEGORY_KEYWORDS.items():
        score = 0

        # Primary keywords: single tokens, lower weight
        primary = kw_data.get("primary", [])
        for kw in primary:
            if kw.lower() in text_lower:
                score += 1

        # Phrases: multi-word sequences, higher weight (3x)
        phrases = kw_data.get("phrases", [])
        for phrase in phrases:
            if phrase.lower() in text_lower:
                score += 3  # Phrases are more specific, weighted higher

        if score > best_score:
            best_score = score
            best_match = category

    # Only return if we have a strong enough match (>= 5 points)
    if best_score >= 5:
        # Scale confidence linearly: 0.50 base + 0.05 per point, capped at 0.85
        confidence = min(0.85, 0.5 + (best_score * 0.05))
        return (best_match, confidence)

    return None


# ============================================================================
# SUB-CATEGORY ASSIGNMENT
# ============================================================================

def get_sub_category(text: str, category: str) -> str:
    """
    Determine the sub-category within a main category based on keyword matching.

    After a ticket has been assigned to one of the eight main categories, this
    function further classifies it into a more granular sub-category using
    simple keyword matching against the ``SUB_CATEGORIES`` config.

    Keywords are weighted by word count (longer keywords are more specific
    and receive higher scores).

    Args:
        text: The ticket text.
        category: The main category already assigned to this ticket.

    Returns:
        The best-matching sub-category name, or ``"General"`` if no specific
        sub-category keywords were found.
    """
    if not text or not category:
        return "General"

    if category not in SUB_CATEGORIES:
        return "General"

    text_lower = str(text).lower()
    sub_cats = SUB_CATEGORIES[category]

    best_sub = "General"
    best_score = 0

    for sub_name, keywords in sub_cats.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                # Longer keywords (more words) get higher weight because
                # they are more specific / less likely to false-positive.
                score += len(kw.split())

        if score > best_score:
            best_score = score
            best_sub = sub_name

    return best_sub


# ============================================================================
# MAIN CLASSIFICATION ENTRY POINT
# ============================================================================

def classify_rows(df: pd.DataFrame, ai, show_progress: bool = True) -> pd.DataFrame:
    """
    Categorize rows using the three-tier hybrid classification cascade.

    This is the main function called by the pipeline orchestrator during
    **Phase 1**.  It processes every row in the DataFrame through the
    three-tier classifier and assigns three new columns:

    - ``AI_Category``     -- the predicted main category (one of 8).
    - ``AI_Confidence``   -- confidence score [0, 1].
    - ``AI_Sub_Category`` -- finer-grained sub-category within the main.

    Classification flow per ticket
    ------------------------------
    ::

        ticket text
            |
            v
        Tier 1: pattern_classify()
            | match? --> AI_Category = matched, confidence = 0.95, DONE
            v
        Tier 2: keyword_classify()
            | match? --> cross-validate with embedding similarity (>= 0.25)
            |            --> AI_Category = matched, confidence = max(kw, sim), DONE
            v
        Tier 3: embedding similarity
            | compute cosine sim to all 8 anchor centroids
            | pick the highest
            | if top-2 are within 0.03 --> use keyword overlap as tie-breaker
            | if best_score < MIN_CLASSIFICATION_CONFIDENCE --> "Unclassified"
            v
        DONE

    Consistency improvements
    ------------------------
    - Categories are sorted alphabetically before scoring so that ties are
      broken deterministically (rather than depending on dict iteration order).
    - When two categories have similarity scores within 0.03 of each other,
      keyword overlap is used as a secondary tie-breaker to improve stability
      across runs.

    Args:
        df: DataFrame with a ``Combined_Text`` column (produced by
            ``prepare_text()`` in the orchestrator).
        ai: An ``OllamaBrain`` instance for embedding generation.
        show_progress: Whether to render a tqdm progress bar.

    Returns:
        The same DataFrame with ``AI_Category``, ``AI_Confidence``, and
        ``AI_Sub_Category`` columns added.
    """
    logger.info("[AI Engine] Categorizing rows using Hybrid Classification...")
    df = df.copy()

    # Compute (or retrieve cached) anchor centroids for each category
    anchor_centroids = get_anchor_centroids(ai)

    # ------------------------------------------------------------------
    # Batch-embed all ticket texts in one call for throughput.
    # This avoids the per-ticket API round-trip overhead.
    # ------------------------------------------------------------------
    logger.info(f"Computing embeddings for {len(df)} rows...")
    all_texts = df['Combined_Text'].tolist()
    all_vecs = ai.get_embeddings_batch(all_texts)

    cats = []      # Predicted category per row
    scores = []    # Confidence score per row
    methods = []   # Classification method used (for debugging / analytics)

    # Sort categories alphabetically for deterministic tie-breaking
    sorted_categories = sorted(anchor_centroids.keys())

    # Counters for logging classification method distribution
    pattern_count = 0
    keyword_count = 0
    embedding_count = 0

    # ------------------------------------------------------------------
    # Iterate over each ticket and apply the three-tier cascade.
    # ------------------------------------------------------------------
    for idx, vec in enumerate(tqdm(all_vecs, desc="  Classifying tickets", unit="ticket",
                                   disable=not show_progress, ncols=80,
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        text = all_texts[idx]

        # ---- Tier 1: Pattern-based classification (fastest, highest confidence) ----
        pattern_result = pattern_classify(text)
        if pattern_result:
            cat, conf = pattern_result
            if cat in anchor_centroids:
                cats.append(cat)
                scores.append(conf)
                methods.append("pattern")
                pattern_count += 1
                continue

        # ---- Tier 2: Keyword-based classification ----
        keyword_result = keyword_classify(text)
        if keyword_result:
            cat, conf = keyword_result
            if cat in anchor_centroids:
                # Cross-validate: the keyword match must have at least
                # minimal embedding support (cosine sim >= 0.25) to prevent
                # keyword-only false positives.
                sim = cosine_similarity_gpu(vec, anchor_centroids[cat], use_gpu=_USE_GPU)
                if sim >= 0.25:
                    # Use the higher of keyword confidence or embedding similarity
                    final_conf = max(conf, sim)
                    cats.append(cat)
                    scores.append(final_conf)
                    methods.append("keyword")
                    keyword_count += 1
                    continue

        # ---- Tier 3: Embedding-based classification (semantic) ----
        # Compute cosine similarity between this ticket's embedding and
        # every anchor centroid.
        category_scores = []
        for cat in sorted_categories:
            anchor_vec = anchor_centroids[cat]
            sim = cosine_similarity_gpu(vec, anchor_vec, use_gpu=_USE_GPU)
            category_scores.append((cat, sim))

        # Sort by score descending; alphabetical name as secondary key
        # ensures deterministic output when scores are identical.
        category_scores.sort(key=lambda x: (-x[1], x[0]))

        best_cat, best_score = category_scores[0]
        second_cat, second_score = category_scores[1] if len(category_scores) > 1 else ("", 0.0)

        # ------------------------------------------------------------------
        # Tie-breaking: if the top two categories are within 0.03 similarity
        # (essentially a statistical tie), use keyword overlap as a secondary
        # signal to choose the more appropriate category.
        # ------------------------------------------------------------------
        if second_cat and second_score > 0 and (best_score - second_score) < 0.03:
            text_lower = str(text).lower()

            # Count keyword matches for the best-scoring category
            best_keywords = 0
            second_keywords = 0

            if best_cat in CATEGORY_KEYWORDS:
                best_kw = CATEGORY_KEYWORDS[best_cat]
                best_keywords = sum(1 for kw in best_kw.get("primary", []) if kw.lower() in text_lower)
                best_keywords += sum(2 for p in best_kw.get("phrases", []) if p.lower() in text_lower)

            if second_cat in CATEGORY_KEYWORDS:
                second_kw = CATEGORY_KEYWORDS[second_cat]
                second_keywords = sum(1 for kw in second_kw.get("primary", []) if kw.lower() in text_lower)
                second_keywords += sum(2 for p in second_kw.get("phrases", []) if p.lower() in text_lower)

            # If the runner-up has stronger keyword support, promote it
            if second_keywords > best_keywords:
                best_cat, best_score = second_cat, second_score

        # Apply minimum confidence threshold -- tickets below this are
        # labelled "Unclassified" rather than forced into a low-confidence bin.
        if best_score < MIN_CLASSIFICATION_CONFIDENCE:
            best_cat = "Unclassified"

        cats.append(best_cat)
        scores.append(best_score)
        methods.append("embedding")
        embedding_count += 1

    # Write classification results back to the DataFrame
    df['AI_Category'] = cats
    df['AI_Confidence'] = scores

    # ------------------------------------------------------------------
    # Sub-category assignment: a second pass that refines each main
    # category assignment into a more granular label using keyword matching
    # against the SUB_CATEGORIES config.
    # ------------------------------------------------------------------
    logger.info("Assigning sub-categories...")
    sub_cats = []
    for idx, row in df.iterrows():
        text = all_texts[idx] if idx < len(all_texts) else ""
        category = cats[idx] if idx < len(cats) else "Unclassified"
        sub_cat = get_sub_category(text, category)
        sub_cats.append(sub_cat)

    df['AI_Sub_Category'] = sub_cats

    # Log classification method distribution for diagnostics
    logger.info(f"Classification complete:")
    logger.info(f"  - Pattern matches: {pattern_count}")
    logger.info(f"  - Keyword matches: {keyword_count}")
    logger.info(f"  - Embedding matches: {embedding_count}")
    logger.info(f"Category distribution: {df['AI_Category'].value_counts().to_dict()}")
    logger.info(f"Sub-category distribution: {df['AI_Sub_Category'].value_counts().head(10).to_dict()}")

    return df


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_centroid_cache():
    """Clear the centroid cache to force recomputation.

    This should be called when anchor phrases or feedback data have changed
    and the centroids need to be recomputed on the next classification run.
    """
    global _anchor_centroids_cache
    _anchor_centroids_cache.clear()
    logger.info("Cleared anchor centroids cache")
