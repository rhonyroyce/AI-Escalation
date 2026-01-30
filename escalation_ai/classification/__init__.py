"""
AI Classification Engine for escalation categorization.
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
    ANCHORS, CATEGORY_KEYWORDS, MIN_CLASSIFICATION_CONFIDENCE, USE_GPU
)
from ..core.gpu_utils import (
    batch_cosine_similarity_gpu,
    cosine_similarity_gpu,
    is_gpu_available
)
from ..feedback.feedback_learning import get_feedback_learner

logger = logging.getLogger(__name__)

# Check GPU availability at module load
_USE_GPU = USE_GPU and is_gpu_available()
if _USE_GPU:
    logger.info("[Classification] GPU acceleration enabled")

# Cache for anchor centroids
_anchor_centroids_cache: Dict[str, np.ndarray] = {}

# Compile regex patterns for faster matching
_COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {}


def _compile_patterns():
    """Compile regex patterns from CATEGORY_KEYWORDS for faster matching."""
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS:
        return _COMPILED_PATTERNS

    for category, keywords in CATEGORY_KEYWORDS.items():
        patterns = keywords.get("patterns", [])
        _COMPILED_PATTERNS[category] = [
            re.compile(p, re.IGNORECASE) for p in patterns
        ]

    return _COMPILED_PATTERNS


def get_anchor_centroids(ai) -> Dict[str, np.ndarray]:
    """Get or compute anchor centroids with caching and feedback adjustment."""
    global _anchor_centroids_cache

    if _anchor_centroids_cache:
        logger.info("Using cached anchor centroids...")
        return _anchor_centroids_cache

    logger.info("Computing anchor centroids for 8-category system (will be cached)...")

    # Use CATEGORY_KEYWORDS for richer phrase set
    all_phrases = []
    phrase_to_category = []

    for category in ANCHORS.keys():
        # Combine all keyword types for embedding
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

    # Single batch call for all anchor phrases
    logger.info(f"  Computing embeddings for {len(all_phrases)} anchor phrases...")
    all_vecs = ai.get_embeddings_batch(all_phrases)

    # Group by category and compute centroids
    category_vecs: Dict[str, List[np.ndarray]] = {cat: [] for cat in ANCHORS.keys()}
    for phrase, category, vec in zip(all_phrases, phrase_to_category, all_vecs):
        if vec is not None:
            category_vecs[category].append(vec)

    for category, vecs in category_vecs.items():
        if vecs:
            _anchor_centroids_cache[category] = np.mean(vecs, axis=0)
        else:
            logger.warning(f"No embeddings for category: {category}")

    logger.info(f"Computed centroids for {len(_anchor_centroids_cache)} categories")

    # Apply feedback adjustments if available
    feedback_learner = get_feedback_learner()
    if feedback_learner.stats['loaded'] > 0 or feedback_learner.stats['custom_categories'] > 0:
        logger.info("Applying user feedback adjustments to centroids...")
        adjusted = feedback_learner.adjust_centroids(_anchor_centroids_cache)
        _anchor_centroids_cache.clear()
        _anchor_centroids_cache.update(adjusted)
        logger.info(f"Final category count: {len(_anchor_centroids_cache)}")

    return _anchor_centroids_cache


def pattern_classify(text: str) -> Optional[Tuple[str, float]]:
    """
    Check if text matches any regex patterns for high-confidence classification.

    Returns (category, confidence) or None if no pattern match.
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


def keyword_classify(text: str) -> Optional[Tuple[str, float]]:
    """
    Check if text contains category-specific phrases or keywords.

    Uses the extended CATEGORY_KEYWORDS taxonomy for matching.
    Returns (category, confidence) or None if no strong keyword match.
    """
    if not text:
        return None

    text_lower = str(text).lower()
    best_match = None
    best_score = 0

    for category, kw_data in CATEGORY_KEYWORDS.items():
        score = 0

        # Check primary keywords (lower weight)
        primary = kw_data.get("primary", [])
        for kw in primary:
            if kw.lower() in text_lower:
                score += 1

        # Check phrases (higher weight)
        phrases = kw_data.get("phrases", [])
        for phrase in phrases:
            if phrase.lower() in text_lower:
                score += 3  # Phrases are more specific, weighted higher

        if score > best_score:
            best_score = score
            best_match = category

    # Only return if we have a strong enough match
    if best_score >= 5:  # At least 5 points (e.g., 1 phrase + 2 keywords)
        confidence = min(0.85, 0.5 + (best_score * 0.05))  # Scale confidence
        return (best_match, confidence)

    return None


def classify_rows(df: pd.DataFrame, ai, show_progress: bool = True) -> pd.DataFrame:
    """
    Categorize rows using hybrid keyword + embedding classification.

    Three-tier classification approach:
    1. Regex pattern matching (highest confidence, fastest)
    2. Keyword/phrase matching (high confidence)
    3. Embedding similarity (semantic understanding)

    Includes consistency improvements:
    - Tie-breaking for close similarity scores
    - Stable sorting of categories for deterministic results
    """
    logger.info("[AI Engine] Categorizing rows using Hybrid Classification...")
    df = df.copy()

    # Get anchor centroids (cached after first call)
    anchor_centroids = get_anchor_centroids(ai)

    # Batch compute embeddings for all texts
    logger.info(f"Computing embeddings for {len(df)} rows...")
    all_texts = df['Combined_Text'].tolist()
    all_vecs = ai.get_embeddings_batch(all_texts)

    cats = []
    scores = []
    methods = []  # Track classification method for debugging

    # Sort categories alphabetically for deterministic tie-breaking
    sorted_categories = sorted(anchor_centroids.keys())

    # Classification counters
    pattern_count = 0
    keyword_count = 0
    embedding_count = 0

    # Classify each row
    for idx, vec in enumerate(tqdm(all_vecs, desc="  Classifying tickets", unit="ticket",
                                   disable=not show_progress, ncols=80,
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        text = all_texts[idx]

        # Tier 1: Pattern-based classification (fastest, highest confidence)
        pattern_result = pattern_classify(text)
        if pattern_result:
            cat, conf = pattern_result
            if cat in anchor_centroids:
                cats.append(cat)
                scores.append(conf)
                methods.append("pattern")
                pattern_count += 1
                continue

        # Tier 2: Keyword-based classification
        keyword_result = keyword_classify(text)
        if keyword_result:
            cat, conf = keyword_result
            if cat in anchor_centroids:
                # Verify with embedding (must be at least 0.25 similar)
                sim = cosine_similarity_gpu(vec, anchor_centroids[cat], use_gpu=_USE_GPU)
                if sim >= 0.25:
                    # Use higher of keyword confidence or embedding similarity
                    final_conf = max(conf, sim)
                    cats.append(cat)
                    scores.append(final_conf)
                    methods.append("keyword")
                    keyword_count += 1
                    continue

        # Tier 3: Embedding-based classification
        category_scores = []
        for cat in sorted_categories:
            anchor_vec = anchor_centroids[cat]
            sim = cosine_similarity_gpu(vec, anchor_vec, use_gpu=_USE_GPU)
            category_scores.append((cat, sim))

        # Sort by score descending, then by category name for tie-breaking
        category_scores.sort(key=lambda x: (-x[1], x[0]))

        best_cat, best_score = category_scores[0]
        second_cat, second_score = category_scores[1] if len(category_scores) > 1 else ("", 0.0)

        # Check for close scores (within 0.03 = essentially a tie)
        if second_cat and second_score > 0 and (best_score - second_score) < 0.03:
            # Use keyword overlap as tie-breaker
            text_lower = str(text).lower()

            # Count keyword matches for each category
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

            if second_keywords > best_keywords:
                best_cat, best_score = second_cat, second_score

        # Apply minimum confidence threshold
        if best_score < MIN_CLASSIFICATION_CONFIDENCE:
            best_cat = "Unclassified"

        cats.append(best_cat)
        scores.append(best_score)
        methods.append("embedding")
        embedding_count += 1

    df['AI_Category'] = cats
    df['AI_Confidence'] = scores

    # Log classification method distribution
    logger.info(f"Classification complete:")
    logger.info(f"  - Pattern matches: {pattern_count}")
    logger.info(f"  - Keyword matches: {keyword_count}")
    logger.info(f"  - Embedding matches: {embedding_count}")
    logger.info(f"Category distribution: {df['AI_Category'].value_counts().to_dict()}")

    return df


def clear_centroid_cache():
    """Clear the centroid cache to force recomputation."""
    global _anchor_centroids_cache
    _anchor_centroids_cache.clear()
    logger.info("Cleared anchor centroids cache")
