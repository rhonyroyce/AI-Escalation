"""
AI Classification Engine for escalation categorization.
Uses embeddings and anchor-based similarity for classification.

GPU-accelerated with RAPIDS cuML when available.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

from ..core.config import ANCHORS, MIN_CLASSIFICATION_CONFIDENCE, USE_GPU
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

# Keyword patterns for deterministic pre-classification
KEYWORD_OVERRIDES = {
    "OSS/NMS & Systems": [
        r'\bnest(?:ing|ed)\b', r'\bnsi\b', r'\bsi\s+nest', r'\bcell\s*planning\b',
        r'\bpci\s*conflict\b', r'\bprovision', r'\binventory\b', r'\boss\b', r'\bnms\b'
    ],
    "RF & Antenna Issues": [
        r'\bantenna\b', r'\bvswr\b', r'\brru\b', r'\bradio\b', r'\bsector\b',
        r'\bcell\s*down\b', r'\boutage\b', r'\brf\b', r'\bbaseband\b'
    ],
    "Transmission & Backhaul": [
        r'\bfiber\b', r'\bmicrowave\b', r'\btransmission\b', r'\bbackhaul\b',
        r'\bcircuit\b', r'\blatency\b', r'\bpacket\s*loss\b'
    ],
    "Power & Environment": [
        r'\bpower\b', r'\bbattery\b', r'\bgenerator\b', r'\brectifier\b',
        r'\bac\s*fail', r'\bcooling\b', r'\btemperature\b'
    ],
    "Site Access & Logistics": [
        r'\baccess\b.*\bden', r'\bkey\b', r'\bgate\b', r'\blandlord\b',
        r'\bpermit\b', r'\bescort\b', r'\binaccessible\b'
    ],
}


def get_anchor_centroids(ai) -> Dict[str, np.ndarray]:
    """Get or compute anchor centroids with caching and feedback adjustment."""
    global _anchor_centroids_cache
    
    if _anchor_centroids_cache:
        logger.info("Using cached anchor centroids...")
        return _anchor_centroids_cache
    
    logger.info("Computing anchor centroids for categories (will be cached)...")
    
    # Batch all anchor phrases for efficiency
    all_phrases = []
    phrase_to_category = []
    for category, phrases in ANCHORS.items():
        for phrase in phrases:
            all_phrases.append(phrase)
            phrase_to_category.append(category)
    
    # Single batch call for all anchor phrases
    all_vecs = ai.get_embeddings_batch(all_phrases)
    
    # Group by category and compute centroids
    category_vecs: Dict[str, List[np.ndarray]] = {cat: [] for cat in ANCHORS.keys()}
    for phrase, category, vec in zip(all_phrases, phrase_to_category, all_vecs):
        category_vecs[category].append(vec)
    
    for category, vecs in category_vecs.items():
        _anchor_centroids_cache[category] = np.mean(vecs, axis=0)
    
    logger.info(f"Computed centroids for {len(_anchor_centroids_cache)} categories")
    
    # Apply feedback adjustments if available
    feedback_learner = get_feedback_learner()
    if feedback_learner.stats['loaded'] > 0:
        logger.info("Applying user feedback adjustments to centroids...")
        adjusted = feedback_learner.adjust_centroids(_anchor_centroids_cache)
        _anchor_centroids_cache.clear()
        _anchor_centroids_cache.update(adjusted)
        logger.info(f"Final category count: {len(_anchor_centroids_cache)}")
    
    return _anchor_centroids_cache


def keyword_classify(text: str) -> Optional[str]:
    """Check if text matches any keyword patterns for pre-classification."""
    if not text:
        return None
    text_lower = str(text).lower()
    
    for category, patterns in KEYWORD_OVERRIDES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category
    return None


def classify_rows(df: pd.DataFrame, ai) -> pd.DataFrame:
    """
    Categorize rows using AI embeddings and anchor-based similarity.
    
    Includes consistency improvements:
    - Keyword-based pre-classification for known patterns
    - Tie-breaking for close similarity scores
    - Stable sorting of categories to ensure deterministic results
    """
    logger.info("[AI Engine] Categorizing rows using Embeddings...")
    df = df.copy()
    
    # Get anchor centroids (cached after first call)
    anchor_centroids = get_anchor_centroids(ai)
    
    # Batch compute embeddings for all texts
    logger.info(f"Computing embeddings for {len(df)} rows...")
    all_texts = df['Combined_Text'].tolist()
    all_vecs = ai.get_embeddings_batch(all_texts)
    
    cats = []
    scores = []
    
    # Sort categories alphabetically for deterministic tie-breaking
    sorted_categories = sorted(anchor_centroids.keys())
    
    # Classify each row
    for idx, vec in enumerate(tqdm(all_vecs, desc="   > Classifying")):
        text = all_texts[idx]
        
        # Step 1: Try keyword-based classification first
        keyword_cat = keyword_classify(text)
        if keyword_cat:
            # Verify with embedding (must be at least 0.3 similar)
            sim = cosine_similarity_gpu(vec, anchor_centroids[keyword_cat], use_gpu=_USE_GPU)
            if sim >= 0.3:
                cats.append(keyword_cat)
                scores.append(sim)
                continue
        
        # Step 2: Embedding-based classification
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
            best_keywords = sum(1 for kw in ANCHORS.get(best_cat, []) if kw in text_lower)
            second_keywords = sum(1 for kw in ANCHORS.get(second_cat, []) if kw in text_lower)
            
            if second_keywords > best_keywords:
                best_cat, best_score = second_cat, second_score
        
        # Apply minimum confidence threshold
        if best_score < MIN_CLASSIFICATION_CONFIDENCE:
            best_cat = "Unclassified"
        
        cats.append(best_cat)
        scores.append(best_score)
    
    df['AI_Category'] = cats
    df['AI_Confidence'] = scores
    
    # Log category distribution
    logger.info(f"Classification complete. Distribution: {df['AI_Category'].value_counts().to_dict()}")
    
    return df


def clear_centroid_cache():
    """Clear the centroid cache to force recomputation."""
    global _anchor_centroids_cache
    _anchor_centroids_cache.clear()
    logger.info("Cleared anchor centroids cache")
