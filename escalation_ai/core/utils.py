"""
Utility functions for text processing and data validation.
"""

import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clean_text(text):
    """Clean and normalize text for processing"""
    if pd.isna(text):
        return ""
    text = str(text).replace("_x000D_", " ").replace("\n", " ")
    return re.sub(r'\s+', ' ', text).strip()


def validate_columns(df, required_cols):
    """Validate that required columns exist in dataframe"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}. Some features may be limited.")
        return False
    return True


def extract_keywords(text):
    """Extract meaningful keywords from text for overlap comparison"""
    if pd.isna(text) or not text:
        return set()
    
    # Remove common stop words and extract key terms
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'this', 'that', 'these', 'those', 'it'
    }
    
    # Tokenize and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
    keywords = {w for w in words if w not in stop_words}
    return keywords


def calculate_keyword_overlap(text1, text2):
    """Calculate Jaccard similarity of keywords between two texts"""
    kw1 = extract_keywords(text1)
    kw2 = extract_keywords(text2)
    
    if not kw1 or not kw2:
        return 0.0
    
    intersection = len(kw1 & kw2)
    union = len(kw1 | kw2)
    
    return intersection / union if union > 0 else 0.0


def enrich_text_for_embedding(text, category=None):
    """Expand text with contextual information for better embedding"""
    if pd.isna(text) or not text:
        return ""
    
    enriched = str(text)
    
    # Add category context if available
    if category and not pd.isna(category):
        enriched = f"{category}: {enriched}"
    
    # Expand common telecom abbreviations
    expansions = {
        'rru': 'remote radio unit equipment',
        'bbu': 'baseband unit equipment', 
        'oss': 'operations support system',
        'nms': 'network management system',
        'ran': 'radio access network',
        'lte': 'long term evolution cellular',
        'nr': 'new radio 5g',
        'mw': 'microwave transmission',
        'tx': 'transmission',
        'rx': 'receive reception',
        'rf': 'radio frequency',
        'vswr': 'voltage standing wave ratio antenna',
        'gps': 'global positioning system synchronization',
        'ptp': 'precision time protocol synchronization'
    }
    
    text_lower = enriched.lower()
    for abbrev, expansion in expansions.items():
        if abbrev in text_lower:
            enriched = f"{enriched} ({expansion})"
    
    return enriched
