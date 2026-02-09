"""
Pulse Dashboard - Ollama AI Integration

Self-contained Ollama client for:
- Text generation (qwen3:14b)
- Embeddings (qwen2:1.5b)
- Semantic search
- Graceful degradation when Ollama is unavailable
"""

import re
import requests
import numpy as np
from typing import List, Optional

OLLAMA_BASE_URL = 'http://localhost:11434'
CHAT_MODEL = 'qwen3:14b'
EMBED_MODEL = 'qwen2:1.5b'


def check_ollama() -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def ollama_generate(prompt: str, temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
    """Generate text using Ollama chat model.

    Returns cleaned response text, or None on failure.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 4096,
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            text = resp.json().get('response', '')
            return strip_thinking_tags(text)
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def ollama_embed(text: str) -> np.ndarray:
    """Get embedding vector for a single text. Returns zero vector on failure."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": str(text)},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            vec = data.get('embedding') or data.get('embeddings', [[]])[0]
            return np.array(vec)
    except Exception:
        pass
    return np.zeros(384)  # Default dim for small models


def ollama_embed_batch(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """Batch embed multiple texts."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": batch},
                timeout=120,
            )
            if resp.status_code == 200:
                embeddings = resp.json().get('embeddings', [])
                results.extend([np.array(e) for e in embeddings])
            else:
                results.extend([np.zeros(384) for _ in batch])
        except Exception:
            results.extend([np.zeros(384) for _ in batch])
    return results


def build_embeddings_index(df, columns=None) -> dict:
    """Build embeddings index for semantic search.

    Returns:
        dict with keys: embeddings (np.array), metadata (list), texts (list)
    """
    if columns is None:
        columns = ['Comments', 'Pain Points']

    texts = []
    metadata = []

    for col in columns:
        if col not in df.columns:
            continue
        for idx, row in df.iterrows():
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            text = str(val).strip()
            if len(text) < 5:
                continue
            texts.append(text[:500])
            metadata.append({
                'index': idx,
                'column': col,
                'project': row.get('Project', ''),
                'region': row.get('Region', ''),
                'area': row.get('Area', ''),
                'pm': row.get('PM Name', ''),
                'score': row.get('Total Score', 0),
            })

    if not texts:
        return {'embeddings': np.array([]), 'metadata': [], 'texts': []}

    embeddings = ollama_embed_batch(texts)
    return {
        'embeddings': np.array(embeddings),
        'metadata': metadata,
        'texts': texts,
    }


def semantic_search(query: str, index: dict, top_k: int = 5) -> list:
    """Search embeddings index for similar texts."""
    if index['embeddings'].size == 0:
        return []

    query_emb = ollama_embed(query)
    if query_emb.sum() == 0:
        return []

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_emb], index['embeddings'])[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
            'similarity': float(similarities[i]),
            'text': index['texts'][i],
            **index['metadata'][i],
        })
    return results
