"""
FAISS-backed Vector Store for O(n log n) similarity search.

Replaces the O(n^2) pairwise cosine similarity matrix used in Phase 3
(recidivism detection) and Phase 5 (similar ticket analysis) with
approximate nearest neighbor search via Facebook AI Similarity Search (FAISS).

Architecture
------------
- For small datasets (<10K tickets): ``faiss.IndexFlatIP`` (exact inner product
  on L2-normalized vectors = cosine similarity).
- For larger datasets (>=10K tickets): ``faiss.IndexIVFFlat`` with nlist=100
  clusters for approximate search with ~5% recall trade-off but 10-50x speedup.

All vectors are L2-normalized before insertion so that inner product equals
cosine similarity.  This avoids maintaining a separate normalization step at
query time.

Persistence
-----------
The FAISS index and its ticket-ID mapping are saved to disk so that subsequent
pipeline runs can skip re-indexing unchanged tickets::

    .cache/ticket_vectors.faiss   — the FAISS index binary
    .cache/ticket_vectors_ids.json — ordered list of ticket IDs

Fallback
--------
If ``faiss`` is not installed, the module exposes ``FAISS_AVAILABLE = False``
and all callers should fall back to the existing sklearn pairwise approach.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "faiss not installed — falling back to O(n^2) pairwise similarity. "
        "Install with: pip install faiss-cpu  (or faiss-gpu for CUDA support)"
    )


class TicketVectorStore:
    """FAISS-backed vector store for O(n log n) similarity search.

    Manages a FAISS index of L2-normalized ticket embedding vectors and
    provides methods for adding embeddings, searching for nearest neighbors,
    finding clusters of similar tickets, and persisting/loading the index.

    Attributes:
        dimension: Embedding vector dimensionality (e.g. 768).
        index: The underlying FAISS index (IndexFlatIP or IndexIVFFlat).
        ticket_ids: Ordered list of ticket IDs matching the index rows.
        _ivf_threshold: Number of vectors above which IVF indexing is used.
    """

    _ivf_threshold: int = 10_000

    def __init__(self, dimension: int = 768) -> None:
        """Initialize FAISS index.

        Args:
            dimension: Embedding vector dimensionality.  Must match the
                dimension of vectors passed to ``add_embeddings``.
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for TicketVectorStore. "
                "Install with: pip install faiss-cpu"
            )
        self.dimension = dimension
        # Start with exact inner-product index; upgrade to IVF if needed
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.ticket_ids: list[str] = []

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    # ------------------------------------------------------------------
    # Add / build
    # ------------------------------------------------------------------

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ticket_ids: list[str],
    ) -> None:
        """Add ticket embeddings to the index.  Supports incremental adds.

        Vectors are L2-normalized before insertion so that inner product
        equals cosine similarity.

        If the total number of vectors exceeds ``_ivf_threshold`` after this
        call and the index is still a flat index, it is automatically
        upgraded to an IVF index for faster search.

        Args:
            embeddings: Array of shape ``(n, dimension)`` with dtype float32.
            ticket_ids: List of ``n`` ticket identifiers matching the rows
                of ``embeddings``.

        Raises:
            ValueError: If ``embeddings`` shape doesn't match ``dimension``
                or lengths mismatch.
        """
        if len(embeddings) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != "
                f"expected {self.dimension}"
            )
        if len(ticket_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Got {len(ticket_ids)} ticket_ids for "
                f"{embeddings.shape[0]} embeddings"
            )

        # L2-normalize so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        normalized = embeddings / norms

        self.index.add(normalized)
        self.ticket_ids.extend(ticket_ids)

        # Upgrade to IVF if we've crossed the threshold
        if (
            self.size >= self._ivf_threshold
            and not isinstance(self.index, faiss.IndexIVFFlat)
        ):
            self._upgrade_to_ivf(normalized)

    def _upgrade_to_ivf(self, sample_vectors: np.ndarray) -> None:
        """Replace the flat index with a trained IVF index for faster search.

        Reconstructs all vectors from the existing flat index, builds an
        IVF index with 100 Voronoi cells, and re-adds everything.

        Args:
            sample_vectors: Recent batch of vectors (used only as fallback
                training data if reconstruction fails).
        """
        nlist = min(100, self.size // 10)
        if nlist < 2:
            return  # not enough data for IVF

        logger.info(
            f"Upgrading FAISS index to IVF (nlist={nlist}, "
            f"vectors={self.size})"
        )

        # Reconstruct all vectors from the flat index
        all_vectors = np.zeros(
            (self.size, self.dimension), dtype=np.float32
        )
        for i in range(self.size):
            all_vectors[i] = self.index.reconstruct(i)

        quantizer = faiss.IndexFlatIP(self.dimension)
        ivf_index = faiss.IndexIVFFlat(
            quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )
        ivf_index.train(all_vectors)
        ivf_index.add(all_vectors)
        # Search more cells for better recall
        ivf_index.nprobe = min(10, nlist)

        self.index = ivf_index

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Find k most similar tickets to the query embedding.

        The query vector is L2-normalized before search so that the inner
        product scores returned by FAISS equal cosine similarity.

        Args:
            query_embedding: 1-D array of shape ``(dimension,)``.
            k: Number of nearest neighbors to return.
            exclude_ids: Optional set of ticket IDs to exclude from results
                (e.g. to prevent self-matching).

        Returns:
            List of ``(ticket_id, similarity_score)`` tuples sorted by
            descending similarity.  May return fewer than ``k`` results if
            the index is small or exclusions remove candidates.
        """
        if self.size == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        # L2-normalize the query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Request extra results to compensate for exclusions
        search_k = min(k + len(exclude_ids or []) + 5, self.size)
        distances, indices = self.index.search(query, search_k)

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.ticket_ids):
                continue  # FAISS returns -1 for missing results
            tid = self.ticket_ids[idx]
            if exclude_ids and tid in exclude_ids:
                continue
            results.append((tid, float(dist)))
            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------
    # Cluster detection (for recidivism)
    # ------------------------------------------------------------------

    def find_nearest_neighbor(
        self, idx: int
    ) -> tuple[Optional[str], float]:
        """Find the single nearest neighbor for the vector at position ``idx``.

        Used by Phase 3 recidivism detection: for each ticket, find its
        closest match and record the similarity score.

        Args:
            idx: Position in the index (0-based).

        Returns:
            ``(ticket_id, similarity)`` of the nearest neighbor, or
            ``(None, 0.0)`` if the index has fewer than 2 vectors.
        """
        if self.size < 2:
            return None, 0.0

        # Reconstruct the vector at this position
        query = self.index.reconstruct(idx).reshape(1, -1)
        # Search for 2 (self + nearest neighbor)
        distances, indices = self.index.search(query, 2)

        for dist, nbr_idx in zip(distances[0], indices[0]):
            if nbr_idx < 0 or nbr_idx == idx:
                continue
            if nbr_idx < len(self.ticket_ids):
                return self.ticket_ids[nbr_idx], float(dist)

        return None, 0.0

    def find_clusters(
        self, threshold: float = 0.85
    ) -> list[list[str]]:
        """Find clusters of similar tickets for recidivism detection.

        Uses a simple single-linkage approach: for each vector, find all
        neighbors above ``threshold`` and merge their clusters.

        Args:
            threshold: Minimum cosine similarity to consider two tickets
                as belonging to the same cluster.

        Returns:
            List of clusters, where each cluster is a list of ticket IDs.
            Singletons (tickets with no similar neighbors) are excluded.
        """
        if self.size < 2:
            return []

        # Union-Find for clustering
        parent = list(range(self.size))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # For each vector, find neighbors above threshold
        search_k = min(20, self.size)
        for i in range(self.size):
            query = self.index.reconstruct(i).reshape(1, -1)
            distances, indices = self.index.search(query, search_k)

            for dist, j in zip(distances[0], indices[0]):
                if j < 0 or j == i:
                    continue
                if float(dist) >= threshold:
                    union(i, j)

        # Collect clusters
        clusters: dict[int, list[str]] = {}
        for i in range(self.size):
            root = find(i)
            clusters.setdefault(root, []).append(self.ticket_ids[i])

        # Return only multi-member clusters
        return [c for c in clusters.values() if len(c) > 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist index and ticket ID mapping to disk.

        Creates two files:
        - ``{path}`` — the FAISS index binary
        - ``{path}_ids.json`` — the ordered ticket ID list

        Args:
            path: File path for the FAISS index (e.g.
                ``.cache/ticket_vectors.faiss``).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path)

        ids_path = path.replace(".faiss", "_ids.json")
        with open(ids_path, "w") as f:
            json.dump(self.ticket_ids, f)

        logger.info(
            f"Saved FAISS index ({self.size} vectors) to {path}"
        )

    def load(self, path: str) -> None:
        """Load index and ticket ID mapping from disk.

        Args:
            path: File path to the FAISS index file.

        Raises:
            FileNotFoundError: If the index or ID file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found: {path}")

        ids_path = path.replace(".faiss", "_ids.json")
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Ticket ID mapping not found: {ids_path}")

        self.index = faiss.read_index(path)
        self.dimension = self.index.d

        with open(ids_path, "r") as f:
            self.ticket_ids = json.load(f)

        logger.info(
            f"Loaded FAISS index ({self.size} vectors) from {path}"
        )
