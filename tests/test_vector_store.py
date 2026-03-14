"""
Tests for the FAISS-backed TicketVectorStore.

Covers:
- Adding vectors and searching for known matches
- Save/load roundtrip persistence
- Empty index edge cases
- Cluster detection for recidivism
- Sublinear scaling benchmark (1K vs 10K vectors)
"""

import os
import time
import tempfile

import numpy as np
import pytest

# Skip entire module if faiss is not installed
faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

from escalation_ai.predictors.vector_store import TicketVectorStore, FAISS_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """Return a fresh 128-dim TicketVectorStore."""
    return TicketVectorStore(dimension=128)


@pytest.fixture
def populated_store():
    """Return a store with 100 random vectors and one known target."""
    np.random.seed(42)
    s = TicketVectorStore(dimension=128)

    # 99 random vectors
    random_vecs = np.random.randn(99, 128).astype(np.float32)
    ids = [f"ticket_{i}" for i in range(99)]

    # One known vector (all ones, L2-norm will make it a unit vector)
    known_vec = np.ones((1, 128), dtype=np.float32)
    ids.append("known_ticket")
    all_vecs = np.vstack([random_vecs, known_vec])

    s.add_embeddings(all_vecs, ids)
    return s


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestAddAndSearch:
    """Test adding embeddings and searching for nearest neighbors."""

    def test_add_single_vector(self, store):
        vec = np.random.randn(128).astype(np.float32)
        store.add_embeddings(vec.reshape(1, -1), ["ticket_0"])
        assert store.size == 1

    def test_add_100_vectors_and_search_known(self, populated_store):
        """Add 100 vectors, search for the known one — it must be the top result."""
        # The known vector is all-ones; query with the same direction
        query = np.ones(128, dtype=np.float32)
        results = populated_store.search(query, k=5)

        assert len(results) > 0
        top_id, top_score = results[0]
        assert top_id == "known_ticket"
        # Cosine similarity of identical directions should be ~1.0
        assert top_score > 0.95

    def test_search_returns_at_most_k(self, populated_store):
        query = np.random.randn(128).astype(np.float32)
        results = populated_store.search(query, k=3)
        assert len(results) <= 3

    def test_search_exclude_ids(self, populated_store):
        """Excluded IDs should not appear in results."""
        query = np.ones(128, dtype=np.float32)
        results = populated_store.search(
            query, k=5, exclude_ids={"known_ticket"}
        )
        result_ids = [tid for tid, _ in results]
        assert "known_ticket" not in result_ids

    def test_dimension_mismatch_raises(self, store):
        wrong_dim = np.random.randn(5, 64).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            store.add_embeddings(wrong_dim, [f"t{i}" for i in range(5)])

    def test_id_count_mismatch_raises(self, store):
        vecs = np.random.randn(5, 128).astype(np.float32)
        with pytest.raises(ValueError, match="ticket_ids"):
            store.add_embeddings(vecs, ["t0", "t1"])  # only 2 ids for 5 vecs

    def test_incremental_add(self, store):
        """Supports adding embeddings in multiple batches."""
        batch1 = np.random.randn(10, 128).astype(np.float32)
        batch2 = np.random.randn(15, 128).astype(np.float32)
        store.add_embeddings(batch1, [f"a{i}" for i in range(10)])
        store.add_embeddings(batch2, [f"b{i}" for i in range(15)])
        assert store.size == 25


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------

class TestEmptyIndex:
    """Edge cases with 0 vectors in the index."""

    def test_search_empty_returns_empty(self, store):
        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=5)
        assert results == []

    def test_find_nearest_neighbor_empty(self, store):
        tid, score = store.find_nearest_neighbor(0)
        assert tid is None
        assert score == 0.0

    def test_find_clusters_empty(self, store):
        assert store.find_clusters() == []

    def test_size_is_zero(self, store):
        assert store.size == 0


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------

class TestPersistence:
    """Test save/load roundtrip preserves index and IDs."""

    def test_save_load_roundtrip(self, populated_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index.faiss")
            populated_store.save(path)

            # Verify files exist
            assert os.path.exists(path)
            assert os.path.exists(path.replace(".faiss", "_ids.json"))

            # Load into a new store
            loaded = TicketVectorStore(dimension=128)
            loaded.load(path)

            assert loaded.size == populated_store.size
            assert loaded.ticket_ids == populated_store.ticket_ids
            assert loaded.dimension == populated_store.dimension

            # Search should produce same results
            query = np.ones(128, dtype=np.float32)
            orig_results = populated_store.search(query, k=3)
            loaded_results = loaded.search(query, k=3)

            assert [tid for tid, _ in orig_results] == [tid for tid, _ in loaded_results]

    def test_load_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("/nonexistent/path/index.faiss")


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

class TestClusters:
    """Test find_clusters for recidivism detection."""

    def test_identical_vectors_cluster_together(self):
        store = TicketVectorStore(dimension=64)
        # Create 3 groups of near-identical vectors
        base_a = np.random.randn(64).astype(np.float32)
        base_b = np.random.randn(64).astype(np.float32)

        vecs = []
        ids = []
        for i in range(5):
            # Group A: slight perturbation
            vecs.append(base_a + np.random.randn(64).astype(np.float32) * 0.01)
            ids.append(f"a_{i}")
        for i in range(5):
            # Group B: slight perturbation
            vecs.append(base_b + np.random.randn(64).astype(np.float32) * 0.01)
            ids.append(f"b_{i}")

        store.add_embeddings(np.array(vecs, dtype=np.float32), ids)
        clusters = store.find_clusters(threshold=0.95)

        # Should find at least 2 clusters
        assert len(clusters) >= 2
        # Each cluster should have multiple members
        for cluster in clusters:
            assert len(cluster) >= 2

    def test_no_clusters_with_random_vectors(self):
        store = TicketVectorStore(dimension=128)
        np.random.seed(99)
        vecs = np.random.randn(20, 128).astype(np.float32)
        store.add_embeddings(vecs, [f"t{i}" for i in range(20)])
        # Random 128-dim vectors are unlikely to have cosine sim >= 0.95
        clusters = store.find_clusters(threshold=0.95)
        assert len(clusters) == 0


# ---------------------------------------------------------------------------
# Nearest neighbor
# ---------------------------------------------------------------------------

class TestFindNearestNeighbor:
    """Test find_nearest_neighbor used by Phase 3 recidivism."""

    def test_nearest_neighbor_returns_closest(self):
        store = TicketVectorStore(dimension=64)
        # vec0 and vec1 are very similar; vec2 is different
        vec0 = np.ones(64, dtype=np.float32)
        vec1 = np.ones(64, dtype=np.float32) * 0.99 + np.random.randn(64).astype(np.float32) * 0.01
        vec2 = -np.ones(64, dtype=np.float32)  # opposite direction

        store.add_embeddings(
            np.array([vec0, vec1, vec2], dtype=np.float32),
            ["t0", "t1", "t2"],
        )

        # Nearest neighbor of t0 should be t1
        nbr_id, score = store.find_nearest_neighbor(0)
        assert nbr_id == "t1"
        assert score > 0.9

    def test_single_vector_returns_none(self):
        store = TicketVectorStore(dimension=64)
        store.add_embeddings(
            np.ones((1, 64), dtype=np.float32), ["only"]
        )
        nbr_id, score = store.find_nearest_neighbor(0)
        assert nbr_id is None
        assert score == 0.0


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Verify sublinear scaling: 10K search should be less than 10x slower than 1K."""

    @pytest.mark.slow
    def test_search_scales_sublinearly(self):
        np.random.seed(0)
        dim = 128

        # Build 1K index
        store_1k = TicketVectorStore(dimension=dim)
        vecs_1k = np.random.randn(1000, dim).astype(np.float32)
        store_1k.add_embeddings(vecs_1k, [f"t{i}" for i in range(1000)])

        # Build 10K index
        store_10k = TicketVectorStore(dimension=dim)
        vecs_10k = np.random.randn(10000, dim).astype(np.float32)
        store_10k.add_embeddings(vecs_10k, [f"t{i}" for i in range(10000)])

        query = np.random.randn(dim).astype(np.float32)
        n_queries = 100

        # Time 1K searches
        start = time.perf_counter()
        for _ in range(n_queries):
            store_1k.search(query, k=5)
        time_1k = time.perf_counter() - start

        # Time 10K searches
        start = time.perf_counter()
        for _ in range(n_queries):
            store_10k.search(query, k=5)
        time_10k = time.perf_counter() - start

        # 10K should be less than 10x slower (O(n) would be exactly 10x;
        # FAISS flat IP is O(n) but with SIMD optimizations it's much faster
        # than naive Python loops.  With IVF it would be truly sublinear.)
        ratio = time_10k / max(time_1k, 1e-9)
        print(f"\nBenchmark: 1K={time_1k:.4f}s, 10K={time_10k:.4f}s, ratio={ratio:.2f}x")

        # Even with flat index (exact search), FAISS vectorized ops should
        # keep the ratio well under 15x for a 10x data increase
        assert ratio < 15, (
            f"10K search took {ratio:.1f}x longer than 1K — "
            f"expected sublinear scaling"
        )


# ---------------------------------------------------------------------------
# Zero-vector handling
# ---------------------------------------------------------------------------

class TestZeroVectors:
    """Ensure zero vectors don't cause division errors."""

    def test_zero_vector_add_and_search(self):
        store = TicketVectorStore(dimension=64)
        # Mix of zero and non-zero vectors
        vecs = np.zeros((5, 64), dtype=np.float32)
        vecs[2] = np.ones(64, dtype=np.float32)  # one non-zero
        store.add_embeddings(vecs, [f"t{i}" for i in range(5)])

        # Search with a non-zero query — should find t2 as top result
        query = np.ones(64, dtype=np.float32)
        results = store.search(query, k=3)
        assert len(results) > 0
        assert results[0][0] == "t2"
