"""
GPU Utilities for RAPIDS Acceleration.

This module provides a hardware abstraction layer that allows the Escalation AI
pipeline to seamlessly run on either **GPU** (via NVIDIA RAPIDS) or **CPU**
(via scikit-learn / pandas) depending on what hardware and libraries are
available.

Architecture: detect-once, adapt-everywhere
--------------------------------------------
At import time (or on first use), the module probes for three capabilities:

1. **NVIDIA GPU** -- detected by running ``nvidia-smi`` as a subprocess.
2. **cuDF**       -- the RAPIDS GPU-accelerated DataFrame library (drop-in
   replacement for pandas).
3. **cuML**       -- the RAPIDS GPU-accelerated machine learning library
   (drop-in replacement for scikit-learn).

The results are cached in module-level flags (``_GPU_AVAILABLE``,
``_CUDF_AVAILABLE``, ``_CUML_AVAILABLE``).  All subsequent code checks these
flags before deciding whether to use GPU or CPU paths.

Components provided
-------------------
1. **Detection functions** -- ``is_gpu_available()``, ``get_gpu_info()``,
   ``get_optimal_embedding_batch_size()``.

2. **GPUDataFrame** -- a thin wrapper that delegates to ``cudf`` or ``pandas``
   transparently.  Used by data loading and manipulation code.

3. **GPURandomForestClassifier** / **GPURandomForestRegressor** -- wrappers
   around ``cuml.ensemble`` or ``sklearn.ensemble`` that handle data
   conversion between numpy / cupy automatically.  Used by:
   - Phase 4 (Recurrence Prediction) -- classifier
   - Phase 6 (Resolution Time Prediction) -- regressor

4. **GPUSimilaritySearch** -- wraps ``cuml.neighbors.NearestNeighbors`` (GPU)
   or ``sklearn.neighbors.NearestNeighbors`` (CPU) for cosine-similarity-
   based nearest-neighbour search.  Used by Phase 5 (Similar Ticket Analysis).

5. **Cosine similarity functions** -- ``cosine_similarity_gpu()`` and
   ``batch_cosine_similarity_gpu()`` for pairwise and one-vs-many similarity
   computation.  Used by Phase 1 (Classification) and Phase 3 (Recidivism).

6. **GPU memory management** -- ``clear_gpu_memory()``,
   ``get_gpu_memory_usage()``.

7. **Fallback decorator** -- ``@gpu_fallback`` catches GPU errors and
   transparently retries on CPU.

Data type conventions
---------------------
- GPU paths use **CuPy arrays** (``cp.ndarray``) and **cuDF DataFrames**.
- CPU paths use **NumPy arrays** (``np.ndarray``) and **pandas DataFrames**.
- All public methods return **NumPy arrays** (or pandas DataFrames) so that
  calling code never needs to know which backend was used.

Provides seamless switching between CPU (pandas/sklearn) and GPU (cudf/cuml)
based on availability and user configuration.
"""

import logging
import os
from typing import Optional, Union, Any
from functools import wraps

logger = logging.getLogger(__name__)

# ==========================================
# GPU AVAILABILITY DETECTION
# ==========================================
# These module-level flags are lazily initialised on first access.
# ``None`` means "not yet checked"; ``True``/``False`` is the cached result.
# ==========================================

_GPU_AVAILABLE = None    # Is an NVIDIA GPU physically present?
_CUDF_AVAILABLE = None   # Is the cuDF library importable?
_CUML_AVAILABLE = None   # Is the cuML library importable?

def _check_gpu():
    """Check if an NVIDIA GPU is available by running ``nvidia-smi``.

    Uses a subprocess call with a 5-second timeout to avoid hanging if the
    GPU driver is in a bad state.  The result is cached so the subprocess is
    only invoked once per process lifetime.

    Returns:
        ``True`` if ``nvidia-smi`` exits with return code 0, ``False`` otherwise.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            _GPU_AVAILABLE = result.returncode == 0
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE

def _check_cudf():
    """Check if the cuDF library (RAPIDS GPU DataFrames) is importable.

    cuDF is a drop-in replacement for pandas that runs on NVIDIA GPUs.
    If available, the ``GPUDataFrame`` wrapper will delegate DataFrame
    operations to it.

    Returns:
        ``True`` if ``import cudf`` succeeds, ``False`` otherwise.
    """
    global _CUDF_AVAILABLE
    if _CUDF_AVAILABLE is None:
        try:
            import cudf
            _CUDF_AVAILABLE = True
            logger.info(f"[GPU] cuDF {cudf.__version__} available")
        except ImportError:
            _CUDF_AVAILABLE = False
    return _CUDF_AVAILABLE

def _check_cuml():
    """Check if the cuML library (RAPIDS GPU Machine Learning) is importable.

    cuML provides GPU-accelerated implementations of scikit-learn algorithms
    (Random Forest, k-NN, etc.).  If available, the GPU wrapper classes will
    use cuML; otherwise they fall back to scikit-learn.

    Returns:
        ``True`` if ``import cuml`` succeeds, ``False`` otherwise.
    """
    global _CUML_AVAILABLE
    if _CUML_AVAILABLE is None:
        try:
            import cuml
            _CUML_AVAILABLE = True
            logger.info(f"[GPU] cuML {cuml.__version__} available")
        except ImportError:
            _CUML_AVAILABLE = False
    return _CUML_AVAILABLE

def is_gpu_available() -> bool:
    """Check if full GPU acceleration is available.

    All three components must be present: NVIDIA GPU hardware, cuDF, and cuML.
    This is the top-level check used by classification and scoring modules to
    decide whether to enable GPU code paths.

    Returns:
        ``True`` only if GPU + cuDF + cuML are all available.
    """
    return _check_gpu() and _check_cudf() and _check_cuml()

def get_gpu_info() -> dict:
    """Get detailed GPU information for logging / diagnostics.

    Queries ``nvidia-smi`` for the GPU name, total VRAM, and free VRAM.
    This information is logged during pipeline initialisation and also
    shown in the Streamlit dashboard sidebar.

    Returns:
        Dict with keys: ``gpu_available``, ``cudf_available``,
        ``cuml_available``, ``rapids_ready``, and optionally ``gpu_name``,
        ``gpu_memory_total``, ``gpu_memory_free``.
    """
    info = {
        'gpu_available': _check_gpu(),
        'cudf_available': _check_cudf(),
        'cuml_available': _check_cuml(),
        'rapids_ready': is_gpu_available(),
    }

    if info['gpu_available']:
        try:
            import subprocess
            # Query GPU name and memory stats in CSV format
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                info['gpu_name'] = parts[0] if len(parts) > 0 else 'Unknown'
                info['gpu_memory_total'] = parts[1] if len(parts) > 1 else 'Unknown'
                info['gpu_memory_free'] = parts[2] if len(parts) > 2 else 'Unknown'
        except Exception:
            pass

    return info


def get_optimal_embedding_batch_size() -> int:
    """Get optimal batch size for Ollama embedding API calls based on GPU VRAM.

    Larger batch sizes improve throughput (fewer HTTP round-trips) but require
    more GPU memory for the embedding model to process simultaneously.  This
    function queries the GPU's total VRAM and returns a batch size that
    balances throughput against memory pressure.

    VRAM-to-batch-size mapping
    --------------------------
    - 24 GB+ VRAM (RTX 5090, A100):       100 items per batch
    - 16-24 GB VRAM (RTX 5080):            50 items per batch
    - 12-16 GB VRAM (RTX 5070 Ti):         20 items per batch
    -  8-12 GB VRAM (RTX 5070):            10 items per batch
    - < 8 GB or no GPU:                     5 items per batch

    Returns:
        Integer batch size.  Falls back to 20 if VRAM cannot be determined.
    """
    if not _check_gpu():
        return 5  # CPU-only: small batches to keep memory usage low

    try:
        import subprocess
        # Query total VRAM in MiB (no units, no header)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            vram_mb = int(result.stdout.strip().split('\n')[0])
            vram_gb = vram_mb / 1024

            if vram_gb >= 24:
                return 100  # RTX 5090, A100, etc.
            elif vram_gb >= 16:
                return 50   # RTX 5080, etc.
            elif vram_gb >= 12:
                return 20   # RTX 5070 Ti, etc.
            elif vram_gb >= 8:
                return 10   # RTX 5070, etc.
            else:
                return 5    # Lower-end GPUs
    except Exception:
        pass

    return 20  # Default fallback when VRAM query fails


# ==========================================
# GPU-ENABLED DATAFRAME OPERATIONS
# ==========================================

class GPUDataFrame:
    """
    Wrapper that provides GPU-accelerated DataFrame operations.

    This class acts as a **facade** in front of either ``cudf`` (GPU) or
    ``pandas`` (CPU).  Callers interact with it through a uniform interface
    and never need to import cuDF directly.

    The wrapper also handles the special case of Excel files: cuDF cannot
    read ``.xlsx`` files natively, so :meth:`read_excel` always loads via
    pandas first and then optionally converts to a cuDF DataFrame.

    Falls back to pandas when cuDF is unavailable.
    """

    def __init__(self, use_gpu: bool = True):
        """Initialize with GPU preference.

        Args:
            use_gpu: If ``True`` and cuDF is available, GPU DataFrames will
                be used.  If ``False`` or cuDF is missing, pandas is used.
        """
        self.use_gpu = use_gpu and _check_cudf()

        if self.use_gpu:
            import cudf
            self.pd = cudf  # All DataFrame ops go through cuDF
            logger.debug("[GPU] Using cuDF for DataFrame operations")
        else:
            import pandas
            self.pd = pandas  # All DataFrame ops go through pandas
            logger.debug("[CPU] Using pandas for DataFrame operations")

    @property
    def DataFrame(self):
        """Get the DataFrame class (cudf.DataFrame or pandas.DataFrame)."""
        return self.pd.DataFrame

    @property
    def Series(self):
        """Get the Series class (cudf.Series or pandas.Series)."""
        return self.pd.Series

    def read_excel(self, *args, **kwargs):
        """Read Excel file (always uses pandas, converts to cuDF if needed).

        cuDF does not support ``.xlsx`` natively, so we always load via
        pandas and then convert.  This is acceptable because Excel I/O is
        a one-time operation at the start of the pipeline.
        """
        import pandas as pd
        df = pd.read_excel(*args, **kwargs)
        if self.use_gpu:
            import cudf
            return cudf.from_pandas(df)
        return df

    def read_csv(self, *args, **kwargs):
        """Read CSV file.

        cuDF can read CSV natively (often faster than pandas for large files),
        so we delegate directly.
        """
        if self.use_gpu:
            import cudf
            return cudf.read_csv(*args, **kwargs)
        else:
            import pandas as pd
            return pd.read_csv(*args, **kwargs)

    def to_pandas(self, df):
        """Convert a (possibly GPU-backed) DataFrame to a pandas DataFrame.

        This is needed before writing to Excel or passing data to libraries
        that do not support cuDF.
        """
        if self.use_gpu and hasattr(df, 'to_pandas'):
            return df.to_pandas()
        return df

    def to_gpu(self, df):
        """Convert a pandas DataFrame to a cuDF DataFrame.

        No-op if already a cuDF DataFrame or if GPU is not available.
        """
        if self.use_gpu:
            import cudf
            if not isinstance(df, cudf.DataFrame):
                return cudf.from_pandas(df)
        return df

    def concat(self, dfs, **kwargs):
        """Concatenate DataFrames using the appropriate backend."""
        if self.use_gpu:
            import cudf
            return cudf.concat(dfs, **kwargs)
        else:
            import pandas as pd
            return pd.concat(dfs, **kwargs)


def get_dataframe_module(use_gpu: bool = True):
    """
    Factory function to get a ``GPUDataFrame`` wrapper instance.

    Args:
        use_gpu: Whether to prefer GPU if available.

    Returns:
        A ``GPUDataFrame`` instance configured for the appropriate backend.
    """
    return GPUDataFrame(use_gpu=use_gpu)


# ==========================================
# GPU-ENABLED ML MODELS
# ==========================================

class GPURandomForestClassifier:
    """
    GPU-accelerated Random Forest Classifier.

    Wraps either ``cuml.ensemble.RandomForestClassifier`` (GPU) or
    ``sklearn.ensemble.RandomForestClassifier`` (CPU) behind a unified API.

    Used by **Phase 4** (Recurrence Prediction) to train a binary classifier
    that predicts whether a ticket is likely to recur.

    GPU-specific details
    --------------------
    - Input data (``X``, ``y``) are converted to CuPy arrays (``float32`` /
      ``int32``) before training.
    - cuML uses ``n_streams=4`` for parallel GPU stream execution.
    - Predictions are converted back to NumPy before returning so that
      downstream code (pandas operations, report generation) works unchanged.

    Uses cuML when available, falls back to sklearn.
    """

    def __init__(self, use_gpu: bool = True, **kwargs):
        """Initialize classifier.

        Args:
            use_gpu: Whether to attempt GPU acceleration.
            **kwargs: Hyperparameters passed to the underlying Random Forest
                (e.g. ``n_estimators``, ``max_depth``, ``random_state``).
        """
        self.use_gpu = use_gpu and _check_cuml()
        self.kwargs = kwargs
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize the underlying model (cuML or sklearn)."""
        if self.use_gpu:
            from cuml.ensemble import RandomForestClassifier
            # cuML uses slightly different parameter names / defaults
            cuml_kwargs = {
                'n_estimators': self.kwargs.get('n_estimators', 100),
                'max_depth': self.kwargs.get('max_depth', 16),
                'random_state': self.kwargs.get('random_state', 42),
                'n_streams': 4,  # Parallel CUDA streams for faster training
            }
            self.model = RandomForestClassifier(**cuml_kwargs)
            logger.info("[GPU] Using cuML RandomForestClassifier")
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**self.kwargs)
            logger.info("[CPU] Using sklearn RandomForestClassifier")

    def fit(self, X, y):
        """Train the model on labelled data.

        On GPU: converts pandas/numpy inputs to CuPy arrays (float32 for
        features, int32 for labels) before calling cuML's fit().

        Args:
            X: Feature matrix (numpy array or pandas DataFrame).
            y: Label vector (numpy array or pandas Series).

        Returns:
            ``self`` for method chaining.
        """
        if self.use_gpu:
            import cudf
            import cupy as cp
            # Convert to CuPy arrays for GPU processing
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            X = cp.asarray(X, dtype=cp.float32)
            y = cp.asarray(y, dtype=cp.int32)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions -- always returns a numpy array.

        On GPU: converts input to CuPy, runs prediction on GPU, then
        converts the result back to a NumPy array for compatibility with
        pandas and downstream code.
        """
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            X = cp.asarray(X, dtype=cp.float32)
            predictions = self.model.predict(X)
            # CuPy array -> NumPy array via .get()
            if hasattr(predictions, 'get'):
                return predictions.get()
            return predictions

        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities -- always returns a numpy array.

        Returns an (n_samples, n_classes) array of class probabilities.
        Used to set the ``AI_Recurrence_Probability`` column.
        """
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            X = cp.asarray(X, dtype=cp.float32)
            predictions = self.model.predict_proba(X)
            # CuPy array -> NumPy array via .get()
            if hasattr(predictions, 'get'):
                return predictions.get()
            return predictions

        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        """Get feature importances from the trained model.

        Returns a 1-D array of shape (n_features,) indicating the relative
        importance of each feature in the trained forest.
        """
        return self.model.feature_importances_


class GPURandomForestRegressor:
    """
    GPU-accelerated Random Forest Regressor.

    Wraps either ``cuml.ensemble.RandomForestRegressor`` (GPU) or
    ``sklearn.ensemble.RandomForestRegressor`` (CPU) behind a unified API.

    Used by **Phase 6** (Resolution Time Prediction) to train a regressor
    that predicts how many hours a ticket will take to resolve.

    The GPU/CPU switching and data conversion logic mirrors
    :class:`GPURandomForestClassifier` -- see that class for detailed
    commentary on the CuPy conversion pattern.

    Uses cuML when available, falls back to sklearn.
    """

    def __init__(self, use_gpu: bool = True, **kwargs):
        """Initialize regressor.

        Args:
            use_gpu: Whether to attempt GPU acceleration.
            **kwargs: Hyperparameters passed to the underlying Random Forest.
        """
        self.use_gpu = use_gpu and _check_cuml()
        self.kwargs = kwargs
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize the underlying model (cuML or sklearn)."""
        if self.use_gpu:
            from cuml.ensemble import RandomForestRegressor
            cuml_kwargs = {
                'n_estimators': self.kwargs.get('n_estimators', 100),
                'max_depth': self.kwargs.get('max_depth', 16),
                'random_state': self.kwargs.get('random_state', 42),
                'n_streams': 4,  # Parallel CUDA streams
            }
            self.model = RandomForestRegressor(**cuml_kwargs)
            logger.info("[GPU] Using cuML RandomForestRegressor")
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**self.kwargs)
            logger.info("[CPU] Using sklearn RandomForestRegressor")

    def fit(self, X, y):
        """Train the model on labelled data.

        On GPU: converts inputs to CuPy float32 arrays before training.

        Args:
            X: Feature matrix.
            y: Target vector (continuous -- resolution hours).

        Returns:
            ``self`` for method chaining.
        """
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            X = cp.asarray(X, dtype=cp.float32)
            y = cp.asarray(y, dtype=cp.float32)  # float32 for regression targets

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions -- always returns a numpy array.

        On GPU: converts input to CuPy, runs prediction, converts back.
        """
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            X = cp.asarray(X, dtype=cp.float32)
            predictions = self.model.predict(X)
            # CuPy array -> NumPy array via .get()
            if hasattr(predictions, 'get'):
                return predictions.get()
            return predictions

        return self.model.predict(X)

    @property
    def feature_importances_(self):
        """Get feature importances from the trained model."""
        return self.model.feature_importances_


# ==========================================
# GPU-ENABLED SIMILARITY SEARCH
# ==========================================

class GPUSimilaritySearch:
    """
    GPU-accelerated similarity search using NearestNeighbors.

    Wraps either ``cuml.neighbors.NearestNeighbors`` (GPU) or
    ``sklearn.neighbors.NearestNeighbors`` (CPU) for cosine-similarity-based
    nearest-neighbour search.

    Used by **Phase 5** (Similar Ticket Analysis) to find the most similar
    *resolved* tickets for each open ticket, enabling resolution-strategy
    suggestions.

    Algorithm: brute-force cosine
    -----------------------------
    Both backends use the ``brute`` algorithm with ``cosine`` metric.  For
    the typical dataset size in this pipeline (hundreds to low thousands of
    tickets), brute-force is sufficiently fast and avoids the overhead of
    building an approximate index (e.g. IVF or HNSW).

    On GPU, the brute-force search benefits from massive SIMD parallelism
    in the dot-product and normalisation steps.

    Falls back to sklearn when unavailable.
    """

    def __init__(self, use_gpu: bool = True, metric: str = 'cosine', n_neighbors: int = 10):
        """Initialize similarity search.

        Args:
            use_gpu: Whether to attempt GPU acceleration.
            metric: Distance metric (``'cosine'`` is the only one used in
                this pipeline).
            n_neighbors: Default number of neighbours to retrieve.
        """
        self.use_gpu = use_gpu and _check_cuml()
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.model = None
        self.embeddings = None  # Stored for re-use in search queries

    def fit(self, embeddings):
        """Build the nearest-neighbour index from a set of embeddings.

        Args:
            embeddings: Array-like of shape ``(n_samples, embedding_dim)``
                containing the embedding vectors to index.

        Returns:
            ``self`` for method chaining.
        """
        import numpy as np

        self.embeddings = np.array(embeddings, dtype=np.float32)

        if self.use_gpu:
            from cuml.neighbors import NearestNeighbors
            import cupy as cp

            # Build a cuML NearestNeighbors index on the GPU
            self.model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(embeddings)),
                metric='cosine',
                algorithm='brute',  # Best for cosine on GPU
            )
            gpu_embeddings = cp.asarray(self.embeddings)
            self.model.fit(gpu_embeddings)
            logger.info(f"[GPU] Built similarity index with {len(embeddings)} embeddings")
        else:
            from sklearn.neighbors import NearestNeighbors

            # Build a sklearn NearestNeighbors index on CPU
            self.model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(embeddings)),
                metric='cosine',
                algorithm='brute',
            )
            self.model.fit(self.embeddings)
            logger.info(f"[CPU] Built similarity index with {len(embeddings)} embeddings")

        return self

    def search(self, query_embedding, k: int = 5):
        """
        Find the k most similar embeddings to a single query.

        Cosine *distance* is returned by the underlying NearestNeighbors
        model (range [0, 2] for normalised vectors).  This method converts
        it to cosine *similarity* (range [-1, 1]) via ``1 - distance``.

        Args:
            query_embedding: 1-D array of shape ``(embedding_dim,)``.
            k: Number of nearest neighbours to retrieve.

        Returns:
            Tuple of ``(similarities, indices)`` where both are 1-D arrays
            of length ``k``.
        """
        import numpy as np

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self.embeddings))

        if self.use_gpu:
            import cupy as cp
            query_gpu = cp.asarray(query)
            distances, indices = self.model.kneighbors(query_gpu, n_neighbors=k)
            # Transfer results from GPU memory back to host (NumPy)
            distances = cp.asnumpy(distances)[0]
            indices = cp.asnumpy(indices)[0]
        else:
            distances, indices = self.model.kneighbors(query, n_neighbors=k)
            distances = distances[0]
            indices = indices[0]

        # Convert cosine distance to cosine similarity: sim = 1 - dist
        similarities = 1 - distances

        return similarities, indices

    def batch_search(self, query_embeddings, k: int = 5):
        """
        Batch search for multiple queries at once.

        More efficient than calling :meth:`search` in a loop because the
        entire query matrix is transferred to GPU in one operation and the
        k-NN search runs as a single GPU kernel launch.

        Args:
            query_embeddings: 2-D array of shape ``(n_queries, embedding_dim)``.
            k: Number of nearest neighbours per query.

        Returns:
            Tuple of ``(similarities, indices)`` where both are 2-D arrays
            of shape ``(n_queries, k)``.
        """
        import numpy as np

        queries = np.array(query_embeddings, dtype=np.float32)
        k = min(k, len(self.embeddings))

        if self.use_gpu:
            import cupy as cp
            queries_gpu = cp.asarray(queries)
            distances, indices = self.model.kneighbors(queries_gpu, n_neighbors=k)
            # Transfer batch results from GPU back to NumPy
            distances = cp.asnumpy(distances)
            indices = cp.asnumpy(indices)
        else:
            distances, indices = self.model.kneighbors(queries, n_neighbors=k)

        # Convert cosine distance to similarity
        similarities = 1 - distances

        return similarities, indices


# ==========================================
# STANDALONE COSINE SIMILARITY FUNCTIONS
# ==========================================

def cosine_similarity_gpu(vec1, vec2, use_gpu: bool = True):
    """
    Compute cosine similarity between two vectors.

    This is the workhorse function used by the classifier (Phase 1) to
    compare each ticket's embedding against every anchor centroid.

    Formula
    -------
    ::

        sim(A, B) = dot(A, B) / (||A|| * ||B||)

    Returns a scalar float in the range [-1, 1].  Identical vectors yield 1.0;
    orthogonal vectors yield 0.0.

    The function attempts GPU computation first (via CuPy) and falls back to
    NumPy on any error.

    Args:
        vec1: First vector (numpy array or array-like).
        vec2: Second vector (numpy array or array-like).
        use_gpu: Whether to attempt GPU acceleration.

    Returns:
        Float cosine similarity in [-1, 1].  Returns 0.0 if either vector
        has zero norm (to avoid division by zero).
    """
    if use_gpu and _check_cuml():
        try:
            import cupy as cp
            # Transfer vectors to GPU and flatten to 1-D
            vec1 = cp.asarray(vec1, dtype=cp.float32).flatten()
            vec2 = cp.asarray(vec2, dtype=cp.float32).flatten()

            # Compute dot product and L2 norms on GPU
            dot = cp.dot(vec1, vec2)
            norm1 = cp.linalg.norm(vec1)
            norm2 = cp.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Transfer scalar result back to host Python float
            return float(cp.asnumpy(dot / (norm1 * norm2)))
        except Exception as e:
            logger.warning(f"[GPU] CuPy failed, falling back to CPU: {e}")
            # Fall through to CPU path

    # CPU fallback using NumPy
    import numpy as np
    vec1 = np.array(vec1, dtype=np.float32).flatten()
    vec2 = np.array(vec2, dtype=np.float32).flatten()

    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def batch_cosine_similarity_gpu(query_vec, matrix, use_gpu: bool = True):
    """
    Compute cosine similarity between a single query vector and every row
    in a matrix.

    This is the vectorised version of :func:`cosine_similarity_gpu`, used
    when we need to compare one ticket against *all* anchor centroids or all
    other tickets at once.  The matrix multiplication is highly parallelisable
    and benefits significantly from GPU acceleration.

    Algorithm (vectorised)
    ----------------------
    1. L2-normalise the query vector.
    2. L2-normalise every row of the matrix.
    3. Compute the dot product of the normalised matrix with the normalised
       query -- this directly yields cosine similarity for each row.

    A small epsilon (``1e-10``) is added to norms to avoid division by zero.

    Args:
        query_vec: 1-D query vector of shape ``(embedding_dim,)``.
        matrix: 2-D matrix of shape ``(n_items, embedding_dim)``.
        use_gpu: Whether to attempt GPU acceleration.

    Returns:
        1-D numpy array of shape ``(n_items,)`` with cosine similarities.
    """
    if use_gpu and _check_cuml():
        try:
            import cupy as cp
            query = cp.asarray(query_vec, dtype=cp.float32).flatten()
            mat = cp.asarray(matrix, dtype=cp.float32)

            # L2-normalise query and matrix rows (epsilon prevents div-by-zero)
            query_norm = query / (cp.linalg.norm(query) + 1e-10)
            mat_norms = cp.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
            mat_normalized = mat / mat_norms

            # Matrix-vector dot product: each row's dot with query = cosine sim
            similarities = cp.dot(mat_normalized, query_norm)

            # Transfer result back to NumPy
            return cp.asnumpy(similarities)
        except Exception as e:
            logger.warning(f"[GPU] CuPy batch similarity failed, falling back to CPU: {e}")
            # Fall through to CPU path

    # CPU fallback using NumPy
    import numpy as np
    query = np.array(query_vec, dtype=np.float32).flatten()
    mat = np.array(matrix, dtype=np.float32)

    query_norm = query / (np.linalg.norm(query) + 1e-10)
    mat_norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    mat_normalized = mat / mat_norms

    similarities = np.dot(mat_normalized, query_norm)

    return similarities


# ==========================================
# GPU MEMORY MANAGEMENT
# ==========================================

def clear_gpu_memory():
    """Clear the CuPy GPU memory pool.

    CuPy uses a memory pool to reduce the overhead of frequent GPU
    allocations.  Over time the pool can grow large.  This function releases
    all cached blocks back to the CUDA driver.

    Called by the ``@gpu_fallback`` decorator after a GPU error to ensure
    a clean state before retrying on CPU, and can also be called manually
    between pipeline phases to reclaim VRAM.
    """
    if _check_cuml():
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            logger.info("[GPU] Cleared GPU memory cache")
        except Exception as e:
            logger.warning(f"[GPU] Failed to clear memory: {e}")


def get_gpu_memory_usage():
    """Get current GPU memory usage from the CuPy memory pool.

    Returns:
        Dict with ``used_bytes``, ``total_bytes``, ``used_mb``,
        ``total_mb``.  Returns ``None`` if cuML/CuPy is not available.
    """
    if _check_cuml():
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            return {
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'used_mb': mempool.used_bytes() / (1024 * 1024),
                'total_mb': mempool.total_bytes() / (1024 * 1024),
            }
        except Exception:
            pass
    return None


# ==========================================
# DECORATOR FOR GPU FALLBACK
# ==========================================

def gpu_fallback(func):
    """
    Decorator that catches GPU errors and falls back to CPU automatically.

    When the decorated function is called with ``use_gpu=True`` and the GPU
    path raises any exception (e.g. out-of-memory, driver error), the
    decorator:

    1. Logs a warning.
    2. Clears the GPU memory pool (to free any leaked allocations).
    3. Re-invokes the same function with ``use_gpu=False``, which forces
       the CPU code path.

    This provides a safety net so that GPU errors never crash the pipeline.

    Usage::

        @gpu_fallback
        def my_gpu_function(data, use_gpu=True):
            if use_gpu:
                # GPU path
                ...
            else:
                # CPU path
                ...
    """
    @wraps(func)
    def wrapper(*args, use_gpu=True, **kwargs):
        if use_gpu and is_gpu_available():
            try:
                return func(*args, use_gpu=True, **kwargs)
            except Exception as e:
                logger.warning(f"[GPU] Operation failed, falling back to CPU: {e}")
                clear_gpu_memory()
                return func(*args, use_gpu=False, **kwargs)
        else:
            return func(*args, use_gpu=False, **kwargs)
    return wrapper


# ==========================================
# INITIALIZATION
# ==========================================

def init_rapids():
    """Initialize RAPIDS and log GPU information.

    Called once during pipeline startup to detect GPU capabilities and log
    a summary.  If all three components (GPU + cuDF + cuML) are present,
    logs "GPU Acceleration Enabled"; otherwise logs "using CPU".

    Returns:
        ``True`` if RAPIDS is fully available, ``False`` otherwise.
    """
    info = get_gpu_info()

    if info['rapids_ready']:
        logger.info("=" * 50)
        logger.info("[RAPIDS] GPU Acceleration Enabled")
        logger.info(f"[RAPIDS] GPU: {info.get('gpu_name', 'Unknown')}")
        logger.info(f"[RAPIDS] Memory: {info.get('gpu_memory_total', 'Unknown')}")
        logger.info("=" * 50)
        return True
    else:
        logger.info("[RAPIDS] GPU not available, using CPU")
        return False
