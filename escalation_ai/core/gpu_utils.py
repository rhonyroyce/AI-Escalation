"""
GPU Utilities for RAPIDS Acceleration.

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

_GPU_AVAILABLE = None
_CUDF_AVAILABLE = None
_CUML_AVAILABLE = None

def _check_gpu():
    """Check if NVIDIA GPU is available."""
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
    """Check if cuDF is available."""
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
    """Check if cuML is available."""
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
    """Check if GPU acceleration is available."""
    return _check_gpu() and _check_cudf() and _check_cuml()

def get_gpu_info() -> dict:
    """Get GPU information."""
    info = {
        'gpu_available': _check_gpu(),
        'cudf_available': _check_cudf(),
        'cuml_available': _check_cuml(),
        'rapids_ready': is_gpu_available(),
    }
    
    if info['gpu_available']:
        try:
            import subprocess
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


# ==========================================
# GPU-ENABLED DATAFRAME OPERATIONS
# ==========================================

class GPUDataFrame:
    """
    Wrapper that provides GPU-accelerated DataFrame operations.
    Falls back to pandas when cuDF is unavailable.
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize with GPU preference."""
        self.use_gpu = use_gpu and _check_cudf()
        
        if self.use_gpu:
            import cudf
            self.pd = cudf
            logger.debug("[GPU] Using cuDF for DataFrame operations")
        else:
            import pandas
            self.pd = pandas
            logger.debug("[CPU] Using pandas for DataFrame operations")
    
    @property
    def DataFrame(self):
        """Get DataFrame class."""
        return self.pd.DataFrame
    
    @property  
    def Series(self):
        """Get Series class."""
        return self.pd.Series
    
    def read_excel(self, *args, **kwargs):
        """Read Excel file (always uses pandas, converts to cudf if needed)."""
        import pandas as pd
        df = pd.read_excel(*args, **kwargs)
        if self.use_gpu:
            import cudf
            return cudf.from_pandas(df)
        return df
    
    def read_csv(self, *args, **kwargs):
        """Read CSV file."""
        if self.use_gpu:
            import cudf
            return cudf.read_csv(*args, **kwargs)
        else:
            import pandas as pd
            return pd.read_csv(*args, **kwargs)
    
    def to_pandas(self, df):
        """Convert to pandas DataFrame."""
        if self.use_gpu and hasattr(df, 'to_pandas'):
            return df.to_pandas()
        return df
    
    def to_gpu(self, df):
        """Convert pandas DataFrame to cuDF."""
        if self.use_gpu:
            import cudf
            if not isinstance(df, cudf.DataFrame):
                return cudf.from_pandas(df)
        return df
    
    def concat(self, dfs, **kwargs):
        """Concatenate DataFrames."""
        if self.use_gpu:
            import cudf
            return cudf.concat(dfs, **kwargs)
        else:
            import pandas as pd
            return pd.concat(dfs, **kwargs)


def get_dataframe_module(use_gpu: bool = True):
    """
    Get the appropriate DataFrame module (cudf or pandas).
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        GPUDataFrame wrapper instance
    """
    return GPUDataFrame(use_gpu=use_gpu)


# ==========================================
# GPU-ENABLED ML MODELS
# ==========================================

class GPURandomForestClassifier:
    """
    GPU-accelerated Random Forest Classifier.
    Uses cuML when available, falls back to sklearn.
    """
    
    def __init__(self, use_gpu: bool = True, **kwargs):
        """Initialize classifier."""
        self.use_gpu = use_gpu and _check_cuml()
        self.kwargs = kwargs
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying model."""
        if self.use_gpu:
            from cuml.ensemble import RandomForestClassifier
            # cuML uses slightly different params
            cuml_kwargs = {
                'n_estimators': self.kwargs.get('n_estimators', 100),
                'max_depth': self.kwargs.get('max_depth', 16),
                'random_state': self.kwargs.get('random_state', 42),
                'n_streams': 4,  # Parallel streams for GPU
            }
            self.model = RandomForestClassifier(**cuml_kwargs)
            logger.info("[GPU] Using cuML RandomForestClassifier")
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**self.kwargs)
            logger.info("[CPU] Using sklearn RandomForestClassifier")
    
    def fit(self, X, y):
        """Train the model."""
        if self.use_gpu:
            import cudf
            import cupy as cp
            # Convert to cuDF/cupy if needed
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            X = cp.asarray(X, dtype=cp.float32)
            y = cp.asarray(y, dtype=cp.int32)
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            X = cp.asarray(X, dtype=cp.float32)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            X = cp.asarray(X, dtype=cp.float32)
        
        return self.model.predict_proba(X)
    
    @property
    def feature_importances_(self):
        """Get feature importances."""
        return self.model.feature_importances_


class GPURandomForestRegressor:
    """
    GPU-accelerated Random Forest Regressor.
    Uses cuML when available, falls back to sklearn.
    """
    
    def __init__(self, use_gpu: bool = True, **kwargs):
        """Initialize regressor."""
        self.use_gpu = use_gpu and _check_cuml()
        self.kwargs = kwargs
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the underlying model."""
        if self.use_gpu:
            from cuml.ensemble import RandomForestRegressor
            cuml_kwargs = {
                'n_estimators': self.kwargs.get('n_estimators', 100),
                'max_depth': self.kwargs.get('max_depth', 16),
                'random_state': self.kwargs.get('random_state', 42),
                'n_streams': 4,
            }
            self.model = RandomForestRegressor(**cuml_kwargs)
            logger.info("[GPU] Using cuML RandomForestRegressor")
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**self.kwargs)
            logger.info("[CPU] Using sklearn RandomForestRegressor")
    
    def fit(self, X, y):
        """Train the model."""
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            X = cp.asarray(X, dtype=cp.float32)
            y = cp.asarray(y, dtype=cp.float32)
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.use_gpu:
            import cupy as cp
            if hasattr(X, 'values'):
                X = X.values
            X = cp.asarray(X, dtype=cp.float32)
        
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        """Get feature importances."""
        return self.model.feature_importances_


# ==========================================
# GPU-ENABLED SIMILARITY SEARCH
# ==========================================

class GPUSimilaritySearch:
    """
    GPU-accelerated similarity search using cuML NearestNeighbors.
    Falls back to sklearn when unavailable.
    """
    
    def __init__(self, use_gpu: bool = True, metric: str = 'cosine', n_neighbors: int = 10):
        """Initialize similarity search."""
        self.use_gpu = use_gpu and _check_cuml()
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.model = None
        self.embeddings = None
    
    def fit(self, embeddings):
        """Build index from embeddings."""
        import numpy as np
        
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.use_gpu:
            from cuml.neighbors import NearestNeighbors
            import cupy as cp
            
            # cuML NearestNeighbors
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
        Find k most similar embeddings.
        
        Returns:
            Tuple of (distances, indices)
        """
        import numpy as np
        
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self.embeddings))
        
        if self.use_gpu:
            import cupy as cp
            query_gpu = cp.asarray(query)
            distances, indices = self.model.kneighbors(query_gpu, n_neighbors=k)
            # Convert back to numpy
            distances = cp.asnumpy(distances)[0]
            indices = cp.asnumpy(indices)[0]
        else:
            distances, indices = self.model.kneighbors(query, n_neighbors=k)
            distances = distances[0]
            indices = indices[0]
        
        # Convert cosine distance to similarity (1 - distance)
        similarities = 1 - distances
        
        return similarities, indices
    
    def batch_search(self, query_embeddings, k: int = 5):
        """
        Batch search for multiple queries.
        
        Returns:
            Tuple of (distances_list, indices_list)
        """
        import numpy as np
        
        queries = np.array(query_embeddings, dtype=np.float32)
        k = min(k, len(self.embeddings))
        
        if self.use_gpu:
            import cupy as cp
            queries_gpu = cp.asarray(queries)
            distances, indices = self.model.kneighbors(queries_gpu, n_neighbors=k)
            distances = cp.asnumpy(distances)
            indices = cp.asnumpy(indices)
        else:
            distances, indices = self.model.kneighbors(queries, n_neighbors=k)
        
        # Convert cosine distance to similarity
        similarities = 1 - distances
        
        return similarities, indices


def cosine_similarity_gpu(vec1, vec2, use_gpu: bool = True):
    """
    Compute cosine similarity between two vectors.
    Uses GPU when available.
    """
    if use_gpu and _check_cuml():
        import cupy as cp
        vec1 = cp.asarray(vec1, dtype=cp.float32).flatten()
        vec2 = cp.asarray(vec2, dtype=cp.float32).flatten()
        
        dot = cp.dot(vec1, vec2)
        norm1 = cp.linalg.norm(vec1)
        norm2 = cp.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(cp.asnumpy(dot / (norm1 * norm2)))
    else:
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
    Compute cosine similarity between query and all rows in matrix.
    Returns array of similarities.
    """
    if use_gpu and _check_cuml():
        import cupy as cp
        query = cp.asarray(query_vec, dtype=cp.float32).flatten()
        mat = cp.asarray(matrix, dtype=cp.float32)
        
        # Normalize
        query_norm = query / (cp.linalg.norm(query) + 1e-10)
        mat_norms = cp.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        mat_normalized = mat / mat_norms
        
        # Dot product
        similarities = cp.dot(mat_normalized, query_norm)
        
        return cp.asnumpy(similarities)
    else:
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
    """Clear GPU memory cache."""
    if _check_cuml():
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            logger.info("[GPU] Cleared GPU memory cache")
        except Exception as e:
            logger.warning(f"[GPU] Failed to clear memory: {e}")


def get_gpu_memory_usage():
    """Get current GPU memory usage."""
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
    Decorator that catches GPU errors and falls back to CPU.
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
    """Initialize RAPIDS and log GPU info."""
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
