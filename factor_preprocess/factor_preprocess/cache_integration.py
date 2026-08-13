"""
Integration of cache_v2 into existing factor_preprocess package.

Provides backward-compatible caching with new multi-level cache system.
"""

from pathlib import Path
from typing import Optional, Any, Set
import hashlib
import numpy as np
import pandas as pd

from quant_evaluator.runtime.cache_v2 import MultiLevelCache


class FactorPreprocessCache:
    """
    Cache manager for factor preprocessing operations.

    Wraps MultiLevelCache with factor-specific key generation and features.
    """

    def __init__(
        self,
        memory_size_mb: float = 256.0,
        disk_cache_dir: Optional[Path] = None,
        redis_url: Optional[str] = None,
        enable_compression: bool = True,
        default_ttl_hours: float = 24.0,
    ):
        """
        Initialize factor preprocessing cache.

        Args:
            memory_size_mb: L1 memory cache size in MB
            disk_cache_dir: L2 disk cache directory (None = temp dir)
            redis_url: L3 Redis URL (None = disabled)
            enable_compression: Enable compression
            default_ttl_hours: Default TTL in hours
        """
        if disk_cache_dir is None:
            import tempfile
            disk_cache_dir = Path(tempfile.gettempdir()) / "factor_preprocess_cache"

        self.cache = MultiLevelCache(
            memory_size_mb=memory_size_mb,
            disk_root=disk_cache_dir,
            redis_url=redis_url,
            compression="lz4" if enable_compression else "none",
            enable_l1=True,
            enable_l2=True,
            enable_l3=(redis_url is not None),
        )

        self.default_ttl_seconds = default_ttl_hours * 3600

    def _generate_key(
        self,
        operation: str,
        factor_data: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
        data_hash: Optional[str] = None,
    ) -> str:
        """
        Generate cache key for factor operation.

        Args:
            operation: Operation name (e.g., "neutralize", "rank", "zscore")
            factor_data: Input factor data (optional, will hash if provided)
            params: Operation parameters
            data_hash: Pre-computed data hash (overrides factor_data)

        Returns:
            Cache key string
        """
        key_parts = [f"op:{operation}"]

        # Add data hash
        if data_hash:
            key_parts.append(f"data:{data_hash}")
        elif factor_data is not None:
            # Hash factor data
            if isinstance(factor_data, np.ndarray):
                data_bytes = factor_data.tobytes()
            elif isinstance(factor_data, pd.DataFrame):
                data_bytes = pd.util.hash_pandas_object(factor_data).values.tobytes()
            else:
                data_bytes = str(factor_data).encode()

            h = hashlib.sha256(data_bytes).hexdigest()[:16]
            key_parts.append(f"data:{h}")

        # Add params hash
        if params:
            params_str = str(sorted(params.items()))
            h = hashlib.sha256(params_str.encode()).hexdigest()[:16]
            key_parts.append(f"params:{h}")

        return ":".join(key_parts)

    def get_or_compute(
        self,
        operation: str,
        compute_fn: callable,
        factor_data: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
        data_hash: Optional[str] = None,
        ttl_hours: Optional[float] = None,
        force_recompute: bool = False,
    ) -> Any:
        """
        Get cached result or compute and cache.

        Args:
            operation: Operation name
            compute_fn: Function to compute result if cache miss
            factor_data: Input factor data
            params: Operation parameters
            data_hash: Pre-computed data hash
            ttl_hours: TTL in hours (None = default)
            force_recompute: Skip cache and recompute

        Returns:
            Computed or cached result
        """
        key = self._generate_key(operation, factor_data, params, data_hash)

        if not force_recompute:
            result = self.cache.get(key)
            if result is not None:
                return result

        # Compute result
        result = compute_fn()

        # Cache result
        ttl_seconds = (ttl_hours * 3600) if ttl_hours else self.default_ttl_seconds
        self.cache.put(key, result, ttl_seconds=ttl_seconds)

        return result

    def invalidate_operation(self, operation: str) -> int:
        """
        Invalidate all cached results for an operation.

        Args:
            operation: Operation name

        Returns:
            Number of entries invalidated
        """
        # Use operation as dependency
        return self.cache.invalidate_dependency(f"op:{operation}")

    def warm_common_operations(self, operations: list, data_loader: callable):
        """
        Warm cache with common operations.

        Args:
            operations: List of operation names to warm
            data_loader: Function that loads data for operation
        """
        def loader(key: str):
            # Extract operation from key
            op = key.split(":")[1]
            return data_loader(op)

        keys = [f"op:{op}" for op in operations]
        self.cache.warm(keys, loader)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear(self):
        """Clear all cache layers."""
        self.cache.clear()


# Backward-compatible wrapper for existing code
class LegacyCacheAdapter:
    """
    Adapter to make new cache compatible with existing code.

    Provides same interface as old cache but uses new multi-level system.
    """

    def __init__(self, cache: FactorPreprocessCache):
        self.cache = cache
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        result = self.cache.cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        return self.cache.cache.put(key, value, ttl_seconds=ttl)

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self.cache.cache.invalidate(key)

    def clear(self):
        """Clear cache."""
        self.cache.clear()

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        base_stats = self.cache.get_stats()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            **base_stats,
        }


# Global cache instance
_global_cache: Optional[FactorPreprocessCache] = None


def get_global_cache() -> FactorPreprocessCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = FactorPreprocessCache()
    return _global_cache


def set_global_cache(cache: FactorPreprocessCache):
    """Set global cache instance."""
    global _global_cache
    _global_cache = cache


def cached(
    operation: str,
    ttl_hours: float = 24.0,
    use_data_hash: bool = True,
):
    """
    Decorator for caching factor operations.

    Args:
        operation: Operation name
        ttl_hours: Cache TTL in hours
        use_data_hash: Include input data in cache key

    Example:
        @cached("zscore", ttl_hours=12)
        def zscore_factor(data, params):
            # Expensive computation
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()

            # Extract params from kwargs
            params = kwargs.copy()

            # Generate data hash if needed
            data_hash = None
            if use_data_hash and args:
                factor_data = args[0]
                if isinstance(factor_data, np.ndarray):
                    data_bytes = factor_data.tobytes()
                    data_hash = hashlib.sha256(data_bytes).hexdigest()[:16]

            # Try cache
            key = cache._generate_key(operation, params=params, data_hash=data_hash)
            result = cache.cache.get(key)

            if result is not None:
                return result

            # Compute
            result = func(*args, **kwargs)

            # Cache
            cache.cache.put(key, result, ttl_seconds=ttl_hours * 3600)

            return result

        return wrapper

    return decorator
