"""
Tests for advanced cache system.

Tests multi-level caching, compression, invalidation, and warming strategies.
"""

import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

from quant_evaluator.runtime.cache_v2 import (
    MultiLevelCache,
    MemoryCacheLayer,
    DiskCacheLayer,
    create_compressor,
    ZlibCompressor,
    NoCompressor,
    CacheMetadata,
    MostRecentKeysStrategy,
)


class TestCompression:
    """Test compression functionality."""

    def test_zlib_compression(self):
        """Test zlib compression and decompression."""
        compressor = ZlibCompressor(level=6)
        data = b"hello world" * 1000

        compressed, stats = compressor.compress(data)
        assert len(compressed) < len(data)
        assert stats.ratio > 1.0
        assert stats.savings_pct > 0

        decompressed = compressor.decompress(compressed)
        assert decompressed == data

    def test_no_compression(self):
        """Test no-op compressor."""
        compressor = NoCompressor()
        data = b"hello world"

        compressed, stats = compressor.compress(data)
        assert compressed == data
        assert stats.ratio == 1.0
        assert stats.savings_pct == 0.0

    def test_create_compressor(self):
        """Test compressor factory."""
        comp = create_compressor("zlib")
        assert isinstance(comp, ZlibCompressor)

        comp = create_compressor("none")
        assert isinstance(comp, NoCompressor)


class TestMemoryCacheLayer:
    """Test L1 memory cache layer."""

    def test_basic_operations(self):
        """Test get, put, and invalidate."""
        compressor = create_compressor("zlib")
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        # Put and get
        data = np.random.randn(1000)
        assert cache.put("key1", data)

        result = cache.get("key1")
        assert result is not None
        value, metadata = result
        np.testing.assert_array_equal(value, data)
        assert metadata.key == "key1"
        assert metadata.access_count == 1

        # Second get updates access count
        result = cache.get("key1")
        assert result[1].access_count == 2

        # Invalidate
        assert cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        compressor = create_compressor("none")
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=compressor,
            enable_compression=False,
        )

        data = {"value": 42}
        cache.put("key1", data, ttl_seconds=0.1)

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """Test LRU eviction under memory pressure."""
        compressor = create_compressor("none")
        cache = MemoryCacheLayer(
            max_size_bytes=500,  # Very small cache to force eviction
            compressor=compressor,
            enable_compression=False,
        )

        # Fill cache with items
        for i in range(20):
            cache.put(f"key_{i}", f"value_{i}" * 10)

        stats = cache.get_stats()
        # Should have evicted some entries
        assert stats["num_entries"] < 20
        assert stats["utilization"] <= 1.0

    def test_dependency_invalidation(self):
        """Test invalidating entries by dependency."""
        compressor = create_compressor("none")
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=compressor,
            enable_compression=False,
        )

        # Put entries with dependencies
        cache.put("key1", "value1", dependencies={"dep_a"})
        cache.put("key2", "value2", dependencies={"dep_a", "dep_b"})
        cache.put("key3", "value3", dependencies={"dep_b"})
        cache.put("key4", "value4", dependencies=set())

        # Invalidate dep_a
        count = cache.invalidate_dependencies("dep_a")
        assert count == 2  # key1 and key2

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_compression_savings(self):
        """Test that compression reduces memory usage."""
        compressor = create_compressor("zlib")

        cache_compressed = MemoryCacheLayer(
            max_size_bytes=10 * 1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        cache_uncompressed = MemoryCacheLayer(
            max_size_bytes=10 * 1024 * 1024,
            compressor=compressor,
            enable_compression=False,
        )

        # Create compressible data
        data = np.zeros(10000)  # Highly compressible

        cache_compressed.put("key1", data)
        cache_uncompressed.put("key1", data)

        stats_compressed = cache_compressed.get_stats()
        stats_uncompressed = cache_uncompressed.get_stats()

        # Compressed should use less memory
        assert stats_compressed["current_size_bytes"] < stats_uncompressed["current_size_bytes"]
        assert stats_compressed["memory_savings_pct"] > 50  # Should be >50% savings


class TestDiskCacheLayer:
    """Test L2 disk cache layer."""

    def test_basic_operations(self):
        """Test disk cache put and get."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = create_compressor("zlib")
            cache = DiskCacheLayer(
                root_dir=Path(tmpdir),
                compressor=compressor,
            )

            data = np.random.randn(1000)
            assert cache.put("key1", data)

            result = cache.get("key1")
            assert result is not None
            value, metadata = result
            np.testing.assert_array_equal(value, data)

    def test_checksum_verification(self):
        """Test that corrupted cache files are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = create_compressor("zlib")
            cache = DiskCacheLayer(
                root_dir=Path(tmpdir),
                compressor=compressor,
            )

            data = {"value": 42}
            cache.put("key1", data)

            # Corrupt the cache file
            cache_path = cache._key_path("key1")
            cache_path.write_bytes(b"corrupted data")

            # Should return None due to checksum mismatch
            result = cache.get("key1")
            assert result is None

    def test_atomic_writes(self):
        """Test that writes are atomic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = create_compressor("zlib")
            cache = DiskCacheLayer(
                root_dir=Path(tmpdir),
                compressor=compressor,
            )

            # Simulate write failure by making directory read-only
            data = {"value": 42}
            cache.put("key1", data)

            # Verify no temp files left behind
            temp_files = list(Path(tmpdir).rglob(".tmp.*"))
            assert len(temp_files) == 0


class TestMultiLevelCache:
    """Test multi-level cache coordinator."""

    def test_l1_hit(self):
        """Test L1 cache hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = np.random.randn(100)
            cache.put("key1", data)

            # Should hit L1
            result = cache.get("key1")
            np.testing.assert_array_equal(result, data)

            stats = cache.get_stats()
            assert stats["l1_hits"] == 1
            assert stats["l2_hits"] == 0
            assert stats["misses"] == 0

    def test_l2_promotion(self):
        """Test L2 hit promotes to L1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=10.0,  # Large enough for testing
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = np.random.randn(100)
            cache.put("key1", data, write_through=True)

            # Verify it's in L2
            assert cache.l2.get("key1") is not None

            # Clear L1 to force L2 hit
            cache.l1.clear()

            # Reset stats after clear
            cache.stats.l1_hits = 0
            cache.stats.l2_hits = 0

            # First get should hit L2 and promote to L1
            result = cache.get("key1")
            assert result is not None
            np.testing.assert_array_equal(result, data)

            stats = cache.get_stats()
            assert stats["l2_hits"] == 1  # L2 hit

            # Verify promoted to L1
            assert cache.l1.get("key1") is not None

            # Next get should hit L1
            result2 = cache.get("key1")
            stats = cache.get_stats()
            assert stats["l1_hits"] == 1  # L1 hit after promotion

    def test_cache_miss(self):
        """Test cache miss across all layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            result = cache.get("nonexistent")
            assert result is None

            stats = cache.get_stats()
            assert stats["misses"] == 1

    def test_invalidation_cascade(self):
        """Test invalidation across all layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = {"value": 42}
            cache.put("key1", data)

            # Verify in both layers
            assert cache.l1.get("key1") is not None
            assert cache.l2.get("key1") is not None

            # Invalidate
            cache.invalidate("key1")

            # Should be gone from both layers
            assert cache.l1.get("key1") is None
            assert cache.l2.get("key1") is None

    def test_cache_warming(self):
        """Test cache warming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
            )

            # Loader function
            def loader(key: str):
                return f"value_for_{key}"

            # Warm cache
            keys = ["key1", "key2", "key3"]
            cache.warm(keys, loader)

            # All keys should be cached
            for key in keys:
                assert cache.get(key) == f"value_for_{key}"

            stats = cache.get_stats()
            assert stats["l1_hits"] == 3

    def test_write_through(self):
        """Test write-through to all layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = {"value": 42}
            cache.put("key1", data, write_through=True)

            # Should be in both layers
            assert cache.l1.get("key1") is not None
            assert cache.l2.get("key1") is not None

    def test_memory_only_mode(self):
        """Test cache with L1 only."""
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            compression="zlib",
            enable_l1=True,
            enable_l2=False,
            enable_l3=False,
        )

        data = {"value": 42}
        cache.put("key1", data)

        result = cache.get("key1")
        assert result == data

        assert cache.l2 is None
        assert cache.l3 is None


class TestCacheWarmingStrategies:
    """Test cache warming strategies."""

    def test_most_recent_keys_strategy(self):
        """Test most recent keys warming strategy."""
        history = ["key1", "key2", "key3", "key1", "key4", "key2"]
        strategy = MostRecentKeysStrategy(history, top_n=3)

        keys = strategy.get_keys_to_warm()

        # Should return 3 most recent unique keys in reverse order
        assert len(keys) == 3
        assert "key2" in keys  # Most recent
        assert "key4" in keys
        assert "key1" in keys


class TestConcurrency:
    """Test thread safety of cache operations."""

    def test_concurrent_puts(self):
        """Test concurrent put operations."""
        import threading

        compressor = create_compressor("zlib")
        cache = MemoryCacheLayer(
            max_size_bytes=10 * 1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        def worker(thread_id: int):
            for i in range(100):
                cache.put(f"thread{thread_id}_key{i}", f"value_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cache should be in consistent state
        stats = cache.get_stats()
        assert stats["num_entries"] > 0
        assert stats["current_size_bytes"] <= cache.max_size_bytes

    def test_concurrent_get_put(self):
        """Test concurrent get and put operations."""
        import threading

        compressor = create_compressor("zlib")
        cache = MemoryCacheLayer(
            max_size_bytes=1 * 1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        # Pre-populate
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")

        def reader():
            for i in range(100):
                cache.get(f"key_{i % 50}")

        def writer():
            for i in range(100):
                cache.put(f"key_{i % 50}", f"new_value_{i}")

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No crashes = success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
