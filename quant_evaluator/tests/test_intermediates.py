"""
Tests for intermediate result caching.
"""

import pytest
import time

from quant_evaluator.runtime.cache_v2 import CacheV2Config
from quant_evaluator.runtime.cache_v2_adapter import V2IntermediateCache
from quant_evaluator.runtime.intermediates import (
    CacheKey,
    CacheEntry,
    IntermediateCache,
    compute_input_hash,
)


class TestCacheKey:
    def test_cache_key_creation(self):
        key = CacheKey(
            metric_id="ic_daily",
            chunk_id=0,
            input_hash="abc123",
            version="v1",
        )

        assert key.metric_id == "ic_daily"
        assert key.chunk_id == 0
        assert key.input_hash == "abc123"

    def test_cache_key_hashable(self):
        key1 = CacheKey("ic_daily", chunk_id=0, input_hash="abc")
        key2 = CacheKey("ic_daily", chunk_id=0, input_hash="abc")
        key3 = CacheKey("ic_daily", chunk_id=1, input_hash="abc")

        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)

    def test_cache_key_str(self):
        key = CacheKey("ic_daily", chunk_id=5, input_hash="abcdef1234")

        key_str = str(key)
        assert "ic_daily" in key_str
        assert "chunk_5" in key_str
        assert "abcdef12" in key_str  # First 8 chars


class TestCacheEntry:
    def test_entry_creation(self):
        key = CacheKey("test_metric")
        value = [1, 2, 3]

        entry = CacheEntry(key=key, value=value)

        assert entry.key == key
        assert entry.value == value
        assert entry.access_count == 0
        assert entry.size_bytes > 0

    def test_mark_accessed(self):
        key = CacheKey("test_metric")
        entry = CacheEntry(key=key, value=[1, 2, 3])

        initial_time = entry.last_accessed
        initial_count = entry.access_count

        time.sleep(0.01)
        entry.mark_accessed()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time


class TestIntermediateCache:
    def test_cache_creation(self):
        cache = IntermediateCache(max_size_mb=100.0, enable=True)

        assert cache.max_size_bytes == 100 * 1024 * 1024
        assert cache.enable is True
        assert cache.num_entries == 0

    def test_put_and_get(self):
        cache = IntermediateCache()
        key = CacheKey("test_metric")
        value = [1, 2, 3, 4, 5]

        result = cache.put(key, value)
        assert result is True

        retrieved = cache.get(key)
        assert retrieved == value

    def test_get_missing_returns_none(self):
        cache = IntermediateCache()
        key = CacheKey("missing_metric")

        result = cache.get(key)
        assert result is None

    def test_has_key(self):
        cache = IntermediateCache()
        key = CacheKey("test_metric")

        assert not cache.has(key)

        cache.put(key, [1, 2, 3])
        assert cache.has(key)

    def test_disabled_cache(self):
        cache = IntermediateCache(enable=False)
        key = CacheKey("test_metric")

        result = cache.put(key, [1, 2, 3])
        assert result is False

        retrieved = cache.get(key)
        assert retrieved is None

    def test_invalidate(self):
        cache = IntermediateCache()
        key = CacheKey("test_metric")

        cache.put(key, [1, 2, 3])
        assert cache.has(key)

        result = cache.invalidate(key)
        assert result is True
        assert not cache.has(key)

    def test_invalidate_missing_returns_false(self):
        cache = IntermediateCache()
        key = CacheKey("missing_metric")

        result = cache.invalidate(key)
        assert result is False

    def test_invalidate_metric(self):
        cache = IntermediateCache()

        key1 = CacheKey("metric_A", chunk_id=0)
        key2 = CacheKey("metric_A", chunk_id=1)
        key3 = CacheKey("metric_B", chunk_id=0)

        cache.put(key1, [1])
        cache.put(key2, [2])
        cache.put(key3, [3])

        count = cache.invalidate_metric("metric_A")

        assert count == 2
        assert not cache.has(key1)
        assert not cache.has(key2)
        assert cache.has(key3)

    def test_clear(self):
        cache = IntermediateCache()

        cache.put(CacheKey("A"), [1])
        cache.put(CacheKey("B"), [2])
        cache.put(CacheKey("C"), [3])

        assert cache.num_entries == 3

        cache.clear()

        assert cache.num_entries == 0
        assert cache.size_mb == 0.0

    def test_lru_eviction(self):
        # Small cache that can only hold ~2 entries
        cache = IntermediateCache(max_size_mb=0.001)  # 1 KB

        key1 = CacheKey("metric1")
        key2 = CacheKey("metric2")
        key3 = CacheKey("metric3")

        # Add entries
        cache.put(key1, [1] * 100)
        time.sleep(0.01)
        cache.put(key2, [2] * 100)
        time.sleep(0.01)

        # Access key1 to make it more recent
        cache.get(key1)
        time.sleep(0.01)

        # Add key3, should evict key2 (least recently used)
        cache.put(key3, [3] * 100)

        # key1 and key3 should be present, key2 should be evicted
        assert cache.has(key1)
        assert cache.has(key3)

    def test_update_existing_entry(self):
        cache = IntermediateCache()
        key = CacheKey("test_metric")

        cache.put(key, [1, 2, 3])
        initial_size = cache.size_mb

        # Update with different value
        cache.put(key, [4, 5, 6, 7, 8])

        # Should still have 1 entry
        assert cache.num_entries == 1

        # Value should be updated
        assert cache.get(key) == [4, 5, 6, 7, 8]

    def test_size_tracking(self):
        cache = IntermediateCache()

        cache.put(CacheKey("A"), [1] * 1000)
        size_after_one = cache.size_mb

        cache.put(CacheKey("B"), [2] * 1000)
        size_after_two = cache.size_mb

        assert size_after_two > size_after_one
        assert cache.num_entries == 2

    def test_get_stats(self):
        cache = IntermediateCache(max_size_mb=100.0)

        cache.put(CacheKey("A"), [1] * 100)
        cache.put(CacheKey("B"), [2] * 100)

        cache.get(CacheKey("A"))
        cache.get(CacheKey("A"))
        cache.get(CacheKey("B"))

        stats = cache.get_stats()

        assert stats["num_entries"] == 2
        assert stats["size_mb"] > 0
        assert stats["max_size_mb"] == 100.0
        assert stats["utilization"] >= 0
        assert stats["total_accesses"] == 3

    def test_value_too_large_for_cache(self):
        cache = IntermediateCache(max_size_mb=0.001)  # 1 KB

        key = CacheKey("large_metric")
        large_value = [1] * 100000  # Large array

        result = cache.put(key, large_value)

        # Should fail to store
        assert result is False
        assert not cache.has(key)


class TestV2IntermediateCache:
    """Tests for the cache_v2-backed intermediate cache."""

    def test_put_get_round_trip(self, tmp_path):
        cache = V2IntermediateCache(
            config=CacheV2Config(
                runtime_mode="research",
                disk_root=tmp_path,
                enable_l2=True,
            )
        )
        key = CacheKey("metric_a", chunk_id=0, input_hash="hash1")

        assert cache.put(key, [1, 2, 3]) is True
        assert cache.get(key) == [1, 2, 3]

    def test_has_does_not_promote_l2_entries(self, tmp_path):
        """``has`` must answer without promoting an evicted L1 entry."""
        cache = V2IntermediateCache(
            config=CacheV2Config(
                runtime_mode="research",
                disk_root=tmp_path,
                enable_l2=True,
            )
        )
        key = CacheKey("metric_a", chunk_id=0, input_hash="hash1")
        cache.put(key, [1, 2, 3])

        # Simulate L1 eviction: the value survives only in L2.
        cache._cache.l1.clear()
        assert cache._cache.l1.get(cache._physical_key(key)) is None

        assert cache.has(key) is True
        # has() must not have promoted the entry back into L1 and must not
        # have recorded any layer hit.
        assert cache._cache.l1.get(cache._physical_key(key)) is None
        stats = cache.get_stats()
        assert stats["l1_hits"] == 0
        assert stats["l2_hits"] == 0

    def test_invalidate_metric_removes_l2_only_entries(self, tmp_path):
        """Entries evicted from L1 into L2 must still be invalidated."""
        cache = V2IntermediateCache(
            config=CacheV2Config(
                runtime_mode="research",
                disk_root=tmp_path,
                enable_l2=True,
            )
        )
        key1 = CacheKey("metric_a", chunk_id=0, input_hash="hash1")
        key2 = CacheKey("metric_a", chunk_id=1, input_hash="hash1")
        key3 = CacheKey("metric_b", chunk_id=0, input_hash="hash1")

        cache.put(key1, [1])
        cache.put(key2, [2])
        cache.put(key3, [3])

        # Evict everything from L1 so the entries live only in L2/L3.
        cache._cache.l1.clear()

        assert cache.invalidate_metric("metric_a") == 2

        # The evicted entries must not resurrect through later reads.
        assert cache.get(key1) is None
        assert cache.get(key2) is None
        assert cache.get(key3) == [3]

    def test_invalidate_metric_count_is_per_logical_entry(self, tmp_path):
        """A metric present in multiple layers is counted and removed once."""
        cache = V2IntermediateCache(
            config=CacheV2Config(
                runtime_mode="research",
                disk_root=tmp_path,
                enable_l2=True,
            )
        )
        key1 = CacheKey("metric_a", chunk_id=0, input_hash="hash1")
        key2 = CacheKey("metric_a", chunk_id=1, input_hash="hash1")

        cache.put(key1, [1])
        cache.put(key2, [2])

        # Entries live in both L1 and L2 simultaneously.
        assert cache.invalidate_metric("metric_a") == 2
        assert cache.has(key1) is False
        assert cache.has(key2) is False
        assert cache.invalidate_metric("metric_a") == 0

    def test_disabled_cache_is_noop(self):
        cache = V2IntermediateCache(enable=False)
        key = CacheKey("metric_a")

        assert cache.put(key, [1]) is False
        assert cache.get(key) is None
        assert cache.has(key) is False


class TestComputeInputHash:
    def test_basic_hash(self):
        hash1 = compute_input_hash(("factor_a", "factor_b"))
        hash2 = compute_input_hash(("factor_a", "factor_b"))

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest    def test_different_factors_different_hash(self):
        hash1 = compute_input_hash(("factor_a",))
        hash2 = compute_input_hash(("factor_b",))

        assert hash1 != hash2

    def test_factor_order_independent(self):
        # Hash should be same regardless of order (sorted internally)
        hash1 = compute_input_hash(("factor_a", "factor_b"))
        hash2 = compute_input_hash(("factor_b", "factor_a"))

        assert hash1 == hash2

    def test_with_slices(self):
        hash1 = compute_input_hash(
            ("factor_a",),
            time_slice=(0, 100),
            asset_slice=(0, 500),
        )
        hash2 = compute_input_hash(
            ("factor_a",),
            time_slice=(0, 100),
            asset_slice=(0, 500),
        )
        hash3 = compute_input_hash(
            ("factor_a",),
            time_slice=(100, 200),
            asset_slice=(0, 500),
        )

        assert hash1 == hash2
        assert hash1 != hash3

    def test_with_kwargs(self):
        hash1 = compute_input_hash(
            ("factor_a",),
            method="spearman",
            min_periods=20,
        )
        hash2 = compute_input_hash(
            ("factor_a",),
            method="spearman",
            min_periods=20,
        )
        hash3 = compute_input_hash(
            ("factor_a",),
            method="pearson",
            min_periods=20,
        )

        assert hash1 == hash2
        assert hash1 != hash3
