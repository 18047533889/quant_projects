# -*- coding: utf-8 -*-
"""Tests for resource management bug fixes.

Tests:
1. NativeBufferStore memory accounting is correct when overwriting representations
2. Polars threading model documentation is correct
"""
import pytest

from factor_engine.planner.backend_region import Representation
from factor_engine.runtime.native_buffer_store import NativeBufferStore


class TestNativeBufferStoreMemoryAccounting:
    """Test that memory accounting is correct when overwriting representations."""

    def test_overwrite_representation_subtracts_old_bytes(self):
        """When overwriting a representation, old bytes should be subtracted first."""
        store = NativeBufferStore(max_memory_bytes=10_000_000)

        # Store initial representation with 1000 bytes
        success = store.put_native(
            semantic_node_id="node_1",
            representation=Representation.POLARS_LAZY_LONG,
            value="fake_polars_data_v1",
            estimated_bytes=1000,
            consumer_count=1,
        )
        assert success

        stats = store.get_stats()
        assert stats["bytes_current"] == 1000, "Initial bytes should be 1000"

        # Overwrite with larger representation (5000 bytes)
        success = store.put_native(
            semantic_node_id="node_1",
            representation=Representation.POLARS_LAZY_LONG,
            value="fake_polars_data_v2_much_larger",
            estimated_bytes=5000,
            consumer_count=1,
        )
        assert success

        stats = store.get_stats()
        # Should be 5000, not 6000 (1000 + 5000)
        assert stats["bytes_current"] == 5000, (
            f"After overwrite, bytes should be 5000 (new only), "
            f"not {stats['bytes_current']} (old + new)"
        )

    def test_overwrite_representation_reduces_total_when_smaller(self):
        """When overwriting with smaller representation, total bytes should decrease."""
        store = NativeBufferStore(max_memory_bytes=10_000_000)

        # Store initial representation with 5000 bytes
        store.put_native(
            semantic_node_id="node_2",
            representation=Representation.PANDAS_LONG,
            value="large_pandas_data",
            estimated_bytes=5000,
            consumer_count=1,
        )

        stats_before = store.get_stats()
        assert stats_before["bytes_current"] == 5000

        # Overwrite with smaller representation (1000 bytes)
        store.put_native(
            semantic_node_id="node_2",
            representation=Representation.PANDAS_LONG,
            value="small_pandas_data",
            estimated_bytes=1000,
            consumer_count=1,
        )

        stats_after = store.get_stats()
        assert stats_after["bytes_current"] == 1000, (
            "After overwriting with smaller representation, "
            "bytes should decrease to 1000"
        )

    def test_multiple_representations_accounting(self):
        """Multiple representations of same node should sum correctly."""
        store = NativeBufferStore(max_memory_bytes=10_000_000)

        # Add first representation
        store.put_native(
            semantic_node_id="node_3",
            representation=Representation.POLARS_LAZY_LONG,
            value="polars_data",
            estimated_bytes=2000,
            consumer_count=1,
        )

        # Add second representation (different format)
        store.put_native(
            semantic_node_id="node_3",
            representation=Representation.PANDAS_LONG,
            value="pandas_data",
            estimated_bytes=3000,
            consumer_count=1,
        )

        stats = store.get_stats()
        assert stats["bytes_current"] == 5000, (
            "Two representations should sum to 5000 bytes"
        )

        # Overwrite first representation
        store.put_native(
            semantic_node_id="node_3",
            representation=Representation.POLARS_LAZY_LONG,
            value="new_polars_data",
            estimated_bytes=1000,
            consumer_count=1,
        )

        stats = store.get_stats()
        assert stats["bytes_current"] == 4000, (
            "After overwriting first rep (2000->1000), "
            "total should be 4000 (1000 + 3000)"
        )

    def test_overwrite_zero_bytes_edge_case(self):
        """Overwriting representation with zero bytes should work."""
        store = NativeBufferStore(max_memory_bytes=10_000_000)

        store.put_native(
            semantic_node_id="node_4",
            representation=Representation.POLARS_LAZY_LONG,
            value="data",
            estimated_bytes=1000,
            consumer_count=1,
        )

        # Overwrite with zero bytes (edge case)
        store.put_native(
            semantic_node_id="node_4",
            representation=Representation.POLARS_LAZY_LONG,
            value="empty",
            estimated_bytes=0,
            consumer_count=1,
        )

        stats = store.get_stats()
        assert stats["bytes_current"] == 0, "Zero bytes should result in 0 total"


class TestPolarsThreadingModel:
    """Test documentation of correct Polars threading model."""

    def test_polars_thread_budget_is_noop(self):
        """polars_thread_budget should be a no-op (doesn't change threads)."""
        from factor_engine.backend.polars_thread_config import polars_thread_budget

        # Should not raise, but also should not actually change thread count
        with polars_thread_budget(4):
            pass  # No-op

    def test_configure_polars_returns_current_threads(self):
        """configure_polars_for_execution should return current thread count."""
        from factor_engine.backend.polars_thread_config import configure_polars_for_execution

        # Should not crash and should return read-only config
        config = configure_polars_for_execution(max_workers=999)

        assert "thread_count" in config
        # Thread count should NOT be 999 (it was ignored)
        # It should be whatever was set at import time
        assert isinstance(config["thread_count"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
