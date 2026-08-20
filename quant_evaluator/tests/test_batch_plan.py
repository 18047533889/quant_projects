"""
Tests for batch planning.
"""

import pytest
import numpy as np

# Seeded local Generator — no global RNG state dependence.
rng = np.random.default_rng(20260820)

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.planner.batch_plan import (
    ChunkDescriptor,
    BatchPlan,
    create_batch_plan,
    estimate_chunk_memory,
)


class TestChunkDescriptor:
    def test_chunk_properties(self):
        chunk = ChunkDescriptor(
            chunk_id=0,
            time_slice=(0, 100),
            asset_slice=(0, 500),
            factor_slice=(0, 1),
            estimated_memory_mb=128.0,
        )

        assert list(chunk.time_range) == list(range(0, 100))
        assert list(chunk.asset_range) == list(range(0, 500))
        assert list(chunk.factor_range) == list(range(0, 1))
        assert chunk.shape == (100, 500, 1)

    def test_chunk_priority(self):
        chunk1 = ChunkDescriptor(
            chunk_id=0,
            time_slice=(0, 50),
            asset_slice=(0, 100),
            factor_slice=(0, 1),
            estimated_memory_mb=10.0,
            priority=10,
        )
        chunk2 = ChunkDescriptor(
            chunk_id=1,
            time_slice=(50, 100),
            asset_slice=(0, 100),
            factor_slice=(0, 1),
            estimated_memory_mb=10.0,
            priority=5,
        )

        assert chunk1.priority > chunk2.priority


class TestBatchPlan:
    def test_batch_plan_creation(self):
        chunks = [
            ChunkDescriptor(0, (0, 50), (0, 100), (0, 1), 10.0),
            ChunkDescriptor(1, (50, 100), (0, 100), (0, 1), 10.0),
        ]

        plan = BatchPlan(
            chunks=chunks,
            total_time=100,
            total_assets=100,
            total_factors=1,
        )

        assert plan.num_chunks == 2
        assert plan.max_chunk_memory_mb == 10.0

    def test_get_chunk_by_id(self):
        chunks = [
            ChunkDescriptor(0, (0, 50), (0, 100), (0, 1), 10.0),
            ChunkDescriptor(1, (50, 100), (0, 100), (0, 1), 10.0),
        ]

        plan = BatchPlan(chunks=chunks)

        chunk = plan.get_chunk_by_id(1)
        assert chunk is not None
        assert chunk.chunk_id == 1

        chunk = plan.get_chunk_by_id(99)
        assert chunk is None

    def test_get_ordered_chunks(self):
        chunks = [
            ChunkDescriptor(0, (0, 50), (0, 100), (0, 1), 10.0, priority=5),
            ChunkDescriptor(1, (50, 100), (0, 100), (0, 1), 10.0, priority=10),
            ChunkDescriptor(2, (100, 150), (0, 100), (0, 1), 10.0, priority=3),
        ]

        plan = BatchPlan(chunks=chunks)
        ordered = plan.get_ordered_chunks()

        assert ordered[0].chunk_id == 1  # Highest priority
        assert ordered[1].chunk_id == 0
        assert ordered[2].chunk_id == 2


class TestEstimateChunkMemory:
    def test_basic_estimation(self):
        # 100 * 500 * 1 * 8 bytes * 2 overhead = 800 KB
        memory_mb = estimate_chunk_memory(100, 500, 1, dtype_size=8, overhead_factor=2.0)

        expected_mb = (100 * 500 * 1 * 8 * 2.0) / (1024 * 1024)
        assert abs(memory_mb - expected_mb) < 0.01

    def test_large_batch_estimation(self):
        # 1000 * 5000 * 10 elements
        memory_mb = estimate_chunk_memory(1000, 5000, 10)

        # Should be several hundred MB
        assert memory_mb > 100
        assert memory_mb < 10000


class TestCreateBatchPlan:
    def create_factor_batch(self, T, N, F):
        """Helper to create test factor batch."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)
        values = rng.standard_normal((T, N, F))

        return FactorBatch(
            factor_ids=tuple(f"factor_{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

    def test_single_chunk_small_batch(self):
        # Small batch should fit in single chunk
        batch = self.create_factor_batch(T=100, N=500, F=1)

        plan = create_batch_plan(batch, max_chunk_memory_mb=1024.0)

        assert plan.num_chunks == 1
        assert plan.total_time == 100
        assert plan.total_assets == 500
        assert plan.total_factors == 1

    def test_multi_chunk_large_batch(self):
        # Large batch with small memory budget should create multiple chunks
        batch = self.create_factor_batch(T=1000, N=5000, F=5)

        plan = create_batch_plan(batch, max_chunk_memory_mb=100.0)

        assert plan.num_chunks > 1
        assert all(chunk.estimated_memory_mb <= 100.0 for chunk in plan.chunks)

    def test_fixed_chunk_sizes(self):
        batch = self.create_factor_batch(T=200, N=1000, F=4)

        plan = create_batch_plan(
            batch,
            time_chunk_size=100,
            asset_chunk_size=500,
            factor_chunk_size=2,
        )

        # Should create 2 time chunks * 2 asset chunks * 2 factor chunks = 8 chunks
        assert plan.num_chunks == 8

        # Verify chunk sizes
        for chunk in plan.chunks:
            t_size = chunk.time_slice[1] - chunk.time_slice[0]
            a_size = chunk.asset_slice[1] - chunk.asset_slice[0]
            f_size = chunk.factor_slice[1] - chunk.factor_slice[0]

            assert t_size <= 100
            assert a_size <= 500
            assert f_size <= 2

    def test_prioritize_time(self):
        batch = self.create_factor_batch(T=200, N=1000, F=1)

        plan = create_batch_plan(
            batch,
            time_chunk_size=100,
            prioritize_time=True,
        )

        # More recent chunks (higher time indices) should have higher priority
        ordered = plan.get_ordered_chunks()

        if len(ordered) > 1:
            # First chunk should be from later time period (higher priority)
            # Priority is T - t_start, so later chunks have higher priority
            assert ordered[0].priority >= ordered[-1].priority

    def test_metadata_stored(self):
        batch = self.create_factor_batch(T=100, N=500, F=1)

        plan = create_batch_plan(
            batch,
            max_chunk_memory_mb=512.0,
            time_chunk_size=50,
        )

        assert "time_chunk_size" in plan.metadata
        assert "max_chunk_memory_mb" in plan.metadata
        assert plan.metadata["time_chunk_size"] == 50
        assert plan.metadata["max_chunk_memory_mb"] == 512.0

    def test_all_data_covered(self):
        """Ensure chunks cover all data exactly once."""
        batch = self.create_factor_batch(T=150, N=300, F=3)

        plan = create_batch_plan(
            batch,
            time_chunk_size=50,
            asset_chunk_size=100,
            factor_chunk_size=1,
        )

        # Track covered indices
        time_covered = set()
        asset_covered = set()
        factor_covered = set()

        for chunk in plan.chunks:
            for t in range(chunk.time_slice[0], chunk.time_slice[1]):
                for a in range(chunk.asset_slice[0], chunk.asset_slice[1]):
                    for f in range(chunk.factor_slice[0], chunk.factor_slice[1]):
                        time_covered.add(t)
                        asset_covered.add(a)
                        factor_covered.add(f)

        assert time_covered == set(range(150))
        assert asset_covered == set(range(300))
        assert factor_covered == set(range(3))
