"""
Batch planning for large-scale evaluation.

Splits large factor batches into chunks for memory-efficient processing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch


@dataclass(frozen=True)
class ChunkDescriptor:
    """
    Descriptor for a chunk of a factor batch.

    Defines slicing boundaries for time, asset, and factor dimensions.
    """
    chunk_id: int
    time_slice: Tuple[int, int]  # [start, end)
    asset_slice: Tuple[int, int]  # [start, end)
    factor_slice: Tuple[int, int]  # [start, end)
    estimated_memory_mb: float
    priority: int = 0

    @property
    def time_range(self) -> range:
        return range(self.time_slice[0], self.time_slice[1])

    @property
    def asset_range(self) -> range:
        return range(self.asset_slice[0], self.asset_slice[1])

    @property
    def factor_range(self) -> range:
        return range(self.factor_slice[0], self.factor_slice[1])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (
            self.time_slice[1] - self.time_slice[0],
            self.asset_slice[1] - self.asset_slice[0],
            self.factor_slice[1] - self.factor_slice[0],
        )


@dataclass
class BatchPlan:
    """
    Execution plan for processing a factor batch.

    Divides batch into chunks that fit memory budget and defines
    execution order.
    """
    chunks: List[ChunkDescriptor] = field(default_factory=list)
    total_time: int = 0
    total_assets: int = 0
    total_factors: int = 0
    estimated_total_memory_mb: float = 0.0
    max_chunk_memory_mb: float = 0.0
    parallelizable: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.chunks:
            self.max_chunk_memory_mb = max(c.estimated_memory_mb for c in self.chunks)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def get_chunk_by_id(self, chunk_id: int) -> Optional[ChunkDescriptor]:
        """Retrieve chunk by its ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_ordered_chunks(self) -> List[ChunkDescriptor]:
        """Return chunks ordered by priority (higher first)."""
        return sorted(self.chunks, key=lambda c: (-c.priority, c.chunk_id))


def estimate_chunk_memory(
    time_size: int,
    asset_size: int,
    factor_size: int,
    dtype_size: int = 8,
    overhead_factor: float = 2.0,
) -> float:
    """
    Estimate memory footprint for a chunk in MB.

    Args:
        time_size: Number of time periods
        asset_size: Number of assets
        factor_size: Number of factors
        dtype_size: Bytes per element (default 8 for float64)
        overhead_factor: Multiplier for intermediate arrays and overhead

    Returns:
        Estimated memory in megabytes
    """
    base_elements = time_size * asset_size * factor_size
    base_bytes = base_elements * dtype_size
    total_bytes = base_bytes * overhead_factor
    return total_bytes / (1024 * 1024)


def create_batch_plan(
    factor_batch: FactorBatch,
    max_chunk_memory_mb: float = 512.0,
    time_chunk_size: Optional[int] = None,
    asset_chunk_size: Optional[int] = None,
    factor_chunk_size: Optional[int] = None,
    prioritize_time: bool = True,
) -> BatchPlan:
    """
    Create execution plan for a factor batch.

    Splits batch into chunks that respect memory budget. Chunking strategy:
    1. Factor dimension first (each factor can be evaluated independently)
    2. Time dimension second (sequential evaluation)
    3. Asset dimension last (cross-sectional, harder to split)

    Args:
        factor_batch: Input factor batch to plan
        max_chunk_memory_mb: Maximum memory per chunk in MB
        time_chunk_size: Fixed time chunk size (None = auto)
        asset_chunk_size: Fixed asset chunk size (None = auto)
        factor_chunk_size: Fixed factor chunk size (None = auto)
        prioritize_time: Process recent time periods first

    Returns:
        BatchPlan with chunk descriptors
    """
    T = factor_batch.num_times
    N = factor_batch.num_assets
    F = factor_batch.num_factors

    # Determine chunk sizes
    if factor_chunk_size is None:
        # Try to keep all factors together if possible
        single_factor_memory = estimate_chunk_memory(T, N, 1)
        if single_factor_memory <= max_chunk_memory_mb:
            factor_chunk_size = F
        else:
            factor_chunk_size = 1

    if time_chunk_size is None:
        # Estimate time chunk size that fits memory budget
        test_memory = estimate_chunk_memory(T, N, factor_chunk_size)
        if test_memory <= max_chunk_memory_mb:
            time_chunk_size = T
        else:
            # Binary search for appropriate time chunk size
            time_chunk_size = T
            while time_chunk_size > 1:
                test_memory = estimate_chunk_memory(time_chunk_size, N, factor_chunk_size)
                if test_memory <= max_chunk_memory_mb:
                    break
                time_chunk_size = time_chunk_size // 2
            time_chunk_size = max(1, time_chunk_size)

    if asset_chunk_size is None:
        # Try to keep all assets together
        test_memory = estimate_chunk_memory(time_chunk_size, N, factor_chunk_size)
        if test_memory <= max_chunk_memory_mb:
            asset_chunk_size = N
        else:
            # Split assets if needed
            asset_chunk_size = N
            while asset_chunk_size > 100:
                test_memory = estimate_chunk_memory(time_chunk_size, asset_chunk_size, factor_chunk_size)
                if test_memory <= max_chunk_memory_mb:
                    break
                asset_chunk_size = asset_chunk_size // 2
            asset_chunk_size = max(100, asset_chunk_size)

    # Generate chunks
    chunks = []
    chunk_id = 0

    for f_start in range(0, F, factor_chunk_size):
        f_end = min(f_start + factor_chunk_size, F)

        for t_start in range(0, T, time_chunk_size):
            t_end = min(t_start + time_chunk_size, T)

            for a_start in range(0, N, asset_chunk_size):
                a_end = min(a_start + asset_chunk_size, N)

                chunk_memory = estimate_chunk_memory(
                    t_end - t_start,
                    a_end - a_start,
                    f_end - f_start,
                )

                # Priority: recent time periods first if prioritize_time
                priority = (T - t_start) if prioritize_time else 0

                chunk = ChunkDescriptor(
                    chunk_id=chunk_id,
                    time_slice=(t_start, t_end),
                    asset_slice=(a_start, a_end),
                    factor_slice=(f_start, f_end),
                    estimated_memory_mb=chunk_memory,
                    priority=priority,
                )
                chunks.append(chunk)
                chunk_id += 1

    total_memory = sum(c.estimated_memory_mb for c in chunks)

    plan = BatchPlan(
        chunks=chunks,
        total_time=T,
        total_assets=N,
        total_factors=F,
        estimated_total_memory_mb=total_memory,
        parallelizable=True,
        metadata={
            "time_chunk_size": time_chunk_size,
            "asset_chunk_size": asset_chunk_size,
            "factor_chunk_size": factor_chunk_size,
            "max_chunk_memory_mb": max_chunk_memory_mb,
        },
    )

    return plan
