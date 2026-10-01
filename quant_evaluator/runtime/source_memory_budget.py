"""Shared conservative logical-memory estimate for tiled COS sources."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


PREFETCH_BYTES_PER_WORKER = 256 * 1024**2


@dataclass(frozen=True)
class SourceMemoryEstimate:
    tile_factors: int
    assembly_bytes: int
    prefetch_bytes: int
    total_bytes: int


def estimate_source_memory(*, time_size: int, asset_size: int, factor_count: int,
                           max_tile_size: int, dtype_itemsize: int,
                           prefetch_workers: int, prefetch_enabled: bool,
                           extra_assembly_bytes_per_cell: int = 0) -> SourceMemoryEstimate:
    """Mirror the source adapter's existing logical-payload formula exactly."""
    values = (time_size, asset_size, factor_count, max_tile_size, dtype_itemsize,
              prefetch_workers, extra_assembly_bytes_per_cell)
    if (any(not isinstance(value, Integral) or isinstance(value, bool)
            for value in values)
            or type(prefetch_enabled) is not bool):
        raise ValueError("source memory estimate inputs must be integers with positive dimensions")
    time_size, asset_size, factor_count, max_tile_size, dtype_itemsize, \
        prefetch_workers, extra_assembly_bytes_per_cell = map(int, values)
    if (min(time_size, asset_size, factor_count, max_tile_size, dtype_itemsize) <= 0
            or prefetch_workers < 0 or extra_assembly_bytes_per_cell < 0):
        raise ValueError("source memory estimate inputs must be integers with positive dimensions")
    cells = time_size * asset_size
    tile_factors = min(max_tile_size, factor_count)
    assembly = (3 * cells * dtype_itemsize * tile_factors
                + 2 * cells * tile_factors
                + extra_assembly_bytes_per_cell * cells * tile_factors)
    prefetch = (prefetch_workers * PREFETCH_BYTES_PER_WORKER
                if prefetch_enabled else 0)
    return SourceMemoryEstimate(tile_factors, assembly, prefetch, assembly + prefetch)
