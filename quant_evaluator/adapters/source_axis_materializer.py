"""Bounded, fail-closed materialization of a pandas factor frame onto target axes."""
from __future__ import annotations

import numpy as np


AXIS_REINDEX_CHUNK_BYTES = 8 * 1024**2


def write_axis_aligned_float64(frame, dates, assets, output, *,
                               required_dates=None, required_assets=None,
                               memory_bounded=False,
                               chunk_bytes=AXIS_REINDEX_CHUNK_BYTES) -> None:
    """Align one frame into ``output`` and preserve fail-closed dtype handling.

    Indexers validate target and optional shared-axis membership. The default
    uses pandas reindex into the supplied view. Set ``memory_bounded=True`` to
    write in row chunks without creating a full aligned panel temporary.
    Input values are converted to float64 during the write, preserving
    fail-closed conversion for nonnumeric objects. Chunk memory is a logical
    row-selection plus column-selection payload bound, not a process-memory
    cap. Pandas allocator/object and conversion overhead are outside it.
    """
    expected_shape = (len(dates), len(assets))
    if (not isinstance(output, np.ndarray) or output.ndim != 2
            or output.shape != expected_shape or output.dtype != np.dtype(np.float64)
            or not output.flags.writeable):
        raise ValueError("output must be a writable float64 array with target-axis shape")
    if type(memory_bounded) is not bool:
        raise ValueError("memory_bounded must be bool")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if (required_dates is None) != (required_assets is None):
        raise ValueError("required_dates and required_assets must be provided together")

    # Keep the earlier shared-axis error priority when both shared and final
    # target axes are absent from this frame.
    if required_dates is not None:
        required_date_positions = frame.index.get_indexer(required_dates)
        required_asset_positions = frame.columns.get_indexer(tuple(required_assets))
        if (np.any(required_date_positions < 0)
                or np.any(required_asset_positions < 0)):
            raise ValueError("indexed shared axes changed between passes")

    date_positions = frame.index.get_indexer(dates)
    asset_positions = frame.columns.get_indexer(assets)
    if np.any(date_positions < 0) or np.any(asset_positions < 0):
        raise ValueError("factor final axes changed between passes")

    if not memory_bounded:
        output[:, :] = frame.reindex(index=dates, columns=assets).to_numpy(
            dtype=np.float64, copy=False)
        return

    if not len(date_positions) or not len(asset_positions):
        return
    bytes_per_row = (len(frame.columns) + len(asset_positions)) * np.dtype(np.float64).itemsize
    if chunk_bytes < bytes_per_row:
        raise ValueError("chunk_bytes must accommodate at least one logical row")
    rows_per_chunk = chunk_bytes // bytes_per_row
    for row_start in range(0, len(date_positions), rows_per_chunk):
        row_stop = min(len(date_positions), row_start + rows_per_chunk)
        row_frame = frame.iloc[date_positions[row_start:row_stop]]
        chunk = row_frame.iloc[:, asset_positions].to_numpy(
            dtype=np.float64, copy=False)
        output[row_start:row_stop, :] = chunk
        # Release temporaries before the next iteration allocates its row
        # selection and conversion payload, keeping the logical chunk bound.
        del chunk, row_frame


__all__ = ("AXIS_REINDEX_CHUNK_BYTES", "write_axis_aligned_float64")
