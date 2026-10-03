"""Bounded, fail-closed materialization of a pandas factor frame onto target axes."""
from __future__ import annotations

import numpy as np
import pandas as pd


AXIS_REINDEX_CHUNK_BYTES = 8 * 1024**2


def _output_shares_frame_storage(output, frame) -> bool:
    # Inspect existing pandas manager buffers instead of converting an entire
    # bounded frame just to check whether the destination aliases its input.
    for values in frame._mgr.arrays:
        candidates = [values]
        for name in ("_ndarray", "_data", "_mask"):
            candidate = getattr(values, name, None)
            if candidate is not None:
                candidates.append(candidate)
        for candidate in candidates:
            if isinstance(candidate, np.ndarray) and np.shares_memory(
                    output, candidate):
                return True
    return False


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
    Memory-bounded writes reject output arrays sharing storage with the input,
    because snapshot-safe overlap handling could exceed the logical chunk budget.
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

    # Exact, unique axes need no indexers or reindexer: every source value is
    # already in its destination position. Keep this after shared-axis checks
    # so their established error priority is unchanged.
    output_aliases_frame = _output_shares_frame_storage(output, frame)
    target_date_index = pd.Index(dates)
    target_asset_index = pd.Index(assets)
    exact_axes = (frame.index.is_unique and frame.columns.is_unique
                  and frame.index.equals(target_date_index)
                  and frame.columns.equals(target_asset_index))
    # In unbounded mode, use reindex's snapshot semantics for aliases.
    if output_aliases_frame and not memory_bounded:
        exact_axes = False
    if exact_axes:
        date_positions = None
        asset_positions = None
        date_slice = slice(0, len(dates))
        asset_slice = slice(0, len(assets))
    else:
        date_positions = frame.index.get_indexer(dates)
        asset_positions = frame.columns.get_indexer(assets)
        if np.any(date_positions < 0) or np.any(asset_positions < 0):
            raise ValueError("factor final axes changed between passes")

        def contiguous_slice(positions):
            if not len(positions):
                return slice(0, 0)
            start = int(positions[0])
            if len(positions) == 1 or np.all(np.diff(positions) == 1):
                return slice(start, int(positions[-1]) + 1)
            return None

        date_slice = contiguous_slice(date_positions)
        asset_slice = contiguous_slice(asset_positions)

    if memory_bounded and output_aliases_frame:
        raise ValueError(
            "memory-bounded output must not share storage with the input frame")

    if not memory_bounded:
        if exact_axes:
            output[:, :] = frame.to_numpy(dtype=np.float64, copy=False)
        else:
            output[:, :] = frame.reindex(index=dates, columns=assets).to_numpy(
                dtype=np.float64, copy=False)
        return

    if not len(dates) or not len(assets):
        return
    bytes_per_row = (len(frame.columns) + len(assets)) * np.dtype(np.float64).itemsize
    if chunk_bytes < bytes_per_row:
        raise ValueError("chunk_bytes must accommodate at least one logical row")
    rows_per_chunk = chunk_bytes // bytes_per_row
    for row_start in range(0, len(dates), rows_per_chunk):
        row_stop = min(len(dates), row_start + rows_per_chunk)
        if exact_axes:
            row_frame = frame.iloc[row_start:row_stop]
            chunk = row_frame.to_numpy(dtype=np.float64, copy=False)
        else:
            if date_slice is not None:
                row_selector = slice(date_slice.start + row_start,
                                     date_slice.start + row_stop)
                row_frame = frame.iloc[row_selector]
            else:
                row_frame = frame.iloc[date_positions[row_start:row_stop]]
            if asset_slice is not None:
                chunk = row_frame.iloc[:, asset_slice].to_numpy(
                    dtype=np.float64, copy=False)
            else:
                chunk = row_frame.iloc[:, asset_positions].to_numpy(
                    dtype=np.float64, copy=False)
        output[row_start:row_stop, :] = chunk
        # Release temporaries before the next iteration allocates its row
        # selection and conversion payload, keeping the logical chunk bound.
        del chunk, row_frame


__all__ = ("AXIS_REINDEX_CHUNK_BYTES", "write_axis_aligned_float64")
