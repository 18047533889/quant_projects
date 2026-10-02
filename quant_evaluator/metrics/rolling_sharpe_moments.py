"""Stable rolling-window moment helpers used by Sharpe metrics."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conditioned_window_moments(
    values: np.ndarray,
    window: int,
    starts: np.ndarray,
    counts: np.ndarray,
    *,
    relative_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute selected window moments with a bounded, accurate fallback.

    First finite observation is the common anchor, so appending future data
    cannot change prior-window anchoring. Centered prefix moments are used
    only when a forward-error estimate fits the requested relative budget;
    all other windows use bounded batches of direct finite-window reductions.
    """
    values = np.asarray(values, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.float64)
    if starts.ndim != 1 or counts.shape != starts.shape:
        raise ValueError("starts and counts must be aligned one-dimensional arrays")
    if window <= 0 or relative_tolerance <= 0:
        raise ValueError("window and relative_tolerance must be positive")
    means = np.full(starts.size, np.nan, dtype=np.float64)
    stds = np.full(starts.size, np.nan, dtype=np.float64)
    if starts.size == 0:
        return means, stds

    counts = counts.astype(np.longdouble, copy=False)
    finite = np.isfinite(values)
    finite_positions = np.flatnonzero(finite)
    stable = np.zeros(starts.size, dtype=bool)
    extended_precision = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    # For short selected windows, bounded direct reductions are cheaper than
    # constructing long-double prefixes and are equally accurate. Keep this
    # branch based only on the requested window length so appending future
    # observations cannot change the route taken by an existing window.
    if window > 20 and finite_positions.size and extended_precision:
        anchor = np.longdouble(values[finite_positions[0]])
        shifted = np.where(finite, values.astype(np.longdouble) - anchor, np.longdouble(0.0))
        sum_prefix = np.concatenate(([np.longdouble(0.0)], np.cumsum(shifted, dtype=np.longdouble)))
        square_prefix = np.concatenate(([np.longdouble(0.0)], np.cumsum(shifted ** 2, dtype=np.longdouble)))
        abs_prefix = np.concatenate(([np.longdouble(0.0)], np.cumsum(np.abs(shifted), dtype=np.longdouble)))
        ends = starts + window
        local_sum = sum_prefix[ends] - sum_prefix[starts]
        local_squares = square_prefix[ends] - square_prefix[starts]
        correction = local_sum ** 2 / counts
        local_m2 = local_squares - correction
        eps = np.finfo(np.longdouble).eps
        output_eps = np.finfo(np.float64).eps
        gamma_end = ends * eps / (1.0 - ends * eps)
        gamma_start = starts * eps / (1.0 - starts * eps)
        sum_error = (
            gamma_end * abs_prefix[ends]
            + gamma_start * abs_prefix[starts]
            + eps * (np.abs(sum_prefix[ends]) + np.abs(sum_prefix[starts]))
        )
        square_error = (
            gamma_end * np.abs(square_prefix[ends])
            + gamma_start * np.abs(square_prefix[starts])
            + eps * (np.abs(square_prefix[ends]) + np.abs(square_prefix[starts]))
        )
        correction_error = (
            (2.0 * np.abs(local_sum) * sum_error + sum_error ** 2) / counts
            + eps * np.abs(correction)
        )
        m2_error = 8.0 * (
            square_error + correction_error
            + eps * (np.abs(local_squares) + np.abs(correction))
        )
        m2_error += 2.0 * output_eps * np.abs(local_m2)
        delta_mean = local_sum / counts
        local_mean = anchor + delta_mean
        mean_error = (
            sum_error / counts
            + eps * (np.abs(anchor) + np.abs(delta_mean))
            + 2.0 * output_eps * np.abs(local_mean)
        )
        # The variance bound is on M2; sqrt halves its relative error. The
        # tighter mean bound reserves room for both terms in Sharpe's ratio.
        stable = (
            np.isfinite(local_m2) & np.isfinite(m2_error)
            & np.isfinite(local_mean) & np.isfinite(mean_error)
            & (local_m2 > m2_error)
            & (m2_error <= 0.5 * relative_tolerance * local_m2)
            & (mean_error <= 0.25 * relative_tolerance * np.abs(local_mean))
        )
        means[stable] = local_mean[stable]
        stds[stable] = np.sqrt(local_m2[stable] / (counts[stable] - 1.0))

    # Bounded materialization: no T-by-window array is kept or created.
    rejected = np.flatnonzero(~stable)
    if rejected.size:
        windows = sliding_window_view(values, window)
        # Estimate six float payloads per row to leave room for masking and
        # NumPy reduction scratch. A single oversized row is the documented
        # exception to the 8 MiB estimated-payload target.
        bytes_per_row = max(1, window * np.dtype(np.float64).itemsize * 6)
        payload_budget = 8 * 1024 * 1024
        rows_per_batch = max(1, payload_budget // bytes_per_row)
        for offset in range(0, rejected.size, rows_per_batch):
            batch = rejected[offset:offset + rows_per_batch]
            block = windows[starts[batch]]
            finite_block = np.isfinite(block)
            complete_rows = finite_block.all(axis=1)
            means[batch[complete_rows]] = np.mean(block[complete_rows], axis=1)
            stds[batch[complete_rows]] = np.std(
                block[complete_rows], axis=1, ddof=1,
            )
            incomplete_rows = ~complete_rows
            if np.any(incomplete_rows):
                masked = np.where(finite_block[incomplete_rows], block[incomplete_rows], np.nan)
                means[batch[incomplete_rows]] = np.nanmean(masked, axis=1)
                stds[batch[incomplete_rows]] = np.nanstd(masked, axis=1, ddof=1)
                del masked
            del incomplete_rows, complete_rows, finite_block, block, batch
    return means, stds


__all__ = ["conditioned_window_moments"]

