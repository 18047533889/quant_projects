"""Bounded shared raw-input diagnosis for research factor batches."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np


DEFAULT_MAX_DIAGNOSTIC_CHUNK_BYTES = 128 * 1024 * 1024


def diagnose_raw_batch_in_chunks(
    batch,
    labels,
    *,
    config,
    diagnose_training_batch: Callable,
    max_chunk_bytes: int = DEFAULT_MAX_DIAGNOSTIC_CHUNK_BYTES,
):
    """Diagnose unchanged raw columns together under an explicit memory bound.

    The bound applies to the additional float64 input panel only. It is not a
    bound on all scratch used inside QE/FO; one factor is the minimum chunk.
    The callback keeps this module independent from the diagnostic authority.
    """
    if type(max_chunk_bytes) is not int or max_chunk_bytes < 1:
        raise ValueError("max_chunk_bytes must be a positive integer")
    if not batch.factor_ids:
        return {}

    bytes_per_factor = batch.values.shape[0] * batch.values.shape[1] * 8
    chunk_size = max(1, min(
        len(batch.factor_ids), max_chunk_bytes // max(1, bytes_per_factor)
    ))
    results = {}
    for start in range(0, len(batch.factor_ids), chunk_size):
        stop = min(start + chunk_size, len(batch.factor_ids))
        ids = batch.factor_ids[start:stop]
        diagnostic_values = np.array(
            batch.values[:, :, start:stop], dtype=float, copy=True
        )
        if batch.validity is not None:
            diagnostic_values[~batch.validity[:, :, start:stop]] = np.nan
        diagnostic_values[~np.isfinite(diagnostic_values)] = np.nan
        diagnostic_chunk = replace(
            batch, factor_ids=ids, values=diagnostic_values,
            validity=np.isfinite(diagnostic_values),
        )
        results.update(diagnose_training_batch(
            diagnostic_chunk, labels, config=config
        ))
        del diagnostic_chunk, diagnostic_values
    return results
