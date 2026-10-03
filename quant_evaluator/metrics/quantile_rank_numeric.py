"""Bounded average-rank correlation for complete quantile profiles.

This module ranks independent factors, never building an F-by-F matrix.
It deliberately preserves SciPy's Pearson [1, 0] normalization order.
"""
from __future__ import annotations

import numpy as np

# Rank products have quarter-integer units. Q*(Q-1)**2 < 2**53 is a
# sufficient bound for exact products/partial sums in those units.
# This conservative cap also bounds the fixed bucket-rank workspace.
_EXACT_RANK_QUANTILE_LIMIT = 65536
_RANK_TILE_ELEMENTS = 262144
_RANK_TILE_FEATURES = 512


def rank_monotonicity(matrix: np.ndarray) -> np.ndarray:
    """Return signed Spearman correlation, preserving whole-column validity.

    The input is the already-normalized float64 (Q, F) matrix. Tiles limit
    rankdata scratch relative to Q and never mutate the input. These are
    admission bounds, not a promise about SciPy allocator peak bytes.
    """
    from scipy.stats import rankdata, spearmanr

    quantiles, features = matrix.shape
    result = np.full(features, np.nan, dtype=np.float64)
    if quantiles < 3 or features == 0:
        return result
    if quantiles > _EXACT_RANK_QUANTILE_LIMIT:
        # Avoid asserting bit equivalence outside the exact-sum proof.
        bucket_index = np.arange(quantiles)
        for feature in range(features):
            column = matrix[:, feature]
            if np.isfinite(column).all() and np.any(column != column[0]):
                result[feature] = spearmanr(bucket_index, column).statistic
        return result

    center = (quantiles + 1) / 2
    bucket_ranks = np.arange(1, quantiles + 1, dtype=np.float64) - center
    reciprocal = np.true_divide(1, quantiles - 1)
    bucket_std = np.sqrt(np.dot(bucket_ranks, bucket_ranks) * reciprocal)
    width = min(_RANK_TILE_FEATURES, max(1, _RANK_TILE_ELEMENTS // quantiles))
    for start in range(0, features, width):
        block = matrix[:, start:start + width]
        complete = np.all(np.isfinite(block), axis=0)
        nonflat = np.any(block != block[0], axis=0)
        selected = np.flatnonzero(complete & nonflat)
        if selected.size == 0:
            continue
        profiles = np.ascontiguousarray(block[:, selected].T)
        centered = rankdata(profiles, method="average", axis=1) - center
        covariance = np.sum(centered * bucket_ranks[None, :], axis=1)
        variance = np.sum(centered * centered, axis=1)
        # np.corrcoef(...)[1, 0] divides by the profile std first, then
        # the bucket-index std. Swapping these divisions changes bits.
        correlation = covariance * reciprocal
        correlation /= np.sqrt(variance * reciprocal)
        correlation /= bucket_std
        np.clip(correlation, -1.0, 1.0, out=correlation)
        result[start + selected] = correlation
    return result
