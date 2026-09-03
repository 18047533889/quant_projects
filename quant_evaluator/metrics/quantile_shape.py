"""Quantile-shape metrics derived from per-quantile returns (spec §29).

All functions consume a ``QuantileReturnArtifact``'s ``values`` array of
shape ``(n_quantiles, F)`` — the time-averaged forward return per quantile
bucket per factor — and return a per-factor scalar array of shape ``(F,)``.

Quantile encoding convention (QE2-P0-001): quantile 0 = LOWEST factor-value
group (bottom), quantile n_quantiles-1 = HIGHEST factor-value group (top).
A positively predictive factor has monotonically increasing returns from
bottom to top.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_quantile_monotonicity",
    "compute_quantile_curvature",
    "compute_quantile_tail_asymmetry",
    "compute_quantile_adjacent_spread",
    "compute_quantile_extreme_cliff",
    "compute_top_quantile_cliff",
    "compute_bottom_quantile_cliff",
]


def _as_matrix(qr: np.ndarray) -> np.ndarray:
    """Coerce to a float64 (n_quantiles, F) matrix."""
    m = np.asarray(qr, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    return m


def _finite_columns(m: np.ndarray) -> np.ndarray:
    """Boolean (F,) mask of columns with >= 2 finite quantile returns."""
    return np.sum(np.isfinite(m), axis=0) >= 2


def compute_quantile_monotonicity(qr: np.ndarray) -> np.ndarray:
    """Fraction of adjacent quantile steps that are monotone increasing, (F,).

    For each factor, count adjacent pairs (q, q+1) where
    ``ret[q+1] > ret[q]`` over the finite pairs, divided by the number of
    finite adjacent pairs.  NaN when fewer than 2 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    ok = _finite_columns(m)
    if nq < 2:
        return out
    for f in np.where(ok)[0]:
        col = m[:, f]
        finite = np.isfinite(col)
        pairs = finite[:-1] & finite[1:]
        n_pairs = int(np.sum(pairs))
        if n_pairs == 0:
            continue
        inc = np.sum((col[1:][pairs] > col[:-1][pairs]))
        out[f] = inc / n_pairs
    return out


def compute_quantile_curvature(qr: np.ndarray) -> np.ndarray:
    """Signed curvature of the quantile-return profile, (F,).

    Estimated as the second difference of the mean quantile returns
    (``ret[q+1] - 2*ret[q] + ret[q-1]``) averaged over interior quantiles.
    Positive curvature = convex (accelerating) profile; negative = concave
    (decelerating).  NaN when fewer than 3 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        # interior positions with all three neighbours finite
        interior = finite[1:-1] & finite[:-2] & finite[2:]
        if not np.any(interior):
            continue
        d2 = col[2:][interior] - 2.0 * col[1:-1][interior] + col[:-2][interior]
        out[f] = float(np.mean(d2))
    return out


def compute_quantile_tail_asymmetry(qr: np.ndarray) -> np.ndarray:
    """Asymmetry between the top and bottom quantile tails, (F,).

    ``(ret[top] - ret[mid]) - (ret[mid] - ret[bottom])`` where mid is the
    median quantile index.  Positive = top tail is stronger than the bottom
    tail.  NaN when the top/bottom/mid quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    mid = nq // 2
    for f in range(F):
        col = m[:, f]
        if not (np.isfinite(col[0]) and np.isfinite(col[-1]) and np.isfinite(col[mid])):
            continue
        out[f] = (col[-1] - col[mid]) - (col[mid] - col[0])
    return out


def compute_quantile_adjacent_spread(qr: np.ndarray) -> np.ndarray:
    """Mean absolute return difference between adjacent quantiles, (F,).

    A measure of how smooth (vs step-like) the quantile profile is.  NaN
    when fewer than 2 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    ok = _finite_columns(m)
    if nq < 2:
        return out
    for f in np.where(ok)[0]:
        col = m[:, f]
        finite = np.isfinite(col)
        pairs = finite[:-1] & finite[1:]
        if not np.any(pairs):
            continue
        out[f] = float(np.mean(np.abs(col[1:][pairs] - col[:-1][pairs])))
    return out


def compute_quantile_extreme_cliff(qr: np.ndarray) -> np.ndarray:
    """Mean of the top and bottom quantile cliffs, (F,).

    ``(cliff_top + cliff_bottom) / 2`` where cliff_top = ret[top] - ret[top-1]
    and cliff_bottom = ret[1] - ret[0].  A large value means the extreme
    quantiles carry most of the spread (a cliff profile).  NaN when the
    required quantile returns are not finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 2:
        return out
    for f in range(F):
        col = m[:, f]
        if not (np.isfinite(col[0]) and np.isfinite(col[1])
                and np.isfinite(col[-1]) and np.isfinite(col[-2])):
            continue
        cliff_top = col[-1] - col[-2]
        cliff_bottom = col[1] - col[0]
        out[f] = (cliff_top + cliff_bottom) / 2.0
    return out


def compute_top_quantile_cliff(qr: np.ndarray) -> np.ndarray:
    """Top-quantile cliff: ret[top] - ret[top-1], (F,)."""
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 2:
        return out
    for f in range(F):
        col = m[:, f]
        if np.isfinite(col[-1]) and np.isfinite(col[-2]):
            out[f] = col[-1] - col[-2]
    return out


def compute_bottom_quantile_cliff(qr: np.ndarray) -> np.ndarray:
    """Bottom-quantile cliff: ret[1] - ret[0], (F,)."""
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 2:
        return out
    for f in range(F):
        col = m[:, f]
        if np.isfinite(col[1]) and np.isfinite(col[0]):
            out[f] = col[1] - col[0]
    return out
