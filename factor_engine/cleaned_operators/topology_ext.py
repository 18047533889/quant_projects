# -*- coding: utf-8 -*-
"""Topological persistence-entropy operators (2026-08 geometry/math expansion).

Each operator builds the Takens delay embedding of the trailing window, computes
a persistence diagram, and returns the normalised Shannon entropy of the
lifetime distribution ``ell = death - birth``:

    p = ell / sum(ell),   H = -sum(p log p) / log(#pairs).

A high persistence entropy means the topological features have a broad range
of lifetimes (heterogeneous structure); a low value means the signal is
dominated by a single long-lived feature (e.g. one persistent cycle).

Homology degree is a **semantic part of the canonical** — 3rd-round audit P0-03.
``ts_persistence_entropy_h1`` uses the pure-numpy Vietoris-Rips H1 reduction
(``cleaned_operators.advanced_topology``); ``ts_persistence_entropy_h0`` uses a
single-linkage union-find over the points sorted by ascending pairwise distance
(every point born at 0, dying at the merge distance).  There is **no silent
H1→H0 fallback**: the two canonicals are environment-independent and mean the
same thing on every machine.  If the H1 kernel is unavailable the H1 operator
raises ``RuntimeError`` (production unsupported) instead of quietly switching
homology degree.

Both operators are trailing-window, prefix-causal and deterministic.  A window
that yields no persistence pairs emits NaN.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12

try:  # H1 Rips persistence kernel from the topology reference module.
    from cleaned_operators.advanced_topology import _rips_h1_pairs, _takens_points

    _HAVE_H1 = True
except Exception:  # pragma: no cover - optional kernel, no silent fallback.
    _rips_h1_pairs = None  # type: ignore
    _takens_points = None  # type: ignore
    _HAVE_H1 = False


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="topology",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "topology", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:topology",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _h0_pairs_union_find(points: np.ndarray) -> list[tuple[float, float]]:
    """H0 persistence pairs via single-linkage union-find."""
    n = int(points.shape[0])
    if n < 2:
        return []
    d = np.sqrt(np.maximum(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1), 0.0))
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(d[i, j]), i, j))
    edges.sort(key=lambda e: (e[0], e[1], e[2]))
    parent = list(range(n))
    size = [1] * n
    comps = n
    pairs: list[tuple[float, float]] = []

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for dist, i, j in edges:
        if comps <= 1:
            break
        ri, rj = _find(i), _find(j)
        if ri == rj:
            continue
        if size[ri] < size[rj]:
            ri, rj = rj, ri
        # component rj dies at this merge distance; it was born at 0.
        pairs.append((0.0, dist))
        parent[rj] = ri
        size[ri] += size[rj]
        comps -= 1
    return pairs


def _h0_persistence_pairs(chunk: np.ndarray, tau: int, dim: int) -> list[tuple[float, float]]:
    """H0 persistence pairs of the robust-normalised Takens embedding."""
    finite = chunk[np.isfinite(chunk)]
    if finite.size < 4:
        return []
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * 1.4826
    spread = mad if mad > 1e-12 else float(np.std(finite))
    if spread <= 1e-12:
        return []
    z = (chunk - med) / spread
    n = int(chunk.shape[0])
    lag = tau * (dim - 1)
    if n < lag + 3:
        return []
    pts = np.stack([z[s - lag : s + 1 : tau] for s in range(lag, n)], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 4:
        return []
    return _h0_pairs_union_find(pts)


def _h1_persistence_pairs(chunk: np.ndarray, tau: int, dim: int) -> list[tuple[float, float]]:
    """H1 Rips persistence pairs.  Raises if the H1 kernel is unavailable —
    the canonical must not silently change homology degree (P0-03)."""
    if not _HAVE_H1:
        raise RuntimeError(
            "ts_persistence_entropy_h1 requires the H1 Rips kernel "
            "(cleaned_operators.advanced_topology) which is unavailable in this "
            "environment. Use ts_persistence_entropy_h0, or install the kernel."
        )
    pts = _takens_points(chunk, tau, dim)
    if pts is None:
        return []
    return _rips_h1_pairs(pts)


def _persistence_pairs(chunk: np.ndarray, tau: int, dim: int, h0: bool) -> list[tuple[float, float]]:
    if h0:
        return _h0_persistence_pairs(chunk, tau, dim)
    return _h1_persistence_pairs(chunk, tau, dim)


def _persistence_entropy(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return np.nan
    lifetimes = np.asarray([death - birth for birth, death in pairs], dtype=float)
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if lifetimes.size == 0:
        return np.nan
    # R6-118: zero-lifetime features (birth == death) contribute zero to the
    # probability mass but still increase ``count``, silently changing the
    # entropy normalisation of the features that DO persist.  Drop them BEFORE
    # computing p / count / entropy so the measure reflects the genuine
    # persistence structure only.
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > _EPS)]
    if lifetimes.size == 0:
        return np.nan
    total = float(lifetimes.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.nan
    p = np.clip(lifetimes / total, 0.0, 1.0)
    ent = -float(np.sum(p * np.log(np.clip(p, 1e-15, 1.0))))
    count = int(lifetimes.size)
    if count > 1:
        ent = ent / np.log(count)
    return float(max(ent, 0.0))


def _persistence_entropy_series(x2d: np.ndarray, window: int, tau: int, dim: int, h0: bool) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            pairs = _persistence_pairs(col[i0 : r + 1], tau, dim, h0)
            out[r, c] = _persistence_entropy(pairs)
    return out


def _register(name: str, description: str, *, h0: bool) -> SeriesOperator:
    suffix = "h0" if h0 else "h1"
    canonical = f"ts_persistence_entropy_{suffix}"

    @register_operator(
        name=canonical,
        category="topology",
        business_category="topology",
        canonical=canonical,
        source="topology_ext",
    )
    class _PersistenceEntropy(SeriesOperator):
        metadata = _metadata(
            canonical,
            description,
            ["x", "window", "tau", "dim"],
            unit="entropy",
            cost=8,
        )

        def _calculate_series(
            self, x: pd.DataFrame, window: int = 120, tau: int = 1, dim: int = 3, **_: Any
        ) -> pd.DataFrame:
            w = int(window)
            if w < 6:
                raise ValueError("window must be >= 6")
            t = int(tau)
            if t < 1:
                raise ValueError("tau must be >= 1")
            d = int(dim)
            if not (2 <= d <= 6):
                raise ValueError("dim must be in [2, 6]")
            return frame_like(x, _persistence_entropy_series(x.to_numpy(dtype=float), w, t, d, h0))

    return _PersistenceEntropy


_register(
    "ts_persistence_entropy_h0",
    "持久性寿命熵（H0：单链 union-find，零维连接分量的合并距离）。环境无关。",
    h0=True,
)
_register(
    "ts_persistence_entropy_h1",
    "持久性寿命熵（H1：Vietoris-Rips 一维环；内核不可用时抛错，不静默降级）。",
    h0=False,
)

_NEW_CANONICALS = (
    "ts_persistence_entropy_h0",
    "ts_persistence_entropy_h1",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    # R5-50: live extend mutator, never a frozenset reassignment.
    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
