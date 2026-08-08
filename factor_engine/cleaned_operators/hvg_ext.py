# -*- coding: utf-8 -*-
"""Horizontal Visibility Graph (HVG) primitives (2026-08-08 Gemini V2 round).

The HVG of a window embeds a time series as a graph: nodes are the sample
points ``i`` and there is an edge ``(i, j)`` (``i < j``) when
``x_k < min(x_i, x_j)`` for every ``i < k < j`` — i.e. the two points can
"see" each other horizontally.  All five operators share one kernel that
builds the *directed* HVG in ``O(W)`` with a monotonic stack (each node's
in-degree from the past and out-degree to the future) and derives:

* ``ts_hvg_degree_entropy``          — Shannon entropy of the degree
  distribution ``H = -Σ p(k) log p(k)`` (P0, daily).
* ``ts_hvg_forward_backward_asymmetry`` — symmetric KL between the in-degree
  and out-degree distributions (a time-irreversibility measure; 0 ≈
  reversible, > 0 = asymmetric).  Laplace-smoothed (``alpha = 0.5``) so the
  KL never diverges on finite support.  The metric is fixed — it is NOT a
  searchable parameter (P0, daily).
* ``ts_hvg_clustering_coefficient``  — mean local clustering ``2E_i /
  (k_i(k_i-1))`` (P1, extended).
* ``ts_hvg_assortativity``           — Pearson degree assortativity over the
  edges (P1, extended).
* ``ts_hvg_motif_entropy``           — entropy of the size-3 motif type
  distribution (P1, extended).

Engineering contract (per the 2026-08-08 review):

* strict ``<`` edge test — ties are handled deterministically (equal values
  connect through the monotonic-stack "connect to remaining top" rule); no
  random jitter is ever added to break ties;
* the time axis is never compressed — only the longest trailing contiguous
  finite run is used and NaN fail-closes the window;
* the graph is built once per window and all five statistics are derived from
  that single structure.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamSpec
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_extended,
)

_EPS = 1e-12

# R5 P1-01: authoritative per-parameter contracts (no silent 5.9 -> 5).
_HVG_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=4, searchable=True),
    "min_periods": ParamSpec(dtype=int, min=4, searchable=True),
}


# ---------------------------------------------------------------------------
# shared HVG kernel
# ---------------------------------------------------------------------------
def _hvg_reference_O_W2(v: np.ndarray) -> list[tuple[int, int]]:
    """Reference HVG edge enumeration, strictly by the mathematical definition.

    Edge ``(i, j)`` (``i < j``) exists iff ``x_k < min(x_i, x_j)`` for EVERY
    intermediate ``i < k < j`` — a strict inequality, so an equal-height
    intermediate (e.g. ``x_k == x_i``) blocks the pair.  ``O(W²)`` per window;
    used as the ground truth for the monotonic-stack implementation.
    """
    n = v.shape[0]
    edges: list[tuple[int, int]] = []
    for i in range(n):
        vi = float(v[i])
        for j in range(i + 1, n):
            vj = float(v[j])
            m = vi if vi < vj else vj
            ok = True
            for k in range(i + 1, j):
                if float(v[k]) >= m:
                    ok = False
                    break
            if ok:
                edges.append((i, j))
    return edges


def _hvg_directed_degrees(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct HVG in/out degrees via a monotonic stack (strict ``<``).

    Returns ``(k_in, k_out)`` (length ``len(v)``).  Every edge ``(j, i)``
    (``j < i``) contributes ``+1`` to ``k_out[j]`` and ``+1`` to ``k_in[i]``.

    R5 P0-06 (ties): the stack keeps indices with *strictly decreasing* values.
    When the current value equals the stack top, the two are adjacent-visible
    (edge ``(top, i)``) and the current node REPLACES the old equal top — the
    old equal-height node can no longer be exposed to a future taller node
    (it would be blocked by the equal intermediate anyway).  Without this rule
    ``[1, 1, 2]`` would wrongly connect nodes 0 and 2.
    """
    n = v.shape[0]
    k_in = np.zeros(n, dtype=float)
    k_out = np.zeros(n, dtype=float)
    stack: list[int] = []
    for i in range(n):
        vi = float(v[i])
        while stack and float(v[stack[-1]]) < vi:
            j = stack.pop()
            k_out[j] += 1.0
            k_in[i] += 1.0
        if stack:
            j = stack[-1]
            k_out[j] += 1.0
            k_in[i] += 1.0
            if float(v[j]) == vi:
                stack[-1] = i          # equal top is replaced, not duplicated
            else:
                stack.append(i)
        else:
            stack.append(i)
    return k_in, k_out


def _hvg_edges(v: np.ndarray) -> list[tuple[int, int]]:
    """Edge list of the undirected HVG (as ``(min_i, max_i)`` pairs).

    Same strictly-decreasing stack with the equal-top replace rule (R5 P0-06)
    as :func:`_hvg_directed_degrees`; the two must agree with
    :func:`_hvg_reference_O_W2` on every input (property-tested).
    """
    n = v.shape[0]
    edges: list[tuple[int, int]] = []
    stack: list[int] = []
    for i in range(n):
        vi = float(v[i])
        while stack and float(v[stack[-1]]) < vi:
            j = stack.pop()
            edges.append((j, i))
        if stack:
            j = stack[-1]
            edges.append((j, i))
            if float(v[j]) == vi:
                stack[-1] = i          # equal top is replaced, not duplicated
            else:
                stack.append(i)
        else:
            stack.append(i)
    return edges


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = _EPS
    out = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        out += pi * (np.log(pi + eps) - np.log(qi + eps))
    return float(out)


def _hvg_stats(v: np.ndarray) -> dict[str, float]:
    """All HVG statistics for one window.  ``v`` must be finite and contiguous."""
    n = v.shape[0]
    if n < 4:
        return {}
    k_in, k_out = _hvg_directed_degrees(v)
    k_deg = k_in + k_out

    # ---- degree entropy (raw counts, no smoothing) ----
    _, counts = np.unique(k_deg, return_counts=True)
    p = counts / counts.sum()
    degree_entropy = -float(np.sum(p * np.log(p)))

    # ---- forward-backward asymmetry (symmetric KL, Laplace-smoothed) ----
    alpha = 0.5
    support = np.unique(np.concatenate((k_in, k_out)))
    p_in = np.zeros(support.size)
    p_out = np.zeros(support.size)
    for idx, k in enumerate(support):
        p_in[idx] = float(np.sum(k_in == k))
        p_out[idx] = float(np.sum(k_out == k))
    n_in = float(p_in.sum())
    n_out = float(p_out.sum())
    if n_in <= 0.0 or n_out <= 0.0:
        asymmetry = np.nan
    else:
        q_in = (p_in + alpha) / (n_in + alpha * support.size)
        q_out = (p_out + alpha) / (n_out + alpha * support.size)
        asymmetry = 0.5 * (_kl(q_in, q_out) + _kl(q_out, q_in))

    # ---- clustering coefficient ----
    edges = _hvg_edges(v)
    edge_set = set(edges)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    clus_vals: list[float] = []
    for i in range(n):
        nb = adj[i]
        k_i = len(nb)
        if k_i < 2:
            continue
        e_i = 0
        nb_list = sorted(nb)
        for a_ in range(len(nb_list)):
            for b_ in range(a_ + 1, len(nb_list)):
                u, vv = nb_list[a_], nb_list[b_]
                if (u, vv) in edge_set:
                    e_i += 1
        clus_vals.append(2.0 * e_i / (k_i * (k_i - 1.0)))
    clustering = float(np.mean(clus_vals)) if clus_vals else np.nan

    # ---- assortativity (degree correlation over edges) ----
    E = len(edges)
    if E >= 2:
        num = 0.0
        denom_k = 0.0
        denom_k2 = 0.0
        for a, b in edges:
            ka, kb = k_deg[a], k_deg[b]
            num += ka * kb
            denom_k += (ka + kb) / 2.0
            denom_k2 += (ka * ka + kb * kb) / 2.0
        mu = denom_k / E
        sxy = num / E - mu * mu
        sxx = denom_k2 / E - mu * mu
        assortativity = float(sxy / sxx) if abs(sxx) > _EPS else np.nan
    else:
        assortativity = np.nan

    # ---- motif entropy (size-3 induced patterns) ----
    # R5 P1-04: the semantics are the entropy of the *induced* 3-node subgraph
    # type distribution over all C(n,3) triples (not consecutive-window
    # visibility motifs).  O(n³) per window by construction — this operator is
    # extended-only (cost 5) and intentionally not on the daily grammar.
    motifs: dict[tuple[bool, bool, bool], int] = {}
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                pattern = (
                    (i, j) in edge_set,
                    (j, k) in edge_set,
                    (i, k) in edge_set,
                )
                motifs[pattern] = motifs.get(pattern, 0) + 1
                total += 1
    if total > 0:
        mp = np.asarray(list(motifs.values()), dtype=float) / total
        motif_entropy = -float(np.sum(mp * np.log(mp)))
    else:
        motif_entropy = np.nan

    return {
        "degree_entropy": degree_entropy,
        "asymmetry": asymmetry,
        "clustering": clustering,
        "assortativity": assortativity,
        "motif_entropy": motif_entropy,
    }


def _hvg_series(x2d: np.ndarray, window: int, min_periods: int, key: str) -> np.ndarray:
    rows, cols = x2d.shape
    w = max(4, int(window))
    mp = max(4, int(min_periods))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < mp:
                continue
            stats = _hvg_stats(v)
            val = stats.get(key, np.nan)
            if np.isfinite(val):
                out[r, c] = float(val)
    return out


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _hvg_check_params(window: int, min_periods: int, canonical: str) -> tuple[int, int]:
    w = int(window)
    mp = int(min_periods)
    if w < 4:
        raise ValueError(f"{canonical} requires window >= 4")
    if mp < 4:
        raise ValueError(f"{canonical} requires min_periods >= 4")
    if mp > w:
        raise ValueError(f"{canonical} requires min_periods <= window (got {mp} > {w})")
    return w, mp


def _ts_hvg_degree_entropy(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    w, mp = _hvg_check_params(window, min_periods, "ts_hvg_degree_entropy")
    out = _hvg_series(x.to_numpy(dtype=float), w, mp, "degree_entropy")
    return frame_like(x, out)


def _ts_hvg_forward_backward_asymmetry(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    w, mp = _hvg_check_params(window, min_periods, "ts_hvg_forward_backward_asymmetry")
    out = _hvg_series(x.to_numpy(dtype=float), w, mp, "asymmetry")
    return frame_like(x, out)


def _ts_hvg_clustering_coefficient(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    w, mp = _hvg_check_params(window, min_periods, "ts_hvg_clustering_coefficient")
    out = _hvg_series(x.to_numpy(dtype=float), w, mp, "clustering")
    return frame_like(x, out)


def _ts_hvg_assortativity(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    w, mp = _hvg_check_params(window, min_periods, "ts_hvg_assortativity")
    out = _hvg_series(x.to_numpy(dtype=float), w, mp, "assortativity")
    return frame_like(x, out)


def _ts_hvg_motif_entropy(x: pd.DataFrame, window: int = 60, min_periods: int = 8) -> pd.DataFrame:
    w, mp = _hvg_check_params(window, min_periods, "ts_hvg_motif_entropy")
    out = _hvg_series(x.to_numpy(dtype=float), w, mp, "motif_entropy")
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# registration (pandas + polars)
# ---------------------------------------------------------------------------
_SPECS: dict[str, dict[str, Any]] = {
    "ts_hvg_degree_entropy": {
        "fn": _ts_hvg_degree_entropy,
        "params": ["x", "window", "min_periods"],
        "category": "time_series_network",
        "domain": "path_geometry",
        "unit": "entropy",
        "cost": 4,
        "tags_extra": ["hvg"],
        "param_specs": _HVG_PARAM_SPECS,
    },
    "ts_hvg_forward_backward_asymmetry": {
        "fn": _ts_hvg_forward_backward_asymmetry,
        "params": ["x", "window", "min_periods"],
        "category": "time_series_network",
        "domain": "path_geometry",
        "unit": "entropy",
        "cost": 4,
        "tags_extra": ["hvg"],
    },
    "ts_hvg_clustering_coefficient": {
        "fn": _ts_hvg_clustering_coefficient,
        "params": ["x", "window", "min_periods"],
        "category": "time_series_network",
        "domain": "path_geometry",
        "unit": "ratio",
        "cost": 5,
        "tags_extra": ["hvg"],
    },
    "ts_hvg_assortativity": {
        "fn": _ts_hvg_assortativity,
        "params": ["x", "window", "min_periods"],
        "category": "time_series_network",
        "domain": "path_geometry",
        "unit": "ratio",
        "cost": 4,
        "tags_extra": ["hvg"],
    },
    "ts_hvg_motif_entropy": {
        "fn": _ts_hvg_motif_entropy,
        "params": ["x", "window", "min_periods"],
        "category": "time_series_network",
        "domain": "path_geometry",
        "unit": "entropy",
        "cost": 5,
        "tags_extra": ["hvg"],
    },
}


def _register() -> None:
    for canonical, spec in _SPECS.items():
        register_dual(
            canonical,
            spec["fn"],
            spec["params"],
            category=spec["category"],
            domain=spec["domain"],
            unit=spec["unit"],
            cost=spec["cost"],
            source="hvg_ext",
            tags_extra=spec["tags_extra"],
            output_unit=spec["unit"],
            param_specs=spec.get("param_specs"),
        )
    union_extended(*_SPECS.keys())


_register()
