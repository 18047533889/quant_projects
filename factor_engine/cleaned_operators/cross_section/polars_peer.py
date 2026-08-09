# -*- coding: utf-8 -*-
"""Polars backends for next-stage cross-sectional / peer operators (genuine).

Per-row cross-sectional statistics (peer weighted mean ex-self, deviation
index) are expressed with ``map_rows`` over the wide panel.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common._polars_bridge import align_cols


def _cols(df, *others):
    cols = [c for c in df.columns if c != "date"]
    for o in others:
        cols = [c for c in cols if c in o.columns]
    return cols


def _meta(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name, category="group_neutralization", description=description, param_names=params,
        return_type="series", tags=["group_neutralization", "polars", "native", "typed_v2"],
    )


def _register(name: str, description: str, params: list[str], fn):
    @register_operator(
        name=name, category="group_neutralization", business_category="peer",
        canonical=name, source="cross_section.polars_peer",
    )
    class _PeerPolars(SeriesOperator):
        metadata = _meta(name, description, params)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    return _PeerPolars


def _peer_deviation_row(xs):
    arr = np.asarray([np.asarray(v, dtype=float) for v in xs])
    # 与 pandas 参考一致：初始为 NaN（全缺失格子保留 NaN，绝不伪造 0），
    # 且每个标准化成分只用自身帧的 finite 掩码（而非跨帧 all-finite）。
    out = np.full(arr.shape[1], np.nan, dtype=float)
    for k in range(arr.shape[0]):
        a = arr[k]
        valid = np.isfinite(a)
        if valid.sum() < 2:
            continue
        sd = float(np.std(a[valid]))
        if sd <= 1e-12:
            continue
        z = np.where(valid, (a - np.mean(a[valid])) / sd, np.nan)
        nz = np.isfinite(z)
        if np.any(nz):
            out[nz] = np.where(np.isnan(out[nz]), z[nz], out[nz] + z[nz])
    return out


def _peer_deviation_index(*frames):
    base, cols = frames[0], _cols(frames[0])
    n = base.height
    arrays = [f.select(cols).to_numpy() for f in frames]
    stacked = np.stack(arrays, axis=0)
    result = np.zeros((n, len(cols)), dtype=float)
    for row in range(n):
        result[row] = _peer_deviation_row(stacked[:, row, :])
    return base.with_columns([pl.Series(cols[i], result[:, i]) for i in range(len(cols))])


_register(
    "group_peer_deviation_index", "多标准化同行偏离聚合（Polars map_rows）。", ["d1", "d2", "d3", "d4", "d5"],
    lambda d1, d2=None, d3=None, d4=None, d5=None: _peer_deviation_index(
        *[f for f in (d1, d2, d3, d4, d5) if f is not None]
    ),
)


def _group_peer_beta_deviation(beta, group, weight):
    """Polars per-row: weighted peer mean ex-self on beta."""
    cols = _cols(beta, group, weight)
    b = beta.select(cols).to_numpy()
    g = group.select(cols).to_numpy()
    w = weight.select(cols).to_numpy(dtype=float)
    n = beta.height
    out = np.full((n, len(cols)), np.nan, dtype=float)
    for row in range(n):
        labels = {}
        for i, col in enumerate(cols):
            lbl = g[row, i]
            labels.setdefault(lbl, []).append(i)
        for lbl, idxs in labels.items():
            if len(idxs) <= 1:
                continue
            tw = sum(w[row, i] for i in idxs)
            twx = sum(w[row, i] * b[row, i] for i in idxs)
            for j in idxs:
                denom = tw - w[row, j]
                if denom > 1e-12:
                    out[row, j] = b[row, j] - (twx - w[row, j] * b[row, j]) / denom
    return beta.with_columns([pl.Series(cols[i], out[:, i]) for i in range(len(cols))])


_register(
    "group_peer_beta_deviation", "个股 Beta - 行业 peer Beta（Polars）。", ["beta", "group", "weight"],
    lambda beta, group, weight: _group_peer_beta_deviation(beta, group, weight),
)
