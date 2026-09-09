# -*- coding: utf-8 -*-
"""Polars backends for next-stage cross-sectional / peer operators (genuine).

Per-row cross-sectional statistics (peer weighted mean ex-self, deviation
index) are expressed with ``map_rows`` over the wide panel.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common._polars_bridge import align_cols


def _cols(df, *others):
    cols = [c for c in df.columns if c not in {"date", "stock_code"}]
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
        if name == "group_peer_beta_deviation":
            from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
            metadata.tags = [tag for tag in metadata.tags if tag != "native"] + ["cpu_udf"]
            _physical_spec = PhysicalImplementationSpec(
                canonical=name, backend="polars", execution_kind=ExecutionKind.DELEGATE_PYTHON,
                materializes_full_panel=True, supports_nulls=True,
                supports_nan=True, supports_inf=True,
                implementation_source_hash="060d00a3a7155cced621bbd14a9e5097d7c90bf46c29c57a6a12e6e24467c78d",
                emitter_identity="polars.DataFrame.with_columns:python_row_kernel",
                kernel_identity="cross_section.peer_ops._peer_weighted_mean_ex_self_row",
                parameter_domain_hash="7c9f11328591ac2cfc7984669d73849f8f6e6bac67c9982a365d358adf94bcbd",
                semantic_contract_hash="21ef13190ef6cb03055cf5f5acb3048ed9661ac4c53946f24a80a436fbbdfb6f",
                notes="Eager per-row authoritative Python kernel; no native-expression certification.",
            )

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
    """Polars container over the authoritative overflow-safe row kernel."""
    from factor_engine.cleaned_operators.cross_section.peer_ops import (
        _peer_weighted_mean_ex_self_row,
    )

    cols = align_cols(beta, group, weight)
    if "stock_code" in beta.columns:
        raise ValueError("peer CPU bridge requires a wide panel; long rows are not cross sections")
    b = beta.select(cols).to_numpy().astype(float)
    g = group.select(cols).to_numpy()
    w = weight.select(cols).to_numpy().astype(float)
    n = beta.height
    out = np.full((n, len(cols)), np.nan, dtype=float)
    for row in range(n):
        peer = _peer_weighted_mean_ex_self_row(b[row], g[row], w[row])
        out[row] = np.where(np.isfinite(peer), b[row] - peer, np.nan)
    return beta.with_columns([pl.Series(cols[i], out[:, i]) for i in range(len(cols))])


_register(
    "group_peer_beta_deviation", "个股 Beta - 行业 peer Beta（Polars）。", ["beta", "group", "weight"],
    lambda beta, group, weight: _group_peer_beta_deviation(beta, group, weight),
)
