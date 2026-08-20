# -*- coding: utf-8 -*-
"""Polars backends for next-stage shareholder operators (genuine expressions).

Ranked-panel and simple-ratio operators expressed directly with ``pl.Expr`` on
wide panels.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common._polars_bridge import align_cols

_EPS = 1e-12


def _meta(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name, category="shareholder", description=description, param_names=params,
        return_type="series", tags=["shareholder", "polars", "native", "typed_v2"],
    )


def _register(name: str, description: str, params: list[str], fn):
    @register_operator(
        name=name, category="shareholder", business_category="shareholder",
        canonical=name, source="shareholder.polars_churn_network",
    )
    class _ShareholderPolars(SeriesOperator):
        metadata = _meta(name, description, params)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    return _ShareholderPolars


def _cols(base, *others):
    cols = [c for c in base.columns if c != "date"]
    for o in others:
        cols = [c for c in cols if c in o.columns]
    return cols


def _safe_div(a, b):
    return a / b.fill_null(0.0).replace(0, None)


_register("holder_pledge_ratio", "股东质押/总股本（Polars）。", ["pledge_shares", "total_capital"],
          lambda ps, tc: _binary(ps, tc, _safe_div))
_register("holder_freeze_ratio", "股东冻结/总股本（Polars）。", ["freeze_shares", "total_capital"],
          lambda fs, tc: _binary(fs, tc, _safe_div))
_register("holder_locked_share_ratio", "限售股份占比（Polars）。", ["locked_shares", "total_capital"],
          lambda ls, tc: _binary(ls, tc, _safe_div))
_register("holder_float_concentration_gap", "前十大集中度差（Polars）。", ["top10_concentration", "top10_float_concentration"],
          lambda a, b: _binary(a, b, lambda x, y: x - y))
_register("holder_pledge_change", "质押率变化（Polars）。", ["pledge_ratio", "lag"],
          lambda pr, lag=1: pr.with_columns([(pr[c] - pr[c].shift(int(lag))).alias(c) for c in _cols(pr)]))
_register("holder_common_holding_peer_return", "共同持股 peer 收益（Polars）。", ["peer_return", "own_return", "overlap"],
          lambda pr, o, ov: _three(pr, o, ov, lambda p, r, l: p - l * r))
_register("holder_peer_return_breadth", "股东跨股票 breadth（Polars；scale 纯乘法参数，不改变 rank 排序，移除搜索面）。", ["breadth"],
          lambda b, scale=1.0: b.with_columns([(b[c] * float(scale)).alias(c) for c in _cols(b)]))
_register("holder_shareholder_network_centrality", "网络中心度（Polars）。", ["degree", "total"],
          lambda d, t: _binary(d, t, _safe_div))
_register("holder_shareholder_overlap_ratio", "共同股东重叠率（Polars）。", ["shared_holders", "total_holders"],
          lambda s, t: _binary(s, t, _safe_div))


def _binary(a, b, expr_fn):
    cols = _cols(a, b)
    return a.with_columns([expr_fn(a[c], b[c]).alias(c) for c in cols])


def _three(a, b, c, expr_fn):
    cols = _cols(a, b, c)
    return a.with_columns([expr_fn(a[col], b[col], c[col]).alias(col) for col in cols])


def _ranked(*frames):
    base = frames[0]
    cols = [c for c in base.columns if c != "date"]
    arrays = [f for f in frames]
    return base, cols, arrays


def _stacked_expr(frames, cols, expr_fn):
    """Build a per-column expression over stacked ranked panels."""
    out = []
    for c in cols:
        series = [f[c] for f in frames]
        out.append(expr_fn(series, c).alias(c))
    return out


def _hhi(*frames):
    base, cols, arrays = _ranked(*frames)
    out = []
    for c in cols:
        filled = [f[c].fill_null(0.0) for f in arrays]
        total = sum(filled)
        # P1-135: NaN (unknown pledge/freeze) must not be zero-filled back to 0;
        # an unknown share makes the top-10 HHI undefined -> fail closed to null.
        any_unknown = pl.any_horizontal([f[c].is_null() for f in arrays])
        shares = [f / total for f in filled]
        hhi = sum((s * s) for s in shares)
        out.append(
            pl.when((~any_unknown) & (total > 0)).then(hhi).otherwise(None).alias(c)
        )
    return base.with_columns(out)


_register("holder_pledge_concentration", "质押份额 HHI（Polars）。", ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
          lambda *a: _hhi(*a))
_register("holder_freeze_concentration", "冻结份额 HHI（Polars）。", ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
          lambda *a: _hhi(*a))


def _pledged_count(*frames):
    base, cols, arrays = _ranked(*frames)
    out = []
    for c in cols:
        # P1-135: NaN (unknown pledge) must not be counted as a non-pledger.
        any_unknown = pl.any_horizontal([f[c].is_null() for f in arrays])
        cnt = sum((f[c].fill_null(0.0) > 0).cast(pl.Float64) for f in arrays)
        out.append(pl.when(~any_unknown).then(cnt).otherwise(None).alias(c))
    return base.with_columns(out)


_register("holder_pledged_holder_count", "质押股东数（Polars）。", ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
          lambda *a: _pledged_count(*a))


# NOTE 2026-08: the historic slot-based (rank-position) polars implementations
# of holder_weighted_churn / holder_entry_share / holder_exit_share /
# holder_net_entry_share misread rank churn as shareholder entry/exit (the same
# defect the pandas layer had).  They were reworked to the ShareholderId-matched
# union computation in churn_network.py (pandas_numpy reference backend).  An
# ID-matched polars expression is not expressible as wide-panel elementwise exprs,
# so these four canonical keep the pandas reference backend only; the polars
# registrations were removed to avoid advertising a semantically-wrong backend.


def _concentration_slope(conc, window):
    w = max(3, int(window))
    out = []
    for c in _cols(conc):
        out.append(
            conc[c].rolling_map(lambda s: _np_slope(s), window_size=w, min_samples=3).alias(c)
        )
    return conc.with_columns(out)


def _np_slope(vals):
    import numpy as np

    v = np.asarray(vals, dtype=float)
    finite_mask = np.isfinite(v)
    t = np.arange(v.shape[0], dtype=float)[finite_mask]
    v = v[finite_mask]
    if len(v) < 3:
        return float("nan")
    # P1-136: centred dot product (population cov/var, SAME ddof) — the old
    # np.cov(ddof=1)/np.var(ddof=0) inflated the slope by n/(n-1).
    tb = float(np.mean(t))
    vb = float(np.mean(v))
    var = float(np.mean((t - tb) ** 2))
    if var <= _EPS:
        return float("nan")
    cov = float(np.mean((t - tb) * (v - vb)))
    return float(cov / var)


def _slope_no_snapshot(concentration, window=8, snapshot_date=None):
    # R4-100: the pandas contract is ``(concentration, window, snapshot_date)``;
    # the snapshot-aligned slope is not implemented in the polars backend, so
    # fail closed instead of silently computing a trading-day slope.
    if snapshot_date is not None:
        raise NotImplementedError(
            "holder_concentration_slope: snapshot_date-aligned slope is not "
            "implemented in the polars backend"
        )
    return _concentration_slope(concentration, int(window))


_register("holder_concentration_slope", "集中度趋势斜率（Polars）。",
          ["concentration", "window", "snapshot_date"], _slope_no_snapshot)
