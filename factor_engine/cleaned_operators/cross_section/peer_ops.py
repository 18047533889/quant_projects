# -*- coding: utf-8 -*-
"""Peer / industry relative-value and liquidity-commonality operators (P0/P1).

Daily panels in, daily panels out.  ``group`` panels carry a per-cell industry
or level label (string or numeric) and share the exact shape of ``x``; weights
are free-float capitalisation panels.  All kernels are causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str], *, unit: str = "ratio") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="group_neutralization",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "group_neutralization", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:peer",
            f"unit:{unit}", "cost:2",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "ratio"):
    metadata = _meta(name, description, params, unit=unit)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"PeerOps_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="group_neutralization",
        business_category="peer",
        canonical=name,
        source="cross_section.peer_ops",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {name}
    )
    return cls


def _aligned(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    base = frames[0]
    result = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            frame = frame.reindex(index=base.index, columns=base.columns)
        result.append(frame)
    return tuple(result)


def _peer_weighted_mean_ex_self_row(x_row: np.ndarray, g_row: np.ndarray, w_row: np.ndarray) -> np.ndarray:
    out = np.full(x_row.shape, np.nan, dtype=float)
    labels = pd.unique(g_row)
    for label in labels:
        idx = (g_row == label) & np.isfinite(x_row) & np.isfinite(w_row) & (w_row > 0)
        count = int(np.sum(idx))
        if count <= 1:
            continue
        total_w = float(np.sum(w_row[idx]))
        total_wx = float(np.sum(w_row[idx] * x_row[idx]))
        for j in np.flatnonzero(g_row == label):
            if not np.isfinite(x_row[j]) or not np.isfinite(w_row[j]) or w_row[j] <= 0:
                continue
            denom = total_w - w_row[j]
            if denom > _EPS:
                out[j] = (total_wx - w_row[j] * x_row[j]) / denom
    return out


def _group_peer_beta_deviation(beta, group, weight):
    beta, group, weight = _aligned(beta, group, weight)
    bv = beta.to_numpy(dtype=float)
    gv = group.to_numpy()
    wv = weight.to_numpy(dtype=float)
    rows, cols = bv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        peer = _peer_weighted_mean_ex_self_row(bv[row], gv[row], wv[row])
        out[row] = np.where(np.isfinite(peer), bv[row] - peer, np.nan)
    return _frame_like(beta, out)


_mk(
    "group_peer_beta_deviation",
    "个股滚动 Beta - 行业内加权 peer Beta（剔除自身）。",
    ["beta", "group", "weight"],
    _group_peer_beta_deviation,
)


def _peer_deviation_index(*args):
    """多标准化偏离聚合：逐日截面 zscore 后求和。

    初始值为 NaN;只有某个成分在当日截面内有有效 zscore 时才累加,全缺失的
    格子保持 NaN,绝不伪造 0 作为有效因子。
    """
    frames = _aligned(*args)
    base = frames[0]
    stacked = np.stack([f.to_numpy(dtype=float) for f in frames], axis=0)
    rows, cols = stacked.shape[1], stacked.shape[2]
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        acc = np.full(cols, np.nan, dtype=float)
        for k in range(stacked.shape[0]):
            a = stacked[k, row]
            valid = np.isfinite(a)
            if valid.sum() < 2:
                continue
            sd = float(np.std(a[valid]))
            if sd <= _EPS:
                continue
            z = np.where(valid, (a - np.mean(a[valid])) / sd, np.nan)
            nz = np.isfinite(z)
            if np.any(nz):
                acc[nz] = np.where(np.isnan(acc[nz]), z[nz], acc[nz] + z[nz])
        out[row] = acc
    return _frame_like(base, out)


_mk(
    "group_peer_deviation_index",
    "多个标准化同行偏离的聚合指数。",
    ["d1", "d2", "d3", "d4", "d5"],
    _peer_deviation_index,
    unit="level",
)


def _rolling_regression(y: pd.DataFrame, x: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    y, x = _aligned(y, x)
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    mp = max(int(min_periods), 3)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            a, b = yv[start : row + 1, col], xv[start : row + 1, col]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < mp or np.var(b[valid]) <= _EPS:
                continue
            a, b = a[valid], b[valid]
            # R5 P1-96: np.cov defaults to ddof=1 while np.var defaults to ddof=0,
            # so the old beta was scaled by n/(n-1).  Use the centered dot product
            # so numerator and denominator share a single ddof.
            xbar = float(np.mean(b))
            denom = float(np.sum((b - xbar) ** 2))
            out[row, col] = float(np.sum((a - np.mean(a)) * (b - xbar)) / denom)
    return _frame_like(y, out)


def _group_ex_self_mean_panel(own_ret: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Per-cell leave-one-out group mean of ``own_ret`` (unweighted)."""
    own_ret, group = _aligned(own_ret, group)
    xv = own_ret.to_numpy(dtype=float)
    gv = group.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        x_row, g_row = xv[row], gv[row]
        for label in pd.unique(g_row):
            idx = np.flatnonzero((g_row == label) & np.isfinite(x_row))
            if len(idx) <= 1:
                continue
            total = float(np.sum(x_row[idx]))
            for j in idx:
                out[row, j] = (total - x_row[j]) / (len(idx) - 1.0)
    return _frame_like(own_ret, out)


def _group_peer_information_diffusion(own_ret, group, window, lag):
    """Peer return is built *from the group panel* (industry ex-self mean), so
    the ``group`` input is actually used and the name matches the contract."""
    peer = _group_ex_self_mean_panel(own_ret, group)
    peer = peer.shift(int(lag))
    return _rolling_regression(own_ret, peer, int(window), max(3, int(window) // 5))


_mk(
    "group_peer_information_diffusion",
    "个股收益对行业 ex-self 同行过去收益的滚动回归 Beta（信息扩散）。",
    ["own_return", "group", "window", "lag"],
    _group_peer_information_diffusion,
    unit="level",
)


def _group_leader_laggard_exposure(own_ret, leader_ret, group, window, lag):
    # R5 P1-97: the ``group`` input must actually drive the exposure.  Derive a
    # per-group leader return by filtering ``leader_ret`` inside each group (max
    # finite leader return per group), then broadcast it to every member.  This
    # works both for a sparse leader panel (only the leader cell filled) and for
    # a broadcast panel (every member already carries the group's leader return).
    own_ret, leader_ret, group = _aligned(own_ret, leader_ret, group)
    lv = leader_ret.to_numpy(dtype=float)
    gv = group.to_numpy()
    rows, cols = lv.shape
    leader = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for label in pd.unique(gv[row]):
            idx = np.flatnonzero((gv[row] == label) & np.isfinite(lv[row]))
            if len(idx) == 0:
                continue
            leader[row, idx] = float(np.max(lv[row][idx]))
    leader = _frame_like(leader_ret, leader).shift(int(lag))
    return _rolling_regression(own_ret, leader, int(window), max(3, int(window) // 5))


_mk(
    "group_leader_laggard_exposure",
    "个股收益对行业领先股滞后收益的 Beta。",
    ["own_return", "leader_return", "group", "window", "lag"],
    _group_leader_laggard_exposure,
    unit="level",
)


def _group_return_dispersion_exposure(own_ret, dispersion, window):
    dchange = dispersion - dispersion.shift(1)
    return _rolling_regression(own_ret, dchange, int(window), max(3, int(window) // 5))


_mk(
    "group_return_dispersion_exposure",
    "个股收益对行业截面收益离散度变化的敏感度。",
    ["own_return", "industry_dispersion", "window"],
    _group_return_dispersion_exposure,
    unit="level",
)


def _within_group_rank(x_row: np.ndarray, g_row: np.ndarray, gvalue: Any, target_col: int) -> float:
    """Rank of the *target stock* (``target_col``) inside its group.

    The previous implementation returned ``order[-1]``, i.e. the rank of the
    last stock of the group in panel order, not the target stock's rank.
    """
    idx = np.flatnonzero(g_row == gvalue)
    if len(idx) < 3:
        return np.nan
    pos = int(np.where(idx == target_col)[0][0]) if target_col in idx else -1
    if pos < 0:
        return np.nan
    vals = x_row[idx]
    # R5 P1-98: exclude non-finite group members from ranking and give tied
    # values the average rank.  The old double argsort ranked NaN (as the
    # largest) and arbitrarily broke ties.
    if not np.isfinite(vals[pos]):
        return np.nan
    finite = vals[np.isfinite(vals)]
    if finite.size < 3:
        return np.nan
    target = vals[pos]
    below = int(np.sum(finite < target))
    equal = int(np.sum(finite == target))  # includes the target itself
    avg_rank_zero_based = below + (equal - 1) / 2.0
    return float(avg_rank_zero_based / (finite.size - 1))


def _group_multi_level_rank_consistency(x, group1, group2, group3):
    x, g1, g2, g3 = _aligned(x, group1, group2, group3)
    xv = x.to_numpy(dtype=float)
    g1v, g2v, g3v = g1.to_numpy(), g2.to_numpy(), g3.to_numpy()
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            ranks = np.array([
                _within_group_rank(xv[row], g1v[row], g1v[row, col], col),
                _within_group_rank(xv[row], g2v[row], g2v[row, col], col),
                _within_group_rank(xv[row], g3v[row], g3v[row, col], col),
            ])
            if np.any(~np.isfinite(ranks)):
                continue
            sd = float(np.std(ranks))
            # R5 P1-99: for three [0,1] ranks the maximum population std is
            # sqrt(2/9) ~= 0.4714; the old sqrt(3)*sd scaling could only reach
            # ~0.1835 at the bottom.  Scale by 3/sqrt(2) for the full 0-1 range
            # and clip defensively against float round-off.
            out[row, col] = float(np.clip(1.0 - sd * (3.0 / np.sqrt(2.0)), 0.0, 1.0))
    return _frame_like(x, out)


_mk(
    "group_multi_level_rank_consistency",
    "特征在 sw_l1/l2/l3 组内排名一致度（0~1）。",
    ["x", "group1", "group2", "group3"],
    _group_multi_level_rank_consistency,
    unit="level",
)


def _liquidity_beta(own_ret, liquidity, window):
    # The operator description promises a beta against the *change* in
    # liquidity; regress on delta(liquidity), not the raw level.
    d_liquidity = liquidity - liquidity.shift(1)
    return _rolling_regression(own_ret, d_liquidity, int(window), max(3, int(window) // 5))


_mk(
    "ts_market_liquidity_beta",
    "个股收益对市场流动性变化的滚动 Beta。",
    ["own_return", "market_liquidity", "window"],
    lambda r, l, window=60: _liquidity_beta(r, l, window),
    unit="level",
)
_mk(
    "ts_industry_liquidity_beta",
    "个股收益对行业流动性变化的滚动 Beta。",
    ["own_return", "industry_liquidity", "window"],
    lambda r, l, window=60: _liquidity_beta(r, l, window),
    unit="level",
)
