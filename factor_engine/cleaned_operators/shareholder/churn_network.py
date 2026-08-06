# -*- coding: utf-8 -*-
"""Shareholder turnover, category / nature entropy, pledge and network metrics.

Ranked-panel convention follows ``relation/*``: ``s1`` = top holder, ``s2`` =
second, …, with zero-fill for absent ranks.  Previous-snapshot panels are
``p1..p10``.  All operators consume pre-aggregated daily panels (source-side
snapshots as-of ``PubDate``) and return one scalar per (TradeDate, Symbol).
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
        category="shareholder",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "shareholder", "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:shareholder",
            f"unit:{unit}", "cost:1",
        ],
    )


def _stack(panels: list[pd.DataFrame]) -> np.ndarray:
    base = panels[0]
    arrays = [np.asarray(p.reindex(index=base.index, columns=base.columns).to_numpy(dtype=float)) for p in panels]
    return np.stack(arrays, axis=0)


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _safe_div(num, den):
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "ratio"):
    metadata = _meta(name, description, params, unit=unit)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"Shareholder_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="shareholder",
        business_category="shareholder",
        canonical=name,
        source="shareholder.churn_network",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {name}
    )
    return cls


def _cur_prev_ranked(args: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Split ``*args`` into current (s1..s10) and previous (p1..p10) arrays."""
    n_cur = min(10, len(args) // 2)
    cur = _stack(list(args[:n_cur]))
    prev = _stack(list(args[n_cur : 2 * n_cur]))
    return cur, prev


def _holder_weighted_churn(*args):
    cur, prev = _cur_prev_ranked(args)
    with np.errstate(invalid="ignore"):
        churn = 0.5 * np.nansum(np.abs(np.nan_to_num(cur) - np.nan_to_num(prev)), axis=0)
    return _frame_like(args[0], churn)


_mk(
    "holder_weighted_churn",
    "按股东 ID 匹配的加权持股变动：0.5*Σ|ShareRatio_cur - ShareRatio_prev|（进入/退出补0）。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    _holder_weighted_churn,
)


def _entry_share(*args):
    cur, prev = _cur_prev_ranked(args)
    prev_zero = np.nan_to_num(prev) == 0.0
    entry = np.nansum(np.where(prev_zero, np.nan_to_num(cur), 0.0), axis=0)
    return _frame_like(args[0], entry)


_mk(
    "holder_entry_share",
    "新进入前十大股东持股比例合计（上期视为0）。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    _entry_share,
)


def _exit_share(*args):
    cur, prev = _cur_prev_ranked(args)
    cur_zero = np.nan_to_num(cur) == 0.0
    exit_ = np.nansum(np.where(cur_zero, np.nan_to_num(prev), 0.0), axis=0)
    return _frame_like(args[0], exit_)


_mk(
    "holder_exit_share",
    "退出前十大股东的上期持股比例合计（本期视为0）。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    _exit_share,
)


def _net_entry_share(*args):
    entry = _entry_share(*args).to_numpy(dtype=float)
    exit_ = _exit_share(*args).to_numpy(dtype=float)
    return _frame_like(args[0], entry - exit_)


_mk(
    "holder_net_entry_share",
    "新进持股比例合计 - 退出持股比例合计。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    _net_entry_share,
)


def _rank_stability(*args):
    cur, prev = _cur_prev_ranked(args)
    cur_v = np.nan_to_num(cur)
    prev_v = np.nan_to_num(prev)
    _, rows, cols = cur_v.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            a = cur_v[:, row, col]
            b = prev_v[:, row, col]
            keep = (a != 0.0) | (b != 0.0)
            a_s, b_s = a[keep], b[keep]
            if len(a_s) < 3 or float(np.std(a_s)) <= _EPS or float(np.std(b_s)) <= _EPS:
                continue
            out[row, col] = float(pd.Series(a_s).corr(pd.Series(b_s), method="spearman"))
    return _frame_like(args[0], out)


_mk(
    "holder_rank_stability",
    "相同股东本期/上期持股比例的 Spearman 相关。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    _rank_stability,
)


def _pledge_ratio(pledge_shares, total_capital):
    return _safe_div(pledge_shares, total_capital)


_mk(
    "holder_pledge_ratio",
    "股东质押股数合计 / 总股本。",
    ["pledge_shares", "total_capital"],
    _pledge_ratio,
)


def _freeze_ratio(freeze_shares, total_capital):
    return _safe_div(freeze_shares, total_capital)


_mk(
    "holder_freeze_ratio",
    "股东冻结股数合计 / 总股本。",
    ["freeze_shares", "total_capital"],
    _freeze_ratio,
)


def _share_hhi(*args):
    stacked = _stack(list(args))
    values = np.nan_to_num(stacked)
    total = values.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = values / total
        hhi = np.sum(shares * shares, axis=0)
    hhi = np.where(total > 0, hhi, np.nan)
    return _frame_like(args[0], hhi)


_mk(
    "holder_pledge_concentration",
    "各股东质押股数份额 HHI。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    _share_hhi,
)
_mk(
    "holder_freeze_concentration",
    "各股东冻结股数份额 HHI。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    _share_hhi,
)


def _pledged_holder_count(*args):
    stacked = np.nan_to_num(_stack(list(args)))
    count = np.sum(stacked > 0, axis=0).astype(float)
    return _frame_like(args[0], count)


_mk(
    "holder_pledged_holder_count",
    "质押股数大于0的前十大股东数量。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    _pledged_holder_count,
)


def _pledge_change(pledge_ratio, lag=1):
    return pledge_ratio - pledge_ratio.shift(int(lag))


_mk(
    "holder_pledge_change",
    "当前质押率 - 上期质押率。",
    ["pledge_ratio", "lag"],
    _pledge_change,
)


def _pledge_churn(*args):
    cur, prev = _cur_prev_ranked(args)
    churn = np.nansum(np.abs(np.nan_to_num(cur) - np.nan_to_num(prev)), axis=0)
    return _frame_like(args[0], churn)


_mk(
    "holder_pledge_churn",
    "按股东 ID 匹配的质押变化绝对值合计。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
     "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    _pledge_churn,
)


def _float_concentration_gap(top10_concentration, top10_float_concentration):
    return top10_concentration - top10_float_concentration


_mk(
    "holder_float_concentration_gap",
    "前十大股东集中度 - 前十大流通股东集中度。",
    ["top10_concentration", "top10_float_concentration"],
    _float_concentration_gap,
)


def _locked_share_ratio(locked_shares, total_capital):
    return _safe_div(locked_shares, total_capital)


_mk(
    "holder_locked_share_ratio",
    "限售/受限股份占比。",
    ["locked_shares", "total_capital"],
    _locked_share_ratio,
)


def _concentration_slope(concentration, window=8):
    w = max(3, int(window))

    def _slope(vals: np.ndarray) -> float:
        finite = vals[np.isfinite(vals)]
        if len(finite) < 3:
            return np.nan
        t = np.arange(len(finite), dtype=float)
        if np.var(t) <= _EPS:
            return np.nan
        return float(np.cov(t, finite)[0, 1] / np.var(t))

    return concentration.rolling(w, min_periods=3).apply(_slope, raw=True)


_mk(
    "holder_concentration_slope",
    "集中度/HHI 多报告期趋势斜率。",
    ["concentration", "window"],
    _concentration_slope,
)


def _concentration_acceleration(concentration, window=8):
    slope = _concentration_slope(concentration, int(window))
    return slope - slope.shift(1)


_mk(
    "holder_concentration_acceleration",
    "集中度趋势二阶变化。",
    ["concentration", "window"],
    _concentration_acceleration,
)


def _weighted_entropy(*args):
    stacked = np.nan_to_num(_stack(list(args)))
    total = stacked.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(total[None, :, :] > 0, stacked / total, np.nan)
        valid = w > 0
        wv = np.where(valid, w, 0.0)
        h = -np.nansum(np.where(valid, wv * np.log(wv), 0.0), axis=0)
    n = np.sum(valid, axis=0).astype(float)
    out = np.where(n >= 2, h / np.log(n), np.nan)
    return _frame_like(args[0], out)


_mk(
    "holder_class_entropy",
    "股东类别持股权重熵（归一化）。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
    _weighted_entropy,
)
_mk(
    "holder_nature_entropy",
    "股份性质持股权重熵（归一化）。",
    ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
    _weighted_entropy,
)


def _common_holding_peer_return(peer_return, own_return, overlap):
    """共同持股 peer 收益：重叠加权的同行收益（剔除自身）。"""
    return peer_return - overlap * own_return


_mk(
    "holder_common_holding_peer_return",
    "共同持股同行收益（重叠加权，剔除自身）。",
    ["peer_return", "own_return", "overlap"],
    _common_holding_peer_return,
)


def _peer_return_breadth(breadth, scale=1.0):
    return breadth * float(scale)


_mk(
    "holder_peer_return_breadth",
    "股东跨股票 breadth（共同持股覆盖度）。",
    ["breadth", "scale"],
    _peer_return_breadth,
    unit="level",
)


def _shareholder_network_centrality(degree, total):
    return _safe_div(degree, total)


_mk(
    "holder_shareholder_network_centrality",
    "股东网络中心度（度中心度代理）。",
    ["degree", "total"],
    _shareholder_network_centrality,
    unit="level",
)


def _shareholder_overlap_ratio(shared_holders, total_holders):
    return _safe_div(shared_holders, total_holders)


_mk(
    "holder_shareholder_overlap_ratio",
    "共同股东 / 总股东数（重叠率）。",
    ["shared_holders", "total_holders"],
    _shareholder_overlap_ratio,
)
