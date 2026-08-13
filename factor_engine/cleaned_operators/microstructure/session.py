# -*- coding: utf-8
"""微观结构算子：按 session 边界 reset 的 rolling / pct_change helper。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def session_key_from_index(index: pd.Index) -> pd.Series:
    """从时间索引提取 session 键（日频边界）。"""
    if isinstance(index, pd.DatetimeIndex):
        return pd.Series(index.normalize(), index=index, dtype="datetime64[ns]")
    return pd.Series(0, index=index)


def _group_by_session(series: pd.Series):
    s = pd.Series(series)
    session = session_key_from_index(s.index)
    if session.nunique() <= 1:
        return None, s
    return session, s


def pct_change_by_session(series: pd.Series, periods: int = 1) -> pd.Series:
    """按 session 分组 pct_change，避免跨日首 bar 引用前一日收盘价。"""
    session, s = _group_by_session(series)
    p = max(1, int(periods))
    if session is None:
        return s.pct_change(p)
    return s.groupby(session, group_keys=False).apply(lambda g: g.pct_change(p))


def rolling_by_session(
    series: pd.Series,
    window: int,
    agg: str,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """按 session 分组 rolling，避免跨日/session 污染。"""
    w = max(1, int(window))
    mp = w if min_periods is None else max(1, int(min_periods))
    session, s = _group_by_session(series)

    def _roll(group: pd.Series) -> pd.Series:
        roller = group.rolling(w, min_periods=mp)
        if agg == "sum":
            return roller.sum()
        if agg == "mean":
            return roller.mean()
        if agg == "std":
            return roller.std()
        if agg == "var":
            return roller.var()
        raise ValueError(f"unsupported rolling agg: {agg}")

    if session is None:
        return _roll(s)
    return s.groupby(session, group_keys=False).apply(_roll)


def rolling_cov_by_session(
    x: pd.Series,
    y: pd.Series,
    window: int,
    *,
    min_periods: int = 2,
) -> pd.Series:
    """按 session 分组 rolling covariance。"""
    w = max(2, int(window))
    mp = max(2, int(min_periods))
    xs, ys = pd.Series(x), pd.Series(y)
    session = session_key_from_index(xs.index)
    if session.nunique() <= 1:
        return xs.rolling(w, min_periods=mp).cov(ys)
    out = pd.Series(index=xs.index, dtype=float)
    for _, idx in xs.groupby(session).groups.items():
        loc = idx
        out.loc[loc] = xs.loc[loc].rolling(w, min_periods=mp).cov(ys.loc[loc])
    return out


def rolling_var_by_session(
    series: pd.Series,
    window: int,
    *,
    min_periods: int = 2,
) -> pd.Series:
    """按 session 分组 rolling variance。"""
    return rolling_by_session(series, window, "var", min_periods=min_periods)


def rolling_panel_by_session(
    panel: pd.DataFrame,
    window: int,
    agg: str,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """宽表 panel 逐列 session-aware rolling。"""
    return panel.apply(
        lambda col: rolling_by_session(col, window, agg, min_periods=min_periods)
    )


def apply_colwise_by_session(panel: pd.DataFrame, fn) -> pd.DataFrame:
    """宽表逐列应用 session-aware 一元变换。"""
    if hasattr(panel, "apply"):
        return panel.apply(fn)
    return fn(pd.Series(panel))


def session_cum_vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    """session 内累计 VWAP（不跨日）。"""
    session, p = _group_by_session(price)
    v = pd.Series(volume, index=p.index).astype(float)
    if session is None:
        pv = p * v
        return np.where(v.cumsum().replace(0, np.nan) != 0, pv.cumsum() / v.cumsum().replace(0, np.nan), np.nan)

    out = pd.Series(np.nan, index=p.index, dtype=float)
    for key in session.dropna().unique():
        mask = session == key
        gp = p[mask]
        gv = v[mask]
        pv = gp * gv
        out.loc[gp.index] = pv.cumsum() / gv.cumsum().replace(0, np.nan)
    return out


def session_vwap_deviation(close: pd.Series, price: pd.Series, volume: pd.Series) -> pd.Series:
    """close 相对 session 累计 VWAP 的偏差：close / vwap - 1。"""
    vwap = session_cum_vwap(price, volume)
    return np.where(vwap.replace(0, np.nan) - 1.0 != 0, close / vwap.replace(0, np.nan) - 1.0, np.nan)
