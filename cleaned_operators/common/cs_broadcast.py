# -*- coding: utf-8
"""截面行统计广播：保持 panel shape 不变。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def broadcast_row_stat(x: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """将每行标量统计广播到所有列（shape/index/columns 与 x 一致）。"""
    row = pd.Series(values, index=x.index, dtype=float)
    arr = np.repeat(row.to_numpy()[:, None], len(x.columns), axis=1)
    return pd.DataFrame(arr, index=x.index, columns=x.columns, dtype=float)


def broadcast_row_stat_all_null_null(x: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """截面广播；该行无有效样本时输出 NULL。"""
    n = x.count(axis=1)
    row = pd.Series(values, index=x.index, dtype=float).where(n > 0, np.nan)
    return broadcast_row_stat(x, row)


def _finite_mask(x: pd.DataFrame) -> pd.DataFrame:
    """NEW-255: a production numeric factor's statistical sample is ``np.isfinite``
    — a ±Inf cell is as unusable as NaN and must never count toward the
    cross-sectional rank base (a ``notna``/``count`` mask would treat Inf as a
    valid extreme member and skew the rank)."""
    arr = x.to_numpy(dtype=float, copy=False)
    return pd.DataFrame(np.isfinite(arr), index=x.index, columns=x.columns)


def cs_rank_01(x: pd.DataFrame) -> pd.DataFrame:
    """截面 0-1 排名：单有效值 → 0.5；缺失/Inf 行保持 NULL。

    NEW-255: validity is ``np.isfinite`` — ±Inf is not a rankable extreme, so
    it is excluded from the cross-sectional base and its cell stays NULL.
    """
    valid = _finite_mask(x)
    masked = x.where(valid)
    r = masked.rank(axis=1, method="average")
    n = valid.sum(axis=1)
    denom = (n - 1).replace(0, np.nan)
    out = r.sub(1, axis=0).div(denom, axis=0)
    singleton = valid & np.broadcast_to((n <= 1).to_numpy()[:, None], out.shape)
    out = out.where(~singleton, 0.5)
    return out.where(valid, np.nan)


def cs_rank_pct(x: pd.DataFrame) -> pd.DataFrame:
    """截面 pandas 百分位排名（``rank_pct`` / ``cs_pct_rank``）。

    NEW-255: rank on the finite-masked panel so ±Inf does not participate in
    the percentile base; its cell stays NULL.
    """
    valid = _finite_mask(x)
    return x.where(valid).rank(pct=True, axis=1).where(valid, np.nan)
