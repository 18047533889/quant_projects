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


def cs_rank_01(x: pd.DataFrame) -> pd.DataFrame:
    """截面 0-1 排名：单有效值行 → 0.5。"""
    r = x.rank(axis=1, method="average")
    n = x.count(axis=1)
    denom = (n - 1).replace(0, np.nan)
    out = r.sub(1, axis=0).div(denom, axis=0)
    return out.where(n.gt(1), 0.5)
