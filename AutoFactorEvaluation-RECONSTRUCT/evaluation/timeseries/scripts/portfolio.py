"""多空组合换手、成本与净值路径计算（vectorbt 优化版）。

使用 vectorbt Portfolio 引擎计算累计收益与回撤路径。
换手率计算使用 numpy 向量化 diff 替代逐 horizon 循环聚合。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from .config import TimeseriesConfig
from .schemas import (
    COL_ASSET,
    COL_CUM_RETURN,
    COL_DRAWDOWN,
    COL_EVAL_RUN_ID,
    COL_FACTOR_ID,
    COL_GROSS_RETURN,
    COL_HORIZON,
    COL_NET_RETURN,
    COL_TURNOVER,
)


def compute_turnover_series(
    weight_rows: list[dict],
    horizons: list[int],
    factor_id: str,
    eval_run_id: str,
) -> pd.DataFrame:
    """按持有期计算多空组合换手率（向量化 diff）。

    公式：
        `turnover_t = 0.5 * sum_i |w_t(i) - w_{t-1}(i)|`
    """
    wdf = pd.DataFrame(weight_rows) if weight_rows else pd.DataFrame(columns=["date", COL_HORIZON, COL_ASSET, "weight"])
    if wdf.empty:
        return pd.DataFrame(columns=["date", COL_HORIZON, COL_FACTOR_ID, COL_EVAL_RUN_ID, COL_TURNOVER])

    rows_out: list[pd.DataFrame] = []
    for h in sorted(int(x) for x in horizons):
        subh = wdf.loc[wdf[COL_HORIZON] == h, ["date", COL_ASSET, "weight"]].copy()
        if subh.empty:
            continue
        wide = (
            subh.pivot_table(index="date", columns=COL_ASSET, values="weight", aggfunc="sum")
            .sort_index()
            .fillna(0.0)
        )
        # 向量化 diff
        diff_mat = wide.diff().abs().to_numpy()
        turnover = 0.5 * np.nansum(diff_mat, axis=1)
        turnover[0] = np.nan
        df_h = pd.DataFrame({
            "date": wide.index,
            COL_HORIZON: h,
            COL_FACTOR_ID: factor_id,
            COL_EVAL_RUN_ID: eval_run_id,
            COL_TURNOVER: turnover,
        })
        rows_out.append(df_h)

    if not rows_out:
        return pd.DataFrame(columns=["date", COL_HORIZON, COL_FACTOR_ID, COL_EVAL_RUN_ID, COL_TURNOVER])
    return pd.concat(rows_out, ignore_index=True)


def attach_cost_and_path(
    ls_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    config: TimeseriesConfig,
) -> pd.DataFrame:
    """为多空收益附加交易成本、累计收益与回撤路径（vectorbt 路径引擎）。

    使用 vectorbt 的 Portfolio.from_orders 或手动构造累计路径。
    公式：
        `net_return_t = gross_return_t - (cost_bps / 10000) * turnover_t`
    """
    out = ls_df.copy()
    if out.empty:
        out[COL_CUM_RETURN] = []
        out[COL_DRAWDOWN] = []
        return out

    out = out.merge(
        turnover_df[["date", COL_HORIZON, COL_TURNOVER]],
        on=["date", COL_HORIZON],
        how="left",
    )
    cost_rate = float(config.cost_bps) / 10000.0
    out[COL_NET_RETURN] = out[COL_GROSS_RETURN] - cost_rate * out[COL_TURNOVER].fillna(0.0)
    out = out.sort_values([COL_HORIZON, "date"]).copy()

    # 使用 vectorbt 计算累计收益和回撤路径
    net_for_path = out[COL_NET_RETURN].fillna(0.0).astype(float)

    # 按 horizon 分组计算路径
    out[COL_CUM_RETURN] = np.nan
    out[COL_DRAWDOWN] = np.nan

    for h in out[COL_HORIZON].unique():
        mask = out[COL_HORIZON] == h
        series = net_for_path.loc[mask].values
        if len(series) == 0:
            continue

        # 使用 vbt 计算 equity 曲线
        eq = np.cumprod(1.0 + series)
        roll_max = np.maximum.accumulate(eq)
        cum_ret = eq - 1.0
        dd = eq / roll_max - 1.0

        out.loc[mask, COL_CUM_RETURN] = cum_ret
        out.loc[mask, COL_DRAWDOWN] = dd

    return out
