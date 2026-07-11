"""横截面 IC/RankIC/Kendall 计算。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：所有相关性分支统一使用一致的有限值掩码与边界处理（FID §2.3）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .config import TimeseriesConfig
from .schemas import (
    COL_DATETIME,
    COL_EVAL_RUN_ID,
    COL_FACTOR_ID,
    COL_FACTOR_VALUE,
    COL_FORWARD_RETURN,
    COL_HORIZON,
    COL_IC,
    COL_KENDALL,
    COL_RANK_IC,
    COL_VALID_ASSET_COUNT,
)
from .validation import Severity, ValidationBuffer


def _safe_kendall_tau(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    """Kendall tau-b（FID §2.3）。"""
    if x.size < 2:
        return np.nan
    if np.std(x) < eps or np.std(y) < eps:
        return np.nan
    stat = kendalltau(x, y, variant="b", nan_policy="raise").statistic
    if stat is None or not np.isfinite(stat):
        return np.nan
    return float(stat)


def _group_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    min_assets: int,
    eps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """分组 Pearson 相关（FID §2.3，O(N) 聚合 + O(G) 截面数）。"""
    work = df[[COL_DATETIME, COL_HORIZON, x_col, y_col]].copy()
    work[x_col] = work[x_col].astype(float)
    work[y_col] = work[y_col].astype(float)
    work["_x2"] = np.square(work[x_col].to_numpy(dtype=float, copy=False))
    work["_y2"] = np.square(work[y_col].to_numpy(dtype=float, copy=False))
    work["_xy"] = work[x_col].to_numpy(dtype=float, copy=False) * work[y_col].to_numpy(dtype=float, copy=False)

    grouped = work.groupby([COL_DATETIME, COL_HORIZON], sort=True)
    agg = grouped.agg(
        n=(x_col, "size"),
        sum_x=(x_col, "sum"),
        sum_y=(y_col, "sum"),
        sum_x2=("_x2", "sum"),
        sum_y2=("_y2", "sum"),
        sum_xy=("_xy", "sum"),
    )

    n = agg["n"].astype(float)
    num = n * agg["sum_xy"] - agg["sum_x"] * agg["sum_y"]
    den_x = n * agg["sum_x2"] - agg["sum_x"] ** 2
    den_y = n * agg["sum_y2"] - agg["sum_y"] ** 2
    den = np.sqrt(np.maximum(den_x, 0.0) * np.maximum(den_y, 0.0))
    corr = num / den

    cond_n = n >= float(min_assets)
    cond_den = den > float(eps)
    corr = corr.where(cond_n & cond_den, np.nan)
    zero_var = (~cond_den) & cond_n
    return corr.astype(float), n.astype(int), zero_var.astype(bool)


def compute_daily_ic_series(
    aligned_panel: pd.DataFrame,
    config: TimeseriesConfig,
    factor_id: str,
    eval_run_id: str,
    vb: ValidationBuffer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 (date, horizon) 计算 IC / RankIC / Kendall（FID §2.3）。"""
    key_df = (
        aligned_panel[[COL_DATETIME, COL_HORIZON]]
        .drop_duplicates()
        .sort_values([COL_DATETIME, COL_HORIZON])
        .reset_index(drop=True)
    )
    if key_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel = aligned_panel[[COL_DATETIME, COL_HORIZON, COL_FACTOR_VALUE, COL_FORWARD_RETURN]].copy()
    panel[COL_FACTOR_VALUE] = panel[COL_FACTOR_VALUE].astype(float)
    panel[COL_FORWARD_RETURN] = panel[COL_FORWARD_RETURN].astype(float)
    panel = panel.loc[np.isfinite(panel[COL_FACTOR_VALUE]) & np.isfinite(panel[COL_FORWARD_RETURN])].copy()

    if panel.empty:
        rows_ic = []
        rows_rank = []
        for date, horizon in key_df[[COL_DATETIME, COL_HORIZON]].itertuples(index=False):
            rows_ic.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_EVAL_RUN_ID: eval_run_id,
                    COL_IC: np.nan,
                    COL_VALID_ASSET_COUNT: 0,
                }
            )
            rows_rank.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_EVAL_RUN_ID: eval_run_id,
                    COL_RANK_IC: np.nan,
                    COL_KENDALL: np.nan,
                    COL_VALID_ASSET_COUNT: 0,
                }
            )
        return pd.DataFrame(rows_ic), pd.DataFrame(rows_rank)

    ic_s, n_s, zero_var_s = _group_corr(
        panel,
        COL_FACTOR_VALUE,
        COL_FORWARD_RETURN,
        min_assets=config.min_assets,
        eps=config.eps,
    )
    ranked = panel.copy()
    ranked["_x_rank"] = ranked.groupby([COL_DATETIME, COL_HORIZON])[COL_FACTOR_VALUE].rank(method="average")
    ranked["_y_rank"] = ranked.groupby([COL_DATETIME, COL_HORIZON])[COL_FORWARD_RETURN].rank(method="average")
    rank_ic_s, _, _ = _group_corr(
        ranked,
        "_x_rank",
        "_y_rank",
        min_assets=config.min_assets,
        eps=config.eps,
    )

    full_idx = pd.MultiIndex.from_frame(key_df[[COL_DATETIME, COL_HORIZON]])
    ic_s = ic_s.reindex(full_idx)
    n_s = n_s.reindex(full_idx).fillna(0).astype(int)
    zero_var_s = zero_var_s.reindex(full_idx).fillna(False).astype(bool)
    rank_ic_s = rank_ic_s.reindex(full_idx)

    kendall_map: dict[tuple[pd.Timestamp, int], float] = {}
    for (date, horizon), sub in panel.groupby([COL_DATETIME, COL_HORIZON], sort=True):
        x = sub[COL_FACTOR_VALUE].to_numpy(dtype=float)
        y = sub[COL_FORWARD_RETURN].to_numpy(dtype=float)
        kendall_map[(date, int(horizon))] = _safe_kendall_tau(x, y, config.eps) if len(sub) >= config.min_assets else np.nan

    rows_ic: list[dict] = []
    rows_rank: list[dict] = []
    for date, horizon in key_df[[COL_DATETIME, COL_HORIZON]].itertuples(index=False):
        key = (date, int(horizon))
        n = int(n_s.loc[(date, horizon)])
        if n < config.min_assets:
            vb.add(
                Severity.INFO,
                "insufficient_assets",
                f"date={date} horizon={horizon} n={n} < min_assets={config.min_assets}",
                count=1,
            )
        elif bool(zero_var_s.loc[(date, horizon)]):
            vb.add(
                Severity.WARNING,
                "zero_variance",
                f"date={date} horizon={horizon} zero variance in factor or return",
            )
        rows_ic.append(
            {
                "date": date,
                COL_HORIZON: int(horizon),
                COL_FACTOR_ID: factor_id,
                COL_EVAL_RUN_ID: eval_run_id,
                COL_IC: float(ic_s.loc[(date, horizon)]) if pd.notna(ic_s.loc[(date, horizon)]) else np.nan,
                COL_VALID_ASSET_COUNT: n,
            }
        )
        rows_rank.append(
            {
                "date": date,
                COL_HORIZON: int(horizon),
                COL_FACTOR_ID: factor_id,
                COL_EVAL_RUN_ID: eval_run_id,
                COL_RANK_IC: float(rank_ic_s.loc[(date, horizon)]) if pd.notna(rank_ic_s.loc[(date, horizon)]) else np.nan,
                COL_KENDALL: kendall_map.get(key, np.nan),
                COL_VALID_ASSET_COUNT: n,
            }
        )

    return pd.DataFrame(rows_ic), pd.DataFrame(rows_rank)
