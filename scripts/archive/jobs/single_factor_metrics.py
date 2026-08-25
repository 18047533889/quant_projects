#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单因子指标计算（独立于批量评估，供 456 详情页渲染逐因子使用）。

对给定因子矩阵 (date × symbol)，用 quant_evaluator 计算：
  IC 时序 / RankIC 均值 / IR / 十分层 NAV / 多空 NAV / Sharpe / 回撤 / 胜率 / 换手
输入：per-factor parquet + close 矩阵。
"""
import json, sys, os, time, math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, "/home/sunhaiwei/quant_projects/vectorbt_qs")

# 中文字体
_CN_FONT = "/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf"
if os.path.exists(_CN_FONT):
    try:
        import matplotlib.font_manager as _fm
        _fm.fontManager.addfont(_CN_FONT)
        _CN_NAME = _fm.FontProperties(fname=_CN_FONT).get_name()
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = [_CN_NAME, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

from quant_evaluator.metrics.ic import compute_daily_ic, _spearman_rank_correlation
from quant_evaluator.metrics.portfolio_stats import (
    compute_sharpe_ratio, compute_maximum_drawdown, compute_win_rate,
    compute_long_short_returns,
)
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

QE_AVAILABLE = True

_HAS_CLOSE = None
def load_close():
    """加载收盘价矩阵 (date × symbol)，全历史。"""
    global _HAS_CLOSE
    if _HAS_CLOSE is not None:
        return _HAS_CLOSE
    try:
        import duckdb
        files = sorted(Path.home().glob("cos_data/StockDailyBar/*.parquet"))
        files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
        con = duckdb.connect()
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol, Close as close
            FROM read_parquet({files_str})
        """).df()
        mat = df.pivot_table(index='date', columns='symbol', values='close', aggfunc='first')
        mat.index = pd.to_datetime(mat.index)
        mat = mat.sort_index()
        _HAS_CLOSE = mat
        return mat
    except Exception:
        _HAS_CLOSE = pd.DataFrame()
        return _HAS_CLOSE


def compute_factor_metrics_single(factor_name: str, matrix: pd.DataFrame) -> dict:
    """单因子全指标计算。

    matrix: date × symbol 因子值矩阵（page_name 对应）。
    """
    close = load_close()
    if close.empty:
        return {}
    common_idx = matrix.index.intersection(close.index)
    if len(common_idx) < 20:
        return {}
    fv = matrix.reindex(index=common_idx)
    close_c = close.loc[common_idx]
    # 对齐列
    common_cols = close_c.columns.intersection(fv.columns)
    fv = fv[common_cols]
    close_c = close_c[common_cols]

    fwd = close_c.pct_change().shift(-1)

    T, N = fv.shape
    if T < 20 or N < 20:
        return {}

    # 每日 IC（手写 spearman 避免 QE FactorBatch 构造开销，逐因子够快）
    fv_a = fv.values.astype(np.float64)
    fwd_a = fwd.values.astype(np.float64)
    ic_arr = np.full(T, np.nan)
    valid_arr = np.zeros(T, dtype=int)
    for t in range(T):
        m = fv_a[t]; r = fwd_a[t]
        mask = np.isfinite(m) & np.isfinite(r)
        if mask.sum() < 20:
            continue
        valid_arr[t] = mask.sum()
        ic_arr[t] = _spearman_rank_correlation(m[mask], r[mask])

    ic_series = pd.Series(ic_arr, index=common_idx)

    # 面板常数因子（同一天所有股票值相同）→ rank 无效 → IC 全 NaN → 返回空（不可评估）
    if np.isfinite(ic_arr).sum() < 20 or not np.isfinite(ic_arr).any():
        return {}
    if np.nanstd(ic_arr) < 1e-12:
        return {}

    # 十分层 NAV
    group_ret = np.zeros((T, 10))
    mat_ranks = fv.rank(axis=1, method='first', pct=True).values
    valid_mask = np.isfinite(fv_a) & np.isfinite(fwd_a)
    group_ids = np.floor(mat_ranks * 10).clip(0, 9).astype(int)
    group_ids[~valid_mask] = -1
    for t in range(T):
        for k in range(10):
            mk = (group_ids[t] == k)
            if mk.any():
                group_ret[t, k] = float(np.nanmean(fwd_a[t, mk]))

    decile_navs = {f"G{k+1}": np.cumprod(1 + group_ret[:, k]) for k in range(10)}
    r_ls = group_ret[:, 9] - group_ret[:, 0]
    decile_navs["LS"] = np.cumprod(1 + r_ls)

    ls_nav = decile_navs["LS"]
    ls_rets = np.diff(ls_nav) / np.maximum(ls_nav[:-1], 1e-12)
    ls_rets_safe = np.concatenate([[0.0], np.nan_to_num(ls_rets)])
    n_years = T / 252.0

    try:
        ls_sharpe = float(compute_sharpe_ratio(ls_rets_safe, periods_per_year=252))
    except Exception:
        ls_sharpe = 0.0
    try:
        mdd_t = compute_maximum_drawdown(ls_rets_safe, missing_return_policy="zero_fill")
        ls_mdd = float(mdd_t[0]) if isinstance(mdd_t, tuple) else float(mdd_t)
    except Exception:
        ls_mdd = 0.0
    try:
        ls_winrate = float(compute_win_rate(ls_rets_safe))
    except Exception:
        ls_winrate = 0.0
    ls_annual = float(ls_nav[-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0.0
    g10_ann = float(decile_navs["G10"][-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0.0
    g1_ann = float(decile_navs["G1"][-1] ** (1.0 / max(n_years, 1e-6)) - 1) if T > 0 else 0.0
    g10_rets = np.diff(decile_navs["G10"]) / np.maximum(decile_navs["G10"][:-1], 1e-12)
    g1_rets = np.diff(decile_navs["G1"]) / np.maximum(decile_navs["G1"][:-1], 1e-12)
    g10_sharpe = float(compute_sharpe_ratio(np.nan_to_num(g10_rets), periods_per_year=252)) if len(g10_rets) > 1 else 0.0
    g1_sharpe = float(compute_sharpe_ratio(np.nan_to_num(g1_rets), periods_per_year=252)) if len(g1_rets) > 1 else 0.0

    turnover = 0.0
    try:
        turnover = float(np.nanmean(estimate_turnover_from_ranks(fv_a[..., None], min_obs=20)))
    except Exception:
        pass

    s = ic_series.dropna()
    mean_ric = float(s.mean()) if len(s) else 0.0
    std_ric = float(s.std()) if len(s) > 1 else 0.0
    ric_ir = mean_ric / std_ric if std_ric > 1e-9 else 0.0
    win_rate_ic = float((s > 0).sum() / len(s)) if len(s) else 0.0

    return {
        "ic_series": ic_series,
        "decile_navs": {k: v.tolist() for k, v in decile_navs.items()},
        "dates_out": [str(d)[:10] for d in common_idx],
        "perf": {
            "ls_sharpe": ls_sharpe, "ls_annual": ls_annual, "ls_mdd": ls_mdd,
            "ls_winrate": ls_winrate, "g10_annual": g10_ann, "g1_annual": g1_ann,
            "g10_sharpe": g10_sharpe, "g1_sharpe": g1_sharpe, "turnover": turnover,
            "n_periods": T, "start_date": str(common_idx[0])[:10], "end_date": str(common_idx[-1])[:10],
        },
        "mean_ic": mean_ric, "ic_ir": ric_ir, "ic_std": std_ric,
        "win_rate": ls_winrate,
        "mean_rankic": mean_ric, "rankic_ir": ric_ir,
    }


if __name__ == "__main__":
    # 自测：对 vol_volume_asym_ewma_flipped
    m = pd.read_parquet("/home/sunhaiwei/quant_projects/weekly_backtest_output/factor_matrices/vol_volume_asym_ewma_flipped.parquet")
    t0 = time.time()
    r = compute_factor_metrics_single("vol_volume_asym_ewma_flipped", m)
    print(f"耗时 {time.time()-t0:.1f}s")
    if r:
        print("mean_rankic:", r["mean_rankic"])
        print("ls_sharpe:", r["perf"]["ls_sharpe"])
        print("ls_annual:", r["perf"]["ls_annual"])
        print("n_periods:", r["perf"]["n_periods"])
        print("start~end:", r["perf"]["start_date"], "~", r["perf"]["end_date"])
    else:
        print("空结果")
