#!/usr/bin/env python3
"""用自实现 + quant_evaluator 子集 算每个因子的指标"""
import sys, os, json, warnings, math
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

PROJECT = Path("/home/sunhaiwei/quant_projects")
FV_PATH = PROJECT / "weekly_backtest_output" / "factor_values.parquet"
LOCAL = Path.home() / "cos_data" / "StockDailyBar"


def load_close_matrix() -> pd.DataFrame:
    files = sorted(LOCAL.glob("*.parquet"))
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as date, Symbol as symbol, Close as close
        FROM read_parquet({files_str})
    """).df()
    mat = df.pivot_table(index='date', columns='symbol', values='close', aggfunc='first')
    mat.index = pd.to_datetime(mat.index)
    mat = mat.sort_index()
    return mat


def compute_annualized_sharpe(daily_rets: np.ndarray, periods: int = 252) -> float:
    s = np.nanstd(daily_rets)
    if s == 0 or np.isnan(s):
        return 0.0
    return float(np.nanmean(daily_rets) / s * math.sqrt(periods))


def compute_max_drawdown(nav: np.ndarray) -> float:
    if len(nav) == 0:
        return 0.0
    nav = np.nan_to_num(nav, nan=1.0)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(np.nanmin(dd))


def compute_win_rate(rets: np.ndarray) -> float:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return 0.0
    return float(np.mean(rets > 0))


def eval_factor(name: str, fv_mat: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Evaluate one factor: IC, decile NAV, LS sharpe, MDD"""
    fv_mat = fv_mat.copy()
    fv_mat.index = pd.to_datetime(fv_mat.index)
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    common = fv_mat.index.intersection(close.index)
    if len(common) < 30:
        return {}
    fv_mat = fv_mat.loc[common]
    close = close.loc[common]

    fwd_ret = close.pct_change().shift(-1)

    fv_arr = fv_mat.values  # (T, N)
    fwd_arr = fwd_ret.values
    T, N = fv_arr.shape

    # Vectorized: cross-section rank + corr per day
    ic_arr = np.zeros(T)
    ric_arr = np.zeros(T)
    for t in range(T):
        m = np.isfinite(fv_arr[t]) & np.isfinite(fwd_arr[t])
        if m.sum() < 20:
            continue
        x = fv_arr[t, m]
        y = fwd_arr[t, m]
        if x.std() > 0 and y.std() > 0:
            ic_arr[t] = np.corrcoef(x, y)[0, 1]
        xr = pd.Series(x).rank().values
        yr = pd.Series(y).rank().values
        if xr.std() > 0 and yr.std() > 0:
            ric_arr[t] = np.corrcoef(xr, yr)[0, 1]

    mean_ic = float(np.nanmean(ic_arr))
    mean_rankic = float(np.nanmean(ric_arr))
    std_rankic = float(np.nanstd(ric_arr))
    rankic_ir = mean_rankic / std_rankic if std_rankic > 0 else 0

    # 十分层收益
    g_ret = np.zeros((T, 10))
    for t in range(T):
        m = np.isfinite(fv_arr[t]) & np.isfinite(fwd_arr[t])
        if m.sum() < 50:
            continue
        x = fv_arr[t, m]
        y = fwd_arr[t, m]
        try:
            ranks = pd.Series(x).rank(method='first').values
            q = pd.qcut(ranks, 10, labels=False, duplicates='drop')
            for k in range(10):
                mask = (q == k)
                if mask.sum() > 0:
                    g_ret[t, k] = float(np.mean(y[mask]))
        except Exception:
            continue

    g_nav = np.cumprod(1 + g_ret, axis=0)
    r_ls = g_ret[:, 9] - g_ret[:, 0]
    ls_nav = np.cumprod(1 + r_ls)
    ls_sharpe = compute_annualized_sharpe(r_ls)
    ls_mdd = compute_max_drawdown(ls_nav)
    ls_winrate = compute_win_rate(r_ls)
    ls_annual = ls_nav[-1] ** (252 / len(ls_nav)) - 1 if len(ls_nav) > 0 else 0
    ls_cum = float(ls_nav[-1] - 1) if len(ls_nav) > 0 else 0

    g_annual = [float(g_nav[-1, k] ** (252 / g_nav.shape[0]) - 1) if g_nav.shape[0] > 0 else 0 for k in range(10)]

    return {
        "mean_ic": mean_ic,
        "mean_rankic": mean_rankic,
        "std_ic": float(np.nanstd(ic_arr)),
        "std_rankic": std_rankic,
        "ic_ir": mean_ic / float(np.nanstd(ic_arr)) if np.nanstd(ic_arr) > 0 else 0,
        "rankic_ir": rankic_ir,
        "ic_winrate": float(np.nanmean(ic_arr > 0)),
        "rankic_winrate": float(np.nanmean(ric_arr > 0)),
        "n_periods": len(common),
        "ls_sharpe": ls_sharpe,
        "ls_mdd": ls_mdd,
        "ls_winrate": ls_winrate,
        "ls_annual": ls_annual,
        "ls_cum": ls_cum,
        "g1_annual": g_annual[0],
        "g10_annual": g_annual[9],
        "g_annual": g_annual,
        "ls_nav": ls_nav.tolist(),
        "g_nav": g_nav.tolist(),
        "dates": [str(d)[:10] for d in common],
        "daily_rankic": ric_arr.tolist(),
    }


def main():
    print("=" * 60)
    print("评估 61 因子: IC / RankIC / 十分层 / Sharpe / MDD")
    print("=" * 60)

    print(f"Load {FV_PATH} ...")
    fv = pd.read_parquet(FV_PATH, engine='pyarrow',
                          thrift_string_size_limit=2**31-1,
                          thrift_container_size_limit=2**31-1)
    if not isinstance(fv.index, pd.DatetimeIndex):
        fv.index = pd.to_datetime(fv.index)
    print(f"  形状: {fv.shape}, {len(fv.columns.get_level_values(0).unique())} 因子")

    print("加载 close 矩阵 ...")
    close = load_close_matrix()
    print(f"  形状: {close.shape}")

    HTML_DIR = PROJECT / "docs" / "reports" / "2026-08-23" / "factors"
    factor_names = sorted([
        f.stem.replace('factor_', '')
        for f in HTML_DIR.glob("factor_*.html")
    ])
    print(f"HTML 因子数: {len(factor_names)}")

    out = {}
    for name in factor_names:
        full_name = f"factor_{name}"
        if full_name not in fv.columns.get_level_values(0):
            out[name] = {"error": "no data in parquet"}
            continue
        cols = [c for c in fv.columns if c[0] == full_name]
        mat = fv[cols]
        mat.columns = [c[1] for c in mat.columns]
        r = eval_factor(name, mat, close)
        r['name'] = name
        out[name] = r
        if r and 'ls_sharpe' in r:
            print(f"  {name}: RankIC={r['mean_rankic']:+.4f} Sharpe={r['ls_sharpe']:+.2f} "
                  f"Annual={r['ls_annual']*100:+.1f}% G10={r['g10_annual']*100:+.1f}% "
                  f"MDD={r['ls_mdd']*100:+.1f}%", flush=True)

    # Save slim version for HTML use
    slim = {}
    for n, r in out.items():
        if not r:
            slim[n] = {}
            continue
        slim[n] = {k: v for k, v in r.items() if k not in ('ls_nav', 'g_nav', 'dates')}
    with open(PROJECT / "docs" / "reports" / "2026-08-23" / "all_eval.json", "w") as f:
        json.dump(slim, f, indent=2, default=str)
    print(f"\nSlim eval: {PROJECT / 'docs' / 'reports' / '2026-08-23' / 'all_eval.json'}")

    # Save full version (with nav/ic)
    with open(PROJECT / "docs" / "reports" / "2026-08-23" / "all_eval_full.json", "w") as f:
        json.dump(out, f, default=str)
    print(f"Full eval: {PROJECT / 'docs' / 'reports' / '2026-08-23' / 'all_eval_full.json'}")

    ranked = [(n, r) for n, r in out.items() if r and 'ls_sharpe' in r]
    ranked.sort(key=lambda x: x[1]['ls_sharpe'], reverse=True)
    print(f"\nTOP 15 by LS Sharpe:")
    for n, r in ranked[:15]:
        print(f"  {n}: Sharpe={r['ls_sharpe']:+.2f} IC={r['mean_rankic']:+.4f} "
              f"G10_ann={r['g10_annual']*100:+.1f}% G1_ann={r['g1_annual']*100:+.1f}%"
              f" IR={r['rankic_ir']:+.2f}")
    print(f"\nBOTTOM 10 by LS Sharpe:")
    for n, r in ranked[-10:]:
        print(f"  {n}: Sharpe={r['ls_sharpe']:+.2f} IC={r['mean_rankic']:+.4f} "
              f"G10_ann={r['g10_annual']*100:+.1f}% G1_ann={r['g1_annual']*100:+.1f}%"
              f" IR={r['rankic_ir']:+.2f}")


if __name__ == "__main__":
    main()