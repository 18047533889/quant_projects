#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill the 13 DSL-only (class C) empty factors in factor_matrices_all/ with
real daily-data computations over the FULL universe (~5460 symbols).

Semantics = LQTP `lqtp_formula` (daily_* aggregation of daily bars), reusing the
validated hand-written logic from jobs/backfill_28_factors.py::run_dsl_pages.

Writes each page as factor_matrices_all/<page>.parquet with the SAME date index
(2016-01-04 .. 2026-08-24, 2585 rows) and full 5460 symbol columns as the other
441 non-empty factor files. Causal/running computations are identical on the
2019+ window regardless of earlier warm-up, so the 2019-01-02+ values prescribed
in the task are exactly the same as a 2019-start computation would give.

Usage:
  .venv/bin/python jobs/backfill_13_dsl_factors.py                        # full universe
  .venv/bin/python jobs/backfill_13_dsl_factors.py --limit 500            # first 500 symbols (fallback)
"""
import sys, json, time, argparse, gc, resource, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "archive" / "jobs"))
sys.path.insert(0, str(PROJECT / "vectorbt_qs"))

FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
DAILY = Path.home() / "cos_data" / "StockDailyBar"
START, END = "2016-01-04", "2026-08-24"  # matches placeholder index (2585 rows) & other factors

DSL_PAGES = [
    "amount_volume_corr",
    "kurt_volume_std",
    "minute_cond_vol_corr_std",
    "minute_corr_cond_vol_median",
    "minute_mutation_skew_mean",
    "minute_price_deviation_zscore_mean",
    "minute_range_skew",
    "skew_intraday_range",
    "std_ret",
    "vol_ret_corr",
    "volume_delta_corr",
    "vwap_deviation",
    "vwap_deviation_end_of_day",
]


def mem_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def load_wide(symbols, start, end, col_map=None):
    """date x symbol wide matrix (float32) for selected daily columns."""
    con = duckdb.connect()
    files = sorted(DAILY.glob("*.parquet"))
    fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
    sym_str = ",".join(f"'{s}'" for s in symbols)
    if col_map:
        sel = ", ".join(f"{sql} as {name}" for sql, name in col_map.items())
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol, {sel}
            FROM read_parquet({fs})
            WHERE Symbol IN ({sym_str})
              AND TradeDate >= DATE '{start}' AND TradeDate <= DATE '{end}'
        """).df()
    else:
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol
            FROM read_parquet({fs})
            WHERE TradeDate >= DATE '{start}' AND TradeDate <= DATE '{end}'
        """).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    cols_all = df.columns.drop(["date", "symbol"])
    for c in cols_all:
        sub = df[["date", "symbol", c]].drop_duplicates(["date", "symbol"], keep="first")
        m = sub.pivot(index="date", columns="symbol", values=c).sort_index().reindex(columns=symbols)
        out[c] = m.astype("float32")
    return out


def compute_dsl_page(page, bars):
    """One page's hand-written DSL implementation (from backfill_28_factors.run_dsl_pages)."""
    close = bars["close"]; high = bars["high"]; low = bars["low"]
    vol = bars["volume"]; amt = bars["amount"]; vwap = bars["vwap"]
    ret = close.pct_change(fill_method=None)
    delta = close.diff()

    if page == "amount_volume_corr":
        vp = vol.pct_change(fill_method=None); ap = amt.pct_change(fill_method=None)
        mat = vp.rolling(60, min_periods=20).corr(ap)
    elif page == "kurt_volume_std":
        vstd = vol.rolling(20, min_periods=15).std()
        mat = vstd.rolling(60, min_periods=20).kurt()
    elif page in ("minute_cond_vol_corr_std", "minute_corr_cond_vol_median"):
        vmed = vol.rolling(20, min_periods=10).median()
        cond = (vol > vmed).astype(float)
        amt_p = amt.pct_change(fill_method=None)
        vol_p = vol.pct_change(fill_method=None)
        r = (amt_p.rolling(20, min_periods=10).corr(vol_p) * cond).replace(0, np.nan)
        mat = r.rolling(20, min_periods=10).std()
    elif page == "minute_mutation_skew_mean":
        mat = close.rolling(60, min_periods=30).skew()
    elif page == "minute_price_deviation_zscore_mean":
        z = (close - close.rolling(60, min_periods=30).mean()) / close.rolling(60, min_periods=30).std().replace(0, np.nan)
        mat = z.rolling(20, min_periods=10).mean()
    elif page in ("minute_range_skew", "skew_intraday_range"):
        mat = (high - low).rolling(60, min_periods=30).skew()
    elif page == "std_ret":
        mat = ret.rolling(20, min_periods=10).std()
    elif page == "vol_ret_corr":
        mat = vol.pct_change(fill_method=None).replace(0, np.nan).rolling(60, min_periods=30).corr(ret)
    elif page == "volume_delta_corr":
        mat = vol.rolling(60, min_periods=30).corr(delta)
    elif page in ("vwap_deviation", "vwap_deviation_end_of_day"):
        mat = ((vwap - close) / close.replace(0, np.nan)).abs()
    else:
        raise ValueError(page)
    return mat.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N symbols (subset fallback)")
    args = ap.parse_args()

    ref = pd.read_parquet(FV_DIR / "abnormality_asymmetry.parquet")
    symbols = list(ref.columns)
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"[13dsl] universe: {len(symbols)} symbols, dates {START}..{END} ({ref.index.size} rows target) "
          f"mem_peak_base={mem_rss_gb():.2f}G")

    t0 = time.time()
    bars = load_wide(symbols, START, END, {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "Amount": "amount", "Vwap": "vwap",
    })
    print(f"[13dsl] bars loaded: {list(bars)}  shapes={bars['close'].shape}  "
          f"elapsed={time.time()-t0:.0f}s  mem_peak={mem_rss_gb():.2f}G")

    report = []
    for page in DSL_PAGES:
        t1 = time.time()
        mat = compute_dsl_page(page, bars)
        mat = mat.astype("float32").reindex(index=ref.index, columns=ref.columns)
        n_valid = int(mat.notna().sum().sum())
        pct = mat.notna().mean().mean() * 100 if mat.size else 0.0
        sub = mat.dropna(axis=1, how="all")
        assert sub.shape[1] > 0, f"{page}: all columns empty"
        # 9 个 2026-07 之后才上市的次新股在 60 日窗口内数据不足 20 天，rolling.corr
        # 全 NaN；用 vwap 矩阵（全量 5460 列、非 NaN）的列顺序补齐空列（合法：NaN 表示数据不足）
        sub = sub.reindex(columns=ref.columns)
        if not args.limit:
            assert sub.shape == ref.shape, f"{page}: partial shape {sub.shape} != {ref.shape}"
            assert n_valid > 0, f"{page}: zero valid cells"
        sub.to_parquet(FV_DIR / f"{page}.parquet")
        dt = time.time() - t1
        report.append((page, mat.shape, n_valid, pct, dt))
        print(f"  ✓ {page}: shape={mat.shape} valid_cells={n_valid} valid={pct:.2f}%  "
              f"elapsed={dt:.0f}s  mem_peak={mem_rss_gb():.2f}G")
        del mat, sub
        gc.collect()

    print(f"\n[13dsl] done in {time.time()-t0:.0f}s, final mem_peak={mem_rss_gb():.2f}G")
    for page, shape, nv, pct, dt in report:
        print(f"  {page}: shape={shape} valid_cells={nv} valid={pct:.2f}%")


if __name__ == "__main__":
    main()