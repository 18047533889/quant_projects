#!/usr/bin/env python3
"""Smoke: run land_minute_9 day_values on one real day for all 9, compare to old matrices."""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/sunhaiwei/quant_projects')
os.environ.setdefault("ASHARE_PARQUET_ROOT", "/home/sunhaiwei/cos_data")
os.environ.setdefault("DATA_ACCESS_COS_READ_MODE", "mirror")
os.environ.setdefault("DATA_ACCESS_AUTO_BOUND_MEMORY", "0")
from data_access import get_store
from jobs.land_minute_9 import day_values, FACTORS

store = get_store()
day = "2024-01-02"
df = store.read_frame("ashare_stock_minute_adj",
                      columns=["TradeDate","QuoteTime","Symbol","AdjClose","AdjVwap","Volume","AdjAmount"],
                      time_range=(day, day)).sort_values(["Symbol","QuoteTime"])
print("rows", len(df))
OLD = Path("/home/sunhaiwei/quant_projects/weekly_backtest_output/factor_matrices_all")
for page in FACTORS:
    oldp = OLD / f"{page}.parquet"
    if not oldp.exists():
        print(page, "NO OLD")
        continue
    old = pd.read_parquet(oldp)
    vals = []
    for sym, g in df.groupby("Symbol", sort=True):
        if len(g) < 10: continue
        v = day_values(g, page)
        if np.isfinite(v):
            vals.append((sym, v, old.loc[day, sym] if day in old.index and sym in old.columns else np.nan))
    # correlation + scale between new and old across symbols
    a = np.array([x[1] for x in vals]); b = np.array([x[2] for x in vals])
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    if valid.sum() >= 5:
        ratio = np.median(a[valid]/b[valid])
        print(f"{page}: n={len(vals)} corr={np.corrcoef(a[valid],b[valid])[0,1]:.4f} med_ratio={ratio:.4g}")
    else:
        print(f"{page}: n={len(vals)} too few comparable old values")
