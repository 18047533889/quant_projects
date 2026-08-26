# -*- coding: utf-8 -*-
"""Build the final training feature matrix from the 25 selected factors.
Output: long-frame (date, asset) x [25 factor cols] + fwd_ret10, saved to parquet.
"""
import glob, os, duckdb, pyarrow.parquet as pq, pandas as pd, numpy as np

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
con = duckdb.connect()

# Selected factors
sel = pd.read_csv(f"{ROOT}/data/build/selected_factors.csv")
# full-panel selected factors already in selected_factors.csv; take all
names = sel["factor"].tolist()
print("selected factors:", len(names))

trad = list(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad.parquet").columns)
fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")  # date x asset

# Load each selected factor from whichever pool holds it (fm247 or fmqa)
def load_factor(name):
    for pool in ("fm247", "fmqa"):
        p = f"{ROOT}/data/factor_pools/{pool}/{name}_neu.parquet"
        if os.path.exists(p):
            return pq.read_table(p, columns=["datetime", "asset", "factor_value"]).to_pandas()
    return None

# Assemble per-asset-time frame
frames = []
for name in names:
    df = load_factor(name)
    if df is None:
        print("  missing", name); continue
    df = df[df.asset.isin(trad)].copy()
    df["fname"] = name
    frames.append(df[["datetime", "asset", "factor_value", "fname"]])
allf = pd.concat(frames, ignore_index=True)
print("allf rows:", len(allf))

# Pivot to (datetime, asset) x fname
wide = allf.pivot_table(index=["datetime", "asset"], columns="fname", values="factor_value")
print("wide:", wide.shape)
wide.to_parquet(f"{ROOT}/data/build/features_selected.parquet")
print("saved features_selected.parquet")
