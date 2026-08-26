# -*- coding: utf-8 -*-
"""Build factor matrix: (date x asset) x factor columns, restricted to investable/tradable universe.
Reads local fm247 pool (full-coverage factors), pivots to a per-factor long frame.
Outputs a master long table joined with forward returns + price for the LightGBM step.
"""
import duckdb, pyarrow.parquet as pq, pandas as pd, os

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
con = duckdb.connect()

trad = list(pd.read_parquet(f"{ROOT}/data/panel/vwap_trad.parquet").columns)
print("tradable assets:", len(trad))

# Factor-matrix: one column per FULL-coverage factor, rows = (date, asset)
# Load full-coverage fm247 factors into a wide-by-factor long table.
files = sorted(pq.ParquetDataset(f"{ROOT}/data/factor_pools/fm247", use_legacy_dataset=False).files) \
        if False else None

import glob, os
factors = []
for f in glob.glob(f"{ROOT}/data/factor_pools/fm247/*.parquet"):
    df = pq.read_table(f, columns=["datetime", "asset", "factor_value"]).to_pandas()
    dmin, dmax = str(df.datetime.min())[:10], str(df.datetime.max())[:10]
    if not (dmin <= "2016-03-01" and dmax >= "2024-01-01"):
        continue
    name = os.path.basename(f).replace("_neu.parquet", "")
    factors.append((name, df))
print("full-coverage fm247 factors:", len(factors))

# Build a wide factor matrix: index=(date,asset), columns=factor_name
frames = []
for name, df in factors:
    df = df[df.asset.isin(trad)].copy()
    df["fname"] = name
    frames.append(df[["datetime", "asset", "factor_value", "fname"]])
allf = pd.concat(frames, ignore_index=True)
print("allf rows:", len(allf))

# Pivot to (datetime, asset) x fname
wide = allf.pivot_table(index=["datetime", "asset"], columns="fname", values="factor_value")
print("wide matrix shape:", wide.shape)
wide.to_parquet(f"{ROOT}/data/build/factor_matrix.parquet")
print("saved factor_matrix.parquet", wide.shape)
