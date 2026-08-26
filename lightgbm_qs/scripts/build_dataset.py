# -*- coding: utf-8 -*-
"""Build final LightGBM training dataset: join features_selected with fwd_ret10.
Output: parquet long-frame with columns [date, asset] + 25 feature cols + fwd_ret10.
Filters to rows where label exists and features are sufficiently populated.
"""
import duckdb, pyarrow.parquet as pq, pandas as pd, numpy as np

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
con = duckdb.connect()

feat = pd.read_parquet(f"{ROOT}/data/build/features_selected.parquet")
feat = feat.reset_index()
feat["datetime"] = pd.to_datetime(feat["datetime"]).dt.date
feat = feat.rename(columns={"datetime": "date"})
print("features:", feat.shape)

fwd = pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet")
fwd.index = pd.to_datetime(fwd.index).date
fwd_long = fwd.stack().rename("fwd_ret10").reset_index().rename(columns={"level_0": "date", "level_1": "asset"})

merged = feat.merge(fwd_long, on=["date", "asset"], how="inner")
print("merged with fwd:", merged.shape)

# Drop rows where forward return is NaN (end of sample)
merged = merged[merged["fwd_ret10"].notna()]
print("after dropping NaN fwd:", merged.shape)

# Feature columns
feat_cols = [c for c in merged.columns if c not in ("date", "asset", "fwd_ret10")]

# Keep rows with >=8 of the 25 features present (avoid sparse rows)
present = merged[feat_cols].notna().sum(axis=1)
merged = merged[present >= 8].copy()
print("after sparse-row filter:", merged.shape)

merged.to_parquet(f"{ROOT}/data/build/train_dataset.parquet")
print("saved train_dataset.parquet")
print("date range:", merged.date.min(), "..", merged.date.max())
print("num factors used:", len(feat_cols))
