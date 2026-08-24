#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 _flipped 因子在 parquet 里取负 (因为 fix_missing_factors 算的 value 没真 flip)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

FV_PATH = Path("/home/sunhaiwei/quant_projects/weekly_backtest_output/factor_values.parquet")

print(f"Loading {FV_PATH}...")
df = pd.read_parquet(FV_PATH)
print(f"shape: {df.shape}")

# 找 _flipped 列
factor_labels = set(c[0] for c in df.columns if isinstance(c, tuple))
flipped = [f for f in factor_labels if f.endswith("_flipped")]
print(f"flipped factors: {len(flipped)}")

# 取负
n_flipped = 0
for f in flipped:
    cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == f]
    if cols:
        df[cols] = -df[cols]
        n_flipped += 1
        print(f"  flipped: {f} ({len(cols)} cols)")

print(f"\nflipped {n_flipped} factors")

# 写回
FV_PATH.unlink()
df.to_parquet(str(FV_PATH), engine='pyarrow', compression='snappy')
print(f"✅ saved to {FV_PATH} ({FV_PATH.stat().st_size/1024/1024:.1f} MB)")