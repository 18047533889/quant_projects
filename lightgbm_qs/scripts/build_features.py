# -*- coding: utf-8 -*-
"""Build the final training feature matrix from the 25 selected factors.
Output: long-frame (date, asset) x [25 factor cols] + fwd_ret10, saved to parquet.

P0-B (2026-08-28): the feature columns come from the walk-forward selection manifest
(data/build/walkforward_selection.json) — the UNION of per-fold lists, so every fold's
own factor columns exist in the matrix. data/build/selected_factors.csv is the legacy
FULL-SAMPLE list (rank_IC computed on ALL dates, i.e. with future OOS labels): it is a
diagnostic only and must never define training features.
"""
import glob, os, sys, duckdb, pyarrow.parquet as pq, pandas as pd, numpy as np

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
con = duckdb.connect()

# Per-fold walk-forward selection manifest (P0-B); selected_factors.csv is diagnostic only
WF_SELECTION_JSON = f"{ROOT}/data/build/walkforward_selection.json"
if not os.path.exists(WF_SELECTION_JSON):
    print("!! data/build/walkforward_selection.json not found — run "
          "`python factor_selection.py --folds-from-train` first. The full-sample "
          "selected_factors.csv is NOT a valid feature source (P0-B selection leakage).")
    sys.exit(2)
sys.path.insert(0, f"{ROOT}/scripts")
from factor_selection import load_selection_manifest  # noqa: E402
folds, meta = load_selection_manifest(path=WF_SELECTION_JSON)
names = sorted({f for lst in folds.values() for f in lst})
print(f"per-fold walk-forward lists: {len(folds)} cuts, union={len(names)} factors "
      f"(purge={meta.get('purge_trading_days')}, label_basis={meta.get('label_basis')})")
print("selected factors (walk-forward union):", len(names))

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
