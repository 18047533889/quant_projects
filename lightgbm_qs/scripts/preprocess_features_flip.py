# -*- coding: utf-8 -*-
"""Step 3: factor_preprocess on the flip-selected feature matrix.

Input : data/build/features_full3_adj.parquet (date, asset, <selected factors>)
Output: data/build/features_full3_adj_prep.parquet (same grid, per-pool transforms)

Pool policy (matches the user's fixed strategy + preprocess_features.py):
  - fm247 / fmqa / cogfull / cogshort / cogneutral : already neutralized -> cs_winsor only
  - delivery / optfac / factmat / lqtp             : raw -> cs_winsor + cs_rank + cs_zscore
The cs_* transforms come from factor_preprocess (library), applied per date
(axis=1), no look-ahead.

Usage:
  python preprocess_features_flip.py [--in path] [--out path]
"""
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
LOG = "/tmp/preprocess_features_flip.log"

IN = os.environ.get("FEATURES_PARQUET", os.path.join(BUILD, "features_full3_adj.parquet"))
OUT = os.environ.get("PREP_OUT", os.path.join(BUILD, "features_full3_adj_prep.parquet"))

sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_preprocess")
from factor_preprocess.transforms.cross_sectional import cs_winsor, cs_rank, cs_zscore

POOL_PREFIXES = {
    "fm247:": "neu",
    "fmqa:": "neu",
    "cogfull:": "neu",
    "cogshort:": "neu",
    "cogneutral:": "neu",
    "delivery:": "raw",
    "optfac:": "raw",
    "factmat:": "raw",
    "lqtp:": "raw",
    "new026:": "raw",
}


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] read {IN}")
    df = pd.read_parquet(IN)
    plog(f"  shape={df.shape}")
    feats = [c for c in df.columns if c not in ("date", "asset")]
    date_arr = df["date"].to_numpy()
    asset_arr = df["asset"].to_numpy()

    uni_dates, date_idx = np.unique(date_arr, return_inverse=True)
    uni_assets = pd.Index(pd.unique(asset_arr))
    a_pos = pd.Series(np.arange(len(uni_assets)), index=uni_assets)
    asset_pos = a_pos.reindex(asset_arr).to_numpy()
    n_date, n_ast = len(uni_dates), len(uni_assets)
    plog(f"  dates={n_date} assets={n_ast} feats={len(feats)}")

    out_block = np.empty((len(df), len(feats)), dtype=np.float32)
    for j, col in enumerate(feats):
        v = df[col].to_numpy(dtype=np.float64)
        prefix = next((p for p in POOL_PREFIXES if col.startswith(p)), "raw")
        kind = POOL_PREFIXES[prefix]
        M = np.full((n_date, n_ast), np.nan, dtype=np.float64)
        ok = np.isfinite(v)
        if ok.sum():
            M[date_idx[ok], asset_pos[ok]] = v[ok]
        if kind == "neu":
            M = cs_winsor(M, lower=0.01, upper=0.99, axis=1)
        else:
            M = cs_winsor(M, lower=0.01, upper=0.99, axis=1)
            M = cs_rank(M, axis=1, pct=True)
            M = cs_zscore(M, axis=1, ddof=1)
        out_block[:, j] = M[date_idx, asset_pos]
        if (j + 1) % 100 == 0:
            plog(f"  {j+1}/{len(feats)} {time.time()-t0:.0f}s")
        del M

    out = pd.DataFrame({"date": date_arr, "asset": asset_arr})
    for j, col in enumerate(feats):
        out[col] = out_block[:, j]
    out.to_parquet(OUT, index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] DONE -> {OUT} shape={out.shape} total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
