# -*- coding: utf-8 -*-
"""Step 2: build the feature matrix from the flip-selected factor list.

Inputs
------
- data/build/selected_factors_flipped.csv  (name, pool, rank_ic, n_dates, flipped)
- data/panel/fwd_ret10_adj.parquet         (the (date, asset) panel grid)
- data/factor_pools/*                      (factor value sources)

Output
------
- data/build/features_full3_adj.parquet    (row = date, asset; col = selected factor,
  flipped ones stored NEGATED so values are all "higher = better")

The grid is exactly the fwd panel grid (767745 = 2585 dates x 297 assets), so the
feature matrix rows align with the label frame fwd_ret10_adj / fwd_adj_neu.

NaN preserved (LightGBM native). Memory-bounded: writes column by column into a
pre-allocated float32 block.

Usage:
  python build_features_flip.py [--limit N]
"""
import os
import sys
import time
import glob
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
PANEL = os.path.join(ROOT, "data/panel")
LOG = "/tmp/build_features_flip.log"

LQTP_FORMULA_MAP = os.path.join(BUILD, "lqtp_formula_map.json")
LQTP_MIN_WIDTH = 5000
FRESH_CUTOFF = pd.Timestamp("2026-08-28").timestamp()


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


_lqtp_map_names = None
def _lqtp_formula_names():
    global _lqtp_map_names
    if _lqtp_map_names is None:
        with open(LQTP_FORMULA_MAP) as f:
            _lqtp_map_names = set(json.load(f).keys())
    return _lqtp_map_names


def _is_wide_lqtp(path):
    try:
        if os.path.getmtime(path) < FRESH_CUTOFF:
            return False
        if len(pq.ParquetFile(path).schema.names) < LQTP_MIN_WIDTH:
            return False
        return True
    except Exception:
        return False


def load_factor(full_path, fmt, trad_set, keep_cols=None):
    try:
        if fmt == "long":
            df = pq.read_table(full_path, columns=["datetime", "asset", "factor_value"]).to_pandas()
            df = df[df.asset.isin(trad_set)].copy()
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
            return df[["date", "asset", "factor_value"]].rename(columns={"factor_value": "fv"})
        if keep_cols is not None:
            cols = [c for c in keep_cols if c in pq.ParquetFile(full_path).schema.names]
            df = pq.read_table(full_path, columns=["timestamp"] + cols).to_pandas()
            if "timestamp" in df.columns:
                datecol = "timestamp"
            else:
                df = df.reset_index()
                datecol = "timestamp"
        else:
            df = pq.read_table(full_path).to_pandas()
            if "timestamp" in df.columns:
                datecol = "timestamp"
            elif "date" in df.columns:
                datecol = "date"
            else:
                df.index = pd.to_datetime(df.index)
                df = df.reset_index()
                datecol = df.columns[0]
            cols = [c for c in df.columns if c in trad_set]
            if not cols:
                return None
        df = df[[datecol] + cols].copy()
        df.columns = ["date"] + cols
        df["date"] = pd.to_datetime(df["date"]).dt.date
        w = df[["date"] + cols].melt(id_vars="date", var_name="asset", value_name="fv")
        w = w[w.asset.isin(trad_set)]
        w["fv"] = pd.to_numeric(w["fv"], errors="coerce")
        return w
    except Exception as e:
        plog(f"    load失败 {os.path.basename(full_path)}: {e}")
        return None


POOL_DIR = {
    "fm247": "fm247",
    "fmqa": "fmqa",
    "cogfull": "cogfull",
    "cogshort": "cogshort",
    "cogneutral": "cogneutral",
    "delivery": "delivery",
    "optfac": "optimized_factors",
    "factmat": "factor_matrices",
    "lqtp": "lqtp",
    "new026": "new_20260830",
}

POOL_FMT = {
    "fm247": "long",
    "fmqa": "long",
    "cogfull": "wide",
    "cogshort": "wide",
    "cogneutral": "wide",
    "delivery": "wide",
    "optfac": "wide",
    "factmat": "wide",
    "lqtp": "wide",
    "new026": "wide",
}


def resolve_path(pool, name):
    if pool == "delivery":
        p = os.path.join(ROOT, "data/factor_pools/delivery", name, "factor_values_test.parquet")
        return p if os.path.exists(p) else None
    base = os.path.join(ROOT, "data/factor_pools", POOL_DIR.get(pool, pool))
    for cand in (os.path.join(base, f"{name}.parquet"), os.path.join(base, f"{name}_neu.parquet")):
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] ===== build_features_flip 开始 limit={args.limit} =====")

    sel = pd.read_csv(os.environ.get("SELECTED_CSV", os.path.join(BUILD, "selected_factors_flipped.csv")))
    sel = sel[(~sel["drop"]) & sel["rank_ic"] > 0]
    if args.limit:
        sel = sel.head(args.limit)
    plog(f"selected factors: {len(sel)}")

    fwd = pd.read_parquet(os.path.join(PANEL, "fwd_ret10_adj.parquet"))
    fwd["date"] = pd.to_datetime(fwd["date"])
    trad = list(pd.read_parquet(os.path.join(PANEL, "vwap_trad_adj.parquet")).columns)
    trad_set = set(trad)

    grid = fwd[["date", "asset"]].copy()
    grid["date"] = grid["date"].dt.date
    grid = grid.set_index(["date", "asset"]).sort_index()
    plog(f"grid: {grid.shape}")

    n = len(grid)
    out = np.full((n, len(sel)), np.nan, dtype=np.float32)
    pos = pd.Series(np.arange(n), index=grid.index)

    for j, row in enumerate(sel.itertuples(index=False)):
        name = row.name
        pool = row.pool
        flipped = bool(row.flipped)
        bare = name.split(":", 1)[1] if ":" in name else name
        fp = resolve_path(pool, bare)
        if pool == "lqtp" and bare not in _lqtp_formula_names():
            # stale legacy exclusion already happened in step 1; defensive skip
            plog(f"  skip {name}: lqtp name not in formula map")
            continue
        keep = trad_set if (pool == "lqtp" and fp and _is_wide_lqtp(fp)) else None
        df = load_factor(fp, POOL_FMT.get(pool, "wide"), trad_set, keep_cols=keep)
        if df is None:
            continue
        df = df.set_index(["date", "asset"])["fv"]
        if flipped:
            df = -df
        idx = pos.reindex(df.index)
        ok = idx.notna()
        if ok.any():
            out[idx[ok].astype(int).values, j] = df.values[ok.values]
        if (j + 1) % 50 == 0:
            plog(f"  {j+1}/{len(sel)} {time.time()-t0:.0f}s")
        del df

    g = grid.reset_index()
    out_df = pd.DataFrame({"date": g["date"].astype(str), "asset": g["asset"]})
    for j, row in enumerate(sel.itertuples(index=False)):
        out_df[row.name] = out[:, j]
    out_df.to_parquet(os.path.join(BUILD, "features_full3_adj.parquet"), index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] DONE {out_df.shape} -> features_full3_adj.parquet total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
