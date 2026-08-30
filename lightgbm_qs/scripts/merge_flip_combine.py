# -*- coding: utf-8 -*-
"""Combine parallel rank_ic partials -> rankic_all_factors_flipped.csv +
selected_factors_flipped.csv, with a FAST full dedup.

Dedup strategy (correct + fast, same semantics as rebuild_selection_adj):
  - For each selected factor, draw a 40k-row sample but store it as
    (row_positions: int64 array into the global (date,asset) grid,
     values: float64 array).
  - For a pair, the inner-join correlation == correlation over the POSITION
    INTERSECTION (both non-NaN at the same grid cell). Compute via
    np.intersect1d (fast) + np.corrcoef on the matching values.
This is mathematically identical to pd.concat(join='inner').dropna().corr() and
~100x faster (no pandas index alignment per pair).

Usage:
  python merge_flip_combine.py <partial_dir> [--limit N]
"""
import os
import sys
import glob
import time
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/home/sunhaiwei/quant_projects/lightgbm_qs/scripts")
from merge_all_factors_flip import (enumerate_candidates, load_factor,
                                     _is_wide_lqtp, load_fwd, load_trad)

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
LOG = "/tmp/merge_flip_combine.log"
CORR_THRESH = 0.98
IC_THRESH = float(os.environ.get("IC_THRESH", "0.015"))  # 2026-08-30: 支持环境变量放开（IC_THRESH=0 → 全部进模）


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("partial_dir")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] combine partials from {args.partial_dir}")
    # 2026-08-30: allow passing either a directory of chunk_*.csv partials or a single merged CSV
    if os.path.isfile(args.partial_dir):
        parts = [pd.read_csv(args.partial_dir)]
    else:
        parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.partial_dir, "chunk_*.csv")))]
    icdf = pd.concat(parts, ignore_index=True)
    icdf = icdf.drop_duplicates(subset=["name"], keep="first")
    icdf["rank_ic_orig"] = np.where(icdf["flipped"], -icdf["rank_ic"], icdf["rank_ic"])
    plog(f"combined {len(icdf)} factors, flipped={int(icdf['flipped'].sum())}")
    if args.limit:
        icdf = icdf.head(args.limit)
    sel = icdf[(icdf["rank_ic"] > IC_THRESH) & (~icdf["drop"])].sort_values("rank_ic", ascending=False)
    plog(f"rank_ic>0.015 selected (flip后诊断): {len(sel)}")

    # ---- global (date,asset) grid positions ----
    fwd = load_fwd()
    fwd["date"] = pd.to_datetime(fwd["date"]).dt.date
    grid_idx = pd.MultiIndex.from_frame(fwd[["date", "asset"]])
    pos_map = pd.Series(np.arange(len(grid_idx)), index=grid_idx)
    trad = load_trad()
    trad_set = set(trad)

    # NO_DEDUP=1: 用户 2026-08-28 定死"不去重，最低 2000+ 个因子进模型"。
    # 跳过相关性去重（也跳过 sample 采集——那是 dedup 专用），全部
    # rank_ic>0.015 的因子直接进 selected。
    if os.environ.get("NO_DEDUP") == "1":
        sel_final = sel.copy()
        sel_final.to_csv(os.path.join(BUILD, "selected_factors_flipped.csv"), index=False)
        icdf.to_csv(os.path.join(BUILD, "rankic_all_factors_flipped.csv"), index=False)
        plog(f"[{time.strftime('%H:%M:%S')}] DONE NO_DEDUP {len(sel_final)} selected total={time.time()-t0:.0f}s")
        return

    cands = enumerate_candidates(trad_set)
    samples = {}
    for key in sel["name"].tolist():
        cand = next((c for c in cands if f"{c[0]}:{c[1]}" == key), None)
        if cand is None:
            continue
        pool, name, fp, fmt = cand
        keep = trad_set if (pool == "lqtp" and fmt == "wide" and _is_wide_lqtp(fp)) else None
        # sample_rows keeps the melt small (wide full-market files are 5460 cols)
        df = load_factor(fp, fmt, trad_set, keep_cols=keep, sample_rows=120 if fmt == "wide" else 0)
        if df is None:
            continue
        if len(df) > 40000:
            df = df.sample(n=40000, random_state=42)
        df = df.dropna(subset=["fv"])
        if len(df) < 500:
            continue
        d = df.set_index(["date", "asset"])["fv"]
        pos = pos_map.reindex(d.index)
        ok = pos.notna()
        samples[key] = (pos[ok].astype(np.int64).to_numpy(), d.values[ok.values])
        del df, d
        if len(samples) % 100 == 0:
            plog(f"  samples {len(samples)}/{len(sel)} {time.time()-t0:.0f}s")
    plog(f"samples ready: {len(samples)} ({time.time()-t0:.0f}s)")

    skeys = list(samples)
    ic_lookup = dict(zip(sel["name"], sel["rank_ic"]))
    to_drop = set()
    for i in range(len(skeys)):
        a = skeys[i]
        pa, va = samples[a]
        for b in skeys[i+1:]:
            pb, vb = samples[b]
            # position intersection
            idx, ia, ib = np.intersect1d(pa, pb, assume_unique=True, return_indices=True)
            if len(idx) < 500:
                continue
            x = va[ia]
            y = vb[ib]
            # correlation with NaN safety (no NaN here: values came from dropna'd per-sample, but
            # the same grid cell in both samples is non-NaN by construction)
            c = np.corrcoef(x, y)[0, 1] if len(x) > 2 else 0.0
            if np.isfinite(c) and c >= CORR_THRESH:
                to_drop.add(a if ic_lookup[a] <= ic_lookup[b] else b)
        if (i + 1) % 60 == 0:
            plog(f"  dedup {i+1}/{len(skeys)} dropped {len(to_drop)} {time.time()-t0:.0f}s")
    plog(f"dedup removed {len(to_drop)}, kept {len(sel) - len(to_drop)}")
    sel_names = [n for n in sel["name"].tolist() if n not in to_drop]
    sel_final = sel[sel["name"].isin(sel_names)].copy()
    sel_final.to_csv(os.path.join(BUILD, "selected_factors_flipped.csv"), index=False)
    icdf.to_csv(os.path.join(BUILD, "rankic_all_factors_flipped.csv"), index=False)
    plog(f"[{time.strftime('%H:%M:%S')}] DONE {len(sel_final)} selected total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
