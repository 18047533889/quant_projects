# -*- coding: utf-8 -*-
"""Parallel rank_ic+flip worker: computes pass-1 (rank_ic per factor, flip flag)
for a chunk of candidates and writes a partial CSV. Combine script merges partials
then runs selection + full dedup.

Usage:
  python merge_flip_chunk.py <chunk_idx> <n_chunks> <out_csv>
"""
import os, sys, time, json, glob
import numpy as np, pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/home/sunhaiwei/quant_projects/lightgbm_qs/scripts")
from merge_all_factors_flip import (enumerate_candidates, load_factor,
                                     _is_wide_lqtp, load_fwd, load_trad,
                                     _lqtp_formula_names)

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"


def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)


def main():
    idx, n_chunks, out_csv = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    trad = load_trad()
    trad_set = set(trad)
    fwd_long = load_fwd()
    cands = enumerate_candidates(trad_set)
    chunk = cands[idx::n_chunks]
    plog(f"[chunk{idx}] candidates={len(chunk)} total={len(cands)}")
    rows = []
    for pool, name, fp, fmt in chunk:
        key = f"{pool}:{name}"
        keep = trad_set if (pool == "lqtp" and fmt == "wide" and _is_wide_lqtp(fp)) else None
        df = load_factor(fp, fmt, trad_set, keep_cols=keep)
        if df is None or len(df) < 500:
            rows.append((key, pool, np.nan, 0, True, False))
            continue
        m = df.merge(fwd_long, on=["date", "asset"], how="inner")
        if len(m) < 1500:
            rows.append((key, pool, np.nan, 0, True, False))
            continue
        try:
            ic = m.groupby("date").apply(
                lambda g: g["fv"].rank().corr(g["fwd"].rank()), include_groups=False)
            ic = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            rows.append((key, pool, np.nan, 0, True, False))
            continue
        if len(ic) < 50:
            rows.append((key, pool, np.nan, 0, True, False))
            continue
        raw = float(ic.mean())
        flip = bool(raw < 0)
        rows.append((key, pool, float(-raw) if flip else raw, len(ic), False, flip))
    out = pd.DataFrame(rows, columns=["name", "pool", "rank_ic", "n_dates", "drop", "flipped"])
    out.to_csv(out_csv, index=False)
    plog(f"[chunk{idx}] done {len(out)} -> {out_csv}")


if __name__ == "__main__":
    main()
