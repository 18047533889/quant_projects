# -*- coding: utf-8 -*-
"""Merge LQTP factor pool (1439 wide parquet, date x 297 asset) into the feature
matrix -> features_full2.parquet.  Vectorized + memory-safe.

KEY FACTS (verified before writing):
- base features_full_filled.parquet: 767745 rows = 2585 dates x 297 assets,
  sorted by (date, asset); within each date the 297 assets are in sorted order.
- lqtp wide files: index = trading dates (sorted DatetimeIndex), columns = the
  same 297 assets in the same sorted order; 1321 files have 2575 dates, 118 have
  2585 (extended to 2026-08-24).
- 993 of the 1439 lqtp files are ALREADY base columns (prefix 'lqtp:'),
  446 are new -> features_full2 = 1720 + 446 = 2166 factors.

Merging: for each lqtp file, reindex its value matrix onto the base row grid with
two numpy searchsorted/position vectors (no per-factor merge/pivot).  New-factor
columns accumulate into a pre-allocated float32 block, then the output is written
in one pass by concatenating base columns + block into a pyarrow table.
Peak RAM ~ base 5.3G + block ~1.4G + table overhead, well within 92G.
"""
import glob, os, sys, time, gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BASE = f"{ROOT}/data/build/features_full_filled.parquet"
OUT  = f"{ROOT}/data/build/features_full2.parquet"
LQDIR = f"{ROOT}/data/factor_pools/lqtp"
LOG = "/tmp/merge_lqtp2.log"

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def used_gb():
    with open("/proc/meminfo") as f:
        d = {l.split()[0].rstrip(':'): int(l.split()[1]) for l in f}
    return (d["MemTotal"] - d["MemAvailable"]) / 1048576.0

def avail_gb():
    with open("/proc/meminfo") as f:
        d = {l.split()[0].rstrip(':'): int(l.split()[1]) for l in f}
    return d["MemAvailable"] / 1048576.0

def mem_guard():
    """If available RAM drops below 20G, pause a moment for GC / co-tenants."""
    a = avail_gb()
    if a < 20:
        time.sleep(2)
    return a

t_start = time.time()
plog(f"[{time.strftime('%H:%M:%S')}] start avail={avail_gb():.0f}G used={used_gb():.0f}G")

# ---------------- load base ----------------
t0 = time.time()
df = pq.read_table(BASE).to_pandas()
base_cols = [c for c in df.columns if c not in ("date", "asset")]
n_row = len(df)
plog(f"base loaded {df.shape} factors={len(base_cols)} {time.time()-t0:.0f}s "
     f"avail={avail_gb():.0f}G used={used_gb():.0f}G")

# grid: base rows are (date sorted x asset sorted), 297 rows per date.
dates = df["date"].drop_duplicates().to_numpy()          # sorted str dates (2585)
asset_cols = df["asset"].drop_duplicates().tolist()       # sorted assets (297)
asset_pos = pd.Series(np.arange(len(asset_cols)), index=asset_cols)

df["_d"] = df["date"].map({d: i for i, d in enumerate(dates)}).to_numpy()
df["_a"] = df["asset"].map(asset_pos).to_numpy()
d_pos = df["_d"].to_numpy(); a_pos = df["_a"].to_numpy()
del df["_d"], df["_a"]
# base row for (date_i, asset_j): since rows sorted by (date, asset) and 297/date
base_row_grid = (d_pos.astype(np.int64) * len(asset_cols) + a_pos.astype(np.int64))
plog(f"grid aligned {time.time()-t0:.0f}s")

# ---------------- merge lqtp files ----------------
lq_files = sorted(glob.glob(f"{LQDIR}/*.parquet"))
existing_stems = {c[len("lqtp:"):] for c in base_cols if c.startswith("lqtp:")}
new_files = [f for f in lq_files if os.path.basename(f)[:-8] not in existing_stems]
new_names = ["lqtp:" + os.path.basename(f)[:-8] for f in new_files]
plog(f"lqtp files: {len(lq_files)}  new (not in base): {len(new_files)}")

BLOCK = np.full((n_row, len(new_files)), np.nan, dtype=np.float32)
plog(f"allocated block {n_row}x{len(new_files)} "
     f"avail={avail_gb():.0f}G used={used_gb():.0f}G")

t1 = time.time(); n_fallback = 0
for j, f in enumerate(new_files):
    w = pd.read_parquet(f)
    fdates = w.index.strftime("%Y-%m-%d").to_numpy()      # sorted date strings
    pos = np.searchsorted(dates, fdates)                   # -> base date position
    ok = (pos < len(dates)) & (dates[pos] == fdates)
    cpos = asset_pos.reindex(w.columns).to_numpy()
    col_ok = ~np.isnan(cpos.astype(float))
    if ok.all() and col_ok.all():
        # base rows for every (factor date, factor asset) cell
        rows = pos[:, None] * len(asset_cols) + cpos[None, :]
        BLOCK[rows, j] = w.to_numpy(dtype=np.float32)
    else:
        # corner-case files: merge on (date, asset) key
        ws = w.stack(dropna=False).rename("v").reset_index()
        ws.columns = ["date", "asset", "v"]
        ws["date"] = ws["date"].astype(str)
        key = ws.set_index(["date", "asset"]).index
        base_key = pd.MultiIndex.from_arrays([df["date"], df["asset"]])
        got = base_key.get_indexer(key)
        BLOCK[got, j] = ws["v"].to_numpy(dtype=np.float32)
        n_fallback += 1
    del w
    if (j + 1) % 100 == 0:
        gc.collect()
        mem_guard()
        plog(f"  {j+1}/{len(new_files)} {time.time()-t1:.0f}s "
             f"avail={avail_gb():.0f}G used={used_gb():.0f}G")
plog(f"lqtp merged {len(new_files)} files {time.time()-t1:.0f}s "
     f"fallback={n_fallback} avail={avail_gb():.0f}G used={used_gb():.0f}G")

# ---------------- write combined ----------------
plog(f"writing {n_row} x {len(base_cols)+len(new_files)} "
     f"avail={avail_gb():.0f}G used={used_gb():.0f}G")
t2 = time.time()
out_names = ["date", "asset"] + base_cols + new_names
cols = []
cols.append(pa.array(df["date"].to_numpy(), type=pa.string()))
cols.append(pa.array(df["asset"].to_numpy(), type=pa.string()))
tbl_base = pq.read_table(BASE, columns=base_cols)
for name in base_cols:
    cols.append(tbl_base.column(name))
for j in range(len(new_names)):
    cols.append(pa.array(BLOCK[:, j], type=pa.float32()))
table = pa.table(cols, names=out_names)
pq.write_table(table, OUT, compression="zstd", compression_level=1)
plog(f"WROTE {OUT} {table.num_rows}x{table.num_columns} {time.time()-t2:.0f}s "
     f"avail={avail_gb():.0f}G used={used_gb():.0f}G")
plog(f"[{time.strftime('%H:%M:%S')}] DONE total {time.time()-t_start:.0f}s")
