# -*- coding: utf-8 -*-
"""Verify + measure the ``compute_rolling_beta`` vectorisation.

Old body: per-column loop calling ``rolling_beta(ret[[col]], bench[[col]])``.
New body: one ``rolling_beta(ret, bench)`` over the whole panel.

Checks (must all hold before the change is admissible):
  * NaN mask identical
  * max abs diff <= 1e-12 (bit-identical expected -- same kernel, same order)
  * shape / index / columns identical
  * the callers' three parameter shapes reproduce: (window, min_periods=None),
    (window, min_periods=5), and the broadcast-benchmark path
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import time
import warnings

warnings.filterwarnings("ignore")

REPO = "/home/sunhaiwei/quant_projects"
import sys

sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.price_volume.beta_helpers import (  # noqa: E402
    align_benchmark_to_ret,
    compute_rolling_beta,
)
from factor_engine.cleaned_operators._rolling_fast import rolling_beta  # noqa: E402

load_all()


def old_compute_rolling_beta(ret, benchmark_ret, window, *, min_periods=None):
    """The exact pre-change body, kept verbatim for the A/B comparison."""
    w = max(2, int(window))
    mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 3)
    bench = align_benchmark_to_ret(ret, benchmark_ret)
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for col in ret.columns:
        out[col] = rolling_beta(ret[[col]], bench[[col]], window=w, min_periods=mp)[col]
    return out


DATA = os.path.expanduser("~/cos_data/StockDailyBarAdj")
files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
files = [f for f in files if "2019-01-01" <= os.path.basename(f)[:10] <= "2022-12-31"]
syms = (
    pl.read_parquet(files[-1])
    .sort("AdjAmount", descending=True)
    .head(30)["Symbol"]
    .to_list()
)
long = (
    pl.scan_parquet(files)
    .select(["TradeDate", "Symbol", "AdjClose", "Volume"])
    .filter(pl.col("Symbol").is_in(syms))
    .collect()
)

panels = {}
for c in ("AdjClose", "Volume"):
    w = (
        long.select(["TradeDate", "Symbol", c])
        .pivot(values=c, index="TradeDate", on="Symbol")
        .sort("TradeDate")
    )
    pdf = w.to_pandas().set_index("TradeDate")
    pdf.index.name = "date"
    panels[c] = pdf

ret = panels["AdjClose"]
bench_wide = panels["Volume"]
# A single-column benchmark exercises the broadcast branch of
# ``align_benchmark_to_ret`` -- that is the shape ``ops.py`` uses.
bench_single = bench_wide.iloc[:, [0]]

report = {"panel_rows": int(len(ret)), "panel_cols": int(ret.shape[1])}
cases = [
    ("wide_bench_mp_default", bench_wide, 20, None),
    ("wide_bench_mp_5", bench_wide, 20, 5),
    ("single_bench_mp_default", bench_single, 60, None),
    ("single_bench_mp_5", bench_single, 20, 5),
]

checks = []
for name, bench, w, mp in cases:
    a = old_compute_rolling_beta(ret, bench, w, min_periods=mp)
    b = compute_rolling_beta(ret, bench, w, min_periods=mp)
    row = {
        "case": name,
        "window": w,
        "min_periods": mp,
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "index_equal": a.index.equals(b.index),
        "columns_equal": list(a.columns) == list(b.columns),
        "dtype_equal": a.dtypes.tolist() == b.dtypes.tolist(),
    }
    av, bv = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
    na, nb = ~np.isfinite(av), ~np.isfinite(bv)
    row["nan_mask_equal"] = bool((na == nb).all())
    both = ~na & ~nb
    row["finite_pairs"] = int(both.sum())
    if both.any():
        d = np.abs(av[both] - bv[both])
        row["max_abs_diff"] = float(d.max())
        row["bit_identical"] = bool((av[both] == bv[both]).all())
    row["verdict"] = (
        "EQUIVALENT"
        if row["shape_equal"]
        and row["index_equal"]
        and row["columns_equal"]
        and row["nan_mask_equal"]
        and row.get("max_abs_diff", 0.0) <= 1e-12
        else "DIFFERS"
    )
    checks.append(row)
    print(f"  {name:<26} {row['verdict']:<11} maxdiff={row.get('max_abs_diff')}", flush=True)
report["checks"] = checks


def t(fn, target=0.6, maxi=40, mini=5):
    fn()
    t0 = time.perf_counter()
    fn()
    one = time.perf_counter() - t0
    n = int(max(mini, min(maxi, target / one if one > 0 else maxi)))
    s = [(lambda _s: (fn(), time.perf_counter() - _s)[1])(time.perf_counter()) for _ in range(n)]
    return statistics.median(s)


timing = {}
for name, bench, w, mp in cases:
    old_s = t(lambda: old_compute_rolling_beta(ret, bench, w, min_periods=mp))
    new_s = t(lambda: compute_rolling_beta(ret, bench, w, min_periods=mp))
    timing[name] = {
        "old_s": round(old_s, 6),
        "new_s": round(new_s, 6),
        "speedup": round(old_s / new_s, 3) if new_s > 0 else None,
    }
    print(
        f"  {name:<26} old={old_s * 1000:.1f}ms new={new_s * 1000:.1f}ms "
        f"speedup={timing[name]['speedup']}x",
        flush=True,
    )
report["timing"] = timing
report["verdict"] = (
    "ALL_EQUIVALENT"
    if all(c["verdict"] == "EQUIVALENT" for c in checks)
    else "DIFFERS"
)
print("\nVERDICT:", report["verdict"], flush=True)
json.dump(report, open("/tmp/beta_equiv.json", "w"), indent=1, default=str)
print(json.dumps(report, indent=1, default=str)[:2500])
