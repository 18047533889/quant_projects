# -*- coding: utf-8 -*-
"""Batch 3: render the measured per-operator verification table for the report."""
from __future__ import annotations

import json
import os

REPO = "/home/sunhaiwei/quant_projects"


def row(tag, r, polars_key, pandas_key, extra=""):
    ps = r[polars_key] * 1000.0
    pds = r[pandas_key] * 1000.0
    ratio = (pds / ps) if ps else float("nan")
    dev = r["max_abs_dev"]
    dev = "0" if dev == 0 else (f"{dev:.2e}" if dev is not None else "n/a")
    print(f"| `{r['operator']}` | {tag} | "
          f"{'✅' if r['nan_mask_identical'] else '❌'} | {dev} | "
          f"{'✅' if r['idempotent'] else '❌'} | {r['to_pandas_calls']} | "
          f"{r['n_finite']} | {ps:.3f} | {pds:.3f} | {ratio:.2f}x{extra} |")


d = json.load(open(f"{REPO}/evidence/_verify_batch3_specs.json"))
recs = d[-1]["results"]
print("== daily panel (25 syms x 972 days, 2019-01-02..2022-12-30) ==")
for r in recs:
    if r["operator"] == "fin_quarter_from_cumulative":
        continue
    row("daily", r, "_delegate_s", "_pandas_s")

f = json.load(open(f"{REPO}/evidence/_verify_batch3_fiscal.json"))["results"][0]
row("fiscal", f, "_polars_s", "_pandas_s")

i = json.load(open(f"{REPO}/evidence/_verify_batch3_intraday.json"))["results"]
print("== minute panel (20 syms x 5 sessions x 240 bars) ==")
for r in i:
    row("minute", r, "_polars_s", "_pandas_s")

c = json.load(open(f"{REPO}/evidence/_batch3_coverage.json"))
print("== coverage ==")
print("baseline", c["baseline_rows_that_run"], c["baseline_pct"])
for s in c["stages"]:
    print(s["stage"], s["newly_coverable"], s["rows_that_run"], s["pct_of_catalog_rows"])
