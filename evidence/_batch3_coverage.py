# -*- coding: utf-8 -*-
"""Batch 3: MEASURE the row-gated backend coverage after the batch-3 declarations.

The published curve in BACKEND_COVERAGE_REPORT_20260919.md is keyed to "convert the
top-N workqueue operators".  Our batches are not a contiguous prefix, so the only
honest way to report cumulative progress is to re-run the same gate over the real
catalog rows with the actual covered set.

Gate definition (identical to evidence/_build_worklist_v2.py `gate`):
    a factor row runs  <=>  every operator name appearing in its r57_formula
    belongs to the covered set
Covered set = operators already real on the clean tree (worklist
``convertible_to == "already_real"``) UNION the operators whose explicit spec this
stream declared (batch 1 + batch 2 + batch 3).

Reads only the catalog CSV and the two evidence JSONs -- no engine import, so it is
safe to run while the regression suite owns the working tree.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

import pandas as pd

REPO = "/home/sunhaiwei/quant_projects"
CATALOG = (f"{REPO}/evidence/factor_catalog_20260916/"
           "r57c_20260919_final/factor_catalog_review_r57c_final.csv.gz")
WORKLIST = f"{REPO}/evidence/factor_catalog_20260916/backend_coverage_worklist.json"

CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

BATCH1 = ["add", "subtract", "multiply", "divide", "abs", "neg", "log", "exp", "sign"]
BATCH2 = ["rank", "cs_pct_rank", "gt", "ts_sum", "ts_zscore", "and_", "tanh", "lt",
          "ts_delta", "ts_sharpe", "ts_delay", "cs_mad_zscore", "cs_mean", "ts_max",
          "ts_min"]
BATCH3 = ["price_impact", "efficiency_ratio", "turnover_zscore",
          "fin_quarter_from_cumulative", "relative_volume", "intra_realized_skewness",
          "amihud_illiquidity", "intra_realized_variance", "ts_pct", "signed_sqrt",
          "sigmoid", "intra_realized_kurtosis", "log_abs"]


def parse_rows():
    df = pd.read_csv(CATALOG, usecols=["id", "r57_formula"])
    rows = []
    for fid, formula in zip(df["id"], df["r57_formula"]):
        rows.append((fid, set(CALL_RE.findall(str(formula)))))
    return df, rows


def gate(rows, covered):
    ok = 0
    for _fid, ops in rows:
        if ops and ops <= covered:
            ok += 1
    return ok


def main():
    wl = json.load(open(WORKLIST))
    already = {p["operator"] for p in wl["operators"]
               if p["convertible_to"] == "already_real"}
    work = [p["operator"] for p in wl["workqueue_by_frequency"]]

    df, rows = parse_rows()
    total = len(rows)
    print(f"catalog rows parsed: {total}  (already_real={len(already)})", flush=True)

    base = gate(rows, already)
    print(f"baseline (already_real only): {base} rows = "
          f"{round(100.0 * base / total, 2)}%  "
          f"[report published 1361 = 1.19%]", flush=True)

    stages = []
    cov = set(already)
    for name, batch in (("batch1", BATCH1), ("batch2", BATCH2), ("batch3", BATCH3)):
        added = []
        for o in batch:
            if o in cov:
                continue
            cov.add(o)
            added.append(o)
        k = gate(rows, cov)
        stages.append({
            "stage": name,
            "batch_size": len(batch),
            "newly_coverable": len(added),
            "already_covered_before": [o for o in batch if o not in added],
            "cumulative_declared": len(BATCH1) + len(BATCH2) + len(BATCH3)
            if name == "batch3" else (len(BATCH1) if name == "batch1" else len(BATCH1) + len(BATCH2)),
            "rows_that_run": k,
            "pct_of_catalog_rows": round(100.0 * k / total, 2),
        })
        print(f"{name}: +{len(added)} newly coverable -> {k} rows = "
              f"{round(100.0 * k / total, 2)}%", flush=True)

    # how many workqueue-prefix entries are now declared
    declared = set(BATCH1) | set(BATCH2) | set(BATCH3)
    prefix = {}
    for n in (1, 5, 10, 20, 30, 50, 75, 100, 300, 1000):
        head = work[:n]
        prefix[f"top{n}"] = {
            "workqueue_entries": len(head),
            "declared": sum(1 for o in head if o in declared),
            "not_in_catalog": sum(1 for o in head if o in
                                  {p["operator"] for p in wl["operators"]
                                   if p["convertible_to"] == "needs_inspection"}),
        }

    occ = collections.Counter()
    for _fid, ops in rows:
        for o in ops:
            occ[o] += 1
    occ_declared = sum(occ[o] for o in declared)

    out = {
        "total_catalog_rows": total,
        "baseline_rows_that_run": base,
        "baseline_pct": round(100.0 * base / total, 2),
        "stages": stages,
        "declared_operators": {
            "batch1": BATCH1, "batch2": BATCH2, "batch3": BATCH3,
            "total": sorted(declared),
        },
        "workqueue_prefix_declared": prefix,
        "occurrence_weighted": {
            "declared_operator_occurrences": occ_declared,
            "pct_of_total_occurrences": None,
        },
        "caveat": ("Row-gated coverage is a MEASUREMENT over the 113893 real catalog "
                   "rows with the exact gate used by _build_worklist_v2.py. It counts "
                   "only the operators this stream declared plus those already real on "
                   "the clean tree; it does NOT include any declaration that has not "
                   "been verified numerically."),
    }
    p = f"{REPO}/evidence/_batch3_coverage.json"
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("baseline_rows_that_run", "baseline_pct",
                                          "stages")}, indent=1, ensure_ascii=False))
    print("WROTE", p)


if __name__ == "__main__":
    sys.exit(main())
