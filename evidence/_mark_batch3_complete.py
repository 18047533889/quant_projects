# -*- coding: utf-8 -*-
"""Batch 3 bookkeeping: mark the batch-3 operators complete in the worklist and
record a batch ledger, without regenerating the frozen batch-2 measurement file.

Only additive keys are written: ``completed`` / ``completed_batch`` /
``completed_at`` / ``completed_execution_kind`` / ``completed_verification`` on the
touched entries, plus top-level ``batch_ledger`` and ``coverage_after_batch3``.
Every pre-existing key is preserved byte-for-byte in value.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO = "/home/sunhaiwei/quant_projects"
WL = f"{REPO}/evidence/factor_catalog_20260916/backend_coverage_worklist.json"
COV = f"{REPO}/evidence/_batch3_coverage.json"

BATCH3_AT = os.environ.get("BATCH3_AT", "2026-09-20T02:35:00+08:00")

# canonical -> (kernel class, source module, verification artifact)
BATCH3 = {
    "price_impact": ("PolarsLiquidityV2_price_impact", "price_volume/polars_liquidity_v2.py", "daily"),
    "efficiency_ratio": ("PolarsTechMisc_efficiency_ratio", "technical/polars_tech_misc.py", "daily"),
    "turnover_zscore": ("PolarsLiquidityV2_turnover_zscore", "price_volume/polars_liquidity_v2.py", "daily"),
    "fin_quarter_from_cumulative": ("PolarsFundamental_fin_quarter_from_cumulative",
                                    "fundamental/polars_fundamental.py", "fiscal"),
    "relative_volume": ("PolarsLiquidityV2_relative_volume", "price_volume/polars_liquidity_v2.py", "daily"),
    "intra_realized_skewness": ("IntradayPolars_intra_realized_skewness",
                                "intraday/polars_next_stage.py", "intraday"),
    "amihud_illiquidity": ("PolarsLiquidityV2_amihud_illiquidity",
                           "price_volume/polars_liquidity_v2.py", "daily"),
    "intra_realized_variance": ("IntradayPolarsFull_intra_realized_variance",
                                "intraday/polars_intraday_full.py", "intraday"),
    "ts_pct": ("TSPctPolars", "common/polars_ops.py", "daily"),
    "signed_sqrt": ("SignedSqrtNative", "common/polars_daily_native.py", "daily"),
    "sigmoid": ("SigmoidPolarsAuto", "common/polars_auto.py", "daily"),
    "intra_realized_kurtosis": ("IntradayPolars_intra_realized_kurtosis",
                                "intraday/polars_next_stage.py", "intraday"),
    "log_abs": ("LogAbsPolarsAuto", "common/polars_auto.py", "daily"),
}

BATCH1 = ["add", "subtract", "multiply", "divide", "abs", "neg", "log", "exp", "sign"]
BATCH2 = ["rank", "cs_pct_rank", "gt", "ts_sum", "ts_zscore", "and_", "tanh", "lt",
          "ts_delta", "ts_sharpe", "ts_delay", "cs_mad_zscore", "cs_mean", "ts_max",
          "ts_min"]

VERIF = {
    "daily": "evidence/_verify_batch3_specs.json",
    "intraday": "evidence/_verify_batch3_intraday.json",
    "fiscal": "evidence/_verify_batch3_fiscal.json",
}


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO, *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main():
    d = json.load(open(WL))
    head = git("rev-parse", "--short", "HEAD")

    marked = 0
    for section in ("operators", "workqueue_by_frequency"):
        for entry in d[section]:
            op = entry.get("operator")
            if op not in BATCH3:
                continue
            kernel, src, kind = BATCH3[op]
            entry["completed"] = True
            entry["completed_batch"] = "batch3"
            entry["completed_at"] = BATCH3_AT
            entry["completed_execution_kind"] = "polars_native_expr"
            entry["completed_kernel"] = kernel
            entry["completed_source"] = f"factor_engine/cleaned_operators/{src}"
            entry["completed_verification"] = VERIF[kind]
            entry["completed_by_commit"] = head
            marked += 1

    ledger = {
        "batch1": {
            "commit": "ac33a820",
            "operators": BATCH1,
            "kind": "declare POLARS_NATIVE_EXPR spec only (kernels already native)",
        },
        "batch2": {
            "commit": "5a4902a9",
            "operators": BATCH2,
            "kind": "declare POLARS_NATIVE_EXPR spec only (kernels already native)",
        },
        "batch3": {
            "commit": head,
            "operators": sorted(BATCH3),
            "kind": "declare POLARS_NATIVE_EXPR spec only (kernels already native)",
            "verification": sorted(set(VERIF.values())),
            "notes": [
                "All 13 kernels verified to make ZERO pl.DataFrame.to_pandas calls at runtime.",
                "supports_lazy / supports_streaming deliberately left False (eager panel API).",
                "No kernel body modified; declarations only.",
                "Also fixed two defects in evidence/_verify_batch_specs.py: the absent-default "
                "sentinel test (type name is _MissingDefaultType, not 'MISSING') and the "
                "positional array comparison (now aligned by column name).",
            ],
        },
        "cumulative_declared_operators": len(set(BATCH1) | set(BATCH2) | set(BATCH3)),
    }
    d["batch_ledger"] = ledger

    if os.path.exists(COV):
        cov = json.load(open(COV))
        d["coverage_after_batch3"] = {
            "measured": True,
            "method": ("re-ran the same row gate as _build_worklist_v2.py over the 113893 "
                       "real catalog rows with covered = already_real UNION declared"),
            "baseline_rows_that_run": cov["baseline_rows_that_run"],
            "baseline_pct": cov["baseline_pct"],
            "stages": cov["stages"],
            "final_rows_that_run": cov["stages"][-1]["rows_that_run"],
            "final_pct": cov["stages"][-1]["pct_of_catalog_rows"],
            "artifact": "evidence/_batch3_coverage.json",
            "caveat": cov["caveat"],
        }

    json.dump(d, open(WL, "w"), indent=1)
    print(f"marked entries: {marked}   commit: {head}")
    print("ledger cumulative declared:", ledger["cumulative_declared_operators"])
    if "coverage_after_batch3" in d:
        c = d["coverage_after_batch3"]
        print(f"coverage: {c['baseline_pct']}% -> {c['final_pct']}% "
              f"({c['final_rows_that_run']} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
