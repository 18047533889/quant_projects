# -*- coding: utf-8 -*-
"""Batch 4: row-gated coverage after the batch-4 declarations (same gate as
_build_worklist_v2.py / _batch3_coverage.py). Reads catalog CSV + worklist JSON
only; safe to run during regression."""
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
BATCH4 = [
    # daily (53)
    "true_turnover_rate", "roll_spread_proxy", "zero_return_ratio", "volume_zscore",
    "turnover_shock", "volume_momentum", "corwin_schultz_spread",
    "turnover_acceleration", "volume_shock", "high_low_spread_proxy",
    "EaseOfMovement", "volume_acceleration", "abnormal_volume", "ts_impulse_strength",
    "KeltnerPosition", "bollinger_pct_b", "candle_close_location", "candle_body_ratio",
    "NATR", "PVO", "CMF", "VortexMinus", "VortexPlus", "PVO_signal", "PVO_hist",
    "TSI_signal", "PPO_signal", "TSI", "DMI_plus", "MFI", "DX", "CMO", "DMI_minus",
    "PPO", "PPO_hist", "parkinson_vol", "rogers_satchell_vol", "ts_channel_position",
    "ts_breakout_high", "ichimoku_tenkan", "ashare_limit_up_touch",
    "ashare_limit_down_touch", "ts_turnover_cost_entropy", "clip", "lerp",
    "minimum", "maximum", "or_", "suspension_frequency", "log_positive_or_nan",
    "ATR_WILDER", "ts_topk_mean", "fin_common_size",
    # intraday (11)
    "intra_jump_ratio", "intra_signed_jump_ratio", "intra_jump_variation",
    "intra_amihud", "intra_volume_profile_jsd", "intra_path_efficiency",
    "intra_same_slot_momentum", "intra_entropy", "intra_segment_return",
    "intraday_impact_asymmetry", "intraday_impact_beta",
    # fiscal (3)
    "fiscal_standardized_surprise", "fin_average_balance", "ttm_from_cumulative",
]


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
    df, rows = parse_rows()
    total = len(rows)
    print(f"catalog rows parsed: {total}", flush=True)

    base = gate(rows, already)
    print(f"baseline: {base} rows = {round(100.0*base/total, 2)}%", flush=True)

    stages = []
    cov = set(already)
    cum = 0
    for name, batch in (("batch1", BATCH1), ("batch2", BATCH2), ("batch3", BATCH3),
                        ("batch4", BATCH4)):
        added = [o for o in batch if o not in cov]
        cov |= set(added)
        cum += len(added)
        k = gate(rows, cov)
        stages.append({
            "stage": name, "batch_size": len(batch),
            "newly_coverable": len(added),
            "cumulative_declared": cum,
            "rows_that_run": k,
            "pct_of_catalog_rows": round(100.0 * k / total, 2),
        })
        print(f"{name}: +{len(added)} -> {k} rows = {round(100.0*k/total, 2)}%", flush=True)

    out = f"{REPO}/evidence/_batch4_coverage.json"
    json.dump({"baseline_rows": base, "baseline_pct": round(100.0*base/total, 2),
               "stages": stages, "batch4_operators": BATCH4},
              open(out, "w"), indent=1)
    print("WROTE", out)


if __name__ == "__main__":
    main()
