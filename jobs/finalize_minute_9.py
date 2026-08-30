#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize land_minute_9: wait for the remaining 5 factors to land, then
rank_ic / ic_ir evaluation of the 9 TRUE-MINUTE factors via quant_evaluator.

Label: vwap-to-vwap 后复权, enterprise shift(-2) caliber
  fwd = AdjVwap.pct_change().shift(-2)   (from jobs.adj_vwap_common.adj_fwd_return)
Evaluation window: 2024-01-02 .. 2024-12-31 (the exact 242-day landing window).
Metric ids: rank_ic (mean daily Spearman IC) + ic_ir (mean/std of daily IC).
Quant evaluator usage follows jobs/ evaluation scripts (rank_ic / ic_ir
through the public evaluate() facade).
"""
import sys, os, json, time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "quant_evaluator"))

from jobs.adj_vwap_common import load_adj_vwap, adj_fwd_return
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

OUT_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
RESULT = Path("/tmp/minute_9_result.json")
RANKIC = Path("/tmp/minute_9_rankic.json")
LOG = Path("/tmp/finalize_minute_9.log")

FACTORS = [
    "amount_weighted_impact",
    "high_volume_amount_weighted_impact",
    "high_volume_amount_weighted_squared_impact",
    "high_volume_squared_return_weighted_impact",
    "high_volume_volume_weighted_squared_impact",
    "skew_weighted_volume_event_intensity",
    "skew_weighted_volume_event_intensity_close",
    "vwap_deviation_intensity_squared_amount_weighted",
    "vwap_squared_deviation_intensity",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def load_result() -> dict:
    if RESULT.exists():
        return json.loads(RESULT.read_text())
    return {}


def main() -> None:
    # ---- phase 1: wait for all 9 landed (main process writes result json) ----
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        res = load_result()
        missing = [f for f in FACTORS if res.get(f, {}).get("status") != "landed"]
        log(f"[wait] landed={len(FACTORS)-len(missing)}/9 missing={missing}")
        if not missing:
            break
        time.sleep(180)
    res = load_result()
    landed = {f: res.get(f) for f in FACTORS if res.get(f, {}).get("status") == "landed"}
    if len(landed) != len(FACTORS):
        missing = [f for f in FACTORS if f not in landed]
        log(f"[wait] TIMEOUT: still missing {missing}")
        # continue with what IS landed

    # ---- phase 2: rank_ic + ic_ir on landed factors ----
    days = sorted(pd.read_parquet(OUT_DIR / f"{FACTORS[0]}.parquet").index)
    T = len(days)
    fwd = adj_fwd_return().reindex(index=days)  # AdjVwap.pct_change().shift(-2)
    # forward return at the LAST trading day is NaN (no t+2); drop it from eval
    last_nan = int(np.isnan(fwd.iloc[-1].values).all() or fwd.index[-1] == days[-1])
    eval_days = days[:-1] if last_nan else days
    T_eval = len(eval_days)
    fwd_eval = fwd.loc[eval_days]

    summary = {}
    for page in FACTORS:
        if page not in landed:
            summary[page] = {"status": "not_landed", "rank_ic": None, "ic_ir": None}
            continue
        p = OUT_DIR / f"{page}.parquet"
        if not p.exists():
            log(f"[warn] {page} marked landed but parquet missing: {p}")
            summary[page] = {"status": "parquet_missing", "rank_ic": None, "ic_ir": None}
            continue
        mat = pd.read_parquet(p).reindex(index=eval_days)
        assets = sorted(mat.columns)
        N = len(assets)
        vals = mat.values.astype(np.float64)
        lab = fwd_eval.reindex(columns=assets).values.astype(np.float64)

        fb = FactorBatch(
            factor_ids=(page,),
            time_axis=AxisRef("t", "datetime64", T_eval,
                              values=np.array(eval_days, dtype="datetime64[ns]")),
            asset_axis=AxisRef("a", "str", N, values=np.array(assets, dtype=object)),
            values=vals[:, :, np.newaxis],
        )
        lb = LabelBundle(
            target_id="fwd_vwap_shift2",
            values=lab,
            horizon=2,
            execution_delay=1,
            decision_time=tuple(range(T_eval)),
            label_start_time=tuple(range(T_eval)),
            label_end_time=tuple(range(1, T_eval + 1)),
            price_convention="vwap_to_vwap",
        )
        bundle = evaluate(fb, lb, metrics=("rank_ic", "ic_ir"))
        mv = bundle.metric_values
        rank_ic = float(mv["rank_ic"].value)
        ic_ir = float(mv["ic_ir"].value)
        n_ic = int(mv["rank_ic"].observation_count)
        summary[page] = {
            "status": landed[page]["status"],
            "shape": landed[page].get("shape"),
            "valid_cells": landed[page].get("valid_cells"),
            "rank_ic": rank_ic,
            "ic_ir": ic_ir,
            "n_ic_days": n_ic,
            "window": [str(eval_days[0]), str(eval_days[-1])],
            "label": "AdjVwap.pct_change().shift(-2)",
            "n_assets": N,
        }
        log(f"[eval] {page}: rank_ic={rank_ic:.5f} ic_ir={ic_ir:.4f} "
            f"n_ic_days={n_ic} n_assets={N}")
        # write-early after each factor
        RANKIC.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        del mat, vals, lab, fb, lb, bundle
        import gc; gc.collect()

    # merge rankic summary back into minute_9_result.json (one dict per factor)
    res = load_result()
    for page, info in summary.items():
        if page in res:
            res[page].update({k: v for k, v in info.items() if k not in ("status",)})
    res["_rankic_summary"] = summary
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    log("DONE rankic evaluation")

    # ---- phase 3: print final table ----
    print("\n=== 9 TRUE-MINUTE factors landed + rank_ic/ic_ir ===")
    print(f"{'factor':<48} {'landed':<8} {'rank_ic':>10} {'ic_ir':>8} {'n_days':>6}")
    for page in FACTORS:
        info = summary.get(page, {})
        st = "YES" if page in landed else ("NO" if info.get("status") in (None, "not_landed") else "PARTIAL")
        rc = info.get("rank_ic")
        ir = info.get("ic_ir")
        nd = info.get("n_ic_days", 0)
        print(f"{page:<48} {st:<8} "
              f"{'-' if rc is None else f'{rc:.5f}':>10} "
              f"{'-' if ir is None else f'{ir:.4f}':>8} "
              f"{nd:>6}")


if __name__ == "__main__":
    main()
