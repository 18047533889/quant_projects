#!/usr/bin/env python3
"""Backfill report aliases whose intake landing stopped after the train window.

The report uses short aliases while the formula registry uses a different
``page_name``.  This job resolves by ``factor_name``, evaluates the registered
FactorEngine expression on adjusted daily data, and writes the result under
the report alias only after it passes basic value-coverage checks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")

import rob26_rebuild_22daily as engine_job
from factor_engine.storage.factory import build_data_source
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.api.factor import Factor

START, END = "2016-01-04", "2026-08-24"
OUT = ROOT / "weekly_backtest_output/factor_matrices_all"
REF = OUT / "abnormality_asymmetry.parquet"
FORMULAS = Path.home() / "factor_delivery_converted/formula_lqtp_all.json"

ALIASES = {
    "amount_rank_pressure": "amount_rank_pressure",
    "amt_vwap_elasticity_power": "amt_vwap_elasticity_power",
    "cs_residual_momentum_rank_winsor": "cs_residual_momentum_rank_winsor",
    "vol_ret_covariance_zscore": "vol_ret_covariance_zscore",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated report aliases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    wanted = [x for x in args.only.split(",") if x] if args.only else list(ALIASES)
    unknown = set(wanted) - set(ALIASES)
    if unknown:
        raise SystemExit(f"unknown aliases: {sorted(unknown)}")

    records = {r.get("factor_name"): r for r in json.loads(FORMULAS.read_text())}
    ref = pd.read_parquet(REF)
    ds = build_data_source({"type": "data_access", "dataset": "ashare_stock_daily_adj",
                            "start_date": START, "end_date": END})
    runner = FactorEngine(build_backend("pandas"), ds, run_mode="research")
    for alias in wanted:
        factor_name = ALIASES[alias]
        record = records.get(factor_name)
        if not record or not record.get("fe_formula"):
            raise RuntimeError(f"{alias}: registered formula missing")
        formula = record["fe_formula"].strip()
        if formula.startswith("{"):
            expr = engine_job.build(json.loads(formula))
        else:
            # Incremental registry entries store the already-translated
            # FactorEngine Python expression (o['ts_mean'](...), col(...)).
            # Evaluate only against the explicit operator/column namespace.
            expr = eval(formula, {"__builtins__": {}},
                        {"o": engine_job.OPS, "col": engine_job.col})
        result = runner.run(Factor(name=f"backfill_{alias}", expr=expr), market="ashare")["result"]
        matrix = result.unstack(level=result.index.names[-1]).reindex(index=ref.index, columns=ref.columns)
        matrix = matrix.astype("float32")
        valid_days = int((matrix.notna().sum(axis=1) >= 30).sum())
        if valid_days < 1800:
            raise RuntimeError(f"{alias}: only {valid_days} usable days; refusing to replace")
        print(f"{alias}: valid_days={valid_days} range={matrix.index.min().date()}..{matrix.index.max().date()}", flush=True)
        if not args.dry_run:
            matrix.to_parquet(OUT / f"{alias}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
