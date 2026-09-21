#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分解定位 ba00f8bd 全 NaN 的根因，并确认 3b40819c 的横截面常数性质。"""
import os, sys, json, warnings
from pathlib import Path
ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "jobs"))
os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
for k, v in dict(FACTOR_REPORT_ROOT_WORKERS="2", DUCKDB_MAX_THREADS="4", OMP_NUM_THREADS="4",
                 POLARS_MAX_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4").items():
    os.environ.setdefault(k, v)
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import incremental_factor_intake as intake

OUT = Path("/tmp/dslfix_probe2"); OUT.mkdir(exist_ok=True)

def rec(name, fe):
    return dict(page_name=name, factor_name=name, fe_formula=fe, lqtp_formula=fe,
                source_formula=fe, can_use_factor_engine=True)

CASES = {
    "eps_only":        "eps",
    "netprofit_only":  "net_profit",
    "isnan_np":        "is_nan(net_profit)",
    "tspct_isnan_np":  "ts_pct(is_nan(net_profit), 60)",
    "tsdelta_tspct":   "ts_delta(ts_pct(is_nan(net_profit), 60), 1)",
    "full":            "(eps - ts_delta(ts_pct(is_nan(net_profit), 60), 1))",
    "csresid_self":    "cs_resid(turnover_ratio, turnover_ratio)",
    "scale_csresid":   "scale(cs_resid(turnover_ratio, turnover_ratio), 100)",
    "rank_scale":      "rank(scale(cs_resid(turnover_ratio, turnover_ratio), 100))",
    "turnover_only":   "turnover_ratio",
}
for tag, fe in CASES.items():
    try:
        intake.land_factor_batch_windowed([rec(f"probe_{tag}", fe)], backend_name="auto",
            start_date="2016-01-04", end_date="2016-03-31", window_years=1,
            warmup_days=40, output_dir=str(OUT))
        m = pd.read_parquet(OUT / f"probe_{tag}.parquet")
        a = m.to_numpy(dtype=float)
        fin = np.isfinite(a)
        cs = np.nanstd(np.where(fin, a, np.nan), axis=1)
        print(f"{tag:16s} 有效行={int((fin.sum(1)>=30).sum()):3d}/{len(m)} "
              f"总有效值={int(fin.sum()):7d} 行内截面std中位={np.nanmedian(cs):.3e} "
              f"唯一值数={len(np.unique(a[fin])) if fin.any() else 0}")
    except Exception as e:
        print(f"{tag:16s} FAIL {type(e).__name__}: {str(e)[:120]}")
