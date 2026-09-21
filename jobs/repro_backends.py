#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单因子 × 各后端 × 短窗口，实测 3 个 PlanParamError / 1 全NaN 公式的落值能力。"""
import os, sys, json, warnings, traceback
from pathlib import Path
ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "jobs"))
os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
for k, v in dict(FACTOR_REPORT_ROOT_WORKERS="2", DUCKDB_MAX_THREADS="4", OMP_NUM_THREADS="4",
                 POLARS_MAX_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                 FACTOR_REPORT_FE_BACKEND="auto").items():
    os.environ.setdefault(k, v)
warnings.filterwarnings("ignore")

import pandas as pd
import incremental_factor_intake as intake

WORK = ROOT / "work/newmining_20260919"
cands = {c["page_name"]: c for c in json.load(open(WORK / "candidates.json"))["candidates"]}

TARGETS = {
    "winsorize": "alphasage_20260910154931_3c69f233",
    "win_bool": "alphasage_20260911220700_5cf157f2",
    "win_rank": "alphasage_20260910154931_f70ac242",
    "allnan": "alphasage_20260910154931_ba00f8bd",
    "degen": "alphasage_20260910154931_3b40819c",
}

OUT = Path("/tmp/dslfix_probe")
OUT.mkdir(exist_ok=True)

for tag, name in TARGETS.items():
    rec = dict(cands[name])
    print(f"\n########## {tag}  {name}")
    print(f"   fe = {rec['fe_formula']}")
    for backend in ["auto", "polars_long", "duckdb_sql", "pandas"]:
        try:
            res = intake.land_factor_batch_windowed(
                [dict(rec)], backend_name=backend,
                start_date="2016-01-04", end_date="2016-03-31",
                window_years=1, warmup_days=40, output_dir=str(OUT / backend))
            m = pd.read_parquet(OUT / backend / f"{name}.parquet")
            fin = (m.notna().sum(axis=1))
            print(f"   {backend:12s} OK  shape={m.shape} 首行有效={fin.iloc[0]} 最大有效={fin.max()} "
                  f"全窗有效行={int((fin>=30).sum())}")
        except Exception as e:
            print(f"   {backend:12s} FAIL {type(e).__name__}: {str(e)[:180]}")
