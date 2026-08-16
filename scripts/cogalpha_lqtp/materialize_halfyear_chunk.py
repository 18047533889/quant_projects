#!/usr/bin/env python3
"""Materialize ONE date-window DSL chunk in an isolated process, then exit.

Memory knobs (set before FE import):
  FACTOR_ENGINE_MAX_MEMORY_MB, FACTOR_ENGINE_SPILL_DIR, FACTOR_ENGINE_RESERVE_GB,
  FACTOR_ENGINE_DISABLE_PANEL_NATIVE, FACTOR_ENGINE_OPERATOR_BACKEND, etc.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

# Cap native thread pools before numpy/fe import.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Use local StockDailyBar mirror; avoid COS fail-closed on calendar holidays.
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
# auto → HybridBackend needs physical plan; pin polars for materialize fills.
os.environ.setdefault("FACTOR_ENGINE_OPERATOR_BACKEND", "polars")
# FactorEngine low-mem defaults for ~32G / no-swap chart fills.
os.environ.setdefault("FACTOR_ENGINE_MAX_WORKERS", "1")
os.environ.setdefault("FACTOR_ENGINE_DISABLE_CSE", "1")
os.environ.setdefault("FACTOR_ENGINE_DISABLE_PANEL_NATIVE", "1")
os.environ.setdefault("FACTOR_ENGINE_RESERVE_GB", "8")
os.environ.setdefault("FACTOR_ENGINE_MAX_MEMORY_MB", "12000")
os.environ.setdefault(
    "FACTOR_ENGINE_SPILL_DIR",
    str(ROOT / "data/cogalpha_lqtp_production/_fe_spill"),
)
os.environ.setdefault("FACTOR_ENGINE_SPILL_BUDGET_BYTES", str(20 * 1024**3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    ap.add_argument("--min-avail-gb", type=float, default=14.0)
    ap.add_argument(
        "--universe",
        default="ALL",
        help="FE universe; cs-rank needs ALL unless source is pre-filtered",
    )
    args = ap.parse_args()

    spill = Path(os.environ["FACTOR_ENGINE_SPILL_DIR"])
    spill.mkdir(parents=True, exist_ok=True)

    from scripts.cogalpha_lqtp.data_access_panel import ashare_materialize_data_source_config
    from scripts.cogalpha_lqtp.materialize import materialize_factor
    from scripts.cogalpha_lqtp.memory_utils import read_mem_available_gb, release_memory

    avail = read_mem_available_gb()
    print(
        f"CHUNK_START {args.name} {args.start}->{args.end} avail={avail:.1f}G "
        f"fe_max_mb={os.environ.get('FACTOR_ENGINE_MAX_MEMORY_MB')} "
        f"spill={os.environ.get('FACTOR_ENGINE_SPILL_DIR')}",
        flush=True,
    )
    if avail < args.min_avail_gb:
        print(f"ABORT_LOW_MEM avail={avail:.1f}G < {args.min_avail_gb:.1f}G", flush=True)
        return 9

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.stat().st_size > 1000:
        print(f"CHUNK_REUSE size={args.out.stat().st_size}", flush=True)
        return 0

    tmp_id = f"{args.name}_chunk_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    lake_root = args.work_dir / "factor_lake"
    cfg = ashare_materialize_data_source_config(start_date=args.start, end_date=args.end)
    try:
        out = materialize_factor(
            factor_id=tmp_id,
            dsl=args.dsl,
            data_source_cfg=cfg,
            lake_root=lake_root,
            universe=args.universe,
            use_subplan_cache=False,
        )
        shutil.copy2(out, args.out)
    finally:
        tmp_dir = lake_root / tmp_id
        try:
            if (tmp_dir / "values.parquet").exists():
                (tmp_dir / "values.parquet").unlink()
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
        release_memory()
        gc.collect()

    print(
        f"CHUNK_DONE {args.out.name} size={args.out.stat().st_size} "
        f"avail={read_mem_available_gb():.1f}G",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
