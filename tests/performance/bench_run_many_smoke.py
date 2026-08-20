#!/usr/bin/env python3
"""轻量 run_many 性能基准（本地 / nightly 可选）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _synthetic_panel(n_days: int, n_inst: int) -> dict:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    inst = [f"S{i:04d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["timestamp", "instrument"])
    rng = np.random.default_rng(42)
    return {
        "close": pd.Series(rng.normal(size=len(idx)), index=idx),
        "volume": pd.Series(rng.normal(size=len(idx)), index=idx),
    }


def bench_run_many(*, n_days: int, n_inst: int, n_factors: int) -> dict:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    qp = str(root.parent)
    if qp not in sys.path:
        sys.path.insert(0, qp)

    from api import rank, ts_mean
    from api.columns import col
    from api.factor import Factor
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    os.environ.setdefault("FACTOR_ENGINE_DISABLE_BOTTLENECK", "1")
    src = InMemorySeriesSource(data=_synthetic_panel(n_days, n_inst))
    eng = FactorEngine(backend=build_backend("pandas"), data_source=src)
    factors = [
        Factor(name=f"f{i}", expr=rank(ts_mean(col("close"), 5 + (i % 10))))
        for i in range(n_factors)
    ]
    t0 = time.perf_counter()
    out = eng.run_many(factors)
    elapsed = time.perf_counter() - t0
    return {
        "n_days": n_days,
        "n_inst": n_inst,
        "n_factors": n_factors,
        "rows": n_days * n_inst,
        "elapsed_sec": round(elapsed, 4),
        "factor_count": len(out.get("results", {})),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run_many 轻量性能基准")
    parser.add_argument("--n-days", type=int, default=120)
    parser.add_argument("--n-inst", type=int, default=50)
    parser.add_argument("--n-factors", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = bench_run_many(
        n_days=args.n_days,
        n_inst=args.n_inst,
        n_factors=args.n_factors,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"rows={result['rows']} factors={result['n_factors']} "
            f"elapsed={result['elapsed_sec']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
