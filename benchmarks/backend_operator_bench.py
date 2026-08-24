#!/usr/bin/env python3
"""Backend 算子 micro-benchmark → ``backend_cost_baseline.json``。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = FE_ROOT / "benchmarks" / "backend_cost_baseline.json"

BENCH_OPS = (
    "ts_mean",
    "ts_std",
    "ts_zscore",
    "ts_corr",
    "rank",
    "zscore",
    "group_mean",
    "group_zscore",
    "vwap",
    "log_returns",
    "volatility",
    "protected_div",
    "where",
)

SIZES = {
    "small": (100, 500),
    "medium": (1000, 3000),
    "large": (3000, 5000),
}


def _bootstrap() -> None:
    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    fe = str(FE_ROOT)
    if fe not in sys.path:
        sys.path.insert(0, fe)
    from factor_engine.cleaned_operators import load_all

    load_all()


def _make_panel(n_dates: int, n_inst: int):
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    insts = [f"S{i:04d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    rng = np.random.default_rng(42)
    close = pd.Series(rng.uniform(10, 100, len(idx)), index=idx)
    volume = pd.Series(rng.uniform(1e5, 1e6, len(idx)), index=idx)
    grp = pd.Series(rng.integers(0, 5, len(idx)), index=idx, dtype=float)
    return {"close": close, "open": close * 0.99, "volume": volume, "grp": grp}


def _expr_for(op: str):
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col

    f = make_cleaned_call_factory
    if op == "ts_mean":
        return f("ts_mean")(col("close"), 20)
    if op == "ts_std":
        return f("ts_std")(col("close"), 20)
    if op == "ts_zscore":
        return f("ts_zscore")(col("close"), 20)
    if op == "ts_corr":
        return f("ts_corr")(col("close"), col("open"), 20)
    if op == "rank":
        return f("rank")(col("close"))
    if op == "zscore":
        return f("zscore")(col("close"))
    if op == "group_mean":
        return f("group_mean")(col("close"), col("grp"))
    if op == "group_zscore":
        return f("group_zscore")(col("close"), col("grp"))
    if op == "vwap":
        return f("vwap")(col("close"), col("volume"), 20)
    if op == "log_returns":
        return f("log_returns")(col("close"))
    if op == "volatility":
        return f("volatility")(col("close"), 20)
    if op == "protected_div":
        return f("protected_div")(col("close"), col("volume"))
    if op == "where":
        return f("where")(f("gt")(col("close"), 0), col("close"), col("volume"))
    raise KeyError(op)


def _bench_backend(source, op: str, backend_name: str, *, repeats: int = 2) -> float:
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine

    expr = _expr_for(op)
    eng = FactorEngine(backend=build_backend(backend_name), data_source=source)
    # warmup
    eng.run(Factor(name="w", expr=expr))
    t0 = time.perf_counter()
    for _ in range(repeats):
        eng.run(Factor(name="b", expr=expr))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    rows = len(source.data["close"])
    return elapsed_ms, rows


def _to_cost(elapsed_ms: float, rows: int) -> dict[str, float]:
    millions = max(rows / 1_000_000.0, 0.001)
    per_million = max(elapsed_ms / millions, 0.01)
    return {
        "fixed_overhead_ms": round(max(0.5, elapsed_ms * 0.05), 3),
        "per_million_rows_ms": round(per_million, 3),
        "memory_factor": 1.0,
    }


def run_benchmark(*, size: str = "small", ops: tuple[str, ...] = BENCH_OPS, with_duckdb: bool = False) -> dict:
    from tests.helpers import InMemorySeriesSource

    n_dates, n_inst = SIZES.get(size, SIZES["small"])
    panel = _make_panel(n_dates, n_inst)
    source = InMemorySeriesSource(data=panel)
    duckdb_source = None
    if with_duckdb:
        duckdb_source = _make_duckdb_source(panel, n_dates, n_inst)
    backends = {
        "pandas_numpy": ("pandas", source),
        "polars_panel": ("polars", source),
        "polars_long": ("polars_long", source),
    }
    if duckdb_source is not None:
        backends["duckdb_sql"] = ("duckdb_sql", duckdb_source)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for op in ops:
        out[op] = {}
        for key, (backend, src) in backends.items():
            try:
                elapsed, rows = _bench_backend(src, op, backend)
                out[op][key] = _to_cost(elapsed, rows)
            except Exception as exc:
                out[op][key] = {"error": str(exc)}
    return {
        "schema_version": 1,
        "generated_by": "backend_operator_bench.py",
        "size": size,
        "with_duckdb": with_duckdb,
        "sizes": {k: {"dates": v[0], "instruments": v[1]} for k, v in SIZES.items()},
        "operators": out,
    }


def _make_duckdb_source(panel: dict, n_dates: int, n_inst: int):
    """临时 parquet + data_access registry → DuckDB 可下推 source。"""
    import tempfile
    from pathlib import Path

    import pandas as pd

    tmp = Path(tempfile.mkdtemp(prefix="fe_bench_duckdb_"))
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    close = panel["close"]
    idx = close.index
    df = pd.DataFrame(
        {
            "TradeDate": idx.get_level_values(0),
            "Symbol": idx.get_level_values(1),
            "Close": close.values,
            "Open": panel["open"].values,
            "Volume": panel["volume"].values,
            "Grp": panel["grp"].values,
        }
    )
    df.to_parquet(data_dir / "panel.parquet", index=False)
    cfg = tmp / "datasets.yaml"
    cfg.write_text(
        f"""
test_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {data_dir}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
    Open: double
    Volume: double
    Grp: int64
""",
        encoding="utf-8",
    )
    import os

    os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
    os.environ["DATA_ACCESS_CONFIG"] = str(cfg)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    from factor_engine.storage.factory import build_data_source

    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="small", choices=sorted(SIZES))
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help="依次跑 small/medium/large 并写入 sizes 字段",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--merge-seed", action="store_true", help="与现有 seed JSON 合并")
    parser.add_argument("--with-duckdb", action="store_true", help="额外跑 duckdb_sql backend")
    args = parser.parse_args()
    _bootstrap()
    if args.all_sizes:
        measured = {
            "schema_version": 1,
            "generated_by": "backend_operator_bench.py",
            "sizes": {k: {"dates": v[0], "instruments": v[1]} for k, v in SIZES.items()},
            "with_duckdb": args.with_duckdb,
            "by_size": {},
        }
        for size in SIZES:
            measured["by_size"][size] = run_benchmark(size=size, with_duckdb=args.with_duckdb)
        payload = measured
    else:
        payload = run_benchmark(size=args.size, with_duckdb=args.with_duckdb)
    if args.merge_seed and args.out.is_file():
        base = json.loads(args.out.read_text(encoding="utf-8"))
        if args.all_sizes:
            ops_root = base.setdefault("operators", {})
            for size, block in payload.get("by_size", {}).items():
                for op, costs in block.get("operators", {}).items():
                    merged = dict(ops_root.get(op, {}))
                    for backend, val in costs.items():
                        if isinstance(val, dict) and "error" not in val:
                            merged[f"{backend}@{size}"] = val
                    ops_root[op] = merged
            base["by_size"] = payload.get("by_size", {})
            base["sizes"] = payload.get("sizes", base.get("sizes"))
            payload = base
        else:
            ops = base.setdefault("operators", {})
            for op, costs in payload["operators"].items():
                merged = dict(ops.get(op, {}))
                for backend, val in costs.items():
                    if "error" not in val:
                        merged[backend] = val
                ops[op] = merged
            payload = base
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.all_sizes:
        print(f"wrote {args.out} sizes={list(SIZES)}")
    else:
        print(f"wrote {args.out} ops={len(payload.get('operators', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
