from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from factor_engine.api import col, ts_mean
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.storage.datasource import DataSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=8), ("A", "B", "C")],
        names=("timestamp", "instrument"),
    )
    base = pd.Series(np.arange(len(index), dtype=float), index=index)

    class Source(DataSource):
        def __init__(self) -> None:
            self.instrument_filter = ("A", "B", "C")
            self.start_date = index.levels[0].min().tz_localize("UTC")
            self.end_date = index.levels[0].max().tz_localize("UTC")
            self.schema = {"close": "float64"}
            self.load_calls = 0

        def load_column(self, name):
            self.load_calls += 1
            return base

        def load_columns(self, names):
            self.load_calls += 1
            return {name: base for name in names}

    source = Source()
    engine = FactorEngine(build_backend("auto"), source, run_mode="research")
    shared = ts_mean(col("close"), 3)
    seen = bytearray(args.factors)
    completed = 0
    digest = hashlib.sha256()
    expected_last = 20.0
    started = time.monotonic()

    def persist(status: str, **extra) -> None:
        payload = {
            "status": status,
            "requested": args.factors,
            "completed_sink_outputs": completed,
            "source_load_calls": source.load_calls,
            "elapsed_seconds": time.monotonic() - started,
            **extra,
        }
        args.progress.write_text(json.dumps(payload, sort_keys=True))

    def sink(name, value):
        nonlocal completed
        position = int(name[1:])
        if not 0 <= position < args.factors or seen[position]:
            raise AssertionError(f"duplicate or invalid sink name: {name}")
        if not np.isclose(value.iloc[-1], expected_last + position):
            raise AssertionError(f"incorrect final value for {name}")
        seen[position] = 1
        completed += 1
        digest.update(name.encode())
        if completed % 1000 == 0:
            persist("RUNNING")

    try:
        factors = [Factor(name=f"f{i}", expr=shared + float(i)) for i in range(args.factors)]
        outcome = engine.run_many(
            factors,
            result_policy="sink",
            sink=sink,
            perf=PerfConfig(max_workers=1, native_fusion=False, result_budget_bytes=16 * 1024**2),
        )
        assert completed == args.factors and all(seen)
        report = {
            "status": "PASS", "requested": args.factors, "completed_sink_outputs": completed,
            "source_load_calls": source.load_calls, "elapsed_seconds": time.monotonic() - started,
            "executor": outcome["executor"], "completed_waves": outcome["completed_waves"],
            "wave_size": outcome["wave_size"], "cse_scope": outcome["cse_scope"],
            "global_cse": outcome["cost_ledger"]["global_cse"], "digest": digest.hexdigest(),
        }
        args.result.write_text(json.dumps(report, indent=2, sort_keys=True))
        persist("PASS", result=str(args.result))
        print(json.dumps(report, sort_keys=True), flush=True)
    except BaseException as exc:
        persist("FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
