"""Tiny-panel real engine benchmark of public defaults; no production publication."""
import hashlib
import json
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api.dsl_parser import DSLParser
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine

index = pd.MultiIndex.from_product(
    [pd.date_range("2025-01-01", periods=8), ["A", "B"]],
    names=["timestamp", "instrument"])
values = pd.Series(np.arange(len(index), dtype=float), index=index)
class CountingSource(Source):
    def __init__(self, x):
        super().__init__(x)
        self.calls = 0
    def load_columns(self, names):
        self.calls += 1
        return super().load_columns(names)
source = CountingSource(values)
source.instrument_filter = ("A", "B")
source.start_date = index.levels[0].min().tz_localize("UTC")
source.end_date = index.levels[0].max().tz_localize("UTC")
source.schema = {"close": "float64"}
engine = FactorEngine(build_backend("auto"), source, run_mode="research")
parser = DSLParser(surface="compat_research")
factors = []
for i in range(1000):
    formula = f"ts_mean(close, 3) + {i}"
    factors.append(Factor(name=f"f{i}", expr=parser.parse(formula),
                          source_expr=formula, surface="compat_research"))
oracle = values.groupby(level="instrument").rolling(3, min_periods=1).mean()
oracle.index = oracle.index.droplevel(0)
oracle = oracle.reorder_levels(values.index.names).sort_index()
seen = set()
lock = threading.Lock()
def sink(name, value):
    pd.testing.assert_series_equal(value, oracle + int(name[1:]), check_names=False)
    with lock:
        assert name not in seen
        seen.add(name)
    return True
started = time.monotonic()
result = engine.run_many(factors, result_policy="sink", sink=sink)
assert len(seen) == 1000
assert not result["results"]
dag = result.get("dag")
report = {
    "roots": 1000, "input_values": len(values),
    "public_entry": "run_many", "backend": "auto",
    "performance_options": "none; public defaults",
    "successful_oracles": len(seen), "source_load_columns_calls": source.calls,
    "seconds": time.monotonic() - started,
    "executor": result.get("executor"),
    "shared_nodes": len(dag.shared_nodes) if dag is not None else None,
    "completed_waves": result.get("completed_waves"),
    "cse_scope": result.get("cse_scope", "single_call_dag"),
    "actual_backends": sorted({str(p.get("actual_backend"))
                              for p in (result.get("backend_paths") or {}).values()}),
    "scope": "1000 distinct offsets of one rolling subtree; not 110k catalog certification",
}
out = Path(__file__).with_name("r13-default-batch1000.json")
with out.open("x") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report))
