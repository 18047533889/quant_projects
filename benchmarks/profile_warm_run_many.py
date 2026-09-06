"""Profile only the warm main-thread pipeline on 64 tiny synthetic factors."""
from collections import defaultdict
from functools import wraps
import cProfile
import json
import pstats
import time

import numpy as np
import pandas as pd

from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
from factor_engine.runtime.resource_governor import ExecutionResourceScope
import factor_engine.runtime.batch_service as bs


def main():
    index = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=8), list("ABCD")],
                                      names=["timestamp", "instrument"])
    values = pd.Series(np.arange(32, dtype=float), index=index)
    engine = FactorEngine(backend=PandasBackend(), data_source=Source(values))
    timings = defaultdict(lambda: {"seconds": 0, "calls": 0})

    def instrument(owner, name):
        original = getattr(owner, name)

        @wraps(original)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings[name]["seconds"] += time.perf_counter() - started
                timings[name]["calls"] += 1

        setattr(owner, name, timed)

    for owner, names in [
        (FactorEngine, ["_dag_from_factors", "compile_many", "compile", "_make_context"]),
        (AdaptiveBatchScheduler, ["plan", "run"]),
        (ExecutionResourceScope, ["__enter__", "__exit__"]),
        (bs, ["_canonicalize_batch_factors", "_execute_run_many_scheduler"]),
    ]:
        for name in names:
            instrument(owner, name)
    factors = [Factor(name=f"f{i}", expr=col("close") + float(i)) for i in range(64)]
    seen = []

    def sink(name, result):
        np.testing.assert_array_equal(result, values + int(name[1:]))
        seen.append(name)

    profiler = cProfile.Profile()
    profiler.enable()
    started = time.perf_counter()
    engine.run_many_parallel(factors, n_jobs=4, perf=PerfConfig(max_workers=4, native_fusion=False),
                             result_policy="sink", sink=sink)
    elapsed = time.perf_counter() - started
    profiler.disable()
    assert len(seen) == 64
    profiler.dump_stats("evidence/r2/runmany_20260906_warm64.prof")
    print(json.dumps({"elapsed": elapsed, "timings": timings}, indent=2), flush=True)
    pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(35)


if __name__ == "__main__":
    main()
