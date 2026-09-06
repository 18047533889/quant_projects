# Large batches

The public single-call entry accepts a generator, so callers do not need to write
an outer batching loop:

```python
summary = engine.run_many_stream(
    factor_generator,
    sink=write_result,
    wave_size=1024,
    n_jobs=workers,
    perf=perf,
    # Supply the input-DQ/warmup/PIT flags required by your run mode.
)
assert summary["results"] == {}
print(summary["completed_factors"], summary["completed_waves"])
```

This call pulls at most one wave ahead, requires globally unique factor names,
and stops on input, compilation, execution or sink errors. A synchronous
`write_result(name, result)` returns `None`/`True` on success; explicit `False`
rejects the output. Sink callbacks are serialized. Already written outputs are not
rolled back on failure, and a duplicate name in a later wave is discovered only
when that wave is consumed. The service retains O(F) names to detect duplicates,
but not all factor definitions or outputs. The compact summary excludes full
DAGs, analyses and per-factor backend paths. CSE and warmup unions are per-wave.

The existing APIs remain available with unchanged return contracts:

For an existing `FactorEngine`, stream independent factor waves through the normal
public execution entry. Supply a synchronous sink that writes the result and returns
`None` or `True`; `False` or an exception aborts the wave.

```python
from itertools import islice
from factor_engine.planner import MAX_DAG_WIDTH
from factor_engine.runtime.perf_config import PerfConfig

wave_size = min(1024, MAX_DAG_WIDTH)
pending = iter(factors)
perf = PerfConfig(max_workers=workers, native_fusion=False)
while batch := list(islice(pending, wave_size)):
    summary = engine.run_many_parallel(
        batch,
        n_jobs=workers,
        perf=perf,
        result_policy="sink",
        sink=write_result,
        # Supply the input-DQ/warmup/PIT flags required by your run mode.
    )
    del summary, batch
```

This preserves the existing result contract and all normal resource admission and
physical-plan gates. Tune the wave size to the actual panel size and resource budget;
1024 is the current synthetic test setting, not an OOM-free guarantee. Each wave
compiles its own CSE definitions. Shared-context execution uses threads; NumPy/Polars
can release the GIL, but Python-heavy work is not guaranteed to saturate all cores.

For callers that already hold a `DAGPlan`, public
`iter_compile_many_chunked(dag, chunk_size=...)` yields one physical DAG at a time.
Every chunk contains its reachable transitive CSE definitions and dependencies.
Roots-only input must also pass `shared_nodes=...` when roots contain `plan_ref`.
Consume/release each physical DAG before requesting the next, and use the normal
scheduler/backend/context and sink lifecycle. The legacy `compile_many_chunked`
returns a list and therefore retains all physical chunks.

Neither API raises the host DAG-width limit or bypasses production certification.
