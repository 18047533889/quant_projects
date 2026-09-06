from types import SimpleNamespace
import weakref

import pytest

from factor_engine.planner.dag import DuplicateFactorNameError
from factor_engine.runtime.streaming_batch_service import execute_run_many_stream


class FakeEngine:
    def __init__(self):
        self.waves = []

    def run_many_parallel(self, factors, *, result_policy, sink, **kwargs):
        assert result_policy == "sink"
        self.waves.append((len(factors), kwargs))
        for factor in factors:
            if sink(factor.name, factor.name) is False:
                raise RuntimeError("sink rejected")
        return {"results": {}}


def test_stream_100k_generator_backpressure_and_summary():
    from factor_engine.api.factor import Factor
    engine = FakeEngine()
    consumed = written = 0
    refs = []

    def factors():
        nonlocal consumed
        for i in range(100_000):
            consumed += 1
            factor = Factor(name=f"f{i}", expr=None)
            if i % 1000 == 0:
                refs.append(weakref.ref(factor))
            yield factor

    def sink(name, result):
        nonlocal written
        assert consumed - written <= 128
        assert name == result == f"f{written}"
        written += 1

    out = execute_run_many_stream(engine, factors(), wave_size=128, sink=sink,
                                  n_jobs=3, pit_enforce=True)
    assert out["requested_factors"] == out["completed_factors"] == 100_000
    assert out["completed_waves"] == 782
    assert written == 100_000 and out["results"] == {}
    assert "dag" not in out and "backend_paths" not in out
    assert all(ref() is None for ref in refs)
    assert engine.waves[0][1] == {"n_jobs": 3, "pit_enforce": True}


@pytest.mark.parametrize("sink", [None, 42])
def test_invalid_sink_before_input_consumption(sink):
    def factors():
        pytest.fail("must not consume input")
        yield
    with pytest.raises(ValueError):
        execute_run_many_stream(FakeEngine(), factors(), sink=sink)


@pytest.mark.parametrize("size", [0, -1, True, 2.5, "2", 10**9])
def test_invalid_wave_size(size):
    with pytest.raises(ValueError):
        execute_run_many_stream(FakeEngine(), [], wave_size=size, sink=lambda *a: None)


def test_empty_does_not_compile():
    engine = FakeEngine()
    out = execute_run_many_stream(engine, [], sink=lambda *a: None)
    assert out["completed_waves"] == out["completed_factors"] == 0
    assert engine.waves == []


def test_duplicate_name_across_waves_stops_before_duplicate_write():
    written = []
    factors = [SimpleNamespace(name=name) for name in ["a", "b", "a", "c"]]
    with pytest.raises(DuplicateFactorNameError):
        execute_run_many_stream(FakeEngine(), factors, wave_size=2,
                                sink=lambda name, result: written.append(name))
    assert written == ["a", "b"]


def test_sink_error_propagates_unchanged_without_prefetch():
    consumed = []
    error = OSError("disk full")

    def factors():
        for i in range(10):
            consumed.append(i)
            yield SimpleNamespace(name=str(i))

    def sink(*args):
        raise error

    with pytest.raises(OSError) as caught:
        execute_run_many_stream(FakeEngine(), factors(), wave_size=2, sink=sink)
    assert caught.value is error
    assert consumed == [0, 1]


def test_rejected_sink_aborts():
    with pytest.raises(RuntimeError, match="sink rejected"):
        execute_run_many_stream(FakeEngine(), [SimpleNamespace(name="a")], sink=lambda *a: False)


def test_missing_delivery_is_not_reported_as_success():
    engine = SimpleNamespace(run_many_parallel=lambda *a, **kw: {"results": {}})
    with pytest.raises(RuntimeError, match="did not deliver"):
        execute_run_many_stream(engine, [SimpleNamespace(name="a")], sink=lambda *a: None)


@pytest.mark.parametrize("second", ["a", "unknown"])
def test_duplicate_delivery_is_not_reported_as_success(second):
    def duplicate(factors, *, sink, **kwargs):
        sink(second, 1)
        sink("a", 1)
        return {"results": {}}
    with pytest.raises(RuntimeError, match="unexpected or duplicate"):
        execute_run_many_stream(SimpleNamespace(run_many_parallel=duplicate),
                                [SimpleNamespace(name="a"), SimpleNamespace(name="b")],
                                sink=lambda *a: None)


def test_concurrent_deliveries_serialize_sink_and_count_exactly():
    from concurrent.futures import ThreadPoolExecutor
    import time

    active = peak = 0

    def sink(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.001)
        active -= 1

    def concurrent(factors, *, sink, **kwargs):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda factor: sink(factor.name, 42), factors))
        return {"results": {}}

    out = execute_run_many_stream(SimpleNamespace(run_many_parallel=concurrent),
                                  (SimpleNamespace(name=str(i)) for i in range(64)),
                                  wave_size=16, sink=sink)
    assert out["completed_factors"] == 64 and out["completed_waves"] == 4
    assert peak == 1 and active == 0


def test_public_stream_executes_real_factor_values_and_preserves_production_gate():
    import numpy as np
    import pandas as pd
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.runtime.perf_config import PerfConfig
    from benchmarks.benchmark_run_many_streaming_20260906 import Source

    values = pd.Series(np.arange(6, dtype=float), index=pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["A", "B"]], names=["timestamp", "instrument"],
    ))
    engine = FactorEngine(backend=PandasBackend(), data_source=Source(values))
    seen = []

    def sink(name, result):
        pd.testing.assert_series_equal(result, values + int(name[1:]), check_names=False)
        seen.append(name)

    out = engine.run_many_stream(
        (Factor(name=f"f{i}", expr=col("close") + i) for i in range(7)),
        sink=sink, wave_size=3, n_jobs=2, perf=PerfConfig(max_workers=2, native_fusion=False),
    )
    assert set(seen) == {f"f{i}" for i in range(7)}
    assert out["completed_factors"] == 7 and out["completed_waves"] == 3
    engine.run_mode = "production"
    with pytest.raises(Exception, match="input_dq_check|auto_warmup|production"):
        engine.run_many_stream([Factor(name="p", expr=col("close"))], sink=sink, wave_size=1)
