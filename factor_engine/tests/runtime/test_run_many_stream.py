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
        # One active wave plus at most one newly-consumed wave; result ownership
        # is bounded separately by the sink queue and outstanding-item gate.
        assert consumed - written <= 256
        assert name == result == f"f{written}"
        written += 1

    out = execute_run_many_stream(engine, factors(), wave_size=128, sink=sink,
                                  sink_queue_bytes=4096, n_jobs=3, pit_enforce=True)
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
    out = execute_run_many_stream(engine, [], sink=lambda *a: None, sink_queue_bytes=1024)
    assert out["completed_waves"] == out["completed_factors"] == 0
    assert engine.waves == []


def test_duplicate_name_across_waves_stops_before_duplicate_write():
    written = []
    factors = [SimpleNamespace(name=name) for name in ["a", "b", "a", "c"]]
    with pytest.raises(DuplicateFactorNameError):
        execute_run_many_stream(FakeEngine(), factors, wave_size=2,
                                sink=lambda name, result: written.append(name), sink_queue_bytes=1024)
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
        execute_run_many_stream(FakeEngine(), factors(), wave_size=2, sink=sink, sink_queue_bytes=100)
    assert caught.value is error
    assert consumed == [0, 1, 2, 3]


def test_rejected_sink_aborts():
    with pytest.raises(RuntimeError, match="sink rejected"):
        execute_run_many_stream(FakeEngine(), [SimpleNamespace(name="a")], sink=lambda *a: False, sink_queue_bytes=1024)


def test_missing_delivery_is_not_reported_as_success():
    engine = SimpleNamespace(run_many_parallel=lambda *a, **kw: {"results": {}})
    with pytest.raises(RuntimeError, match="did not deliver"):
        execute_run_many_stream(engine, [SimpleNamespace(name="a")], sink=lambda *a: None, sink_queue_bytes=1024)


@pytest.mark.parametrize("second", ["a", "unknown"])
def test_duplicate_delivery_is_not_reported_as_success(second):
    def duplicate(factors, *, sink, **kwargs):
        sink(second, 1)
        sink("a", 1)
        return {"results": {}}
    with pytest.raises(RuntimeError, match="unexpected or duplicate"):
        execute_run_many_stream(SimpleNamespace(run_many_parallel=duplicate),
                                [SimpleNamespace(name="a"), SimpleNamespace(name="b")],
                                sink=lambda *a: None, sink_queue_bytes=1024)


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
                                  wave_size=16, sink=sink, sink_queue_bytes=4096)
    assert out["completed_factors"] == 64 and out["completed_waves"] == 4
    assert peak == 1 and active == 0


def test_slow_sink_does_not_hold_delivery_state_lock_or_block_next_wave():
    import threading

    writer_started = threading.Event()
    release_writer = threading.Event()
    second_wave_started = threading.Event()

    class Engine:
        calls = 0

        def run_many_parallel(self, factors, *, sink, **kwargs):
            self.calls += 1
            if self.calls == 2:
                second_wave_started.set()
            for factor in factors:
                sink(factor.name, factor.name)
            return {"results": {}}

    def slow_sink(name, result):
        writer_started.set()
        assert release_writer.wait(5)

    outcome = {}

    def execute():
        outcome.update(execute_run_many_stream(
            Engine(), [SimpleNamespace(name="a"), SimpleNamespace(name="b")],
            wave_size=1, sink=slow_sink, sink_queue_bytes=1024,
        ))

    thread = threading.Thread(target=execute)
    thread.start()
    assert writer_started.wait(2)
    assert second_wave_started.wait(2), "next wave was serialized behind slow sink I/O"
    release_writer.set()
    thread.join(5)
    assert not thread.is_alive()
    assert outcome["completed_factors"] == 2


def test_stream_cost_ledger_is_honest_about_per_wave_cse_and_name_growth():
    out = execute_run_many_stream(
        FakeEngine(), [SimpleNamespace(name="alpha"), SimpleNamespace(name="beta")],
        wave_size=1, sink=lambda *args: None, sink_queue_bytes=1024,
    )
    ledger = out["cost_ledger"]
    assert ledger["cse_scope"] == "per_wave"
    assert ledger["global_cse"] is False
    assert ledger["name_metadata_entries"] == 2
    assert ledger["name_utf8_payload_bytes"] == len(b"alphabeta")
    assert ledger["name_set_container_bytes"] is None
    assert ledger["sink_accepted"] == ledger["sink_committed"] == 2
    assert ledger["sink_queue_wait_seconds"] is None
    assert ledger["sink_submit_wall_seconds"] >= 0
    assert ledger["sink_finish_wait_seconds"] >= 0


def test_sink_false_is_a_durable_failure_not_success_or_replay():
    calls = []

    def refusing(name, result):
        calls.append(name)
        return False

    with pytest.raises(RuntimeError, match="sink rejected|writer fatal"):
        execute_run_many_stream(
            FakeEngine(), [SimpleNamespace(name="a")], sink=refusing,
            sink_queue_bytes=1024,
        )
    assert calls == ["a"]


def test_missing_result_queue_budget_refuses_before_input_consumption():
    consumed = []

    def factors():
        consumed.append(True)
        yield SimpleNamespace(name="a")

    with pytest.raises(ValueError, match="explicit budget"):
        execute_run_many_stream(FakeEngine(), factors(), sink=lambda *args: None)
    assert consumed == []


def test_timeout_from_unreceipted_user_sink_is_never_replayed():
    calls = 0

    def timeout(*args):
        nonlocal calls
        calls += 1
        raise TimeoutError("ambiguous remote timeout")

    with pytest.raises(TimeoutError, match="ambiguous"):
        execute_run_many_stream(
            FakeEngine(), [SimpleNamespace(name="a")], sink=timeout,
            sink_queue_bytes=1024,
        )
    assert calls == 1


def test_compute_failure_is_not_masked_by_sink_finish():
    error = LookupError("compute failed")

    def fail(*args, **kwargs):
        raise error

    with pytest.raises(LookupError) as caught:
        execute_run_many_stream(
            SimpleNamespace(run_many_parallel=fail), [SimpleNamespace(name="a")],
            sink=lambda *args: None, sink_queue_bytes=1024,
        )
    assert caught.value is error


def test_input_generator_is_closed_on_compute_failure():
    closed = False

    def factors():
        nonlocal closed
        try:
            yield SimpleNamespace(name="a")
            yield SimpleNamespace(name="b")
        finally:
            closed = True

    error = RuntimeError("cancelled compute")

    def fail(*args, **kwargs):
        raise error

    with pytest.raises(RuntimeError) as caught:
        execute_run_many_stream(
            SimpleNamespace(run_many_parallel=fail), factors(), wave_size=1,
            sink=lambda *args: None, sink_queue_bytes=1024,
        )
    assert caught.value is error
    assert closed


def test_oversize_result_is_refused_before_ownership_transfer():
    calls = []
    huge = bytearray(4096)

    class Engine:
        def run_many_parallel(self, factors, *, sink, **kwargs):
            sink(factors[0].name, huge)
            return {"results": {}}

    with pytest.raises(Exception, match="budget|capacity|large"):
        execute_run_many_stream(
            Engine(), [SimpleNamespace(name="a")],
            sink=lambda *args: calls.append(args), sink_queue_bytes=64,
        )
    assert calls == []


def test_unknown_result_charge_refuses_before_user_sink():
    calls = []

    class Unknown:
        pass

    class Engine:
        def run_many_parallel(self, factors, *, sink, **kwargs):
            sink(factors[0].name, Unknown())
            return {"results": {}}

    with pytest.raises(Exception, match="unknown|estimate"):
        execute_run_many_stream(
            Engine(), [SimpleNamespace(name="a")],
            sink=lambda *args: calls.append(args), sink_queue_bytes=1024,
        )
    assert calls == []


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
        sink=sink, wave_size=3, n_jobs=2,
        perf=PerfConfig(max_workers=2, native_fusion=False, result_budget_bytes=4096),
    )
    assert set(seen) == {f"f{i}" for i in range(7)}
    assert out["completed_factors"] == 7 and out["completed_waves"] == 3
    engine.run_mode = "production"
    with pytest.raises(Exception, match="input_dq_check|auto_warmup|production"):
        engine.run_many_stream(
            [Factor(name="p", expr=col("close"))], sink=sink, wave_size=1,
            perf=PerfConfig(result_budget_bytes=1024),
        )
