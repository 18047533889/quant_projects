import os
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.batch_service import _consume_bounded_roots


class Broker:
    def summary(self):
        return {"kind": "v7-direct-test"}

    def automatic_result_queue_budget(self):
        return 4 * 1024 * 1024

    def acquire_protected_egress(self, *args, **kwargs):
        return Lease(), Lease()

    def acquire_memory(self, *args, **kwargs):
        return Lease()

    def try_reserve(self, *args, **kwargs):
        return Lease()

    def current_read_budget(self):
        return 16 * 1024 * 1024

    def current_sink_budget(self):
        return 1024 * 1024

    def execution_budget(self):
        return 16 * 1024 * 1024

    def cpu_budget(self):
        return 2

    hard_cpu_slots = 2
    hard_memory_limit = 64 * 1024 * 1024


class Lease:
    def release(self):
        pass


class TinyQueueBroker(Broker):
    def __init__(self, allow_extra=True):
        self.allow_extra = allow_extra

    def automatic_result_queue_budget(self):
        return 64 * 1024

    def acquire_memory(self, *args, **kwargs):
        return Lease() if self.allow_extra else None


class Factor:
    def __init__(self, name):
        self.name = name


class AckLossEngine:
    run_mode = "research"

    def __init__(self, config):
        self.marker = Path(config["marker"])
        self.modes = Path(config["modes"])

    def compile(self, factor):
        return factor

    @staticmethod
    def _value(name):
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "000001.SZ")],
            names=["timestamp", "instrument"],
        )
        return pd.Series([1.25], index=index, name=name)

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        with self.modes.open("a") as stream:
            stream.write(f"{result_policy}:{os.getpid()}\n")
        results = {factor.name: self._value(factor.name) for factor in factors}
        if result_policy == "sink":
            for name, value in results.items():
                sink(name, value)
            if not self.marker.exists():
                self.marker.write_text("artifact-written-before-ack-loss")
                os._exit(17)
            return {"results": {}}
        return {"results": results}

    def close(self):
        pass


def build_ack_loss_engine(config):
    return AckLossEngine(config)


class LargeDirectEngine(AckLossEngine):
    @staticmethod
    def _value(name):
        index = pd.MultiIndex.from_arrays(
            [
                pd.date_range("2025-01-01", periods=20_000, freq="min"),
                ["000001.SZ"] * 20_000,
            ],
            names=["timestamp", "instrument"],
        )
        return pd.Series(range(20_000), index=index, dtype=float, name=name)

    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        with self.modes.open("a") as stream:
            stream.write(f"{result_policy}:{os.getpid()}\n")
        results = {factor.name: self._value(factor.name) for factor in factors}
        if result_policy == "sink":
            for name, value in results.items():
                sink(name, value)
            return {"results": {}}
        return {"results": results}


def build_large_direct_engine(config):
    return LargeDirectEngine(config)


def build_real_engine_with_pid_probe(config):
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    modes = Path(config.pop("__modes__"))
    engine = FactorEngine(PandasBackend(), InMemorySeriesSource(data=config))
    original = engine.run_many_parallel

    def probed(factors, *, result_policy, sink=None, **kwargs):
        with modes.open("a") as stream:
            stream.write(f"{result_policy}:{os.getpid()}\n")
        return original(
            factors, result_policy=result_policy, sink=sink, **kwargs
        )

    engine.run_many_parallel = probed
    return engine


def test_default_spawn_sinks_values_and_lost_descriptor_ack_exactly_reconciles(tmp_path):
    marker = tmp_path / "ack-lost"
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None,
        [Factor("direct")],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
        run_kwargs={"broker": Broker()},
        engine_factory=build_ack_loss_engine,
        engine_factory_config={"marker": str(marker), "modes": str(modes)},
    )

    assert receipt["status"] == "SUCCEEDED"
    assert receipt["counts"] == {"SUCCEEDED": 1}
    mode_records = modes.read_text().splitlines()
    assert [item.split(":")[0] for item in mode_records] == ["sink", "return"]
    assert mode_records[0].split(":")[1] != mode_records[1].split(":")[1]
    with sqlite3.connect(receipt["state_path"]) as db:
        attempts, commit_state = db.execute(
            "SELECT attempts,commit_state FROM outcomes WHERE ordinal=0"
        ).fetchone()
    assert attempts == 2
    assert commit_state == "VERIFIED"
    manifests = list((tmp_path / receipt["run_id"] / "values").glob("*/manifest.json"))
    assert len(manifests) == 1


def test_bounded_root_queue_overlaps_next_compute_with_current_sink():
    sink_started = threading.Event()
    second_compute_finished = threading.Event()
    consumed = []

    def execute(item):
        if item == 1:
            assert sink_started.wait(1)
            second_compute_finished.set()
        return item, f"value-{item}"

    def consume(item, value):
        consumed.append((item, value))
        if item == 0:
            sink_started.set()
            assert second_compute_finished.wait(1)
            time.sleep(0.01)

    _consume_bounded_roots([0, 1], execute, consume, max_workers=2)
    assert sorted(consumed) == [(0, "value-0"), (1, "value-1")]


def test_large_default_spawn_output_bypasses_result_pickle_queue(tmp_path):
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None,
        [Factor("large")],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
        run_kwargs={"broker": TinyQueueBroker(allow_extra=True)},
        engine_factory=build_large_direct_engine,
        engine_factory_config={"marker": str(tmp_path / "unused"), "modes": str(modes)},
    )
    assert receipt["status"] == "SUCCEEDED"
    assert [item.split(":")[0] for item in modes.read_text().splitlines()] == ["sink"]


def test_atomic_large_output_is_rejected_before_write_without_extra_lease(tmp_path):
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None,
        [Factor("large")],
        policy=resolve_default_policy(),
        artifact_root=tmp_path,
        run_kwargs={"broker": TinyQueueBroker(allow_extra=False)},
        engine_factory=build_large_direct_engine,
        engine_factory_config={"marker": str(tmp_path / "unused"), "modes": str(modes)},
    )
    assert receipt["status"] == "COMPLETED_WITH_ERRORS"
    assert receipt["counts"] == {"FAILED": 1}
    assert receipt["error_groups"][0]["code"] == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
    values = tmp_path / receipt["run_id"] / "values"
    assert not values.exists() or not any(values.iterdir())


def test_two_default_spawn_waves_retire_pid_and_bind_fresh_generation(tmp_path):
    marker = tmp_path / "already-normal"
    marker.write_text("do-not-inject-ack-loss")
    modes = tmp_path / "modes"
    receipt = execute_run_many_durable(
        None,
        [Factor("first"), Factor("second")],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path,
        run_kwargs={"broker": Broker()},
        engine_factory=build_ack_loss_engine,
        engine_factory_config={"marker": str(marker), "modes": str(modes)},
    )
    assert receipt["status"] == "SUCCEEDED"
    records = modes.read_text().splitlines()
    assert [item.split(":")[0] for item in records] == ["sink", "sink"]
    assert records[0].split(":")[1] != records[1].split(":")[1]


def test_real_engine_two_wave_default_uses_fresh_compute_pid(tmp_path):
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor as RealFactor

    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    modes = tmp_path / "real-modes"
    receipt = execute_run_many_durable(
        None,
        [RealFactor("close_factor", col("close")), RealFactor("open_factor", col("open"))],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path,
        run_kwargs={"broker": Broker()},
        engine_factory=build_real_engine_with_pid_probe,
        engine_factory_config={
            "close": pd.Series([1.0, 2.0, 3.0, 4.0], index=index),
            "open": pd.Series([2.0, 3.0, 4.0, 5.0], index=index),
            "__modes__": str(modes),
        },
    )
    assert receipt["status"] == "SUCCEEDED"
    records = modes.read_text().splitlines()
    assert [item.split(":")[0] for item in records] == ["sink", "sink"]
    assert records[0].split(":")[1] != records[1].split(":")[1]
