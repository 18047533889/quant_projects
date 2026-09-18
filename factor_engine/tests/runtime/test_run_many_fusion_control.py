from types import SimpleNamespace

import pytest

from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.planner.native_fusion import native_fusion_capability_map


class Backend:
    runtime_backend_label = "polars_long"
    def execute_multi_roots(self, nodes, ctx):
        raise AssertionError("capability discovery must not execute")


def test_missing_certification_never_enables_callable_backend(monkeypatch):
    from factor_engine.backend import backend_certification
    monkeypatch.delattr(backend_certification, "backend_certifies", raising=False)
    assert not any(native_fusion_capability_map(SimpleNamespace(backend=Backend())).values())


def test_certification_failure_never_enables_callable_backend(monkeypatch):
    from factor_engine.backend import backend_certification
    def fail(*a):
        raise RuntimeError("evidence unavailable")
    monkeypatch.setattr(backend_certification, "backend_certifies", fail, raising=False)
    assert not any(native_fusion_capability_map(SimpleNamespace(backend=Backend())).values())


def test_capability_only_enables_the_actual_backend(monkeypatch):
    from factor_engine.backend import backend_certification
    monkeypatch.setattr(backend_certification, "backend_certifies", lambda *a: True, raising=False)
    caps = native_fusion_capability_map(SimpleNamespace(backend=Backend()))
    assert {key for key, value in caps.items() if value} == {"polars_long"}


def test_explicit_fusion_disable_wins_over_capability(monkeypatch):
    from factor_engine.backend import backend_certification
    monkeypatch.setattr(backend_certification, "backend_certifies", lambda *a: True, raising=False)
    ctx = SimpleNamespace(backend=Backend(), perf=PerfConfig(native_fusion=False))
    assert not any(native_fusion_capability_map(ctx).values())


@pytest.mark.parametrize("enabled", [True, False])
def test_real_run_many_passes_fusion_preference_to_scheduler(monkeypatch, enabled):
    import pandas as pd
    from benchmarks.benchmark_run_many_streaming_20260906 import Source
    from factor_engine.api import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    engine = FactorEngine(PandasBackend(), Source(pd.Series([1., 2., 3.], index=index)))
    observed = []
    original = AdaptiveBatchScheduler.plan
    def capture(self, *args, **kwargs):
        observed.append(kwargs.get("fusion_backend_capability", "MISSING"))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(AdaptiveBatchScheduler, "plan", capture)
    output = engine.run_many(
        [Factor("a", col("close") + 1), Factor("b", col("close") + 2)],
        perf=PerfConfig(max_workers=1, native_fusion=enabled),
    )
    assert set(output["results"]) == {"a", "b"}
    assert observed == ([None] if enabled else [{}])
