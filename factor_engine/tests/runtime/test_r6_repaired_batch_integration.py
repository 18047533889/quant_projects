"""Repaired multi-panel operators through real run_many, not Registry calls only."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource
from dataclasses import replace
from factor_engine.runtime.resource_broker import ResourceBroker

def _coherent_broker(monkeypatch):
    # Isolated test resource snapshot, not evidence of live-host admission.
    import polars as pl
    broker = ResourceBroker(hard_memory_limit=2*1024**3,cpu_slots=max(4,pl.thread_pool_size()),
        min_host_reserve_gb=0,min_host_reserve_fraction=0)
    original = broker._refresh
    def coherent(*, force=False):
        return replace(original(force=force),hard_memory_limit=broker.hard_memory_limit,
            cgroup_memory_current=0,host_mem_available=broker.hard_memory_limit,
            process_rss=0,worker_rss=0,process_family_rss=0,process_family_pss=0,
            host_mem_available_known=True)
    monkeypatch.setattr(broker,"_refresh",coherent)
    decision=replace(broker._conservative_decision(),pressure_state="NORMAL",
        memory_constrained=False,target_cpu_tokens=broker.hard_cpu_slots,target_concurrency=2,
        reasons=("coherent_test_snapshot",))
    monkeypatch.setattr(broker,"resource_decision",lambda **_kwargs:decision)
    return broker

CASES = (
    ("ts_kurt", ("x",), {}),
    ("ts_sma_cn", ("x",), {}),
    ("lqtp_historical_cvar", ("x",), {}),
    ("cs_rank_gaussian", ("x",), {}),
    ("ts_days_since", ("condition",), {}),
    ("ts_vector_path_efficiency", ("x", "y"), {"window": 8}),
    ("ts_hsic", ("x", "y"), {"window": 8}),
    ("ts_structural_level_density", ("price",),
     {"window": 20, "confirmation": 1, "min_periods": 5}),
)

@pytest.mark.parametrize("backend", ["pandas", "polars_long"])
def test_repaired_operators_reach_real_batch_sink(backend, monkeypatch):
    load_all()
    dates = pd.date_range("2025-01-01", periods=32)
    t = np.arange(32, dtype=float)
    frames = {
        "x": pd.DataFrame({"A": np.sin(t*.7)+t*.1, "B": np.cos(t*.4)}, index=dates),
        "y": pd.DataFrame({"A": np.cos(t*.6)+t*.04, "B": np.sin(t*.3)}, index=dates),
        "price": pd.DataFrame({"A": np.tile([100.,110.,100.,90.],8),
                               "B": np.tile([90.,100.,90.,80.],8)}, index=dates),
    }
    frames["condition"] = (frames["x"] > 0).astype(float)
    data = {}
    for name, frame in frames.items():
        series = frame.stack()
        series.index.names = ["timestamp", "instrument"]
        data[name] = series
    engine = FactorEngine(backend=build_backend(backend),
                          data_source=InMemorySeriesSource(data=data), run_mode="research")
    engine.resource_broker = _coherent_broker(monkeypatch)
    factors = [
        Factor(name=name, expr=F(name)(*(col(p) for p in panels), **kwargs))
        for name, panels, kwargs in CASES
    ]
    written = {}
    def sink(name, result):
        assert name not in written
        written[name] = result
        return True
    result = engine.run_many(factors, result_policy="sink", sink=sink)
    assert result["results"] == {}
    assert set(written) == {f.name for f in factors}
    for name, panels, kwargs in CASES:
        reference = OperatorRegistry.get(name, "pandas_numpy", mode="research").calculate(
            *(frames[p] for p in panels), **kwargs)
        expected = reference.stack(future_stack=True)
        expected.index.names = ["timestamp", "instrument"]
        assert expected.notna().any()
        pd.testing.assert_series_equal(written[name].sort_index(), expected.sort_index(),
                                       check_names=False, check_dtype=False,
                                       atol=1e-10, rtol=1e-10)
    assert engine.resource_broker.summary()["running_tasks"] == 0
