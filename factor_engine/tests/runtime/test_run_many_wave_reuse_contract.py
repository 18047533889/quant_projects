from types import SimpleNamespace

import pandas as pd
import pytest

from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api import col, ts_mean
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig


def test_warmup_waves_reuse_one_full_batch_compilation(monkeypatch):
    """Cost waves slice the compiled DAG instead of compiling every wave again."""
    from factor_engine.runtime import batch_service
    from factor_engine.runtime import batch_warmup_plan

    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    values = pd.Series(range(len(index)), index=index, dtype=float)
    source = Source(values)
    engine = FactorEngine(PandasBackend(), source)
    shared = ts_mean(ts_mean(col("close"), 2), 2)
    factors = [Factor(f"f{i}", shared + i) for i in range(4)]

    original_compile = engine._dag_from_factors
    compile_batches = []

    def counting_compile(batch, **kwargs):
        compile_batches.append(tuple(f.name for f in batch))
        return original_compile(batch, **kwargs)

    monkeypatch.setattr(engine, "_dag_from_factors", counting_compile)
    monkeypatch.setattr(
        batch_warmup_plan,
        "compute_batch_warmup_plan",
        lambda *args, **kwargs: SimpleNamespace(groups=()),
    )
    monkeypatch.setattr(
        batch_service, "_cluster_factors_by_cost",
        lambda _engine, roots, analyses, plan=None: [list(roots[:2]), list(roots[2:])],
    )
    monkeypatch.setattr(
        batch_service, "_maybe_prepare_batch_warmup",
        lambda current, *args, **kwargs: (current, None),
    )

    result = engine.run_many(
        factors,
        auto_warmup=True,
        warmup_clusters=True,
        perf=PerfConfig(max_workers=1, native_fusion=False),
    )

    assert compile_batches == [("f0", "f1", "f2", "f3")]
    assert set(result["results"]) == {"f0", "f1", "f2", "f3"}
    assert result["warmup_waves"] == [["f0", "f1"], ["f2", "f3"]]
    expected = values.groupby(level="instrument").rolling(2, min_periods=1).mean()
    expected.index = expected.index.droplevel(0)
    expected = expected.reorder_levels(values.index.names).sort_index()
    expected = expected.groupby(level="instrument").rolling(2, min_periods=1).mean()
    expected.index = expected.index.droplevel(0)
    expected = expected.reorder_levels(values.index.names).sort_index()
    for i in range(4):
        pd.testing.assert_series_equal(
            result["results"][f"f{i}"], expected + i, check_names=False,
        )


@pytest.mark.parametrize("broken", ["dangling", "cycle"])
def test_warmup_wave_rejects_invalid_shared_graph_before_sink(monkeypatch, broken):
    from factor_engine.planner.dag import DAGPlan, FactorPlan
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime import batch_service, batch_warmup_plan

    factors = [Factor("f0", col("close")), Factor("f1", col("close"))]
    if broken == "dangling":
        shared = {}
        root = PlanNode("plan_ref", attrs={"sid": "missing"})
    else:
        shared = {
            "a": PlanNode("plan_ref", attrs={"sid": "b"}),
            "b": PlanNode("plan_ref", attrs={"sid": "a"}),
        }
        root = PlanNode("plan_ref", attrs={"sid": "a"})
    dag = DAGPlan(
        roots=[FactorPlan("f0", root), FactorPlan("f1", root)],
        shared_nodes=shared,
    )
    engine = FactorEngine(PandasBackend(), Source(pd.Series(dtype=float)))
    monkeypatch.setattr(
        engine, "_dag_from_factors",
        lambda roots, **kwargs: (dag, {factor.name: object() for factor in roots}),
    )
    monkeypatch.setattr(
        batch_warmup_plan, "compute_batch_warmup_plan",
        lambda *args, **kwargs: SimpleNamespace(groups=()),
    )
    monkeypatch.setattr(
        batch_service, "_cluster_factors_by_cost",
        lambda _engine, roots, analyses, plan=None: [[roots[0]], [roots[1]]],
    )
    calls = []
    with pytest.raises(RuntimeError, match="cannot reuse the compiled DAG"):
        engine.run_many(
            factors, auto_warmup=True, warmup_clusters=True,
            result_policy="sink", sink=lambda *args: calls.append(args),
            perf=PerfConfig(max_workers=1, native_fusion=False),
        )
    assert calls == []


def test_default_oversize_sink_is_bounded_and_cse_is_per_wave(monkeypatch):
    """Document the default: auto waves are bounded; cross-wave CSE is not retained."""
    from factor_engine.planner import physical_lowerer

    monkeypatch.setattr(physical_lowerer, "_get_adaptive_dag_width_limit", lambda: 2)
    monkeypatch.setattr(physical_lowerer, "_get_adaptive_chunk_size", lambda: 2)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["A"]],
        names=["timestamp", "instrument"],
    )

    class CountingSource(Source):
        def __init__(self, values):
            super().__init__(values)
            self.loads = 0

        def load_columns(self, names):
            self.loads += 1
            return super().load_columns(names)

    source = CountingSource(pd.Series(range(len(index)), index=index, dtype=float))
    engine = FactorEngine(PandasBackend(), source)
    shared = ts_mean(col("close"), 2)
    seen = []
    result = engine.run_many(
        [Factor(f"f{i}", shared + i) for i in range(5)],
        result_policy="sink",
        sink=lambda name, value: seen.append(name),
        perf=PerfConfig(
            max_workers=1, native_fusion=False, result_budget_bytes=1024 * 1024,
        ),
    )

    assert seen == ["f0", "f1", "f2", "f3", "f4"]
    assert result["executor"] == "streaming_waves"
    assert result["completed_waves"] == 3
    assert result["wave_size"] == 2
    assert result["cse_scope"] == "per_wave"
    assert source.loads == 3
