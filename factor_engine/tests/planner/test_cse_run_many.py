"""CSE、run_many、常量折叠与 plan_ref 执行。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.api import rank, ts_mean
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.backend.context import ExecutionContext
from factor_engine.planner.cse import apply_cse
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.lowerer import Lowerer
from factor_engine.planner.optimizer import Optimizer
from factor_engine.ir.analyzer import Analyzer
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _data():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {
        "close": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx),
    }


def test_duplicate_dependency_edges_do_not_release_root_early():
    from factor_engine.planner.physical_factor_dag import PhysicalFactorDAG, PhysicalFactorTask
    dag = PhysicalFactorDAG()
    for name, inputs, consumers in [('a',(),('b','b','c')),('b',('a','c'),()),('c',('a',),('b',))]:
        dag.tasks[name] = PhysicalFactorTask(task_id=name,op='x',task_type='compute',inputs=inputs,consumers=consumers)
    assert dag.topological_order() == ['a','c','b']
    assert dag.critical_path_remaining_ms('a',dict(a=1.,b=2.,c=3.)) == 6.


def test_cse_duplicate_subtree_becomes_ref_and_shared_nodes():
    expr = ts_mean(col("close"), 2)
    ir = Analyzer().lower(expr).ir
    p0 = Optimizer().optimize(Lowerer().to_logical_plan(ir))
    roots, shared = apply_cse([p0, p0])
    assert len(shared) >= 1
    assert any(n.op == "plan_ref" for n in _walk(roots[0]))
    assert any(n.op == "plan_ref" for n in _walk(roots[1]))


@pytest.mark.parametrize('label', ['hybrid_long', 'auto_long', 'polars_long'])
def test_long_consumer_requests_native_wave_representation(label):
    from types import SimpleNamespace
    from factor_engine.planner.physical_lowerer import backend_context_for
    from factor_engine.planner.source_representation import (
        consumer_backend_mask, preferred_representation_for_mask,
    )
    ir = Analyzer().lower(ts_mean(col('close'),2)).ir
    plan = Lowerer().to_logical_plan(ir)
    stage = backend_context_for(plan,ctx=SimpleNamespace(selected_backend=label))
    assert stage.preferred_backend == 'polars'
    assert not stage.materializes_full_panel
    assert preferred_representation_for_mask(consumer_backend_mask([stage.preferred_backend])).value == 'polars_lazy'


def _walk(root: PlanNode):
    for c in root.inputs:
        yield from _walk(c)
    yield root


def test_run_many_matches_separate_runs():
    data = _data()
    src = InMemorySeriesSource(data=data)
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    out = eng.run_many([f1, f2])
    r1 = eng.run(f1)["result"]
    r2 = eng.run(f2)["result"]
    pd.testing.assert_series_equal(out["results"]["a"], r1, check_names=False)
    pd.testing.assert_series_equal(out["results"]["b"], r2, check_names=False)
    assert len(out["dag"].shared_nodes) >= 1


def test_optimizer_folds_literal_add():
    from factor_engine.api import add
    from factor_engine.expr.literal import Literal

    e = add(Literal(2.0), Literal(3.0))
    analysis = Analyzer().lower(e)
    plan = Optimizer().optimize(Lowerer().to_logical_plan(analysis.ir))
    assert plan.op == "literal"
    assert float(plan.attrs["value"]) == 5.0


@pytest.mark.parametrize('dispatch', ['parallel', 'default', 'serial'])
def test_global_roots_share_materialization_across_worker_dispatch(monkeypatch, dispatch):
    from collections import Counter
    from threading import Lock
    from factor_engine.api import add
    from factor_engine.expr.literal import Literal
    from factor_engine.runtime import batch_service
    calls = Counter()
    guard = Lock()
    original = batch_service._materialize_shared_subplan
    def tracked(backend, node, ctx, sid):
        with guard: calls[sid] += 1
        return original(backend,node,ctx,sid)
    monkeypatch.setattr(batch_service,'_materialize_shared_subplan',tracked)
    engine = FactorEngine(backend=PandasBackend(),data_source=InMemorySeriesSource(data=_data()))
    shared = ts_mean(col('close'),2)
    factors = [Factor(name=f'global_{i}',expr=add(shared,Literal(float(i)))) for i in range(32)]
    if dispatch == 'parallel':
        result = engine.run_many_parallel(factors,n_jobs=2,enable_cse=True)
    elif dispatch == 'serial':
        from factor_engine.runtime.perf_config import PerfConfig
        result = engine.run_many(factors, perf=PerfConfig(max_workers=1))
    else:
        result = engine.run_many(factors)
    assert len(result['results']) == 32
    assert calls and max(calls.values()) == 1
    first = result['results']['global_0']
    for i in range(32):
        pd.testing.assert_series_equal(result['results'][f'global_{i}'],first+i,check_names=False)


def test_plan_ref_requires_cache():
    ir = Analyzer().lower(ts_mean(col("close"), 2)).ir
    plan = Lowerer().to_logical_plan(ir)
    ref = PlanNode(op="plan_ref", attrs={"sid": "missing"}, inputs=[])
    backend = PandasBackend()
    ctx = ExecutionContext(
        data_source=InMemorySeriesSource(data=_data()),
        shared_result_cache={},
    )
    try:
        backend.execute(ref, ctx)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for missing sid")
