"""CSE、run_many、常量折叠与 plan_ref 执行。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.columns import col
from api.factor import Factor
from api import rank, ts_mean
from backend.pandas_backend import PandasBackend
from backend.context import ExecutionContext
from planner.cse import apply_cse
from planner.logical_plan import PlanNode
from planner.lowerer import Lowerer
from planner.optimizer import Optimizer
from ir.analyzer import Analyzer
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _data():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return {
        "close": pd.Series(np.arange(4, dtype=float) + 1.0, index=idx),
    }


def test_cse_duplicate_subtree_becomes_ref_and_shared_nodes():
    expr = ts_mean(col("close"), 2)
    ir = Analyzer().lower(expr).ir
    p0 = Optimizer().optimize(Lowerer().to_logical_plan(ir))
    roots, shared = apply_cse([p0, p0])
    assert len(shared) >= 1
    assert any(n.op == "plan_ref" for n in _walk(roots[0]))
    assert any(n.op == "plan_ref" for n in _walk(roots[1]))


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
    from api import add
    from expr.literal import Literal

    e = add(Literal(2.0), Literal(3.0))
    analysis = Analyzer().lower(e)
    plan = Optimizer().optimize(Lowerer().to_logical_plan(analysis.ir))
    assert plan.op == "literal"
    assert float(plan.attrs["value"]) == 5.0


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
