from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.batch_service import (
    _ensure_worker_local_fit_failure_sink,
    _execute_root_with_path,
    _materialize_shared_subplan,
)
from tests.helpers import InMemorySeriesSource


def _source_and_plan():
    dates = pd.date_range("2026-03-01", periods=18)
    instruments = ("GOOD", "BAD")
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["timestamp", "instrument"])
    x = []
    y = []
    for row in range(len(dates)):
        x.extend((float(row), 1.0))
        y.extend((2.0 + 3.0 * row, float(row)))
    source = InMemorySeriesSource(data={
        "target": pd.Series(y, index=index),
        "feature": pd.Series(x, index=index),
    })
    plan = PlanNode(
        op="ts_multi_regression_coeff",
        inputs=[
            PlanNode(op="column", inputs=[], attrs={"name": "target"}),
            PlanNode(op="column", inputs=[], attrs={"name": "feature"}),
        ],
        attrs={
            "window": 10,
            "coefficient_index": 1,
            "min_periods": 2,
            "add_intercept": True,
            "warmup_policy": "expanding",
        },
    )
    return source, plan, dates


def test_real_root_bridge_registry_and_dynamic_producer_end_to_end():
    source, plan, dates = _source_and_plan()
    ctx = ExecutionContext(
        data_source=source,
        run_mode="research",
        execution_id="exec-integration",
        profile_id="profile-integration",
        run_id="run-integration",
        shared_result_cache={},
        runtime_stats={},
    )
    sink = _ensure_worker_local_fit_failure_sink(ctx)
    backend = PandasBackend()

    winner = OperatorRegistry.get(
        "ts_multi_regression_coeff", "pandas_numpy", mode="research")
    assert winner is not None
    result, _path = _execute_root_with_path(
        backend,
        plan,
        ctx,
        run_mode="research",
        factor_name="factor-alpha",
        task_id="root-task-alpha",
        factor_id="factor-alpha",
    )

    panel = result.unstack("instrument")
    np.testing.assert_allclose(panel["GOOD"].dropna(), 3.0, rtol=1e-12, atol=1e-12)
    assert panel["BAD"].isna().all()
    failures = [r for r in sink.page() if r.status.reason == "singular"]
    assert failures
    receipt = failures[-1]
    assert receipt.scope_kind == "factor_window"
    assert receipt.scope.canonical == "ts_multi_regression_coeff"
    assert receipt.scope.backend == "pandas_numpy"
    assert receipt.scope.task_id == "root-task-alpha"
    assert receipt.scope.factor_id == "factor-alpha"
    assert receipt.scope.instrument == "BAD"
    assert receipt.scope.window_start in dates
    assert receipt.scope.window_end in dates
    assert receipt.scope.output_row in dates
    assert receipt.scope.fit_cutoff == receipt.scope.window_end
    assert receipt.scope.maturity_cutoff == receipt.scope.window_end


def test_real_shared_cse_bridge_receipts_cannot_claim_factor_owner():
    source, plan, _dates = _source_and_plan()
    ctx = ExecutionContext(
        data_source=source,
        run_mode="research",
        execution_id="exec-shared",
        profile_id="profile-shared",
        run_id="run-shared",
        shared_result_cache={},
        runtime_stats={},
    )
    sink = _ensure_worker_local_fit_failure_sink(ctx)

    assert _materialize_shared_subplan(
        PandasBackend(), plan, ctx, "shared-multi-regression") is True
    assert "shared-multi-regression" in ctx.shared_result_cache
    assert sink.page()
    assert all(r.scope_kind == "kernel_only" for r in sink.page())
    assert all(r.scope.task_id is None for r in sink.page())
    assert all(r.scope.factor_id is None for r in sink.page())
