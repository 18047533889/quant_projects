from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from factor_engine.planner.logical_plan import PlanNode


def _m36(window=120):
    attrs = {} if window is None else {"window": window, "purge_gap": 3}
    columns = tuple(
        PlanNode(op="column", attrs={"name": name}) for name in ("x", "y", "z")
    )
    return PlanNode(op="ts_residualized_hsic", inputs=columns, attrs=attrs)


def test_m36_workspace_default_quadratic_and_non_m36_unchanged():
    from factor_engine.backend.operator_cost import estimate_plan_cost

    default = estimate_plan_cost(_m36(None), rows=250)
    w120 = estimate_plan_cost(_m36(120), rows=250)
    w240 = estimate_plan_cost(_m36(240), rows=250)
    assert default["declared_workspace_bytes"] == 12 * 8 * 120 * 120
    assert w120["declared_workspace_bytes"] == default["declared_workspace_bytes"]
    assert w240["declared_workspace_bytes"] == 4 * w120["declared_workspace_bytes"]
    assert w240["hard_peak_floor_bytes"] == w240["declared_workspace_bytes"] + 4 * 250 * 8
    ordinary = estimate_plan_cost(PlanNode(op="ts_mean", attrs={"window": 240}), rows=250)
    assert ordinary["declared_workspace_bytes"] == 0
    assert ordinary["hard_peak_floor_bytes"] == 0
    assert ordinary["panel_input_output_floor_bytes"] == 0
    assert ordinary["peak_live_memory_bytes"] == 250 * 8


def test_m36_workspace_invalid_and_extreme_integer_boundaries():
    from factor_engine.backend.operator_cost import estimate_plan_cost

    for bad in (True, 0, -1, 120.0, "120"):
        with pytest.raises(ValueError, match="window"):
            estimate_plan_cost(_m36(bad), rows=250)
    huge = estimate_plan_cost(_m36(sys.maxsize), rows=250)
    assert huge["declared_workspace_bytes"] == sys.maxsize
    assert huge["peak_live_memory_bytes"] == sys.maxsize


def test_nested_m36_workspaces_use_peak_not_sum():
    from factor_engine.backend.operator_cost import estimate_plan_cost

    parent = PlanNode(op="add", inputs=(_m36(120), _m36(240)))
    summary = estimate_plan_cost(parent, rows=250)
    assert summary["declared_workspace_bytes"] == 12 * 8 * 240 * 240


def test_m36_fixed_window_adds_simultaneously_live_input_output_rows():
    from factor_engine.backend.operator_cost import estimate_plan_cost

    small = estimate_plan_cost(_m36(120), rows=750)
    large = estimate_plan_cost(_m36(120), rows=30_000)
    assert large["hard_peak_floor_bytes"] - small["hard_peak_floor_bytes"] == (
        30_000 - 750
    ) * 4 * 8
    assert small["panel_input_output_floor_bytes"] == 4 * 750 * 8


def test_m36_default_matches_canonical_param_default():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.backend.operator_cost import estimate_plan_cost
    from factor_engine.cleaned_operators.base import _kernel_param_defaults
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    op = OperatorRegistry.get("ts_residualized_hsic", "pandas_numpy", mode="any")
    default = _kernel_param_defaults(op)["window"]
    assert default == 120
    assert estimate_plan_cost(_m36(None), rows=250) == estimate_plan_cost(
        _m36(default), rows=250
    )


def test_scheduler_passes_flattened_rows_and_zero_instruments_to_lowerer(monkeypatch):
    from factor_engine.planner import physical_lowerer
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.runtime.resource_broker import ResourceBroker

    captured = {}

    class StopAfterCapture(Exception):
        pass

    def capture(_dag, **kwargs):
        captured.update(kwargs)
        raise StopAfterCapture

    monkeypatch.setattr(physical_lowerer, "lower_batch_dag", capture)
    broker = ResourceBroker(
        hard_memory_limit=4 * 1024**3, cpu_slots=2,
        min_host_reserve_gb=0.0, min_host_reserve_fraction=0.0,
    )
    scheduler = AdaptiveBatchScheduler(broker=broker, max_concurrency=1)
    ctx = SimpleNamespace(runtime_stats={"row_count_estimate": 750})
    with pytest.raises(StopAfterCapture):
        scheduler.plan(SimpleNamespace(), ctx=ctx)
    assert captured["rows"] == 750
    assert captured["instruments"] == 0


def test_calibration_cannot_reduce_hard_workspace(monkeypatch):
    import factor_engine.runtime.runtime_calibration as calibration
    from factor_engine.backend.operator_cost import calibrated_plan_peak_bytes

    monkeypatch.setattr(calibration, "calibrated_peak_bytes", lambda *_: (1, 0.01))
    peak, uncertainty = calibrated_plan_peak_bytes(_m36(240), rows=250, window=240)
    hard = 12 * 8 * 240 * 240 + 4 * 250 * 8
    assert int(peak * uncertainty) >= hard
    assert uncertainty == 1.0


@pytest.mark.parametrize("small", [0.01, 0.999999, sys.float_info.min])
def test_m36_subunit_uncertainty_is_integer_safe(monkeypatch, small):
    import factor_engine.runtime.runtime_calibration as calibration
    from factor_engine.backend.operator_cost import calibrated_plan_peak_bytes

    monkeypatch.setattr(calibration, "calibrated_peak_bytes", lambda *_: (1, small))
    peak, uncertainty = calibrated_plan_peak_bytes(_m36(240), rows=250, window=240)
    hard = 12 * 8 * 240 * 240 + 4 * 250 * 8
    assert uncertainty == 1.0
    assert int(peak * uncertainty) >= hard


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_m36_invalid_calibration_uncertainty_falls_back_safely(monkeypatch, bad):
    import factor_engine.runtime.runtime_calibration as calibration
    from factor_engine.backend.operator_cost import calibrated_plan_peak_bytes

    monkeypatch.setattr(calibration, "calibrated_peak_bytes", lambda *_: (1, bad))
    peak, uncertainty = calibrated_plan_peak_bytes(_m36(240), rows=250, window=240)
    hard = 12 * 8 * 240 * 240 + 4 * 250 * 8
    assert peak >= hard
    assert uncertainty == 1.30


def test_non_m36_calibration_result_is_unchanged(monkeypatch):
    import factor_engine.runtime.runtime_calibration as calibration
    from factor_engine.backend.operator_cost import calibrated_plan_peak_bytes

    monkeypatch.setattr(calibration, "calibrated_peak_bytes", lambda *_: (7, 0.01))
    peak, uncertainty = calibrated_plan_peak_bytes(
        PlanNode(op="ts_mean", attrs={"window": 20}), rows=250, window=20
    )
    assert (peak, uncertainty) == (7, 0.01)


def test_physical_lowerer_contract_is_admitted_by_existing_broker(monkeypatch):
    from factor_engine.planner.physical_lowerer import lower_root_plan
    from factor_engine.runtime.resource_broker import ResourceBroker

    tasks = lower_root_plan(
        _m36(240), factor_name="m36", rows=750, instruments=0,
        preferred_backend="pandas_numpy",
    )
    root = next(task for task in tasks if task.task_id == "root:m36")
    contract = root.resource_contract
    assert contract is not None
    hard = 12 * 8 * 240 * 240 + 4 * 750 * 8
    assert contract.admissible_peak_bytes >= hard
    alternate = next(
        task.resource_contract
        for task in lower_root_plan(
            _m36(240), factor_name="m36-alt", rows=750, instruments=3,
            preferred_backend="pandas_numpy",
        )
        if task.task_id == "root:m36-alt"
    )
    assert alternate.admissible_peak_bytes == contract.admissible_peak_bytes
    broker = ResourceBroker(
        hard_memory_limit=4 * 1024**3, cpu_slots=4,
        min_host_reserve_gb=0.0, min_host_reserve_fraction=0.0,
    )
    monkeypatch.setattr(broker, "pressure_stage", lambda: "NORMAL")
    # Exercise the authoritative execution pool, not an incomplete legacy
    # live-headroom snapshot. No model-sized buffers are allocated here.
    monkeypatch.setattr(
        broker, "execution_budget", lambda: hard - 1
    )
    assert broker.try_reserve(contract, task_id="m36-too-small") is None
    monkeypatch.setattr(
        broker, "execution_budget",
        lambda: 2 * contract.admissible_peak_bytes,
    )
    lease = broker.try_reserve(contract, task_id="m36-root")
    assert lease is not None
    assert len(broker._running) == 1
    assert len(broker._memory_leases) == 0
    lease.release()
    assert len(broker._running) == 0
