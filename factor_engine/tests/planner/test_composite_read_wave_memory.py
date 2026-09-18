from types import SimpleNamespace as NS

import pytest

from factor_engine.planner.batch_data_request import CompositeExecutionScanCost
from factor_engine.planner.read_wave_planner import (
    ReadWavePlanner,
    _ReadRequest,
    build_waves_from_dag,
)
from factor_engine.runtime.resource_errors import ResourceBudgetExceeded


def _request(scope, bound):
    return _ReadRequest(
        task_id=f"task:{scope}:{bound}",
        dataset="logical",
        source_scope=scope,
        snapshot_id="snap",
        time_range=("2026-01-01", "2026-01-02"),
        columns=frozenset({"value"}),
        estimated_scan_bytes=bound,
        estimated_memory_bytes=bound,
        composite_scope_memory_bytes=bound,
    )


def test_composite_bound_is_indivisible_and_reused_by_same_scope_requests():
    planner = ReadWavePlanner(wave_memory_budget=20_000, rows_estimate=1)
    for task_id in ("a", "b"):
        planner.register_scan_task(
            task_id,
            dataset="logical",
            source_scope="anchor",
            snapshot_id="snap",
            time_range=("2026-01-01", "2026-01-02"),
            columns={task_id},
            estimated_scan_bytes=100,
            estimated_memory_bytes=10_000,
            composite_scope_memory_bytes=10_000,
            composite_scope_cost_identity=123,
        )

    plan = planner.plan()

    assert len(plan.waves) == 1
    wave = plan.waves[0]
    assert wave.composite_scope_memory_bytes == 10_000
    assert wave.estimated_memory_bytes >= 10_000
    assert wave.estimated_memory_bytes < 20_000


def test_composite_bound_participates_in_atomic_budget_gate():
    planner = ReadWavePlanner(wave_memory_budget=9_999, rows_estimate=1)
    planner.register_scan_task(
        "a",
        dataset="logical",
        source_scope="anchor",
        snapshot_id="snap",
        time_range=None,
        columns={"value"},
        estimated_scan_bytes=100,
        estimated_memory_bytes=10_000,
        composite_scope_memory_bytes=10_000,
    )

    with pytest.raises(ResourceBudgetExceeded, match="atomic read request"):
        planner.plan()


def test_independent_scope_bounds_are_additive_when_costed_together():
    planner = ReadWavePlanner(rows_estimate=1)
    reqs = [_request("rank:1", 6_000), _request("rank:2", 7_000)]

    assert planner._composite_scope_memory_bytes(reqs) == 13_000
    assert planner._wave_memory_bytes({"value"}, reqs) >= 13_000


def test_same_scope_different_cost_identity_is_additive_not_reused():
    planner = ReadWavePlanner(rows_estimate=1)
    first = _request("anchor", 6_000)
    second = _request("anchor", 7_000)
    first.composite_scope_cost_identity = 1
    second.composite_scope_cost_identity = 2

    assert planner._composite_scope_memory_bytes([first, second]) == 13_000


def test_ordinary_request_keeps_existing_footprint_calculation():
    planner = ReadWavePlanner(rows_estimate=2)
    planner.register_scan_task(
        "ordinary",
        dataset="daily",
        source_scope="daily",
        snapshot_id="snap",
        time_range=None,
        columns={"close"},
        estimated_memory_bytes=999_999,
    )

    wave = planner.plan().waves[0]

    assert wave.composite_scope_memory_bytes == 0
    assert wave.estimated_memory_bytes == planner.per_column_bytes


def test_dag_builder_only_promotes_typed_composite_cost_to_scope_bound():
    task = NS(
        task_id="scan",
        task_type="SOURCE_SCAN",
        source_scan_spec=None,
        required_columns=("encoded_ref",),
        source_scope="anchor",
        source_snapshot_id="snap",
        time_range=None,
        instrument_scope=(),
        preferred_backend="pandas_numpy",
    )
    dag = NS(tasks={"scan": task})
    cost = CompositeExecutionScanCost(
        dataset="daily",
        file_count=2,
        total_bytes=12_000,
        estimated_rows=100,
        projected_columns=2,
        total_columns=None,
        remote=False,
        selected_files=2,
        selected_bytes=12_000,
        projection_bytes=10_000,
        component_scope_keys=("anchor", "rank:1"),
    )

    wave = build_waves_from_dag(
        dag, rows_estimate=1, scope_scan_cost_map={"anchor": cost}
    ).waves[0]

    assert wave.composite_scope_memory_bytes == 10_000
    assert wave.estimated_memory_bytes >= 10_000
