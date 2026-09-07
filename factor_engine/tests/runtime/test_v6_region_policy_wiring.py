from types import SimpleNamespace

from factor_engine.runtime.batch_service import (
    _record_region_candidate_ledger,
    _v2_planner_policy,
    resolve_batch_execution_control_plane,
)
from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy


def test_default_policy_supplies_bounded_region_planner_limits():
    policy = DefaultExecutionPolicy()
    engine = SimpleNamespace(default_execution_policy=policy)

    resolved = _v2_planner_policy(engine)

    assert resolved is policy
    assert resolved.optimization_budget_ms_per_group == 250.0
    assert resolved.candidate_limit_per_group == 128


def test_candidate_ledger_binds_policy_and_physical_authority():
    policy = DefaultExecutionPolicy(
        optimization_budget_ms_per_group=17.0,
        candidate_limit_per_group=9,
    )
    optimization = SimpleNamespace(
        candidate_plan_count=7,
        optimization_elapsed_ms=3.5,
        optimization_basis="dp_global_exact",
        selected_incumbent="single:pandas_numpy:pandas_long",
        production_ready=True,
        readiness_reason="",
    )
    ctx = SimpleNamespace(runtime_stats={})

    _record_region_candidate_ledger(ctx, optimization, policy)

    ledger = ctx.runtime_stats["region_candidate_ledger"]
    assert ledger["authority"] == "PhysicalBatchGlobalOptimizer"
    assert ledger["candidate_plan_count"] == 7
    assert ledger["candidate_limit"] == 9
    assert ledger["optimization_budget_ms"] == 17.0
    assert ledger["policy_digest"] == policy.digest
    assert ledger["production_ready"] is True


def test_legacy_environment_cannot_bypass_v2_physical_admission(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LAYER_LOOP", "1")
    engine = SimpleNamespace(
        run_mode="production",
        default_execution_policy=DefaultExecutionPolicy(),
    )

    assert resolve_batch_execution_control_plane(engine=engine) == "adaptive_scheduler"

    legacy = SimpleNamespace(run_mode="research", default_execution_policy=None)
    assert resolve_batch_execution_control_plane(engine=legacy) == "legacy"
