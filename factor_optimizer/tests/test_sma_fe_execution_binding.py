"""Execution identity coverage for FactorEngine-backed trailing SMA."""

from factor_optimizer.adapters.repair_execution import compile_value_repair
from factor_optimizer.search.execution_dedup import (
    _execution_binding, _execution_signature, deduplicate_proposals,
)


def _plan():
    return compile_value_repair(
        "CAUSAL_SMOOTHING",
        {"method": "SMA", "natural_time_scale_relative": 1.0},
        natural_time_scale=3,
        training_context_ref="sma-execution-binding-test",
    )


def _replacement_lagged_mean(frame, *, window, min_periods=None, **kwargs):
    """A distinct implementation identity for replacement tests."""
    raise AssertionError("the identity test must not execute the replacement")


def test_sma_binding_tracks_the_function_resolved_after_warm_import(monkeypatch):
    import factor_engine.backend.long_smoothing as long_smoothing

    plan = _plan()
    before = _execution_signature(plan, 1, None)
    before_binding = _execution_binding(plan)
    assert before_binding["route"] == "factor_engine.backend.long_smoothing.lagged_mean"
    assert before_binding["binding"]["selected_function"]["qualname"] == "lagged_mean"
    assert before_binding["binding"]["selected_function"]["module_source_hash"]
    assert before_binding["binding"]["versions"]["polars_version"]
    assert "not transitively hashed" in before_binding["binding"]["identity_scope"]

    # fe_smoothing and long_smoothing have already been imported: only replace
    # the module attribute that execute_lagged_sma resolves on each call.
    monkeypatch.setattr(long_smoothing, "lagged_mean", _replacement_lagged_mean)

    after_binding = _execution_binding(plan)
    assert after_binding["binding"]["selected_function"]["implementation_hash"] != (
        before_binding["binding"]["selected_function"]["implementation_hash"])
    assert _execution_signature(plan, 1, None) != before


def test_sma_dedup_resolves_replacement_after_each_compile_callback(monkeypatch):
    import factor_engine.backend.long_smoothing as long_smoothing

    plan = _plan()
    original = long_smoothing.lagged_mean
    replacements = (original, _replacement_lagged_mean)
    calls = iter(replacements)

    def compile_with_runtime_switch(_family, _parameters):
        monkeypatch.setattr(long_smoothing, "lagged_mean", next(calls))
        return plan

    proposals = [
        ("CAUSAL_SMOOTHING", {}, None, 1),
        ("CAUSAL_SMOOTHING", {}, None, 1),
    ]
    retained = deduplicate_proposals(proposals, compile_plan=compile_with_runtime_switch)

    assert len(retained) == 2
    assert all(len(item.aliases) == 1 for item in retained)

