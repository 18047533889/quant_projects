"""Purpose contracts; these synthetic profiles grant no real-data access."""
import json
import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from factor_engine.runtime.default_execution_policy import (
    DefaultExecutionPolicy, ExecutionPurpose, operator_admission_mode,
)
from factor_engine.runtime.default_engine import (
    ApprovedDeploymentProfile, DeploymentConfigurationError,
    ExecutionCoreWorkerConfig, build_execution_core_from_worker_config, get_engine,
)


def _profile(tmp_path, **extra):
    return {"profile_id": "r3-contract-only", "approval_id": "synthetic-no-data-grant",
            "data_source": {"type": "data_access", "dataset": "ashare_stock_daily_adj",
                            "start_date": "2024-01-01", "end_date": "2024-01-31",
                            "instrument_filter": ["000001.SZ"]},
            "artifact_root": str(tmp_path), "market": "ashare", "calendar_id": "SSE",
            "timezone": "Asia/Shanghai", "frequency": "1d", "universe_id": "fixture",
            "adjustment": "hfq", "expected_source_content_digests": {
                "ashare_stock_daily_adj": "a" * 32}, **extra}


def test_purpose_is_immutable_and_separate_from_resource_policy():
    research = ExecutionPurpose()
    production = ExecutionPurpose("production_compute")
    assert research.digest != production.digest
    assert research.to_dict()["assurance"] == production.to_dict()["assurance"] == "UNVERIFIED"
    assert research.to_dict()["publication_authorized"] is False
    with pytest.raises(FrozenInstanceError):
        research.purpose = "production_compute"
    assert operator_admission_mode(SimpleNamespace(run_mode="production")) == "production"
    assert operator_admission_mode(SimpleNamespace(run_mode="production", execution_purpose=research)) == "research"
    with pytest.raises(TypeError):
        operator_admission_mode(SimpleNamespace(execution_purpose="research"))
    assert DefaultExecutionPolicy().memory_fraction == .8


@pytest.mark.parametrize("purpose", ["research", "diagnostic", "", None, False])
def test_unimplemented_or_invalid_purpose_is_not_silently_reinterpreted(tmp_path, purpose):
    with pytest.raises(DeploymentConfigurationError):
        ApprovedDeploymentProfile.from_mapping(_profile(tmp_path, purpose=purpose))


@pytest.mark.parametrize("purpose", [None, "production_compute"])
def test_parent_worker_clone_context_keep_purpose_and_strict_input(tmp_path, purpose):
    mapping = _profile(tmp_path, **({} if purpose is None else {"purpose": purpose}))
    deployment = ApprovedDeploymentProfile.from_mapping(mapping)
    config = ExecutionCoreWorkerConfig(deployment, DefaultExecutionPolicy())
    core = build_execution_core_from_worker_config(pickle.loads(pickle.dumps(config)))
    try:
        assert core.run_mode == "production"
        assert core.data_source.production and core.data_source.pit_enforce
        assert core.data_source.strict_unknown_fields
        assert core.execution_purpose.purpose == (purpose or "research_compute")
        clone = core.with_data_source(core.data_source)
        assert clone.execution_purpose is core.execution_purpose
        assert clone.resource_broker is core.resource_broker
        assert clone.execution_scope is core.execution_scope
        context = clone._make_context()
        assert context.execution_purpose == core.execution_purpose
        assert context.run_mode == "production"
        assert context.execution_scope == core.execution_scope
        assert context.data_source.execution_scope == core.execution_scope
        assert not hasattr(core.data_source, "execution_scope")
    finally:
        core.data_source.close()


def test_forged_worker_wrapper_is_revalidated_before_source_creation(tmp_path):
    raw = _profile(tmp_path)
    raw["data_source"]["pit_enforce"] = False
    config = ExecutionCoreWorkerConfig(ApprovedDeploymentProfile(json.dumps(raw)), DefaultExecutionPolicy())
    with pytest.raises(DeploymentConfigurationError):
        build_execution_core_from_worker_config(config)


def test_changed_core_purpose_rejected_before_input_iteration(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile(tmp_path)))
    with get_engine(profile_path=path) as engine:
        engine._engine.execution_purpose = ExecutionPurpose("production_compute")
        with pytest.raises(DeploymentConfigurationError, match="changed"):
            engine.run_many(iter(()))


def test_changed_core_scope_rejected_before_input_iteration(tmp_path):
    from factor_engine.api.factor import FactorExecutionScopeHint
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile(tmp_path)))
    with get_engine(profile_path=path) as engine:
        engine._engine.execution_scope = FactorExecutionScopeHint(
            market="US", calendar_id="NYSE", frequency="1d", universe_id="sp500"
        )
        with pytest.raises(DeploymentConfigurationError, match="scope"):
            engine.run_many(iter(()))


def test_real_model_compile_keeps_typed_pit_without_production_certificate(tmp_path):
    from dataclasses import replace
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.api.factor import FactorExecutionScopeHint
    from factor_engine.runtime.production_policy import assert_production_factors
    deployment = ApprovedDeploymentProfile.from_mapping(_profile(tmp_path))
    core = build_execution_core_from_worker_config(
        ExecutionCoreWorkerConfig(deployment, DefaultExecutionPolicy()))
    try:
        factor = replace(parse_factor("cs_huber_resid(close, open)", name="model", surface="all"),
                         semantic_identity=FactorExecutionScopeHint(
                             market="ashare", frequency="1d", calendar_id="SSE", universe_id="fixture"))
        plan, analysis = core.compile(factor)
        assert plan is not None and analysis.ir is not None
        assert_production_factors([factor], mode=core.run_mode,
                                  execution_purpose=core.execution_purpose)
        forged = replace(factor, source_expr="close + open")
        with pytest.raises(Exception, match="does not match"):
            assert_production_factors([forged], mode=core.run_mode,
                                      execution_purpose=core.execution_purpose)
        # Python Expr has no text to compare, but still goes through the same
        # typed compile/PIT analyzer; no alternate engine or fake operator.
        expression_only = replace(factor, source_expr=None)
        assert_production_factors([expression_only], mode=core.run_mode,
                                  execution_purpose=core.execution_purpose)
        core.compile(expression_only)
        assert core.run_mode == "production"
    finally:
        core.data_source.close()


def test_research_purpose_does_not_launder_future_dependency_through_rank():
    from factor_engine.ir.nodes import IRNode
    from factor_engine.runtime.quality.pit_audit import audit_ir
    future = IRNode("ts_delay", inputs=(IRNode("column", attrs={"name": "close"}),),
                    attrs={"n": -1})
    expression = IRNode("rank", inputs=(future,))
    report = audit_ir(expression, execution_purpose=ExecutionPurpose())
    assert not report.passed
    assert any("ts_delay(n=-1)" in item for item in report.violations)


def test_structural_temporal_policy_does_not_promote_production_evidence():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
    load_all(include_research=True)
    op = OperatorRegistry.get("cs_huber_resid", "pandas_numpy", mode="any")
    production = infer_operator_policy(op, canonical="cs_huber_resid")
    research = infer_operator_policy(op, canonical="cs_huber_resid", certification_required=False)
    assert research.pit_safe and research.scope == "cs"
    assert not production.pit_safe


@pytest.mark.parametrize("expression,passed", [
    ("cs_huber_resid(close, open)", True),
    ("ts_delay(close, -1)", False),
])
def test_derived_field_audit_retains_purpose_and_future_gate(monkeypatch, expression, passed):
    from factor_engine.runtime.derived_field_registry import DerivedFieldDefinition
    from factor_engine.runtime.quality.pit_audit import _audit_derived_field
    monkeypatch.setattr(
        "factor_engine.runtime.derived_field_registry.load_derived_field_definition",
        lambda name: DerivedFieldDefinition(name, 1, expression, "fixture"))
    violations, checked = [], []
    _audit_derived_field("example", forbid_forward_fill=False, fail_on_missing=True,
                         violations=violations, checked=checked, source_guard=set(),
                         execution_purpose=ExecutionPurpose())
    assert bool(not violations) == passed, violations
    assert checked


def test_logical_wrapper_cannot_reuse_other_purpose():
    from factor_engine.api.factor import FactorExecutionScopeHint
    from factor_engine.backend.context import ExecutionContext
    class Source:
        pass
    scope = FactorExecutionScopeHint(
        market="ashare", calendar_id="SSE", frequency="1d", universe_id="fixture"
    )
    source = Source()
    first = ExecutionContext(
        source, run_mode="production", execution_purpose=ExecutionPurpose(),
        execution_scope=scope,
    )
    assert first.data_source.execution_purpose == ExecutionPurpose()
    with pytest.raises(ValueError, match="cross execution purposes"):
        ExecutionContext(first.data_source, run_mode="production",
                         execution_purpose=ExecutionPurpose("production_compute"),
                         execution_scope=scope)
    other = ExecutionContext(source, run_mode="production",
                             execution_purpose=ExecutionPurpose("production_compute"),
                             execution_scope=scope)
    assert first.data_source is not other.data_source


def test_managed_context_requires_complete_scope_and_rejects_cross_scope_reuse():
    from factor_engine.api.factor import FactorExecutionScopeHint
    from factor_engine.backend.context import ExecutionContext
    class Source:
        pass
    purpose = ExecutionPurpose()
    with pytest.raises(ValueError, match="complete execution scope"):
        ExecutionContext(Source(), run_mode="production", execution_purpose=purpose)
    with pytest.raises(ValueError, match="calendar_id"):
        ExecutionContext(
            Source(), run_mode="production", execution_purpose=purpose,
            execution_scope=FactorExecutionScopeHint(
                market="ashare", frequency="1d", universe_id="fixture"
            ),
        )
    scope = FactorExecutionScopeHint(
        market="ashare", calendar_id="SSE", frequency="1d", universe_id="fixture"
    )
    first = ExecutionContext(
        Source(), run_mode="production", execution_purpose=purpose,
        execution_scope=scope,
    )
    conflicting = FactorExecutionScopeHint(
        market="US", calendar_id="NYSE", frequency="1d", universe_id="sp500"
    )
    with pytest.raises(ValueError, match="cross execution scopes"):
        ExecutionContext(
            first.data_source, run_mode="production", execution_purpose=purpose,
            execution_scope=conflicting,
        )


def test_worker_scope_is_frozen_and_pickle_stable(tmp_path):
    from factor_engine.api.factor import FactorExecutionScopeHint
    deployment = ApprovedDeploymentProfile.from_mapping(_profile(tmp_path))
    core = build_execution_core_from_worker_config(
        pickle.loads(pickle.dumps(
            ExecutionCoreWorkerConfig(deployment, DefaultExecutionPolicy())
        ))
    )
    try:
        expected = FactorExecutionScopeHint(
            market="ashare", calendar_id="SSE", frequency="1d", universe_id="fixture"
        )
        assert core.execution_scope == expected
        assert pickle.loads(pickle.dumps(core.execution_scope)) == expected
        with pytest.raises(FrozenInstanceError):
            core.execution_scope.market = "US"
    finally:
        core.data_source.close()
