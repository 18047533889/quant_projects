"""Synthetic derived-field scope propagation tests; no external data is read."""
from types import SimpleNamespace

import pandas as pd
import pytest

from factor_engine.api.factor import FactorExecutionScopeHint
from factor_engine.runtime.default_execution_policy import ExecutionPurpose
from factor_engine.runtime.derived_field_registry import (
    DerivedFieldDefinition,
    evaluate_derived_field,
)


def _scope(**changes):
    values = {
        "market": "ashare",
        "calendar_id": "SSE",
        "frequency": "1d",
        "universe_id": "fixture",
    }
    values.update(changes)
    return FactorExecutionScopeHint(**values)


def _definition(monkeypatch):
    monkeypatch.setattr(
        "factor_engine.runtime.derived_field_registry.load_derived_field_definition",
        lambda name: DerivedFieldDefinition(name, 1, "close", "fixture"),
    )


def test_managed_derived_field_binds_exact_scope_and_reaches_real_compile(monkeypatch):
    _definition(monkeypatch)
    expected = _scope()
    seen = {}

    def compile_then_return(self, factor, **kwargs):
        plan, analysis = self.compile(factor)
        seen.update(factor=factor, plan=plan, analysis=analysis, engine=self)
        return {"result": pd.Series([1.0], name=factor.name)}

    monkeypatch.setattr("factor_engine.runtime.engine.FactorEngine.run", compile_then_return)
    source = SimpleNamespace(_cache_broker=object())
    result = evaluate_derived_field(
        "fixture", source,
        execution_purpose=ExecutionPurpose(), execution_scope=expected,
    )
    assert list(result) == [1.0]
    assert seen["factor"].semantic_identity is expected
    assert seen["engine"].execution_scope is expected
    assert seen["plan"] is not None and seen["analysis"].ir is not None


def test_managed_derived_field_rejects_missing_or_partial_scope_before_execution(monkeypatch):
    _definition(monkeypatch)
    source = SimpleNamespace(_cache_broker=object())
    with pytest.raises(ValueError, match="complete execution scope"):
        evaluate_derived_field("fixture", source, execution_purpose=ExecutionPurpose())
    with pytest.raises(ValueError, match="calendar_id"):
        evaluate_derived_field(
            "fixture", source, execution_purpose=ExecutionPurpose(),
            execution_scope=_scope(calendar_id=None),
        )


def test_unmanaged_derived_field_does_not_receive_market_default(monkeypatch):
    _definition(monkeypatch)

    def compile_only(self, factor, **kwargs):
        self.compile(factor)

    monkeypatch.setattr("factor_engine.runtime.engine.FactorEngine.run", compile_only)
    with pytest.raises(Exception, match="requires an explicit market"):
        evaluate_derived_field("fixture", SimpleNamespace())
