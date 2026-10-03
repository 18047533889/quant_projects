import numpy as np
import pandas as pd
import pytest

from factor_optimizer.adapters.repair_execution import compile_value_repair
from factor_optimizer.search.execution_dedup import (
    _execution_binding, _execution_signature, deduplicate_proposals,
)


def _sign_plan(direction):
    return compile_value_repair(
        "SIGN_ORIENTATION", {"direction": direction}, natural_time_scale=8,
        training_context_ref="train:neg-binding-test")


def _value_frame(values):
    return pd.DataFrame({
        "asset_id": ["A"] * len(values),
        "date": pd.date_range("2026-01-01", periods=len(values)),
        "value": values,
    })


class _ReplacementNeg:
    def __init__(self, delegate):
        self._delegate = delegate
        self.metadata = delegate.metadata

    def calculate(self, values, **kwargs):
        return self._delegate.calculate(values, **kwargs)


def _registry_get_with_calls(monkeypatch, *, hide_polars=False, hide_all=False):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    original_get = OperatorRegistry.get
    calls = []

    def tracked_get(name, backend="pandas_numpy", *, mode="production"):
        if name == "neg":
            calls.append(backend)
            if hide_all or (hide_polars and backend == "polars"):
                return None
        return original_get(name, backend=backend, mode=mode)

    monkeypatch.setattr(OperatorRegistry, "get", tracked_get)
    return calls, original_get


def test_sign_flip_binding_and_execution_select_polars_first(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    plan = _sign_plan("flip")
    calls, original_get = _registry_get_with_calls(monkeypatch)
    polars_operator = original_get("neg", backend="polars", mode="any")
    if polars_operator is None:
        pytest.skip("this FactorEngine installation has no Polars neg registration")

    binding = _execution_binding(plan)
    assert binding["route"] == "factor_engine.operator_registry"
    assert binding["binding"]["canonical"] == OperatorRegistry.resolve_canonical_strict("neg")
    assert binding["binding"]["backend"] == "polars"
    assert binding["binding"]["mode"] == "any"
    assert binding["binding"]["operator_implementation_hash"]
    assert binding["binding"]["operator_contract_hash"]
    assert binding["binding"]["adapter_implementation_hash"]
    assert binding["binding"]["numpy_version"] == np.__version__
    assert binding["binding"]["pandas_version"] == pd.__version__
    assert binding["binding"]["polars_version"]
    assert calls[0] == "polars"

    calls_before_execution = len(calls)
    result = plan.execute(_value_frame([1.0, -2.0]), allow_research=True)
    np.testing.assert_array_equal(result.to_numpy(), [-1.0, 2.0])
    assert calls[calls_before_execution] == "polars"


def test_sign_flip_binding_and_execution_use_pandas_fallback(monkeypatch):
    plan = _sign_plan("flip")
    calls, _ = _registry_get_with_calls(monkeypatch, hide_polars=True)
    binding = _execution_binding(plan)
    assert binding["binding"]["backend"] == "pandas_numpy"
    assert calls[:2] == ["polars", "pandas_numpy"]

    calls_before_execution = len(calls)
    result = plan.execute(_value_frame([3.0, -4.0]), allow_research=True)
    np.testing.assert_array_equal(result.to_numpy(), [-3.0, 4.0])
    assert calls[calls_before_execution:calls_before_execution + 2] == [
        "polars", "pandas_numpy"]


def test_sign_flip_without_fe_backend_fails_closed_and_does_not_deduplicate(monkeypatch):
    plan = _sign_plan("flip")
    calls, _ = _registry_get_with_calls(monkeypatch, hide_all=True)
    with pytest.raises(RuntimeError, match="no executable backend"):
        _execution_binding(plan)
    assert calls == ["polars", "pandas_numpy"]

    proposals = [(plan.family, {}, plan, 1), (plan.family, {}, plan, 1)]
    retained = deduplicate_proposals(proposals, compile_plan=lambda *_: plan)
    assert len(retained) == 2
    assert all(not item.aliases for item in retained)
    assert calls[-4:] == ["polars", "pandas_numpy", "polars", "pandas_numpy"]


def test_sign_flip_operator_replacement_changes_identity_and_executes(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    plan = _sign_plan("flip")
    original_binding = _execution_binding(plan)
    original_signature = _execution_signature(plan, 1, None)
    actual_backend = original_binding["binding"]["backend"]
    actual_operator = OperatorRegistry.get("neg", backend=actual_backend, mode="any")
    assert actual_operator is not None

    original_get = OperatorRegistry.get

    def replacement_get(name, backend="pandas_numpy", *, mode="production"):
        if name == "neg" and backend == actual_backend:
            return _ReplacementNeg(actual_operator)
        return original_get(name, backend=backend, mode=mode)

    monkeypatch.setattr(OperatorRegistry, "get", replacement_get)
    changed_binding = _execution_binding(plan)
    assert changed_binding["binding"]["backend"] == actual_backend
    assert changed_binding["binding"]["operator_implementation_hash"] != (
        original_binding["binding"]["operator_implementation_hash"])
    changed_signature = _execution_signature(plan, 1, None)

    assert changed_signature != original_signature
    result = plan.execute(_value_frame([5.0, -6.0]), allow_research=True)
    np.testing.assert_array_equal(result.to_numpy(), [-5.0, 6.0])


def test_sign_identity_is_pure_multiply_for_positive_direction(monkeypatch):
    plan = _sign_plan("keep")
    calls, _ = _registry_get_with_calls(monkeypatch, hide_all=True)
    binding = _execution_binding(plan)
    assert binding == {
        "route": "pandas.multiply.v1", "mapping_version": plan.mapping_version,
    }
    assert calls == []
    result = plan.execute(_value_frame([5.0, -6.0]), allow_research=True)
    np.testing.assert_array_equal(result.to_numpy(), [5.0, -6.0])
    assert calls == []


def test_uncertified_sign_multiplier_does_not_get_multiply_identity():
    from dataclasses import replace

    invalid = replace(_sign_plan("keep"), parameters=(("multiplier", 0.5),))
    with pytest.raises(ValueError, match="uncertified SIGN_ORIENTATION multiplier"):
        _execution_binding(invalid)
