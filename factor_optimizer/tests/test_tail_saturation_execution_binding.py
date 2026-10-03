import pytest

from factor_optimizer.adapters.repair_execution import compile_value_repair
from factor_optimizer.search.execution_dedup import (
    _execution_binding, _execution_signature, deduplicate_proposals,
)


def _plan():
    return compile_value_repair(
        "TAIL_SATURATION",
        {"saturation_quantile": .95, "saturate": "top"},
        natural_time_scale=5, training_context_ref="train:tail-binding")


def test_tail_saturation_identity_binds_actual_fe_winsorize():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    plan = _plan()
    binding = _execution_binding(plan)
    assert binding["route"] == "factor_engine.operator_registry"
    actual = OperatorRegistry.get("winsorize", backend="pandas_numpy", mode="any")
    assert actual is not None
    details = binding["binding"]
    assert details["canonical"] == OperatorRegistry.resolve_canonical_strict("winsorize")
    assert details["backend"] == "pandas_numpy"
    assert details["mode"] == "any"
    assert isinstance(details["semantic_version"], str)
    assert details["operator_type"] == {
        "module": type(actual).__module__, "qualname": type(actual).__qualname__}
    assert details["operator_implementation_hash"]
    assert details["operator_contract_hash"]
    assert details["adapter_implementation_hash"]
    assert details["python_version"]
    assert details["numpy_version"]
    assert details["pandas_version"]


def test_tail_saturation_operator_replacement_changes_execution_signature(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    plan = _plan()
    before = _execution_signature(plan, 1, None)
    operator = OperatorRegistry.get("winsorize", backend="pandas_numpy", mode="any")
    assert operator is not None
    original_get = OperatorRegistry.get
    import factor_engine.cleaned_operators.registry as registry_module
    original_impl_hash = registry_module._impl_source_hash
    original_contract_hash = registry_module._contract_hash

    class ReplacementWinsorize:
        metadata = operator.metadata

        def calculate(self, values, **kwargs):
            return operator.calculate(values, **kwargs)

    replacement = ReplacementWinsorize()

    def get(name, backend="pandas_numpy", *, mode="production"):
        if name == "winsorize" and backend == "pandas_numpy":
            return replacement
        return original_get(name, backend=backend, mode=mode)

    monkeypatch.setattr(registry_module, "_impl_source_hash",
                        lambda selected: "replacement-impl" if selected is replacement
                        else original_impl_hash(selected))
    monkeypatch.setattr(registry_module, "_contract_hash",
                        lambda selected: "replacement-contract" if selected is replacement
                        else original_contract_hash(selected))
    monkeypatch.setattr(OperatorRegistry, "get", get)
    assert _execution_binding(plan)["binding"]["operator_implementation_hash"] != (
        _execution_binding_before(operator))
    assert _execution_signature(plan, 1, None) != before


def _execution_binding_before(operator):
    from factor_engine.cleaned_operators.registry import _impl_source_hash
    return _impl_source_hash(operator)


def test_tail_saturation_missing_fe_binding_keeps_proposals_separate(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    original_get = OperatorRegistry.get
    monkeypatch.setattr(
        OperatorRegistry, "get",
        lambda name, backend="pandas_numpy", *, mode="production":
        None if name == "winsorize" else original_get(name, backend=backend, mode=mode))
    plan = _plan()
    with pytest.raises(RuntimeError, match="no pandas_numpy backend"):
        _execution_binding(plan)
    proposals = [(plan.family, {}, plan, 1), (plan.family, {}, plan, 1)]
    retained = deduplicate_proposals(proposals, compile_plan=lambda *_: plan)
    assert len(retained) == 2
    assert all(not item.aliases for item in retained)


def test_tail_dedup_resolves_registry_replacement_between_proposals(monkeypatch):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    import factor_engine.cleaned_operators.registry as registry_module

    plan = _plan()
    original_operator = OperatorRegistry.get(
        "winsorize", backend="pandas_numpy", mode="any")
    assert original_operator is not None
    original_get = OperatorRegistry.get
    original_impl_hash = registry_module._impl_source_hash
    original_contract_hash = registry_module._contract_hash

    class ReplacementWinsorize:
        metadata = original_operator.metadata

        def calculate(self, values, **kwargs):
            return original_operator.calculate(values, **kwargs)

    replacement = ReplacementWinsorize()
    active = {"operator": original_operator, "compiled": 0}
    selected = []

    def get(name, backend="pandas_numpy", *, mode="production"):
        if name == "winsorize" and backend == "pandas_numpy":
            operator = active["operator"]
            selected.append(operator)
            return operator
        return original_get(name, backend=backend, mode=mode)

    def compile_plan(*_):
        active["compiled"] += 1
        active["operator"] = (original_operator if active["compiled"] == 1
                              else replacement)
        return plan

    monkeypatch.setattr(
        registry_module, "_impl_source_hash",
        lambda operator: "replacement-impl" if operator is replacement
        else original_impl_hash(operator))
    monkeypatch.setattr(
        registry_module, "_contract_hash",
        lambda operator: "replacement-contract" if operator is replacement
        else original_contract_hash(operator))
    monkeypatch.setattr(OperatorRegistry, "get", get)
    proposals = [(plan.family, {}, None, 1), (plan.family, {}, None, 1)]
    retained = deduplicate_proposals(proposals, compile_plan=compile_plan)

    assert active["compiled"] == 2
    assert original_operator in selected and replacement in selected
    assert len(retained) == 2
    assert all(len(item.aliases) <= 1 for item in retained)

