"""Discovery defaults must match the real callable without inventing nullability."""
from types import SimpleNamespace
from factor_engine.cleaned_operators.base import ParamSpec
from factor_engine.runtime.operator_snapshot import _parameter_contract

def _op(fn, specs):
    return SimpleNamespace(
        _contract_callable=fn,
        metadata=SimpleNamespace(param_names=list(specs), param_specs=specs),
    )

def test_authored_kernel_default_is_optional_in_agent_schema():
    def fn(window: int = 3):
        pass
    schema, _, verified = _parameter_contract(_op(fn, {"window": ParamSpec(dtype=int, min=1)}), ())
    assert verified and schema["required"] == []
    assert schema["properties"]["window"]["default"] == 3
    assert schema["required"] == []
    assert schema["properties"]["window"]["type"] == "integer"

def test_only_explicit_nullable_spec_admits_null():
    def fn(window: int = None):
        pass
    explicit = _op(fn, {"window": ParamSpec(dtype=int, min=1, default=None)})
    schema, _, _ = _parameter_contract(explicit, ())
    assert schema["properties"]["window"]["type"] == ["integer", "null"]
    assert schema["properties"]["window"]["default"] is None
    missing = _op(fn, {"window": ParamSpec(dtype=int, min=1)})
    schema, _, _ = _parameter_contract(missing, ())
    assert schema["required"] == []
    assert schema["properties"]["window"]["type"] == "integer"

def test_required_scalar_remains_required():
    def fn(window: int):
        pass
    schema, _, _ = _parameter_contract(_op(fn, {"window": ParamSpec(dtype=int, min=1)}), ())
    assert schema["required"] == ["window"]

def test_unknown_parameter_not_promoted_by_kernel_default():
    def fn(window: int = 3):
        pass
    schema, _, verified = _parameter_contract(_op(fn, {"window": None}), ())
    assert not verified
    assert schema["properties"]["window"]["x-factor-engine-verification"] == "unknown"
