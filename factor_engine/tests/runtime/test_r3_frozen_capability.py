from dataclasses import dataclass
import pytest
from factor_engine.backend.operator_capability import _json_contract_value, CapabilityInfrastructureError
from factor_engine.cleaned_operators.registry import _deepfreeze_catalog
from factor_engine.cleaned_operators.base import ParamSpec, RelationalParamSpec


@pytest.mark.parametrize("spec", [ParamSpec(dtype=int, min=1, default=8),
                                  ParamSpec(dtype=float),
                                  RelationalParamSpec("window >= 4*k+1")])
def test_frozen_contract_has_same_capability_identity(spec):
    live = {"parameter": spec}
    assert _json_contract_value(_deepfreeze_catalog(live)) == _json_contract_value(live)


def test_frozen_unknown_dataclass_does_not_bypass_contract_allowlist():
    @dataclass
    class Unknown:
        expression: str = "window >= 0"
    with pytest.raises(CapabilityInfrastructureError):
        _json_contract_value(_deepfreeze_catalog({"x": Unknown()}))
