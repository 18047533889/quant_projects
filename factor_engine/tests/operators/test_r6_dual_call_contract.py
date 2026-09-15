"""Function-backed operators must retain their real call contract."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators.base import ParamSpec
from factor_engine.cleaned_operators.gemini_v2_common import register_dual
from factor_engine.cleaned_operators.operator_spec import _infer_panel_params
from factor_engine.cleaned_operators.registry import OperatorRegistry

@pytest.fixture
def pair(monkeypatch):
    captured = {}
    def capture(op, *, backend, **kwargs):
        captured[backend] = op
    monkeypatch.setattr(OperatorRegistry, "register", capture)
    def kernel(x: pd.DataFrame, y: pd.DataFrame, window: int = 2):
        return (x + y).rolling(window, min_periods=1).mean()
    register_dual(
        "r6_transport_probe", kernel, ["x", "y", "window"],
        category="test", domain="price_volume", unit="same_as:x", cost=1,
        param_specs={"window": ParamSpec(dtype=int, min=1, max=4)},
    )
    return captured

def test_contract_exposes_real_inputs(pair):
    for op in pair.values():
        assert _infer_panel_params(op, op.metadata, {}) == ("x", "y")
        assert op.metadata.param_names == ["x", "y", "window"]
        assert op.metadata.param_specs["window"].max == 4

@pytest.mark.parametrize("style", ["default", "positional", "keywords"])
def test_real_bridge_calls_match_reference(pair, style):
    x = pd.DataFrame({"A": [1., 3., np.nan, 8.], "B": [2., 4., 6., 10.]})
    y = x * 2
    expected = (x + y).rolling(2, min_periods=1).mean()
    for backend, op in pair.items():
        a, b = (pl.from_pandas(x), pl.from_pandas(y)) if backend == "polars" else (x, y)
        if style == "default":
            result = op.calculate(a, b)
        elif style == "positional":
            result = op.calculate(a, b, 2)
        else:
            result = op.calculate(x=a, y=b, window=2)
        if backend == "polars":
            result = result.to_pandas()
        pd.testing.assert_frame_equal(result, expected)

@pytest.mark.parametrize("bad", [True, 2.5, 0, 5])
def test_scalar_contract_is_enforced_in_both_backends(pair, bad):
    for backend, op in pair.items():
        panel = pl.DataFrame({"A": [1., 2.]}) if backend == "polars" else pd.DataFrame({"A": [1., 2.]})
        with pytest.raises((ValueError, TypeError)):
            op.calculate(panel, panel, window=bad)

def test_keyword_panel_misalignment_rejected(pair):
    op = pair["polars"]
    with pytest.raises((ValueError, TypeError)):
        op.calculate(x=pl.DataFrame({"A": [1., 2.]}), y=pl.DataFrame({"B": [1., 2.]}))

def test_private_default_kernel_preserves_required_scalar_role():
    from types import SimpleNamespace
    def kernel(panel: pd.DataFrame, degree: int, window: int = 3):
        return panel
    class Adapter:
        metadata = SimpleNamespace(
            param_names=["panel", "degree", "window"], param_specs={},
            panel_params=(), input_fields=(),
        )
        def _calculate_series(self, *args, _fn=kernel, **kwargs):
            return _fn(*args, **kwargs)
    op = Adapter()
    assert _infer_panel_params(op, op.metadata, {}) == ("panel",)


def test_private_transport_parameter_is_not_a_contract():
    from factor_engine.cleaned_operators.operator_spec import _operator_contract_signature
    class Adapter:
        def _calculate_series(self, *args, _fn=None, **kwargs):
            pass
    assert _operator_contract_signature(Adapter()) is None
