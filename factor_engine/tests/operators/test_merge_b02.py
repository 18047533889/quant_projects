import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import spectral_ext
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.ir.analyzer import Analyzer, TypedInputContractError, validate_input_type_contracts
from factor_engine.ir.nodes import IRNode
from factor_engine.api.columns import col
from factor_engine.expr.cleaned_call import CleanedCall


@pytest.mark.parametrize("canonical,kind", [
    ("ts_return_spectral_entropy", "ReturnDecimal"),
    ("ts_spectral_entropy", "ReturnDecimal"),
    ("ts_detrended_level_spectral_entropy", "PriceContinuous"),
])
def test_prefix_and_security_independence_from_numeric_shape(canonical, kind):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    prefix = pd.DataFrame({"A": np.sin(np.arange(80.)/3)})
    future = pd.DataFrame({"A": np.arange(80., 160.)**3})
    combined = pd.concat([prefix, future], ignore_index=True)
    combined["B"] = np.arange(160.)**4
    a = op.calculate(prefix, window=20, input_kind=kind)
    b = op.calculate(combined, window=20, input_kind=kind)
    pd.testing.assert_frame_equal(a, b.iloc[:80][["A"]])
    with pytest.raises(ValueError):
        op.calculate(prefix, window=20)
    wrong = "PriceContinuous" if kind == "ReturnDecimal" else "ReturnDecimal"
    with pytest.raises(ValueError):
        op.calculate(prefix, window=20, input_kind=wrong)


@pytest.mark.parametrize("kind", [None, "DailySeries", "PriceRaw", "NonNegativeActivity"])
def test_declared_type_gate_rejects_unknown_or_wrong_production_type(kind):
    node = IRNode(op="ts_return_spectral_entropy", inputs=(
        IRNode(op="column", attrs={"name": "X"}, semantic_attrs={"semantic_kind": kind}),))
    assert validate_input_type_contracts(node, strict_unknown=True)


def test_declared_type_gate_accepts_real_return_type():
    node = IRNode(op="ts_return_spectral_entropy", inputs=(
        IRNode(op="column", attrs={"name": "X"}, semantic_attrs={"semantic_kind": "ReturnDecimal"}),))
    assert not validate_input_type_contracts(node, strict_unknown=True)


def test_analyzer_forwards_immediate_child_not_caller_stamp(monkeypatch):
    # Controlled catalog-kind resolver; exercises actual lowering/stamping,
    # without pretending this synthetic field is an approved market profile.
    from factor_engine.fields import spec
    original = spec.semantic_kind_of_field
    def resolve(value):
        if isinstance(value, str) and value == "typed_test_return":
            return "ReturnDecimal"
        return original(value)
    monkeypatch.setattr(spec, "semantic_kind_of_field", resolve)
    call = CleanedCall("ts_return_spectral_entropy", (col("typed_test_return"),),
                       (("window", 20), ("input_kind", "ReturnDecimal")))
    result = Analyzer().lower(call)
    assert result.ir.attrs["input_kind"] == "ReturnDecimal"
    wrong_call = CleanedCall("ts_detrended_level_spectral_entropy", (col("typed_test_return"),),
                             (("window", 20), ("input_kind", "PriceContinuous")))
    with pytest.raises(TypedInputContractError):
        Analyzer().lower(wrong_call)
