from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from factor_engine.api.columns import col, field
from factor_engine.backend.operator_types import OPERATOR_SIGNATURES
from factor_engine.cleaned_operators import spectral_ext
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.ir.analyzer import Analyzer, TypedInputContractError, UnknownRawColumnError


@pytest.fixture(scope="session", autouse=True)
def strict_fiscal_parameter_domain_certification_guard():
    return None


def _call(canonical, leaf, **kwargs):
    return CleanedCall(canonical, (leaf,), tuple(kwargs.items()))


def test_real_catalog_production_return_and_adjusted_level_paths():
    analyzer = Analyzer(production=True, market="ashare")
    ret = analyzer.lower(_call("ts_return_spectral_entropy", field("ret"), window=20))
    assert ret.ir.inputs[0].attrs["field_id"] == "StockDailyBarAdj.ret"
    assert ret.ir.attrs["input_kind"] == "ReturnDecimal"

    level = analyzer.lower(
        _call("ts_detrended_level_spectral_entropy", field("close"), window=20)
    )
    assert level.ir.inputs[0].attrs["field_id"] == "StockDailyBarAdj.close"
    assert level.ir.inputs[0].semantic_attrs["price_basis"] == "ADJUSTED"
    assert level.ir.inputs[0].semantic_attrs["semantic_kind"] == "PriceContinuous"
    assert level.ir.attrs["input_kind"] == "PriceContinuous"


def test_production_unknown_and_mismatch_fail_closed_despite_caller_stamp():
    analyzer = Analyzer(production=True, market="ashare")
    with pytest.raises(UnknownRawColumnError):
        analyzer.lower(
            _call("ts_return_spectral_entropy", col("not_cataloged"),
                  window=20, input_kind="ReturnDecimal")
        )
    with pytest.raises(TypedInputContractError):
        analyzer.lower(
            _call("ts_return_spectral_entropy", field("close"),
                  window=20, input_kind="ReturnDecimal")
        )
    with pytest.raises(TypedInputContractError):
        analyzer.lower(
            _call("ts_detrended_level_spectral_entropy", field("ret"),
                  window=20, input_kind="PriceContinuous")
        )


def test_explicit_return_kind_is_not_overwritten_by_inherited_adjusted_basis(monkeypatch):
    analyzer = Analyzer(production=True, market="ashare")
    ret_ref = field("ret")
    original = analyzer._resolve_field_ir(ret_ref)
    assert original is not None
    explicit_return = replace(
        original, semantic_kind="ReturnDecimal", price_basis="ADJUSTED"
    )
    monkeypatch.setattr(analyzer, "_resolve_field_ir", lambda node: explicit_return)
    lowered = analyzer.lower(
        _call("ts_return_spectral_entropy", ret_ref, window=20,
              input_kind="PriceContinuous")
    )
    assert lowered.ir.inputs[0].semantic_attrs["semantic_kind"] == "ReturnDecimal"
    assert lowered.ir.attrs["input_kind"] == "ReturnDecimal"
    with pytest.raises(TypedInputContractError):
        analyzer.lower(
            _call("ts_detrended_level_spectral_entropy", ret_ref, window=20,
                  input_kind="PriceContinuous")
        )


@pytest.mark.parametrize("alias", ["returns", "pct_change"])
def test_real_adjusted_price_return_alias_flows_to_return_spectral(alias):
    analyzer = Analyzer(production=True, market="ashare")
    derived_return = _call(alias, field("close"), d=1)
    lowered_return = analyzer.lower(derived_return)
    assert lowered_return.ir.op == "ts_pct"
    assert lowered_return.ir.semantic_attrs["semantic_kind"] == "ReturnDecimal"
    assert lowered_return.ir.semantic_attrs["price_basis"] == "RETURN"
    spectral = analyzer.lower(
        _call("ts_return_spectral_entropy", derived_return, window=20)
    )
    assert spectral.ir.inputs[0].semantic_attrs["semantic_kind"] == "ReturnDecimal"
    assert spectral.ir.attrs["input_kind"] == "ReturnDecimal"


def test_return_composite_preserves_kind_but_generic_pct_change_is_not_promoted():
    analyzer = Analyzer(production=True, market="ashare")
    derived_return = _call("returns", field("close"), d=1)
    smoothed = _call("ts_mean", derived_return, window=5)
    spectral = analyzer.lower(
        _call("ts_return_spectral_entropy", smoothed, window=20)
    )
    assert spectral.ir.inputs[0].semantic_attrs["semantic_kind"] == "ReturnDecimal"
    assert spectral.ir.attrs["input_kind"] == "ReturnDecimal"

    research = Analyzer().lower(_call("ts_pct", col("generic_signal"), d=1))
    assert research.ir.semantic_attrs.get("semantic_kind") != "ReturnDecimal"


def test_research_raw_column_requires_explicit_assumption():
    analyzer = Analyzer()
    lowered = analyzer.lower(
        _call("ts_return_spectral_entropy", col("research_signal"),
              window=20, input_kind="ReturnDecimal")
    )
    assert lowered.ir.attrs["input_kind"] == "ReturnDecimal"
    with pytest.raises(TypedInputContractError):
        analyzer.lower(_call("ts_return_spectral_entropy", col("research_signal"), window=20))


def test_spectral_signatures_match_registered_runtime_parameters():
    signatures = OPERATOR_SIGNATURES
    for canonical in (
        "ts_return_spectral_entropy",
        "ts_spectral_entropy",
        "ts_detrended_level_spectral_entropy",
    ):
        assert [arg.name for arg in signatures[canonical].inputs] == [
            "x", "window", "input_kind"
        ]
        assert signatures[canonical].inputs[-1].required is False
        for backend in ("pandas_numpy", "polars"):
            op = OperatorRegistry.get(canonical, backend, mode="any")
            assert op is not None
            assert op.metadata.param_names == ["x", "window", "input_kind"]


def test_polars_bridge_propagates_input_kind_and_matches_pandas():
    values = np.sin(np.arange(80.0) / 4.0) * 0.01
    pdf = pd.DataFrame({"A": values})
    pldf = pl.DataFrame({"date": pd.date_range("2024-01-01", periods=80), "A": values})
    pandas_op = OperatorRegistry.get("ts_return_spectral_entropy", "pandas_numpy", mode="any")
    polars_op = OperatorRegistry.get("ts_return_spectral_entropy", "polars", mode="any")
    expected = pandas_op.calculate(pdf, window=20, input_kind="ReturnDecimal")
    actual = polars_op.calculate(pldf, window=20, input_kind="ReturnDecimal")
    np.testing.assert_allclose(actual["A"].to_numpy(), expected["A"].to_numpy(), equal_nan=True)
    with pytest.raises(ValueError, match="not an allowed choice|requires input_kind"):
        polars_op.calculate(pldf, window=20, input_kind="PriceContinuous")
