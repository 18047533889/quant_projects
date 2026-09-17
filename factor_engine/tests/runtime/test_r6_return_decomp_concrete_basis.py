"""Concrete adjusted-price basis propagation for return decompositions."""

import pytest

from factor_engine.api.columns import field
from factor_engine.cleaned_operators import load_all
from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.ir.analyzer import Analyzer, TypedInputContractError


NAMES_AND_FIELDS = (
    ("overnight_return", "open", "pre_close"),
    ("open_close_return", "open", "close"),
    ("open_to_vwap_return", "open", "vwap"),
    ("vwap_to_close_return", "vwap", "close"),
)


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()


@pytest.mark.parametrize(("name", "left", "right"), NAMES_AND_FIELDS)
def test_adjusted_daily_return_operator_carries_concrete_backward_basis(name, left, right):
    expr = CleanedCall(name, args=(field(left), field(right)))
    ir = Analyzer(production=True, market="ashare").lower(expr).ir
    assert ir.attrs["price_basis"] == "BACKWARD_ADJUSTED"
    assert {child.semantic_attrs["price_basis"] for child in ir.inputs} == {
        "BACKWARD_ADJUSTED"
    }


def test_return_operator_rejects_raw_and_backward_adjusted_inputs():
    expr = CleanedCall(
        "open_close_return",
        args=(
            field("open", table="StockDailyBar"),
            field("close", table="StockDailyBarAdj"),
        ),
        kwargs=(("price_basis", "RAW"),),
    )
    with pytest.raises(TypedInputContractError, match="price basis"):
        Analyzer(production=True, market="ashare").lower(expr)


def test_return_operator_rejects_forged_basis_over_consistent_typed_children():
    expr = CleanedCall(
        "open_close_return",
        args=(field("open"), field("close")),
        kwargs=(("price_basis", "RAW"),),
    )
    with pytest.raises(
        TypedInputContractError,
        match="explicit price basis.*conflicts with typed price inputs",
    ):
        Analyzer(production=True, market="ashare").lower(expr)
