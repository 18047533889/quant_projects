import pytest
from factor_engine.ir.types import SemanticType, semantic_type_of


@pytest.mark.parametrize("basis", ("ADJUSTED", "FORWARD_ADJUSTED", "BACKWARD_ADJUSTED"))
def test_adjusted_basis_is_price_not_shape(basis):
    assert semantic_type_of(price_basis=basis, frequency="daily") == SemanticType.PRICE_CONTINUOUS
    assert semantic_type_of(price_basis=basis, frequency="minute") == SemanticType.PRICE_CONTINUOUS


def test_unknown_basis_not_invented():
    assert semantic_type_of(price_basis="UNVERIFIED") is None


def test_adjusted_close_compiles_typed_dollar_volume_without_erasing_basis():
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.cleaned_operators import load_all
    from factor_engine.ir.analyzer import Analyzer, TypedInputContractError
    load_all()
    parser = DSLParser(surface="compat_research")
    ir = Analyzer(production=True, market="ashare").lower(
        parser.parse("dollar_volume_zscore(close,volume,20)")
    ).ir
    assert ir.inputs[0].semantic_attrs["semantic_kind"] == "PriceContinuous"
    assert ir.inputs[0].semantic_attrs["price_basis"] == "BACKWARD_ADJUSTED"
    assert ir.inputs[1].semantic_attrs["semantic_kind"] == "NonNegativeActivity"
    with pytest.raises(TypedInputContractError):
        Analyzer(production=True, market="ashare").lower(
            parser.parse("dollar_volume_zscore(ret,volume,20)")
        )
