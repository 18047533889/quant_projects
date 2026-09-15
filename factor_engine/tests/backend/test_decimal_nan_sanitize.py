from decimal import Decimal
import polars as pl
import pytest
from factor_engine.backend.polars_expr_emitter import _sanitize_nan_for_compute


@pytest.mark.parametrize("drop_inf",[True,False])
def test_decimal_sanitize_preserves_exact_values_and_nulls(drop_inf):
    source=pl.DataFrame({"_v":pl.Series(
        [Decimal("9007199254740992.1"),Decimal("9007199254740992.2"),None],
        dtype=pl.Decimal(38,1))})
    actual=_sanitize_nan_for_compute(source.lazy(),drop_inf=drop_inf).collect()
    assert actual.equals(source)
    assert actual.schema==source.schema


def test_float_sanitize_still_handles_nan_and_infinity():
    source=pl.DataFrame({"_v":[1.,float("nan"),float("inf"),None]}).lazy()
    assert _sanitize_nan_for_compute(source).collect()["_v"].to_list()==[1.,None,float("inf"),None]
    assert _sanitize_nan_for_compute(source,drop_inf=True).collect()["_v"].to_list()==[1.,None,None,None]


@pytest.mark.parametrize("dtype",[pl.Int64,pl.Boolean,pl.Null])
def test_non_ieee_types_preserved(dtype):
    source=pl.DataFrame({"_v":pl.Series([None],dtype=dtype)})
    assert _sanitize_nan_for_compute(source.lazy(),drop_inf=True).collect().equals(source)
