from decimal import Decimal
import numpy as np
import pytest
from factor_engine.api.intraday_daily import intraday_realized_vol
from factor_engine.api.source_ref import decode_source_ref


@pytest.mark.parametrize('value', [np.bool_(True), np.array(5), np.array([5])])
def test_source_integer_controls_require_numeric_scalar_not_boolean_or_array(value):
    with pytest.raises(ValueError, match='bar_minutes'):
        intraday_realized_vol(bar_minutes=value)


@pytest.mark.parametrize('value', [np.int64(5), np.float64(5), Decimal('5')])
def test_integral_numeric_scalar_types_are_supported(value):
    params = decode_source_ref(intraday_realized_vol(bar_minutes=value).name).transform_params_dict()
    assert params['bar_minutes'] == 5
    assert type(params['bar_minutes']) is int
