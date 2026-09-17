import pytest
from factor_engine.api.intraday_daily import intraday_realized_vol, intraday_first_nmin_return
from factor_engine.api.source_ref import decode_source_ref


@pytest.mark.parametrize(('name', 'value'), [
    ('bar_minutes', 1.5), ('bar_minutes', True),
    ('min_bars', 2.5), ('history_days', 0.5), ('history_days', True),
    ('minutes', 1.5), ('minutes', True),
    ('bar_minutes', float('inf')), ('min_bars', float('nan')),
])
def test_source_controls_reject_non_integer_values_before_reference_creation(name, value):
    with pytest.raises(ValueError, match=name):
        intraday_first_nmin_return(**{name: value})


@pytest.mark.parametrize('value', [5, 5.0, '5'])
def test_integral_source_controls_preserve_existing_supported_values(value):
    expr = intraday_first_nmin_return(bar_minutes=value, min_bars=value, history_days=value, minutes=value)
    spec = decode_source_ref(expr.name)
    params = spec.transform_params_dict()
    assert {k: params[k] for k in ('bar_minutes', 'min_bars', 'history_days', 'minutes')} == {
        'bar_minutes': 5, 'min_bars': 5, 'history_days': 5, 'minutes': 5,
    }
    assert all(type(params[k]) is int for k in ('bar_minutes', 'min_bars', 'history_days', 'minutes'))


def test_default_intraday_source_controls_are_unchanged():
    params = decode_source_ref(intraday_realized_vol().name).transform_params_dict()
    assert params['bar_minutes'] == 5
    assert params['min_bars'] == 2
    assert params['history_days'] == 0
