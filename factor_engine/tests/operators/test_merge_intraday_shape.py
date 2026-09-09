import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.intraday_vol_ext import (
    _time_centroid, _concentration, _entropy, _semi_balance, _rv_curvature,
    _rv_scale_cohorts, IntradayVolatilityTimeCentroid,
)


@pytest.mark.parametrize("fn", [_time_centroid, _concentration, _entropy, _semi_balance, _rv_curvature])
@pytest.mark.parametrize("scale", [1e-200, 1e-8, 1., 1e200])
def test_intraday_shape_return_unit_invariance(fn, scale):
    v = np.sin(np.arange(40.)*.53)+.1
    reference = fn(v)
    assert np.isfinite(reference)
    with np.errstate(over="raise",invalid="raise"):
        assert fn(v*scale) == pytest.approx(reference,abs=1e-12,rel=1e-12)


def test_zero_variance_is_undefined_and_analytic_balanced_mass():
    for fn in (_time_centroid, _concentration, _entropy, _semi_balance, _rv_curvature):
        assert np.isnan(fn(np.zeros(20)))
    assert _semi_balance(np.array([1e-200,-1e-200])) == 0.
    assert _concentration(np.ones(20)*1e-200) == pytest.approx(1/20)
    assert _entropy(np.ones(20)*1e-200) == pytest.approx(1.)


def test_row_coordinate_gaps_not_compressed():
    v = np.array([np.nan, 1., 0., 0., 0.])
    assert _time_centroid(v) == pytest.approx(-.5)
    v = np.ones(30)
    v[10] = np.nan
    assert np.isnan(_rv_curvature(v))


@pytest.mark.parametrize("n", [20,21,24,25])
def test_legacy_rv_cohorts_are_explicit_not_silently_redefined(n):
    assert _rv_scale_cohorts(n) == tuple((s,0,n//s*s) for s in (1,2,5,10))
    if n == 21:
        assert {stop for _,_,stop in _rv_scale_cohorts(n)} == {20,21}


def test_cross_day_output_is_declared_rolling_not_current_session():
    index = pd.date_range("2026-01-05 09:30",periods=10,freq="min").append(
        pd.date_range("2026-01-06 09:30",periods=10,freq="min"))
    values = np.r_[np.ones(10)*3, np.ones(10)]
    op = IntradayVolatilityTimeCentroid()
    out = op.calculate(pd.DataFrame(values,index=index),window=20)
    assert out.iloc[-1,0] == pytest.approx(_time_centroid(values))
    assert out.iloc[-1,0] != pytest.approx(_time_centroid(values[-10:]))
    assert op.metadata.input_grain == op.metadata.output_grain == "minute"
    assert "not_session_aggregate" in op.metadata.tags
