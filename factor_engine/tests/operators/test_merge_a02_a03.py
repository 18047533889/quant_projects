import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.weighted_tail import (
    TsWeightedSemivariance, TsWeightedDownsideDeviation,
    TsWeightedExpectedShortfall, TsWeightedDrawdownArea, _weighted_es_tail,
)


@pytest.mark.parametrize("cls", [TsWeightedSemivariance, TsWeightedDownsideDeviation,
                                   TsWeightedExpectedShortfall, TsWeightedDrawdownArea])
@pytest.mark.parametrize("weight_scale", [1e-14, 1e-200, 1e100, 1e300])
def test_four_public_weighted_risk_entries_unit_invariance(cls, weight_scale):
    x = pd.DataFrame(np.linspace(-.1, .1, 20))
    if cls is TsWeightedDrawdownArea:
        x = pd.DataFrame(np.linspace(20., 1., 20))
    weights = pd.DataFrame(np.ones(20))
    kwargs = dict(window=20)
    if cls is TsWeightedExpectedShortfall:
        kwargs.update(quantile=.25, min_tail_count=3)
    reference = cls().calculate(x, weights, **kwargs)
    actual = cls().calculate(x, weights*weight_scale, **kwargs)
    assert np.isfinite(reference.iloc[-1, 0])
    np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=0., equal_nan=True)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_zero_mass_cannot_invent_tail_members(side):
    values = np.array([-2., -1., 1., 2.])
    weights = np.ones(4)
    assert np.isnan(_weighted_es_tail(values, weights, .5, side, 3))
    padded = np.r_[np.full(100, -100. if side == "lower" else 100.), values]
    assert np.isnan(_weighted_es_tail(padded, np.r_[np.zeros(100), weights], .5, side, 3))


def test_selected_tail_kish_not_whole_window_gate():
    values = np.arange(-3., 3.)
    weights = np.array([8., 1., 1., 10., 10., 10.])
    assert np.isnan(_weighted_es_tail(values, weights, .5, "lower", 3))


@pytest.mark.parametrize("side,expected", [("lower", -2.), ("upper", 2.)])
def test_fractional_ties_and_weight_scale(side, expected):
    values = np.array([-2.]*5 + [2.]*5)
    for scale in (1., 1e-200, 1e200):
        assert _weighted_es_tail(values, np.ones(10)*scale, .4, side, 3) == pytest.approx(expected)


def test_all_zero_mass_is_undefined():
    assert np.isnan(_weighted_es_tail(np.arange(10.), np.zeros(10), .5, "lower", 3))
