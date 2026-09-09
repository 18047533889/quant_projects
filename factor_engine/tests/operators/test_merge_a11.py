import numpy as np
import pytest
from factor_engine.cleaned_operators.glr_change import _mean_shift_score, _variance_shift_score


@pytest.mark.parametrize("fn", [_mean_shift_score, _variance_shift_score])
@pytest.mark.parametrize("scale", [1e-200, 1e-7, 1., 1e100, 1e300])
def test_glr_dimensionless_without_changing_scan_policy(fn, scale):
    t = np.arange(40.)
    values = np.sin(t) * np.r_[np.ones(20), np.full(20, 2.)]
    values[20:] += 2.
    reference = fn(values, 5)
    assert np.isfinite(reference)
    with np.errstate(over="raise", invalid="raise"):
        assert fn(values*scale, 5) == pytest.approx(reference, rel=1e-11, abs=1e-12)


def test_perfect_split_cap_and_constant_behavior_preserved():
    assert _mean_shift_score(np.r_[np.zeros(20), np.ones(20)]*1e-200, 5) == pytest.approx(np.sqrt(40))
    assert np.isnan(_mean_shift_score(np.ones(40)*1e308, 5))
    assert np.isnan(_variance_shift_score(np.ones(40)*1e-200, 5))


def test_opposite_extremes_and_large_translation():
    x = np.r_[np.full(20, -1e308), np.full(20, 1e308)]
    with np.errstate(over="raise", invalid="raise"):
        assert _mean_shift_score(x, 5) == pytest.approx(np.sqrt(40))
    x = np.arange(40.) + np.sin(np.arange(40.))
    assert _mean_shift_score(x+1e9, 5) == pytest.approx(_mean_shift_score(x,5), rel=1e-7)
