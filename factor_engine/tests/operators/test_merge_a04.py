import math
import numpy as np
import pytest
from factor_engine.cleaned_operators.jump_robust import _jump_z, _rolling_returns


@pytest.mark.parametrize("scale", [1., 1e-5, 1e-100, 1e100, -1., 1e-300, 1e300])
def test_declared_jump_stat_dimensionless_reference(scale):
    v = np.array([1., -1., 1., -1., 10., 1., -1., 1., -1., 1.])
    # Independent scalar reference in ordinary units.
    rv = math.fsum(x*x for x in v)
    bv = math.pi / 2 * math.fsum(abs(a*b) for a,b in zip(v[1:], v[:-1]))
    quarticity = math.fsum(x**4 for x in v)
    reference = (rv-bv)/math.sqrt(((math.pi/2)**2 + math.pi-5)/3 * quarticity)
    assert reference == pytest.approx(1.4772639171387154)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert _jump_z(v*scale) == pytest.approx(reference, rel=2e-14)


def test_zero_variation_is_undefined_not_zero_jump():
    assert np.isnan(_jump_z(np.zeros(10)))


def test_signed_finite_sample_result_not_clipped():
    assert _jump_z(np.ones(20)) < 0


def test_nonfinite_window_is_undefined():
    for value in (np.nan, np.inf, -np.inf):
        v = np.ones(10)
        v[5] = value
        assert np.isnan(_jump_z(v))


def test_interior_missing_does_not_reconnect_bars():
    v = np.ones((20, 1))
    v[15, 0] = np.nan
    assert np.isnan(_rolling_returns(v, 20, _jump_z)[-1, 0])
