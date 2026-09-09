import numpy as np
import pytest
from factor_engine.cleaned_operators.rough_vol import _coverages_balanced, _pv_roughness, _scaling_break
from factor_engine.cleaned_operators.distribution_break import _energy_distance


def test_coverage_ceiling_is_independent_of_per_scale_floor():
    assert not _coverages_balanced([.1, .8], 4., .1)
    assert _coverages_balanced([.2, .8], 4., .1)
    assert not _coverages_balanced([.199, .8], 4., .1)


@pytest.mark.parametrize("scale", [1e-200, 1e-7, 1., 1e200])
def test_roughness_common_window_scale_preserves_outputs(scale):
    x = np.sin(np.arange(90.)/4)+np.arange(90.)/100
    a = _pv_roughness(x, 2., (1,2,4,8,16), 8, .5, 4.)
    b = _scaling_break(x, 2., 8, .5, 4.)
    assert np.isfinite(a) and np.isfinite(b)
    assert _pv_roughness(x*scale, 2., (1,2,4,8,16), 8, .5, 4.) == pytest.approx(a, rel=1e-12)
    assert _scaling_break(x*scale, 2., 8, .5, 4.) == pytest.approx(b, rel=1e-12)


def test_linear_path_analytic_hurst_and_no_scaling_break():
    x = np.arange(100.)*1e-200
    assert _pv_roughness(x, 2., (1,2,4,8,16), 8, .5, 4.) == pytest.approx(1.)
    assert _scaling_break(x, 2., 8, .5, 4.) == pytest.approx(0., abs=1e-12)


def test_clipped_energy_u_policy_keeps_negative_raw_statistic_at_zero():
    x = np.array([[0.], [1.]])
    # Cross average=.5, within-sample U averages=1 each: raw U=-1.
    assert _energy_distance(x, x) == 0.
    assert "NOT an unbiased" in _energy_distance.__doc__
