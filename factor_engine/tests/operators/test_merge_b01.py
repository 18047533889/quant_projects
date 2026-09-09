import numpy as np
import pytest
from factor_engine.cleaned_operators.memory_ext import _geyer_ims_tau, _geyer_ips_tau, _sample_acf


@pytest.mark.parametrize("phi", [-.8, -.3, 0., .3, .8])
def test_paired_geyer_matches_ar_population(phi):
    rho = phi**np.arange(202)
    expected = (1+phi)/(1-phi)
    assert _geyer_ips_tau(rho) == pytest.approx(expected, abs=1e-13)
    assert _geyer_ims_tau(rho) == pytest.approx(expected, abs=1e-13)


def test_positive_pair_not_individual_acf_and_monotone_rule():
    # pairs [1.5, .3, .5, -.1]; the last pair is excluded.
    rho = np.array([1., .5, -.1, .4, .2, .3, -.3, .2])
    assert _geyer_ips_tau(rho) == pytest.approx(-1+2*(1.5+.3+.5))
    assert _geyer_ims_tau(rho) == pytest.approx(-1+2*(1.5+.3+.3))
    # An unpaired tail is neither padded nor counted.
    assert _geyer_ips_tau(rho[:6]) == _geyer_ips_tau(np.r_[rho[:6], 999.])


def test_negative_finite_sample_estimate_not_forced_positive():
    assert _geyer_ips_tau(np.array([1., -.9])) == pytest.approx(-.8)
    assert _geyer_ims_tau(np.array([1., -.9])) == pytest.approx(-.8)


@pytest.mark.parametrize("scale", [1., 1e-200, 1e200])
def test_acf_unit_invariance_against_scalar_n_minus_k_reference(scale):
    x = np.array([0., 1., -2., 5., 1., -1., 3., 2.])
    mu = sum(x)/len(x)
    variance = sum((v-mu)**2 for v in x)/len(x)
    expected = [1.] + [sum((x[i]-mu)*(x[i+k]-mu) for i in range(len(x)-k)) /
        (len(x)-k)/variance for k in range(1, 5)]
    np.testing.assert_allclose(_sample_acf(x*scale, 4), expected, rtol=1e-13, atol=1e-15)


def test_unidentifiable_acf_and_short_sequence():
    assert _sample_acf(np.ones(5), 3).size == 0
    assert np.isnan(_geyer_ips_tau(np.array([1.])))
    assert np.isnan(_geyer_ims_tau(np.array([1., np.nan])))

def test_acf_large_representable_offset_against_scalar_reference():
    x = np.array([0., 1., -2., 5., 1., -1., 3., 2.])
    mu = sum(x)/len(x)
    variance = sum((v-mu)**2 for v in x)/len(x)
    expected = [1.] + [sum((x[i]-mu)*(x[i+k]-mu) for i in range(len(x)-k)) /
        (len(x)-k)/variance for k in range(1, 5)]
    np.testing.assert_allclose(_sample_acf(x + 1e12, 4), expected,
                               rtol=1e-13, atol=1e-15)
