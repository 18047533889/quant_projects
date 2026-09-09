import numpy as np
import pytest
from scipy.spatial.distance import cdist
from factor_engine.cleaned_operators.research_spectral import _rbf, _EPS


@pytest.mark.parametrize("offset", [0.,1e8,1e12])
def test_large_common_offset_preserves_geometry_and_psd(offset):
    a=(offset+np.arange(8.))[:,None]
    actual=_rbf(a,a,1.)
    reference=np.exp(-cdist(a,a,"sqeuclidean")/(2.+_EPS))
    np.testing.assert_allclose(actual,reference,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(actual,actual.T,rtol=0,atol=0)
    np.testing.assert_array_equal(np.diag(actual),np.ones(8))
    assert np.linalg.eigvalsh(actual).min() >= -1e-12


def test_cross_gram_uses_joint_geometry_not_separate_centers():
    a=np.array([[1e8,1e8],[1e8+2,1e8+3]])
    b=np.array([[1e8+11,1e8-2],[1e8+1,1e8+4],[1e8,1e8]])
    both=np.vstack([a,b])
    np.testing.assert_allclose(_rbf(a,b,2.),_rbf(both,both,2.)[:2,2:],rtol=1e-14)
    np.testing.assert_allclose(_rbf(a,b,2.),np.exp(-cdist(a,b,"sqeuclidean")/(8.+_EPS)),rtol=1e-14)


def test_duplicate_and_large_finite_coordinates_remain_defined():
    a=np.array([[1e100],[1e100],[1e100+1e90]])
    actual=_rbf(a,a,1e90)
    reference=np.exp(-cdist(a,a,"sqeuclidean")/(2e180+_EPS))
    np.testing.assert_allclose(actual,reference,rtol=1e-14)
    assert actual[0,1] == 1.


@pytest.mark.parametrize("scale", [1e200,1e308])
def test_finite_coordinates_and_bandwidth_do_not_square_to_inf_over_inf(scale):
    a=np.array([[-scale],[0.],[scale]])
    expected=np.exp(-np.square(np.array([-1.,0.,1.])[:,None]-np.array([-1.,0.,1.])[None,:])/2.)
    np.testing.assert_allclose(_rbf(a,a,scale),expected,rtol=1e-14,atol=1e-14)
