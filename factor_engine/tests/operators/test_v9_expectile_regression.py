import numpy as np
import pytest
from scipy.optimize import least_squares
from factor_engine.cleaned_operators.advanced_expectile import _expectile_slope


def sample():
    rng=np.random.default_rng(903)
    x=rng.normal(size=100)
    y=2+3*x+rng.lognormal(size=100)
    return y,x


def reference(y,x,tau):
    a=np.column_stack([x,np.ones(len(x))])
    def residual(beta):
        r=y-a@beta
        return np.sqrt(np.where(r>=0,tau,1-tau))*r
    def jac(beta):
        r=y-a@beta
        return -np.sqrt(np.where(r>=0,tau,1-tau))[:,None]*a
    result=least_squares(residual,[0.,0.],jac=jac,ftol=1e-13,xtol=1e-13,gtol=1e-13)
    assert np.max(np.abs(jac(result.x).T@residual(result.x))) < 1e-8
    return result.x[0]


@pytest.mark.parametrize("tau", [.1,.5,.9])
def test_asymmetric_loss_reference_and_separate_large_translations(tau):
    y,x=sample()
    expected=reference(y,x,tau)
    assert _expectile_slope(y,x,tau)==pytest.approx(expected,abs=1e-8)
    assert _expectile_slope(y+1e8,x,tau)==pytest.approx(expected,abs=1e-7)
    assert _expectile_slope(y,x+1e8,tau)==pytest.approx(expected,abs=1e-7)


@pytest.mark.parametrize("scale", [1e-100,1e100])
def test_predictor_units_preserve_slope_units(scale):
    y,x=sample()
    expected=reference(y,x,.1)
    assert _expectile_slope(y,x*scale,.1)*scale==pytest.approx(expected,abs=1e-8)


def test_identifiability_and_constant_response_are_distinct():
    y,x=sample()
    assert np.isnan(_expectile_slope(y,np.ones(len(x)),.1))
    assert _expectile_slope(np.full(len(x),1e200),x,.1)==0.
    mask=x.copy(); mask[-1]=np.inf
    assert _expectile_slope(y,mask,.1)==pytest.approx(_expectile_slope(y[:-1],x[:-1],.1),abs=1e-10)


def test_registered_polars_preference_matches_independent_asymmetric_loss():
    pl=pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    load_all()
    y,x=sample()
    op=OperatorRegistry.get("ts_expectile_beta",backend="polars",mode="research")
    out=op.calculate(pl.DataFrame({'A':y+1e8}),pl.DataFrame({'A':x}),window=100,tau=.1)
    assert out['A'][-1] == pytest.approx(reference(y,x,.1),abs=1e-7)
