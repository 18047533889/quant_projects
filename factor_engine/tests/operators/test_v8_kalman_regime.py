"""Independent numerical acceptance for V8 M01-M04."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model import complexity, state_space


def panel(x):
    return pd.DataFrame({"A": np.asarray(x, dtype=float)})


def stable_vars(x):
    n = 0; mean = np.longdouble(0); m2 = np.longdouble(0); out = []
    for raw in x:
        if np.isfinite(raw):
            n += 1; value = np.longdouble(raw); delta = value - mean
            mean += delta / n; m2 += delta * (value - mean)
        out.append(float(m2 / n) if n >= 2 and m2 > 0 else None)
    return out


def level_ref(x, q=.01, r=1.):
    out = np.full(len(x), np.nan); mu = np.nan; p = np.nan
    for i, (value, var) in enumerate(zip(x, stable_vars(x))):
        if var is None: continue
        qe, re = q*var, r*var
        if not np.isfinite(value):
            if np.isfinite(mu): p += qe; out[i] = mu
            continue
        if not np.isfinite(mu): mu, p, out[i] = value, re, value; continue
        pp = p + qe; gain = pp / (pp + re); mu += gain*(value-mu); p=(1-gain)*pp; out[i]=mu
    return out


@pytest.mark.parametrize("offset", [0., 1e6, 1e8, 1e10])
def test_level_stable_translation_and_independent_reference(offset):
    x = np.array([0,1,2,1,3,np.nan,2,5,1,0,4.]) + offset
    got = state_space._kalman_level(x,.01,1.,"level","dimensionless")
    np.testing.assert_allclose(got, level_ref(x), rtol=2e-7, atol=2e-6, equal_nan=True)


@pytest.mark.parametrize("scale", [.01,100.])
def test_trend_dimensionless_units(scale):
    x=np.array([0,1,2,1,3,np.nan,2,5,1,0,4.])
    a=state_space._kalman_trend_slope(x,.01,.001,1.,"dimensionless")
    b=state_space._kalman_trend_slope(x*scale,.01,.001,1.,"dimensionless")
    np.testing.assert_allclose(b/scale,a,rtol=3e-10,atol=3e-10,equal_nan=True)


def test_trend_absolute_preserves_legacy_fixed_parameter_recursion():
    x=np.array([0.,1.,2.,np.nan,1.]); ql=.01; qt=.001; r=.5
    expected=[]; level=np.nan; trend=0.; p11=p12=p22=1.
    for value in x:
        if not np.isfinite(value):
            level += trend; p11,p12,p22=p11+ql+2*p12+p22,p12+p22,p22+qt
            expected.append(trend); continue
        if not np.isfinite(level): level=value; expected.append(0.); continue
        lp=level+trend; p11p=p11+ql+2*p12+p22; p12p=p12+p22; p22p=p22+qt
        k1=p11p/(p11p+r); k2=p12p/(p11p+r); innovation=value-lp
        level=lp+k1*innovation; trend += k2*innovation
        p11=(1-k1)*p11p; p12=(1-k1)*p12p; p22=p22p-k2*p12p
        expected.append(trend)
    got=state_space._kalman_trend_slope(x,ql,qt,r,"absolute")
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-14,equal_nan=True)


@pytest.mark.parametrize("xs,ys,bs", [(100.,100.,1.),(100.,1.,.01),(1.,100.,100.),(1e-8,1.,1e8)])
def test_beta_dimensionless_units(xs,ys,bs):
    rng=np.random.default_rng(487); x=rng.normal(0,.02,80); y=2.5*x+rng.normal(0,.01,80)
    x[[8,29]]=np.nan; y[[15,29]]=np.nan
    a=state_space._kalman_beta(y,x,.01,1.,"beta","dimensionless",5)
    b=state_space._kalman_beta(y*ys,x*xs,.01,1.,"beta","dimensionless",5)
    np.testing.assert_allclose(b/bs,a,rtol=3e-10,atol=3e-10,equal_nan=True)
    assert np.flatnonzero(np.isfinite(a))[0] == 4


def test_beta_uncertainty_is_beta_squared_not_standard_error():
    rng=np.random.default_rng(2); x=rng.normal(0,.02,50); y=1.7*x+rng.normal(0,.01,50)
    a=state_space._kalman_beta(y,x,.01,1.,"uncertainty","dimensionless",5)
    b=state_space._kalman_beta(y,100*x,.01,1.,"uncertainty","dimensionless",5)
    np.testing.assert_allclose(b*100**2,a,rtol=3e-10,atol=3e-10,equal_nan=True)
    tiny=state_space._kalman_beta(y,1e-8*x,.01,1.,"uncertainty","dimensionless",5)
    np.testing.assert_allclose(tiny*(1e-8)**2,a,rtol=3e-10,atol=3e-10,equal_nan=True)


def regime_ref(x,switch=.05):
    x=np.asarray(x,dtype=np.longdouble); var=np.var(x)
    hi=np.sqrt(max(2*var,np.longdouble("1e-12"))); lo=np.sqrt(max(var/2,np.longdouble("1e-12")))
    p=np.longdouble(.5); switch=np.longdouble(switch)
    for value in x:
        pred=switch*(1-p)+(1-switch)*p
        lh=np.log(pred)-np.log(hi)-np.longdouble(.5)*(value/hi)**2
        ll=np.log(1-pred)-np.log(lo)-np.longdouble(.5)*(value/lo)**2
        p=np.exp(lh-np.logaddexp(lh,ll))
    return min(max(float(p), 1e-6), 1-1e-6)


@pytest.mark.parametrize("spread", [1e-6,1e-5])
def test_regime_log_domain_independent_reference(spread):
    x=.01+np.linspace(-spread,spread,30)
    assert complexity._regime_filter(x,30,"prob",.05) == pytest.approx(regime_ref(x),abs=2e-12)


def test_public_winners_and_state_schema():
    x=np.array([0,1,2,1,3,2,5,1,0,4.])
    got=OperatorRegistry.get("ts_kalman_level").calculate(panel(x+1e8),q=.01,r=1.,scale_mode="dimensionless")["A"]
    assert got.notna().sum() == len(x)-1
    p=OperatorRegistry.get("ts_two_state_regime_probability").calculate(panel(.01+np.linspace(-1e-6,1e-6,30)),window=30,transition_prob=.05)["A"].iloc[-1]
    assert p == pytest.approx(.5,abs=2e-12)
    for name in state_space.KALMAN_STATEFUL_CANONICALS:
        assert state_space.kalman_stateful_contract(name)["state_schema_version"] == f"{name}.v2"
