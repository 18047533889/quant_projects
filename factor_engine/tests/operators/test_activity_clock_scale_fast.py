import numpy as np
import pytest
from factor_engine.cleaned_operators.activity_clock import _activity_clock_kernel


def reference(a, sw, lookback, budget, include):
    scaled=np.full(len(a),np.nan)
    for s in range(1,len(a)):
        p=a[max(0,s-sw):s]
        p=p[np.isfinite(p)]
        if np.any(p<0) or len(p)<max(5,sw//2):continue
        scale=float(np.median(p))
        if not np.isfinite(scale) or scale<=1e-12:continue
        if np.isfinite(a[s]) and a[s]>=0:scaled[s]=a[s]/scale
    out=np.full(len(a),np.nan)
    for r in range(len(a)):
        total=0.
        for back in range(0 if include else 1,lookback+1):
            s=r-back
            if s<0 or np.isnan(scaled[s]):break
            total+=scaled[s]
            if total>=budget:
                out[r]=float(back);break
    return out


@pytest.mark.parametrize("sw",[2,5,20,100])
@pytest.mark.parametrize("include",[True,False])
def test_activity_clock_matches_prior_median_and_missing_rules(sw,include):
    a=np.random.default_rng(81).lognormal(size=250)
    a[[1,45,91]]=np.nan
    a[61]=np.inf
    a[100]=-1
    a[130:140]=0
    for budget in [.5,1.,7.25,100.]:
        np.testing.assert_array_equal(_activity_clock_kernel(a,sw,60,budget,include),
                                      reference(a,sw,60,budget,include))


def test_current_activity_does_not_enter_scale_estimate():
    a=np.ones(30);a[20]=100
    np.testing.assert_array_equal(_activity_clock_kernel(a,10,20,2,False),
                                  reference(a,10,20,2,False))
