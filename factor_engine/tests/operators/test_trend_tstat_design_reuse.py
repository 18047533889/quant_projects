import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.overhaul.regression import _fit_1d, pd_trend_tstat


@pytest.mark.parametrize("window", [3, 12, 80])
def test_trend_design_reuse_matches_original_fit(window):
    rng=np.random.default_rng(6)
    a=rng.normal(size=(60,4))
    a[:,1]=np.arange(60)*.3+7
    a[:,2]=1.
    a[::7,3]=np.nan
    a[5,3]=np.inf
    expected=np.full(a.shape,np.nan)
    for col in range(a.shape[1]):
        for row in range(len(a)):
            v=a[max(0,row-window+1):row+1,col]
            if np.isfinite(v).sum()<3:
                continue
            fit=_fit_1d(v,np.arange(len(v),dtype=float),True)
            if fit is not None:expected[row,col]=fit[4]
    result=pd_trend_tstat(pd.DataFrame(a),window,min_periods=3)
    np.testing.assert_array_equal(result.to_numpy(),expected)


def test_fixed_geometry_is_computed_once(monkeypatch):
    calls=[]
    original=np.linalg.pinv
    def observed(x,*a,**kw):
        calls.append(x.shape)
        return original(x,*a,**kw)
    monkeypatch.setattr(np.linalg,'pinv',observed)
    pd_trend_tstat(pd.DataFrame(np.random.default_rng(9).normal(size=(80,2))),12,min_periods=12)
    assert calls==[(2,2)]
