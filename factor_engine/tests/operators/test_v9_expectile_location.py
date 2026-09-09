import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from factor_engine.cleaned_operators.advanced_expectile import _expectile, TsExpectile


def oracle(v,tau):
    def score(e):
        residual=v-e
        return np.dot(np.where(residual>=0,tau,1-tau),residual)
    return brentq(score,float(v.min()),float(v.max()),xtol=1e-13)


@pytest.mark.parametrize("tau", [.1,.5,.9])
def test_translation_invariance_and_independent_monotone_score_root(tau):
    v=np.random.default_rng(902).lognormal(size=100)
    expected=oracle(v,tau)
    assert _expectile(v,tau) == pytest.approx(expected,abs=1e-10)
    shifted=_expectile(v+1e8,tau)-1e8
    assert shifted == pytest.approx(expected,abs=3e-8)


@pytest.mark.parametrize("scale", [1e-200,1e200])
def test_response_units_do_not_change_normalized_solution(scale):
    v=np.arange(100.)-50
    assert _expectile(v*scale,.1)/scale == pytest.approx(oracle(v,.1),abs=1e-10)


def test_nonfinite_and_constant_support_policies_preserved():
    assert _expectile(np.r_[np.full(60,3e307),np.nan,np.inf],.1) == pytest.approx(3e307)
    assert np.isnan(_expectile(np.ones(5),.1))


def test_public_prefix_uses_only_current_training_window():
    v=pd.DataFrame({'A':np.random.default_rng(902).lognormal(size=100)+1e8})
    op=TsExpectile()
    full=op.calculate(v,window=60,tau=.1)
    prefix=op.calculate(v.iloc[:80],window=60,tau=.1)
    pd.testing.assert_frame_equal(full.iloc[:80],prefix)
    assert full.iloc[-1,0]-1e8 == pytest.approx(oracle(v.A.iloc[-60:].to_numpy()-1e8,.1),abs=3e-8)


def test_registered_polars_preference_preserves_current_expectile_math():
    pl=pytest.importorskip("polars")
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    load_all()
    v=np.random.default_rng(902).lognormal(size=100)+1e8
    op=OperatorRegistry.get("ts_expectile",backend="polars",mode="research")
    out=op.calculate(pl.DataFrame({'A':v}),window=60,tau=.1)
    assert out['A'][-1]-1e8 == pytest.approx(oracle(v[-60:]-1e8,.1),abs=3e-8)
