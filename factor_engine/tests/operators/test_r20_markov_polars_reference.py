"""Prevent obsolete Markov surrogates and backend-specific parameter defaults."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.contracts import ExecutionKind

@pytest.mark.parametrize("name,params",[
    ("ts_kramers_moyal_local_stability",dict(window=40,bins=5,lag=1,min_count=3,min_state_support=2,min_history=10)),
    ("ts_markov_entropy_production",dict(window=120,bins=5,lag=1,min_periods=60)),
])
def test_exact_reference_and_prefix(name,params):
    pl=pytest.importorskip("polars")
    load_all()
    rng=np.random.default_rng(207)
    x=pd.DataFrame(rng.normal(size=(180,2)).cumsum(axis=0),columns=["A","B"])
    pandas_op=OperatorRegistry.get(name,"pandas_numpy")
    polars_op=OperatorRegistry.get(name,"polars")
    expected=pandas_op.calculate(x,**params).to_numpy()
    actual=polars_op.calculate(pl.from_pandas(x),**params).to_numpy()
    assert np.isfinite(expected).any()
    np.testing.assert_allclose(actual,expected,equal_nan=True,rtol=1e-12,atol=1e-12)
    prefix=polars_op.calculate(pl.from_pandas(x.iloc[:150]),**params).to_numpy()
    np.testing.assert_allclose(prefix,actual[:150],equal_nan=True,rtol=1e-12,atol=1e-12)
    assert polars_op.physical_spec().execution_kind==ExecutionKind.POLARS_PANDAS_DELEGATE

@pytest.mark.parametrize("name",["ts_kramers_moyal_local_stability","ts_markov_entropy_production"])
def test_invalid_history_rejected_on_both_backends(name):
    pl=pytest.importorskip("polars")
    load_all()
    x=pd.DataFrame({"A":np.arange(40,dtype=float)})
    for backend,frame in [("pandas_numpy",x),("polars",pl.from_pandas(x))]:
        with pytest.raises((ValueError,TypeError)):
            OperatorRegistry.get(name,backend).calculate(frame,window=20,min_history=30)
