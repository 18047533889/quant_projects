import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

CASES = [
    ("ts_km_diffusion_gradient", dict(window=60,bins=5,lag=1,min_count=3,min_state_support=3,min_history=10)),
    ("ts_km_quasipotential_depth", dict(window=120,bins=5,lag=1,min_count=3,min_state_support=3,min_history=10)),
    ("ts_markov_mean_first_passage_time", dict(window=60,bins=3,lag=1,min_count=3,min_state_support=3,min_history=10,target="upper")),
    ("ts_active_information_storage", dict(window=60,bins=2,history_length=1,min_history=10)),
]

@pytest.mark.parametrize("name,params", CASES)
def test_exact_delegate_matches_reference_and_is_prefix_stable(name, params):
    pl=pytest.importorskip("polars"); load_all()
    rng=np.random.default_rng(211)
    x=pd.DataFrame(rng.normal(size=(220,2)).cumsum(0),columns=["A","B"])
    ref=OperatorRegistry.get(name,"pandas_numpy"); got=OperatorRegistry.get(name,"polars")
    expected=ref.calculate(x,**params).to_numpy()
    actual=got.calculate(pl.from_pandas(x),**params).to_numpy()
    assert np.isfinite(expected).sum() > 0
    assert np.isfinite(actual).sum() == np.isfinite(expected).sum()
    np.testing.assert_allclose(actual,expected,equal_nan=True,rtol=1e-12,atol=1e-12)
    prefix=got.calculate(pl.from_pandas(x.iloc[:180]),**params).to_numpy()
    np.testing.assert_allclose(prefix,actual[:180],equal_nan=True,rtol=1e-12,atol=1e-12)
    assert got.metadata.param_specs == ref.metadata.param_specs
    assert got.physical_spec().execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE

@pytest.mark.parametrize("name,params", CASES)
def test_invalid_or_insufficient_history_matches_reference(name, params):
    pl=pytest.importorskip("polars"); load_all()
    x=pd.DataFrame({"A":np.arange(16,dtype=float)})
    bad=dict(params); bad["min_history"]=100
    outcomes = []
    for backend,frame in [("pandas_numpy",x),("polars",pl.from_pandas(x))]:
        try:
            outcomes.append(("value", OperatorRegistry.get(name,backend).calculate(frame,**bad).to_numpy()))
        except (ValueError,TypeError) as exc:
            outcomes.append(("error", type(exc), str(exc)))
    assert outcomes[0][0] == outcomes[1][0]
    if outcomes[0][0] == "error":
        assert outcomes[0][1:] == outcomes[1][1:]
    else:
        np.testing.assert_allclose(outcomes[0][1], outcomes[1][1], equal_nan=True, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize("name,params", CASES)
def test_invalid_bins_raise_same_error(name, params):
    pl=pytest.importorskip("polars"); load_all()
    x=pd.DataFrame({"A":np.arange(80,dtype=float)})
    bad=dict(params); bad["bins"]=1
    errors=[]
    for backend,frame in [("pandas_numpy",x),("polars",pl.from_pandas(x))]:
        with pytest.raises((ValueError,TypeError)) as caught:
            OperatorRegistry.get(name,backend).calculate(frame,**bad)
        errors.append((type(caught.value),str(caught.value)))
    assert errors[0] == errors[1]

@pytest.mark.parametrize("name,params", CASES)
def test_nan_inf_fixture_matches_reference(name, params):
    pl=pytest.importorskip("polars"); load_all()
    rng=np.random.default_rng(223)
    values=rng.normal(size=(220,2)).cumsum(0)
    values[17,0]=np.nan; values[93,1]=np.inf; values[141,0]=-np.inf
    x=pd.DataFrame(values,columns=["A","B"])
    expected=OperatorRegistry.get(name,"pandas_numpy").calculate(x,**params).to_numpy()
    actual=OperatorRegistry.get(name,"polars").calculate(pl.from_pandas(x),**params).to_numpy()
    np.testing.assert_allclose(actual,expected,equal_nan=True,rtol=1e-12,atol=1e-12)
