"""Multi-component score contracts, slot binding and exact backend numerics."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def frames():
    idx=pd.date_range("2025-01-01",periods=4,tz="Asia/Hong_Kong")
    return [pd.DataFrame({"A":v},index=idx) for v in ([1.,0.,-2.,np.nan],[-1.,2.,np.nan,np.nan])]
def run(backend,**kwargs):
    op=OperatorRegistry.get("fin_component_score",backend,mode="research")
    schema,_,verified=_parameter_contract(op,op.metadata.panel_params)
    assert verified
    assert "component_2" not in schema["required"]
    if backend=="polars":
        kwargs={k:pl.from_pandas(v.rename_axis("date").reset_index()) if isinstance(v,pd.DataFrame) else v
            for k,v in kwargs.items()}
    out=op.calculate(**kwargs)
    return out.to_pandas().set_index("date").rename_axis(None) if backend=="polars" else out

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_nonadjacent_optional_slots_directions_weights_and_policy(backend):
    a,b=frames()
    got=run(backend,component_2=a,component_8=b,component_directions=("up","down"),score_weights=[2.,3.])
    np.testing.assert_allclose(got.A,[5.,0.,0.,np.nan],equal_nan=True)
    full=run(backend,component_2=a,component_8=b,component_directions=("up","down"),score_weights=[2.,3.],missing_policy="require_full")
    np.testing.assert_allclose(full.A,[5.,0.,np.nan,np.nan],equal_nan=True)
    single=run(backend,component_1=a)
    np.testing.assert_allclose(single.A,[1.,0.,0.,np.nan],equal_nan=True)
    if backend=="pandas_numpy":
        np.testing.assert_array_equal(got.attrs["effective_component_count"].ravel(),[2,2,1,0])
    for weights in ([True,1.],[np.inf,1.],["1",2.],[1.]):
        with pytest.raises((TypeError,ValueError)):
            run(backend,component_1=a,component_2=b,score_weights=weights)
    with pytest.raises((TypeError,ValueError)):
        run(backend,component_1=a,component_directions=["invalid"])
    with pytest.raises((TypeError,ValueError)):
        run(backend)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_cancelling_large_contributions_and_axes(backend):
    a,_=frames();a=a.fillna(1.).abs()+1
    kwargs={f"component_{i}":a for i in range(1,5)}
    out=run(backend,**kwargs,score_weights=[1e308,1e308,-1e308,-1e308])
    np.testing.assert_allclose(out,0.)
    overflow=run(backend,**kwargs,score_weights=[1e308]*4)
    assert overflow.isna().all().all()
    with pytest.raises((TypeError,ValueError)):
        run(backend,component_1=a,component_2=a.rename(columns={"A":"B"}))

def test_native_component_path_no_pandas_conversion(monkeypatch):
    op=OperatorRegistry.get("fin_component_score","polars",mode="research")
    x=pl.DataFrame({"A":[1.,0.,-1.]})
    monkeypatch.setattr(pl.DataFrame,"to_pandas",lambda *a,**k:pytest.fail("Pandas conversion"))
    out=op.calculate(component_3=x)
    assert out["A"].to_list()==[1.,0.,0.]
    assert op.physical_spec().execution_kind.value=="polars_numpy_kernel"
