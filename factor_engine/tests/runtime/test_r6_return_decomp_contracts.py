"""Supplied-price return decompositions: real axes, basis, and final backend numerics."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES={
 "overnight_return":("open_px","pre_close",False),
 "open_close_return":("open_px","close",True),
 "open_to_vwap_return":("open_px","vwap",True),
 "vwap_to_close_return":("vwap","close",True),
}
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def panels():
    idx=pd.date_range("2025-01-01",periods=4,tz="Asia/Hong_Kong")
    return [pd.DataFrame({"A":v,"B":np.array(v)*2},index=idx)
            for v in ([10.,12.,15.,20.],[11.,15.,21.,22.])]
def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index())
def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_supplied_price_oracle_contract_and_basis(name,backend):
    a,b=panels(); p,q,reverse=NAMES[name]
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None
    assert _parameter_contract(op,(p,q))[2]
    inputs=[convert(v) if backend=="polars" else v for v in (a,b)]
    expected=(b/a if reverse else a/b)-1
    got=result(op.calculate(**{p:inputs[0],q:inputs[1],"price_basis":"raw"}))
    pd.testing.assert_frame_equal(got,expected,check_freq=False)
    pd.testing.assert_frame_equal(result(op.calculate(*inputs,"RAW")),expected,check_freq=False)
    short=[v.head(3) for v in inputs]
    pd.testing.assert_frame_equal(result(op.calculate(*short,price_basis="RAW")),expected.head(3),check_freq=False)
    for basis in (None,"RETURN","EITHER","unknown",True):
        with pytest.raises((ValueError,TypeError)):
            op.calculate(*inputs,price_basis=basis)
    shifted=b.copy();shifted.index=shifted.index+pd.Timedelta(days=1)
    with pytest.raises((ValueError,TypeError)):
        op.calculate(inputs[0],convert(shifted) if backend=="polars" else shifted,price_basis="RAW")
    bad=a.copy();bad.iloc[:,0]=[0.,-1.,np.inf,np.nan]
    out=result(op.calculate(convert(bad) if backend=="polars" else bad,inputs[1],price_basis="RAW"))
    assert out.A.isna().all()
    assert np.isfinite(out.B).all()
    for scale in (1e-200,1e200):
        scaled=[convert(v*scale) if backend=="polars" else v*scale for v in (a,b)]
        np.testing.assert_allclose(result(op.calculate(*scaled,price_basis="RAW")),expected,rtol=1e-13,atol=1e-14)

@pytest.mark.parametrize("name",NAMES)
def test_native_does_not_convert_pandas(name,monkeypatch):
    a,b=[convert(v) for v in panels()]
    op=OperatorRegistry.get(name,"polars",mode="research")
    def forbidden(*a,**kw):
        raise AssertionError("native return must not convert to a Pandas panel")
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    monkeypatch.setattr(pd.DataFrame,"__init__",forbidden)
    out=op.calculate(a,b,price_basis="RAW")
    assert out.shape==a.shape and out["date"].equals(a["date"])

def test_common_ohlc_direct_owner_uses_supplied_preclose_without_shift():
    # Execute the exact authored class without re-registering into the frozen Registry.
    import ast
    from pathlib import Path
    import factor_engine
    from factor_engine.cleaned_operators.base_polars import SeriesOperator
    path=Path(factor_engine.__file__).parent/"cleaned_operators/common/polars_ohlc_basic.py"
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="OvernightReturnNative")
    cls.decorator_list=[]
    namespace={"SeriesOperator":SeriesOperator}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),"exec"),namespace)
    a,b=panels()
    got=result(namespace["OvernightReturnNative"]().calculate(convert(a),convert(b),price_basis="RAW"))
    pd.testing.assert_frame_equal(got,a/b-1,check_freq=False)
