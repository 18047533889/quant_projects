"""Robust rolling statistics: independent oracles, support, finite scales and native parity."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES=("ts_quantile_range","ts_trimmed_mean","ts_robust_zscore_inclusive","ts_robust_zscore_prior")
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def panel():
    x=np.array([1.,3.,2.,6.,4.,8.,5.,10.,9.,7.,13.,11.,12.,20.,15.])
    return pd.DataFrame({"A":x,"B":x[::-1]+1},index=pd.date_range("2025-01-01",periods=len(x)))
def convert(x):
    return pl.from_pandas(x.rename_axis("date").reset_index())
def result(x):
    return x.to_pandas().set_index("date").rename_axis(None) if isinstance(x,pl.DataFrame) else x
def run(name,backend,x,**kw):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None and _parameter_contract(op,("x",))[2]
    return result(op.calculate(x=convert(x) if backend=="polars" else x,**kw))

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_robust_oracle_defaults_prefix_and_scale(name,backend):
    x=panel(); w=8
    got=run(name,backend,x,window=w)
    a=x.A.to_numpy()[-w-1:-1] if name.endswith("_prior") else x.A.to_numpy()[-w:]
    if name=="ts_quantile_range":
        expected=np.quantile(a,.75)-np.quantile(a,.25)
    elif name=="ts_trimmed_mean":
        expected=np.mean(np.sort(a)[int(.1*w):w-int(.1*w)])
    else:
        expected=(x.A.iloc[-1]-np.median(a))/(1.4826*np.median(np.abs(a-np.median(a))))
    assert got.A.iloc[-1]==pytest.approx(expected,rel=1e-12,abs=1e-12)
    np.testing.assert_allclose(run(name,backend,x.iloc[:11],window=w),got.iloc[:11],equal_nan=True)
    np.testing.assert_allclose(run(name,backend,x),run(name,"pandas_numpy",x),equal_nan=True,atol=1e-12)
    for scale in (1e-200,1e200):
        scaled=run(name,backend,x*scale,window=w)
        if name in NAMES[:2]:
            scaled=scaled/scale
        np.testing.assert_allclose(scaled,got,rtol=1e-12,atol=1e-12,equal_nan=True)
    for invalid in (True,2.5,0,np.nan):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,x,window=invalid)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_prior_support_and_clip_policy(backend):
    x=panel()
    prior=run("ts_robust_zscore_prior",backend,x,window=5,min_periods=5)
    assert prior.iloc[:5].isna().all().all()
    assert prior.iloc[5:].notna().all().all()
    z=x.copy();z.iloc[-1]=1e300
    clipped=run("ts_robust_zscore_prior",backend,z,window=5,min_periods=5,clip=3.)
    np.testing.assert_allclose(clipped.iloc[-1],3.)
    z.iloc[-1]=np.nan
    assert run("ts_robust_zscore_prior",backend,z,window=5,min_periods=5).iloc[-1].isna().all()
    for name in NAMES[2:]:
        for bad in (-1.,0.,np.inf,True):
            with pytest.raises((ValueError,TypeError)):
                run(name,backend,x,window=8,clip=bad)
        for center,scale in (("mean","std"),("median","mad"),("mean","mad"),("MEDIAN","STD")):
            got=run(name,backend,x,window=8,center=center,scale=scale)
            expected=run(name,"pandas_numpy",x,window=8,center=center,scale=scale)
            np.testing.assert_allclose(got,expected,atol=1e-12,equal_nan=True)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_finite_huge_location_and_unknown_support(backend):
    x=panel()*0+1e308
    np.testing.assert_allclose(run("ts_trimmed_mean",backend,x,window=8).iloc[4:],1e308)
    np.testing.assert_allclose(run("ts_quantile_range",backend,x,window=8).iloc[4:],0.)
    x.iloc[-3:]=np.nan
    assert run("ts_quantile_range",backend,x,window=5,min_periods=3).iloc[-1].isna().all()
    for name in NAMES:
        if name!="ts_robust_zscore_inclusive":
            with pytest.raises((ValueError,TypeError)):
                run(name,backend,x,window=8,min_periods=2.5)

@pytest.mark.parametrize("name",NAMES)
def test_native_forbids_pandas_panel_conversion(name,monkeypatch):
    x=convert(panel())
    op=OperatorRegistry.get(name,"polars",mode="research")
    def forbidden(*a,**kw):
        raise AssertionError("no Pandas panel in native robust kernel")
    monkeypatch.setattr(pd.DataFrame,"__init__",forbidden)
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    out=op.calculate(x,window=8)
    assert out.shape==x.shape and out["date"].equals(x["date"])

@pytest.mark.parametrize("center",["median","mean"])
@pytest.mark.parametrize("scale_factor",[1e-200,1.,1e200])
def test_actual_duckdb_robust_std_scale(center,scale_factor):
    import duckdb
    from factor_engine.backend.sql_pushdown.emitter import SqlDialect,compile_plan_to_sql
    from factor_engine.planner.logical_plan import PlanNode
    x=panel()*scale_factor
    x.iloc[6,0]=np.nan
    attrs=dict(window=8,center=center,scale="std",clip=2.5)
    node=PlanNode(op="ts_robust_zscore_inclusive",inputs=[PlanNode(op="column",attrs={"name":"x"})],attrs=attrs)
    compiled=compile_plan_to_sql(node,table="observations",time_column="t",instrument_column="i",dialect=SqlDialect.DUCKDB)
    assert compiled is not None
    def execute(frame):
        long=frame.rename_axis("t").reset_index().melt(id_vars="t",var_name="i",value_name="x")
        con=duckdb.connect(":memory:")
        try:
            con.register("observations",long)
            return con.execute(compiled.query).fetchdf().sort_values(["ts","inst"])["value"].to_numpy()
        finally:
            con.close()
    expected=run("ts_robust_zscore_inclusive","pandas_numpy",x,**attrs).to_numpy().ravel()
    got=execute(x)
    np.testing.assert_allclose(got,expected,atol=1e-12,rtol=1e-12,equal_nan=True)
    np.testing.assert_allclose(execute(x.iloc[:11]),got[:22],atol=1e-12,equal_nan=True)
    node=PlanNode(op=node.op,inputs=node.inputs,attrs={**attrs,"scale":"mad"})
    assert compile_plan_to_sql(node,table="observations",time_column="t",instrument_column="i",dialect=SqlDialect.DUCKDB) is None

@pytest.mark.parametrize("positional",[False,True])
def test_polars_long_quantile_range_real_support_and_finite_scale(positional):
    from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars
    from factor_engine.planner.logical_plan import PlanNode
    w=8;lo=.2;hi=.8;mp=3
    col=PlanNode(op="column",attrs={"name":"x"})
    node=(PlanNode(op="ts_quantile_range",inputs=[col,*[PlanNode(op="literal",attrs={"value":v}) for v in (w,lo,hi,mp)]])
          if positional else PlanNode(op="ts_quantile_range",inputs=[col],attrs=dict(window=w,q_low=lo,q_high=hi,min_periods=mp)))
    for magnitude in (1e-200,1.,1e200):
        x=panel()*magnitude;x.iloc[6,0]=np.inf;x.iloc[9,1]=np.nan
        long=x.rename_axis("ts").reset_index().melt(id_vars="ts",var_name="inst",value_name="x")
        compiled=compile_plan_to_polars(node,pl.from_pandas(long).lazy())
        assert compiled is not None
        got=compiled.frame.collect().sort(["ts","inst"])[compiled.value_col].to_numpy()
        expected=run("ts_quantile_range","pandas_numpy",x,window=w,q_low=lo,q_high=hi,min_periods=mp).to_numpy().ravel()
        np.testing.assert_allclose(got/magnitude,expected/magnitude,atol=1e-12,equal_nan=True)

@pytest.mark.parametrize("positional",[False,True])
@pytest.mark.parametrize("bad",[True,3.0,2.5,0])
def test_polars_plan_rejects_invalid_min_periods_without_coercion(positional,bad):
    from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars
    from factor_engine.planner.logical_plan import PlanNode
    col=PlanNode(op="column",attrs={"name":"x"})
    node=(PlanNode(op="ts_quantile_range",inputs=[col,*[
        PlanNode(op="literal",attrs={"value":v}) for v in (8,.2,.8,bad)]])
        if positional else PlanNode(op="ts_quantile_range",inputs=[col],
                                   attrs=dict(window=8,min_periods=bad)))
    base=pl.DataFrame({"ts":[1,2],"inst":["A","A"],"x":[1.,2.]}).lazy()
    with pytest.raises((ValueError,TypeError)):
        compile_plan_to_polars(node,base)
