"""Exact contract and independent oracle for multifractal asymmetry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAME = "ts_multifractal_asymmetry"


def _panel(v): return pd.DataFrame({"A": np.asarray(v, dtype=float)})
def _backend(p, b):
    if b == "pandas_numpy": return p
    import polars as pl
    return pl.DataFrame({"A": p["A"].to_list()})
def _values(x): return x.to_numpy(dtype=float) if isinstance(x, pd.DataFrame) else x.to_numpy()
def _ops():
    load_all(); bs = OperatorRegistry.backends_for(NAME); assert set(bs) == {"pandas_numpy", "polars"}
    return [(b, OperatorRegistry.get(NAME, b, mode="any")) for b in bs]


def _hurst(v, q):
    v = np.asarray(v, float); v = v / np.max(np.abs(v)); lx=[]; ly=[]
    for lag in (1, 2, 4, 8):
        d=np.abs(v[lag:]-v[:-lag]); required=max(8, int(np.ceil(8*abs(q))))
        if len(d)<required: continue
        with np.errstate(all="ignore"): moment=np.mean(d**q)
        if np.isfinite(moment) and moment>0: lx.append(np.log(lag)); ly.append(np.log(moment))
    slope,intercept=np.polyfit(lx,ly,1); fitted=slope*np.asarray(lx)+intercept
    r2=1-np.sum((np.asarray(ly)-fitted)**2)/np.sum((np.asarray(ly)-np.mean(ly))**2)
    return slope/q if r2>=.9 else np.nan


def _oracle(v):
    return np.mean([_hurst(v,-q)-_hurst(v,q) for q in (1.,2.,4.)])


def test_exact_contract_defaults_topology_units_and_backend():
    for backend,op in _ops():
        m=op.metadata
        assert tuple(m.panel_params)==("x",) and m.panel_arity==1
        assert tuple(m.scalar_params)==("window","min_periods")
        assert m.param_specs["window"].default==120 and m.param_specs["window"].min==40
        assert m.param_specs["window"].history_semantics=="max_rows"
        assert m.param_specs["window"].param_role is ParamRole.HORIZON
        assert m.param_specs["min_periods"].default==40 and m.param_specs["min_periods"].param_role is ParamRole.SUPPORT_POLICY
        assert m.input_units=={"x":"level"} and m.output_unit=="dimensionless"
        if backend=="polars": assert op._physical_spec.execution_kind.value=="polars_pandas_delegate"


def test_oracle_call_forms_prefix_backend_and_scale_invariance():
    values=np.cumsum(np.random.default_rng(0).lognormal(mean=0,sigma=.4,size=240)); p=_panel(values)
    expected=_oracle(values[-120:]); reference=None
    for backend,op in _ops():
        bp=_backend(p,backend); a=_values(op.calculate(bp,120,40)); b=_values(op.calculate(x=bp,window=120,min_periods=40)); c=_values(op.calculate(bp,window=120,min_periods=40))
        prefix=_values(op.calculate(_backend(p.iloc[:201],backend),120,40))
        np.testing.assert_allclose(a,b,equal_nan=True); np.testing.assert_allclose(a,c,equal_nan=True); np.testing.assert_allclose(prefix,a[:201],equal_nan=True)
        np.testing.assert_allclose(a[-1,0],expected,rtol=1e-9,atol=1e-9)
        scaled=_values(op.calculate(_backend(_panel(values*1e250),backend),120,40))[-1,0]
        np.testing.assert_allclose(scaled,expected,rtol=1e-8,atol=1e-8)
        if reference is None: reference=a
        else: np.testing.assert_allclose(a,reference,equal_nan=True)


def test_gap_not_compressed_and_current_nan_fail_closed():
    values=np.cumsum(np.random.default_rng(0).lognormal(mean=0,sigma=.4,size=160)); values[-35]=np.nan
    for backend,op in _ops(): assert np.isnan(_values(op.calculate(_backend(_panel(values),backend),120,40))[-1,0])
    values[-1]=np.nan
    for backend,op in _ops(): assert np.isnan(_values(op.calculate(_backend(_panel(values),backend),120,40))[-1,0])


@pytest.mark.parametrize("kwargs", [{"window":39},{"window":40.5},{"min_periods":39},{"window":60,"min_periods":61}])
def test_invalids_and_required_panel(kwargs):
    p=_panel(np.arange(80.0))
    for backend,op in _ops():
        with pytest.raises(Exception): op.calculate(_backend(p,backend),**kwargs)
        with pytest.raises(Exception): op.calculate(**kwargs)
