"""Final contracts and independent numerical checks for AR mean-reversion operators."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = (
    "ts_mean_reversion_half_life",
    "ts_mean_reversion_ou_approx_half_life",
    "ts_mean_reversion_ou_approx_half_life_prior",
    "ts_variance_ratio_slope",
)


def _panel(v):
    return pd.DataFrame({"A": np.asarray(v, dtype=float)}, index=pd.bdate_range("2024-01-02", periods=len(v)))


def _backend(p, backend):
    if backend == "pandas_numpy": return p
    import polars as pl
    return pl.DataFrame({"A": p["A"].to_list()})


def _values(v):
    return v.to_numpy(dtype=float) if isinstance(v, pd.DataFrame) else v.to_numpy()


def _ops(name):
    load_all(); backends=OperatorRegistry.backends_for(name)
    assert set(backends)=={"pandas_numpy","polars"}
    return [(b,OperatorRegistry.get(name,b,mode="any")) for b in backends]


def test_exact_contracts_defaults_history_units_and_backend_honesty():
    expected={
        "ts_mean_reversion_half_life":{"window":120,"min_periods":20},
        "ts_mean_reversion_ou_approx_half_life":{"window":120,"min_periods":20},
        "ts_mean_reversion_ou_approx_half_life_prior":{"window":120,"min_periods":20},
        "ts_variance_ratio_slope":{"window":120,"max_q":10,"min_periods":20},
    }
    for name,defaults in expected.items():
        for backend,op in _ops(name):
            m=op.metadata
            assert tuple(m.panel_params)==("x",) and m.panel_arity==1
            assert tuple(m.scalar_params)==tuple(defaults)
            assert {k:m.param_specs[k].default for k in defaults}==defaults
            assert m.param_specs["window"].history_semantics=="max_rows"
            assert m.param_specs["window"].param_role is ParamRole.HORIZON
            if name.endswith("_prior"):
                assert m.param_specs["window"].history_formula=="window + 1"
            if name=="ts_variance_ratio_slope":
                assert m.output_unit=="dimensionless" and "unit:dimensionless" in m.tags
            else:
                assert m.input_units=={"x":"spread_or_residual_or_stationary"}
            if backend=="polars":
                assert op._physical_spec.execution_kind.value=="polars_pandas_delegate"


def test_call_forms_prefix_and_backend_parity():
    rng=np.random.default_rng(4); vals=np.empty(90); vals[0]=1
    for i in range(1,len(vals)): vals[i]=.8*vals[i-1]+rng.normal(scale=.1)
    p=_panel(vals)
    cases={
        "ts_mean_reversion_half_life":((40,10),{"window":40,"min_periods":10}),
        "ts_mean_reversion_ou_approx_half_life":((40,10),{"window":40,"min_periods":10}),
        "ts_mean_reversion_ou_approx_half_life_prior":((40,10),{"window":40,"min_periods":10}),
        "ts_variance_ratio_slope":((40,6,10),{"window":40,"max_q":6,"min_periods":10}),
    }
    for name,(args,kwargs) in cases.items():
        ref=None
        for backend,op in _ops(name):
            bp=_backend(p,backend)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a=_values(op.calculate(bp,*args)); b=_values(op.calculate(x=bp,**kwargs)); c=_values(op.calculate(bp,**kwargs))
                prefix=_values(op.calculate(_backend(p.iloc[:71],backend),*args))
            np.testing.assert_allclose(a,b,equal_nan=True); np.testing.assert_allclose(a,c,equal_nan=True)
            np.testing.assert_allclose(prefix,a[:71],equal_nan=True)
            if ref is None: ref=a
            else: np.testing.assert_allclose(a,ref,rtol=1e-12,atol=1e-12,equal_nan=True)


@pytest.mark.parametrize("name,param,bad",[
    ("ts_mean_reversion_half_life","window",4),
    ("ts_mean_reversion_half_life","min_periods",3),
    ("ts_mean_reversion_ou_approx_half_life","window",20.5),
    ("ts_mean_reversion_ou_approx_half_life_prior","min_periods",np.inf),
    ("ts_variance_ratio_slope","max_q",2),
    ("ts_variance_ratio_slope","max_q",3.5),
    ("ts_variance_ratio_slope","window",np.nan),
    ("ts_variance_ratio_slope","min_periods",2),
])
def test_invalid_scalars_rejected_every_backend(name,param,bad):
    p=_panel(np.arange(30.0))
    for backend,op in _ops(name):
        with pytest.raises(Exception): op.calculate(_backend(p,backend),**{param:bad})


def test_relational_domains_and_required_panel():
    p=_panel(np.arange(30.0))
    for name in NAMES:
        for backend,op in _ops(name):
            with pytest.raises(Exception): op.calculate()
            with pytest.raises(Exception): op.calculate(_backend(p,backend),window=10,min_periods=10)
    for backend,op in _ops("ts_variance_ratio_slope"):
        with pytest.raises(Exception): op.calculate(_backend(p,backend),window=10,max_q=9,min_periods=5)


def _manual_vr_slope(levels,max_q):
    levels=np.asarray(levels,float); levels=levels/np.max(np.abs(levels)); rets=np.diff(levels); v1=np.var(rets)
    lx=[]; vr=[]
    for q in range(2,max_q+1):
        lx.append(np.log(q)); vr.append(np.var(levels[q:]-levels[:-q])/(q*v1)-1)
    lx=np.asarray(lx)-np.mean(lx); vr=np.asarray(vr)-np.mean(vr)
    return float(lx@vr/(lx@lx))


def test_known_ar_half_lives_and_variance_ratio_oracle():
    phi=.8; vals=np.empty(50); vals[0]=1.3
    for i in range(1,len(vals)): vals[i]=.4+phi*(vals[i-1]-.4)
    expected_exact=np.log(.5)/np.log(phi); expected_ou=-np.log(2)/(phi-1)
    for name,expected in (("ts_mean_reversion_half_life",expected_exact),("ts_mean_reversion_ou_approx_half_life",expected_ou),("ts_mean_reversion_ou_approx_half_life_prior",expected_ou)):
        for backend,op in _ops(name):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); got=_values(op.calculate(_backend(_panel(vals),backend),40,10))[-1,0]
            np.testing.assert_allclose(got,expected,rtol=1e-8,atol=1e-8)
    rng=np.random.default_rng(11); increments=rng.normal(size=80); levels=np.cumsum(increments)
    expected=_manual_vr_slope(levels[-40:],6)
    for backend,op in _ops("ts_variance_ratio_slope"):
        got=_values(op.calculate(_backend(_panel(levels),backend),40,6,10))[-1,0]
        np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12)


def test_scale_invariance_tiny_and_huge_without_false_nan():
    rng=np.random.default_rng(88); vals=np.empty(80); vals[0]=.4
    for i in range(1,80): vals[i]=.75*vals[i-1]+rng.normal(scale=.05)
    levels=np.cumsum(rng.normal(size=80)); cases={
        "ts_mean_reversion_half_life":(vals,(40,10)),
        "ts_mean_reversion_ou_approx_half_life":(vals,(40,10)),
        "ts_mean_reversion_ou_approx_half_life_prior":(vals,(40,10)),
        "ts_variance_ratio_slope":(levels,(40,6,10)),
    }
    for name,(base,args) in cases.items():
        for backend,op in _ops(name):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs=[_values(op.calculate(_backend(_panel(base*scale),backend),*args))[-1,0] for scale in (1.0,1e-300,1e300)]
            assert np.isfinite(outputs).all()
            np.testing.assert_allclose(outputs,outputs[0],rtol=1e-10,atol=1e-10)
