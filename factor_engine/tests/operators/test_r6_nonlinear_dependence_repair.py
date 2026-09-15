"""Final-backend contracts and numerical oracles for nonlinear dependence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.nonlinear_dependence import _fractional_tail_membership
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES = (
    "ts_mutual_information_nats", "ts_normalized_mutual_information", "ts_lagged_mutual_information",
    "ts_upper_tail_coexceedance_probability", "ts_lower_tail_coexceedance_probability",
)


def _panel(v, start="2024-01-02"):
    return pd.DataFrame({"A": np.asarray(v, dtype=float)}, index=pd.date_range(start, periods=len(v)))


def _backend(p, backend):
    if backend == "pandas_numpy": return p
    import polars as pl
    return pl.DataFrame({"A": p["A"].to_list()})


def _values(v):
    return v.to_numpy(dtype=float) if isinstance(v, pd.DataFrame) else v.to_numpy()


def _ops(name):
    load_all(); backends = OperatorRegistry.backends_for(name)
    assert set(backends) == {"pandas_numpy", "polars"}
    return [(b, OperatorRegistry.get(name, b, mode="any")) for b in backends]


def test_exact_contracts_defaults_history_and_final_backend_honesty():
    expected = {
        "ts_mutual_information_nats": ({"window":40,"estimator":"quantile_hist","bins":5,"min_periods":10,"bias_correction":True}),
        "ts_normalized_mutual_information": ({"window":40,"estimator":"quantile_hist","bins":5,"min_periods":10,"bias_correction":True}),
        "ts_lagged_mutual_information": ({"window":40,"lag":1,"bins":5,"min_periods":10,"bias_correction":True}),
        "ts_upper_tail_coexceedance_probability": ({"window":60,"q":0.9,"min_tail_count":5}),
        "ts_lower_tail_coexceedance_probability": ({"window":60,"q":0.1,"min_tail_count":5}),
    }
    for name, defaults in expected.items():
        for backend, op in _ops(name):
            m = op.metadata
            assert tuple(m.panel_params) == ("x", "y") and m.panel_arity == 2
            assert tuple(m.scalar_params) == tuple(defaults)
            assert {k:m.param_specs[k].default for k in defaults} == defaults
            assert m.param_specs["window"].param_role is ParamRole.HORIZON
            assert m.param_specs["window"].history_semantics == "max_rows"
            if name == "ts_lagged_mutual_information":
                assert m.param_specs["window"].history_formula == "window + lag"
            if backend == "polars":
                assert op._physical_spec.execution_kind.value == "polars_pandas_delegate"


def test_call_forms_prefix_and_backend_parity():
    rng=np.random.default_rng(9); x=_panel(rng.normal(size=55)); y=_panel(np.roll(x["A"],1)+rng.normal(scale=.2,size=55))
    cases={
        "ts_mutual_information_nats":((20,"quantile_hist",4,10,False),{"window":20,"estimator":"quantile_hist","bins":4,"min_periods":10,"bias_correction":False}),
        "ts_normalized_mutual_information":((20,"quantile_hist",4,10,False),{"window":20,"estimator":"quantile_hist","bins":4,"min_periods":10,"bias_correction":False}),
        "ts_lagged_mutual_information":((20,2,4,10,False),{"window":20,"lag":2,"bins":4,"min_periods":10,"bias_correction":False}),
        "ts_upper_tail_coexceedance_probability":((20,.8,2),{"window":20,"q":.8,"min_tail_count":2}),
        "ts_lower_tail_coexceedance_probability":((20,.2,2),{"window":20,"q":.2,"min_tail_count":2}),
    }
    for name,(args,kwargs) in cases.items():
        ref=None
        for backend,op in _ops(name):
            bx,by=_backend(x,backend),_backend(y,backend)
            a=_values(op.calculate(bx,by,*args)); b=_values(op.calculate(x=bx,y=by,**kwargs))
            c=_values(op.calculate(bx,y=by,**kwargs))
            np.testing.assert_allclose(a,b,equal_nan=True); np.testing.assert_allclose(a,c,equal_nan=True)
            px,py=_backend(x.iloc[:43],backend),_backend(y.iloc[:43],backend)
            np.testing.assert_allclose(_values(op.calculate(px,py,*args)),a[:43],equal_nan=True)
            if ref is None: ref=a
            else: np.testing.assert_allclose(a,ref,rtol=1e-12,atol=1e-12,equal_nan=True)


@pytest.mark.parametrize("name,param,bad",[
    ("ts_normalized_mutual_information","estimator","pearson"),("ts_mutual_information_nats","bins",2.5),
    ("ts_mutual_information","normalized",1),("ts_lagged_mutual_information","window",np.inf),
    ("ts_lagged_mutual_information","lag",1.5),("ts_lagged_mutual_information","min_periods",9),
    ("ts_upper_tail_coexceedance_probability","q",.5),("ts_upper_tail_coexceedance_probability","min_tail_count",2.5),
    ("ts_lower_tail_coexceedance_probability","q",np.nan),("ts_lower_tail_coexceedance_probability","window",3),
])
def test_invalid_scalars_fail_every_backend(name,param,bad):
    p=_panel(np.arange(20.0))
    for backend,op in _ops(name):
        with pytest.raises(Exception): op.calculate(_backend(p,backend),_backend(p,backend),**{param:bad})


def test_required_panels_and_axis_identity():
    p=_panel(np.arange(20.0)); shifted=_panel(np.arange(20.0),"2024-01-03")
    for name in NAMES:
        for backend,op in _ops(name):
            with pytest.raises(Exception): op.calculate()
            if backend == "pandas_numpy":
                with pytest.raises(Exception): op.calculate(p,shifted)


def test_fractional_membership_has_exact_target_mass_with_and_without_ties():
    for vals in (np.arange(10.0), np.array([0,0,0,1,1,2,2,2,2,3],float)):
        upper=_fractional_tail_membership(vals,.73,"upper")
        lower=_fractional_tail_membership(vals,.27,"lower")
        assert np.all((upper>=0)&(upper<=1)) and np.all((lower>=0)&(lower<=1))
        assert upper.sum() == pytest.approx(2.7) and lower.sum() == pytest.approx(2.7)
        for value in np.unique(vals):
            assert np.unique(upper[vals==value]).size == 1
            assert np.unique(lower[vals==value]).size == 1


def test_tail_tiny_scale_and_mi_float_max_are_finite_and_informative():
    tiny=_panel(np.arange(40.0)*1e-300); tiny_y=_panel(np.arange(40.0)[::-1]*1e-300)
    for name,q in (("ts_upper_tail_coexceedance_probability",.8),("ts_lower_tail_coexceedance_probability",.2)):
        for backend,op in _ops(name):
            got=_values(op.calculate(_backend(tiny,backend),_backend(tiny_y,backend),20,q,2))[-1,0]
            assert np.isfinite(got) and 0 <= got <= 1
    maxv=np.finfo(float).max
    x=_panel(np.linspace(-.9,.9,40)*maxv); y=_panel(np.sin(np.arange(40.0))*.9*maxv)
    for name,args in (("ts_normalized_mutual_information",(20,"quantile_hist",4,10,False)),
                      ("ts_lagged_mutual_information",(20,1,4,10,False))):
        for backend,op in _ops(name):
            got=_values(op.calculate(_backend(x,backend),_backend(y,backend),*args))[-1,0]
            assert np.isfinite(got) and got >= 0


def test_independent_mi_and_lagged_direction_oracles():
    rng=np.random.default_rng(2026); n=100; x=rng.normal(size=n); noise=rng.normal(size=n)
    linked=x**2; independent=noise
    for backend,op in _ops("ts_normalized_mutual_information"):
        strong=_values(op.calculate(_backend(_panel(x),backend),_backend(_panel(linked),backend),80,"quantile_hist",5,20,False))[-1,0]
        weak=_values(op.calculate(_backend(_panel(x),backend),_backend(_panel(independent),backend),80,"quantile_hist",5,20,False))[-1,0]
        assert strong > weak
    y=np.r_[0.0,x[:-1]]
    for backend,op in _ops("ts_lagged_mutual_information"):
        lag1=_values(op.calculate(_backend(_panel(x),backend),_backend(_panel(y),backend),60,1,5,20,False))[-1,0]
        lag3=_values(op.calculate(_backend(_panel(x),backend),_backend(_panel(y),backend),60,3,5,20,False))[-1,0]
        assert lag1 > lag3
