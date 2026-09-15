"""Exact threshold-cycle contracts and hand-computed state-machine oracles."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry

NAMES=("ts_threshold_cycle_period","ts_threshold_cycle_asymmetry")
def _panel(v): return pd.DataFrame({"A":np.asarray(v,float)})
def _backend(p,b):
    if b=="pandas_numpy": return p
    import polars as pl
    return pl.DataFrame({"A":p["A"].to_list()})
def _values(x): return x.to_numpy(dtype=float) if isinstance(x,pd.DataFrame) else x.to_numpy()
def _ops(n):
    load_all(); bs=OperatorRegistry.backends_for(n); assert set(bs)=={"pandas_numpy","polars"}
    return [(b,OperatorRegistry.get(n,b,mode="any")) for b in bs]


def test_exact_contracts_and_backend_honesty():
    for name in NAMES:
        for backend,op in _ops(name):
            m=op.metadata; assert tuple(m.panel_params)==("x",) and m.panel_arity==1
            assert tuple(m.scalar_params)==("lower","upper","window")
            assert m.param_specs["window"].default==120 and m.param_specs["window"].history_semantics=="max_rows"
            assert m.param_specs["window"].param_role is ParamRole.HORIZON
            assert m.param_specs["lower"].param_role is ParamRole.STATE_THRESHOLD
            assert m.input_units=={"x":"level","lower":"level","upper":"level"}
            assert m.output_unit==("bars" if name.endswith("period") else "dimensionless")
            if backend=="polars": assert op._physical_spec.execution_kind.value=="polars_pandas_delegate"


def test_hand_computed_cycles_call_forms_prefix_and_parity():
    values=[0,0,2,2,2,0,0,2,2,2,2,0,0]; p=_panel(values)
    expected={NAMES[0]:5.5,NAMES[1]:3/11}
    for name in NAMES:
        reference=None
        for backend,op in _ops(name):
            bp=_backend(p,backend); a=_values(op.calculate(bp,.5,1.5,20)); b=_values(op.calculate(x=bp,lower=.5,upper=1.5,window=20)); c=_values(op.calculate(bp,lower=.5,upper=1.5,window=20))
            np.testing.assert_allclose(a,b,equal_nan=True); np.testing.assert_allclose(a,c,equal_nan=True); np.testing.assert_allclose(a[-1,0],expected[name],rtol=0,atol=1e-15)
            prefix=_values(op.calculate(_backend(p.iloc[:10],backend),.5,1.5,20)); np.testing.assert_allclose(prefix,a[:10],equal_nan=True)
            if reference is None: reference=a
            else: np.testing.assert_allclose(a,reference,equal_nan=True)


def test_gap_breaks_cycle_instead_of_bridging():
    values=np.asarray([0,0,2,2,2,0,np.nan,2,2,2,2,0,0])
    for name in NAMES:
        for backend,op in _ops(name): assert np.isnan(_values(op.calculate(_backend(_panel(values),backend),.5,1.5,20))[-1,0])


@pytest.mark.parametrize("lower,upper,window",[(1.,1.,20),(2.,1.,20),(np.nan,1.,20),(0.,np.inf,20),(0.,1.,1),(0.,1.,2.5)])
def test_invalid_thresholds_window_and_required_panel(lower,upper,window):
    p=_panel(np.arange(20.0))
    for name in NAMES:
        for backend,op in _ops(name):
            with pytest.raises(Exception): op.calculate(_backend(p,backend),lower,upper,window)
            with pytest.raises(Exception): op.calculate(lower=lower,upper=upper,window=window)
