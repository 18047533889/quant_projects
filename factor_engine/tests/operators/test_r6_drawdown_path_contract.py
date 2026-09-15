from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()
NAMES=("ts_recovery_fraction","ts_current_drawdown_area")

def op(name,backend="pandas_numpy"):
    value=OperatorRegistry.get(name,backend); assert value is not None; return value

def call(name,values,backend="pandas_numpy",window=6):
    frame=pd.DataFrame({"A":values,"B":np.asarray(values)*2.})
    if backend=="polars": frame=pl.from_pandas(frame)
    out=op(name,backend).calculate(frame,window=window)
    return out.to_pandas() if isinstance(out,pl.DataFrame) else out

def test_fresh_contract_defaults_topology_and_delegate():
    for name in NAMES:
        p,q=op(name),op(name,"polars")
        assert p.metadata==q.metadata
        m=p.metadata
        assert m.panel_params==("x",) and m.panel_arity==1
        assert m.scalar_params==("window",) and m.output_unit=="dimensionless"
        assert m.param_specs["window"].default==60
        assert m.param_specs["window"].history_semantics=="max_rows"
        assert q._physical_spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_peak_trough_recovery_and_current_episode_area(backend):
    x=np.array([100.,80.,90.,100.,120.,90.,100.])
    rf=call(NAMES[0],x,backend,window=7)["A"].to_numpy()
    area=call(NAMES[1],x,backend,window=7)["A"].to_numpy()
    assert rf[2]==pytest.approx(.5) and rf[3]==pytest.approx(1.)
    assert rf[-1]==pytest.approx(1/3)
    assert area[3]==pytest.approx(0.)
    assert area[-1]==pytest.approx(.25+1/6)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_gap_nonpositive_prefix_causality_and_strict_window(backend):
    x=np.array([100.,90.,80.,np.nan,70.,60.,65.])
    for name in NAMES:
        base=call(name,x,backend,window=6)
        assert np.isnan(base.iloc[4,0])
        longer=call(name,np.r_[x,1e9],backend,window=6)
        pd.testing.assert_frame_equal(base,longer.iloc[:-1].reset_index(drop=True))
        bad=x.copy(); bad[5]=0.
        assert np.isnan(call(name,bad,backend,window=6).iloc[-1,0])
        with pytest.raises((TypeError,ValueError)):
            call(name,x,backend,window=2.5)
