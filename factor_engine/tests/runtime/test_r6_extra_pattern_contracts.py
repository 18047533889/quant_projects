"""Final twelve supplemental chart patterns: native parity and authored bounds."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract
from factor_engine.cleaned_operators.price_volume.structure_extra_contracts import EXTRA_PATTERNS

@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
VALUES=dict(left_window=2,right_window=2,history_window=60,tolerance=.1,min_depth=.001,
            min_spacing=1,max_spacing=50,min_swing=.001,window=15,min_fit=.001,
            max_edge_diff=.1,cup_window=15,handle_window=4,max_handle_retracement=.2,
            impulse_window=5,pennant_window=6,min_impulse=.001,max_width=.4,
            volume_decay_threshold=0.,max_wait=10)
def inputs(name):
    t=np.arange(100)
    close=pd.DataFrame({"A":100+5*np.sin(t*.5)+.03*t,"B":90+4*np.cos(t*.45)+.02*t},
                       index=pd.date_range("2026-01-01",periods=100))
    panels=dict(close=close,high=close+1,low=close-1,volume=close*3+100)
    op=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    return {n:panels[n] if n in panels else VALUES[n] for n in op.metadata.param_names}
def convert(v):
    return pl.from_pandas(v.rename_axis("date").reset_index()) if isinstance(v,pd.DataFrame) else v
def result(v):
    return v.to_pandas().set_index("date").rename_axis(None) if isinstance(v,pl.DataFrame) else v
@pytest.mark.parametrize("name",sorted(EXTRA_PATTERNS))
def test_final_pattern_native_parity_keywords_prefix_and_scalar_bounds(name):
    values=inputs(name)
    reference=OperatorRegistry.get(name,"pandas_numpy",mode="research")
    expected=reference.calculate(**values)
    assert np.isfinite(expected.to_numpy()).any(),name
    for backend in ("pandas_numpy","polars"):
        op=OperatorRegistry.get(name,backend,mode="research")
        assert op is not None
        kwargs={k:convert(v) if backend=="polars" else v for k,v in values.items()}
        actual=result(op.calculate(**kwargs))
        pd.testing.assert_frame_equal(actual,expected,check_freq=False,rtol=1e-8,atol=1e-9)
        pd.testing.assert_frame_equal(result(op.calculate(*kwargs.values())),actual,check_freq=False)
        short={k:convert(v.iloc[:80]) if backend=="polars" else v.iloc[:80] for k,v in values.items() if isinstance(v,pd.DataFrame)}
        scalars={k:v for k,v in values.items() if not isinstance(v,pd.DataFrame)}
        pd.testing.assert_frame_equal(result(op.calculate(**short,**scalars)),actual.iloc[:80],check_freq=False,rtol=1e-8,atol=1e-9)
        _,_,verified=_parameter_contract(op,tuple(k for k,v in values.items() if isinstance(v,pd.DataFrame)))
        assert verified,(name,backend)
        for key in scalars:
            bad=dict(kwargs);bad[key]=-1
            with pytest.raises((TypeError,ValueError)):
                op.calculate(**bad)
@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_rounding_independent_quadratic_reference_and_unknown_warmup(backend):
    coord=np.linspace(-1,1,15)
    x=pd.DataFrame({"A":100+10*coord**2},index=pd.date_range("2026-01-01",periods=15))
    op=OperatorRegistry.get("pattern_rounding_bottom",backend,mode="research")
    y=result(op.calculate(convert(x) if backend=="polars" else x,15,0.))
    assert y.iloc[:14].isna().all().all()
    assert y.iloc[-1,0]==pytest.approx(10/x.A.mean(),rel=1e-10)
    broken=x.copy();broken.iloc[5,0]=np.nan
    z=result(op.calculate(convert(broken) if backend=="polars" else broken,15,0.))
    assert z.isna().all().all()


@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_triple_top_requires_fifth_pivot_confirmation(backend):
    close=pd.DataFrame({"A":[100.,110.,100.,110.,100.,110.,100.]},
                       index=pd.date_range("2026-01-01",periods=7))
    high,low=close+1,close-1
    op=OperatorRegistry.get("pattern_triple_top",backend,mode="research")
    args=[convert(high),convert(low)] if backend=="polars" else [high,low]
    y=result(op.calculate(*args,1,1,20,.1,.05,1,5))
    assert y.iloc[:6].isna().all().all()
    assert y.iloc[6,0]==pytest.approx(1.)
    with pytest.raises((TypeError,ValueError)):
        op.calculate(*args,1,1,20,.1,.05,5,1)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
def test_123_signal_requires_trailing_low_high_low_order(backend):
    close=pd.DataFrame({"A":[110.,100.,110.,102.,112.,105.]},
                       index=pd.date_range("2026-01-01",periods=6))
    op=OperatorRegistry.get("pattern_123_bull",backend,mode="research")
    high,low=close+1,close-1
    args=[convert(high),convert(low)] if backend=="polars" else [high,low]
    y=result(op.calculate(*args,1,1,20,.05))
    assert y.A.tolist()==[0.,0.,0.,0.,1.,0.]
