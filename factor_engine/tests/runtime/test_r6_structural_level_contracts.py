"""Confirmed structural levels: full-history contracts and bounded pivot queries."""
import math
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import structural_levels as source
from factor_engine.cleaned_operators.common._pivot_ledger import confirmed_pivot_events, PivotLedgerResult
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES=("ts_structural_level_density","ts_nearest_structural_level_distance","ts_structural_level_strength")
@pytest.fixture(scope="module",autouse=True)
def registry():
    load_all()
def prices():
    return pd.DataFrame({"A":np.tile([100.,110.,100.,90.],20)},
                        index=pd.date_range("2025-01-01",periods=80,tz="Asia/Hong_Kong"))
def run(name,backend,x,**kwargs):
    op=OperatorRegistry.get(name,backend,mode="research")
    assert op is not None and op.metadata.panel_params==("price",)
    assert _parameter_contract(op,("price",))[2]
    if backend=="polars":
        out=op.calculate(price=pl.from_pandas(x.rename_axis("date").reset_index()),**kwargs)
        return out.to_pandas().set_index("date").rename_axis(None)
    return op.calculate(price=x,**kwargs)

@pytest.mark.parametrize("backend",["pandas_numpy","polars"])
@pytest.mark.parametrize("name",NAMES)
def test_independent_confirmed_pivot_oracle_prefix_and_units(name,backend):
    x=prices();v=x.A.to_numpy();w=20;t=len(v)-1
    kwargs=dict(window=w,confirmation=1,min_periods=5)
    out=run(name,backend,x,**kwargs)
    # This alternating fixed-amplitude path has strict ±1 pivots at odd bars;
    # every opposite-side move exceeds prominence, so no replacement ambiguity.
    pivots=[i for i in range(1,t) if i%2==1 and i>=t-w+1]
    distances=np.array([math.log(v[t]/v[i]) for i in pivots])
    if name==NAMES[0]:
        expected=np.exp(-.5*(distances/.03)**2).sum()/w
    elif name==NAMES[1]:
        returns=np.log(v[1:]/v[:-1])[-w:]
        expected=np.min(np.abs(distances))/(np.std(returns)+1e-12)
    else:
        expected=sum(math.exp(-.05*(t-i-1))*(1-abs(d)/.05)
                     for i,d in zip(pivots,distances) if abs(d)<=.05)
    assert out.A.iloc[-1]==pytest.approx(expected,abs=1e-12)
    np.testing.assert_allclose(run(name,backend,x.iloc[:61],**kwargs),out.iloc[:61],equal_nan=True)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(run(name,backend,x*scale,**kwargs),out,atol=1e-11,equal_nan=True)
    default=run(name,backend,x)
    assert default.iloc[:59].isna().all().all()  # 50% of nominal 120 bars
    assert default.iloc[59:].notna().all().all()
    for bad in (True,2.5,1):
        with pytest.raises((ValueError,TypeError)):
            run(name,backend,x,window=bad)
    with pytest.raises((ValueError,TypeError)):
        run(name,backend,x,window=10,min_periods=20)
    with pytest.raises((ValueError,TypeError)):
        run(name,backend,x,window=20,confirmation=20)

def test_log_ratios_never_overflow_or_underflow_before_log():
    assert source._log_price_ratio(1e308,1e-308)==pytest.approx(math.log(1e308)-math.log(1e-308))
    assert source._log_price_ratio(1e-308,1e308)==pytest.approx(math.log(1e-308)-math.log(1e308))
    for name in NAMES:
        x=prices();x.A=np.tile([1e-308,1e308],40)
        out=run(name,"pandas_numpy",x,window=20,confirmation=1,min_periods=5)
        assert out.iloc[-1].notna().all()

def test_pivot_index_exactly_matches_original_temporal_scan():
    rng=np.random.default_rng(71)
    x=100+np.cumsum(rng.normal(size=180))
    ledger=confirmed_pivot_events(x,2,.005,positive_only=True)
    for t in range(len(x)):
        for w in (None,1,7,30):
            lo=t-w+1 if w is not None else 0
            expected=tuple(ev for ev in ledger.events if ev.confirmed_at<=t and ev.pivot_at>=lo
                and (ledger._superseder_at.get(id(ev)) is None or ledger._superseder_at[id(ev)]>t))
            assert ledger.active_at(t,window=w)==expected

def test_pivot_query_slices_only_the_requested_recent_interval():
    ledger=confirmed_pivot_events(np.tile([100.,110.,100.,90.],500),1,.02,positive_only=True)
    class ObservedTuple(tuple):
        last_slice=None
        def __getitem__(self,key):
            if isinstance(key,slice):
                self.last_slice=key
            return super().__getitem__(key)
    events=ObservedTuple(ledger.events)
    indexed=PivotLedgerResult(events,ledger.superseded,ledger.confirmation)
    got=indexed.active_at(1999,window=20)
    selection=events.last_slice
    assert selection is not None
    assert selection.start>len(events)-25
    assert selection.stop-selection.start<=20
    assert len(got)>0

@pytest.mark.parametrize("name",NAMES)
def test_structural_ledger_requires_complete_history_not_only_output_window(name):
    from factor_engine.runtime.execution_contract import execution_contract,history_requirement
    contract=execution_contract(name)
    assert contract.state_model=="recursive"
    assert contract.chunking=="required_full_history"
    assert history_requirement(name,{"window":20,"confirmation":1}).kind=="full_history"
