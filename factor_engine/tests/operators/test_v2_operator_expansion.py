from __future__ import annotations
import numpy as np
import pandas as pd


def _panel(values):
    idx=pd.date_range("2024-01-01",periods=len(values),freq="B")
    return pd.DataFrame({"A":values},index=idx,dtype=float)

def test_fundamental_lag_uses_report_period_not_trading_rows():
    from cleaned_operators.fundamental.transforms_v2 import fin_lag
    x=_panel([10,10,11,11,20,20,20,30,30])
    period=pd.DataFrame({"A":["2023Q1"]*4+["2023Q2"]*3+["2023Q3"]*2},index=x.index)
    out=fin_lag(x,period,1)
    assert np.isnan(out.iloc[3,0])
    assert out.iloc[4,0]==11.0  # latest visible Q1 revision, not yesterday's row
    assert out.iloc[7,0]==20.0

def test_revision_event_is_same_period_only():
    from cleaned_operators.fundamental.transforms_repairs_v2 import fin_revision_delta
    x=_panel([10,10,11,20,20])
    period=pd.DataFrame({"A":["Q1","Q1","Q1","Q2","Q2"]},index=x.index)
    out=fin_revision_delta(x,period)
    assert list(out["A"])[1:]==[0.0,1.0,0.0,0.0]

def test_bounded_fundamental_streak():
    from cleaned_operators.fundamental.transforms_repairs_v2 import fin_positive_streak
    x=_panel([1,2,3,4,5,6]);period=pd.DataFrame({"A":[f"Q{i}" for i in range(6)]},index=x.index)
    out=fin_positive_streak(x,period,4)
    assert out.iloc[-1,0]==3.0

def test_structure_pattern_is_prefix_invariant():
    from cleaned_operators.price_volume.structure_patterns_v2 import pattern_sym_triangle
    n=80;t=np.arange(n,dtype=float);high=_panel(110-0.08*t+np.sin(t/2));low=_panel(90+0.08*t-np.sin(t/2))
    full=pattern_sym_triangle(high,low,2,2,50,3,0.001)
    cut=60;prefix=pattern_sym_triangle(high.iloc[:cut],low.iloc[:cut],2,2,50,3,0.001)
    pd.testing.assert_series_equal(full.iloc[:cut,0],prefix.iloc[:,0],check_names=False)

def test_kama_is_prefix_invariant():
    from cleaned_operators.technical.indicators_v2 import KAMA
    x=_panel(np.linspace(10,20,100)+np.sin(np.arange(100)/3))
    full=KAMA(x,10,2,30);prefix=KAMA(x.iloc[:70],10,2,30)
    np.testing.assert_allclose(full.iloc[:70,0],prefix.iloc[:,0],equal_nan=True,rtol=1e-12,atol=1e-12)

def test_candle_geometry_window_is_prior_based():
    from cleaned_operators.price_volume.candle_geometry_v2 import candle_body_zscore
    o=_panel(np.arange(30)+10);c=o.copy();c.iloc[:,0]+=np.linspace(0.1,3,30)
    full=candle_body_zscore(o,c,10);prefix=candle_body_zscore(o.iloc[:20],c.iloc[:20],10)
    np.testing.assert_allclose(full.iloc[:20,0],prefix.iloc[:,0],equal_nan=True)

def test_intraday_clock_does_not_compress_lunch_gap():
    from storage.sources.intraday_feature_runtime_v2 import _ordinal
    ts=pd.Series(pd.to_datetime(["2024-01-02 11:30","2024-01-02 13:00","2024-01-02 13:05"]))
    ordinal=_ordinal(ts,"ashare_stock_minute","09:30","15:00")
    assert ordinal.iloc[0]==120
    assert ordinal.iloc[1]==120
    # boundary may share the continuous-session coordinate, but 13:05 must
    # advance rather than be compressed by missing lunch observations.
    assert ordinal.iloc[2]==125

def test_cutoff_expected_minutes_excludes_future_session():
    from storage.sources.intraday_feature_runtime_v2 import _effective_minutes
    assert _effective_minutes("us_stock_minute","09:30","16:00","15:50")==380
    assert _effective_minutes("ashare_stock_minute","09:30","15:00","14:00")==180

def test_liquidity_operator_is_prefix_invariant():
    from cleaned_operators.price_volume.liquidity_v2 import amihud_illiquidity
    close=_panel(np.linspace(10,12,80));volume=_panel(np.linspace(1e6,2e6,80));ret=close.pct_change()
    full=amihud_illiquidity(ret,close,volume,20);prefix=amihud_illiquidity(ret.iloc[:60],close.iloc[:60],volume.iloc[:60],20)
    np.testing.assert_allclose(full.iloc[:60,0],prefix.iloc[:,0],equal_nan=True)
