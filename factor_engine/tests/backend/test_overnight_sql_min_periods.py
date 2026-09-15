"""Execute actual emitted DuckDB SQL against the canonical pandas kernels."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.sql_pushdown.emitter import _compile_layer, SqlDialect
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.cleaned_operators.intraday.overnight import (
    _ts_overnight_intraday_spread,
    _ts_overnight_intraday_sign_agreement,
    _ts_overnight_intraday_cov,
    _ts_opening_mispricing_score,
)


def sample():
    rng=np.random.default_rng(45)
    index=pd.date_range('2024-01-01',periods=70)
    pre=pd.DataFrame(rng.uniform(60,130,(70,2)),index=index,columns=['A','B'])
    opening=pre*(1+rng.normal(0,.01,(70,2)))
    close=opening*(1+rng.normal(0,.015,(70,2)))
    return close,opening,pre


def sql_result(close,opening,pre,window=20,op='ts_overnight_intraday_spread'):
    duckdb=pytest.importorskip('duckdb')
    node=PlanNode(op=op,inputs=[PlanNode(op='column',attrs={'name':n})
                  for n in ['close','open','pre_close']],attrs={'window':window})
    compiled=_compile_layer(node,dialect=SqlDialect.DUCKDB)
    assert compiled is not None
    base=pd.concat([pd.DataFrame({'ts':close.index,'inst':col,'close':close[col].values,
         'open':opening[col].values,'pre_close':pre[col].values}) for col in close.columns],ignore_index=True)
    # Deliberately scramble rows: SQL must partition by asset and order by time.
    base=base.sample(frac=1,random_state=6)
    with duckdb.connect() as con:
        con.register('base',base)
        result=con.execute(compiled.sql).df().pivot(index='ts',columns='inst',values='_v')
    return result.reindex(index=close.index,columns=close.columns)


@pytest.mark.parametrize('window',[1,5,20])
def test_full_window_and_warmup(window):
    c,o,p=sample()
    actual=sql_result(c,o,p,window)
    expected=_ts_overnight_intraday_spread(c,o,p,window)
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13,equal_nan=True)
    assert actual.iloc[:window-1].isna().all().all()


@pytest.mark.parametrize('field',['close','open','pre_close'])
@pytest.mark.parametrize('bad',[np.nan,0.0])
def test_missing_or_zero_price_matches_reference(field,bad):
    c,o,p=sample()
    {'close':c,'open':o,'pre_close':p}[field].iloc[25,0]=bad
    actual=sql_result(c,o,p)
    expected=_ts_overnight_intraday_spread(c,o,p,20)
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13,equal_nan=True)
    if np.isnan(bad) or field!='close':
        assert actual.iloc[25:45,0].isna().all()
        assert np.isfinite(actual.iloc[45,0])


def test_disjoint_missing_legs_cannot_average_different_samples():
    c,o,p=sample();c.iloc[25,0]=np.nan;p.iloc[28,0]=np.nan
    actual=sql_result(c,o,p)
    np.testing.assert_allclose(actual,_ts_overnight_intraday_spread(c,o,p,20),atol=1e-13,equal_nan=True)
    assert actual.iloc[25:48,0].isna().all()


def test_prefix_causality():
    c,o,p=sample();actual=sql_result(c,o,p)
    c.iloc[40:]*=5
    np.testing.assert_allclose(actual.iloc[:40],sql_result(c,o,p).iloc[:40],atol=1e-13,equal_nan=True)
    np.testing.assert_allclose(actual.iloc[:40],sql_result(c.iloc[:40],o.iloc[:40],p.iloc[:40]),atol=1e-13,equal_nan=True)


def test_sign_agreement_full_frame_but_missing_sign_counts_as_false():
    c,o,p=sample();c.iloc[25,0]=np.nan
    actual=sql_result(c,o,p,op='ts_overnight_intraday_sign_agreement')
    expected=_ts_overnight_intraday_sign_agreement(c,o,p,20)
    np.testing.assert_allclose(actual,expected,atol=1e-13,equal_nan=True)
    assert actual.iloc[:19].isna().all().all()
    assert np.isfinite(actual.iloc[25,0])


def test_covariance_keeps_its_distinct_five_pair_minimum():
    c,o,p=sample();c.iloc[25,0]=np.nan
    actual=sql_result(c,o,p,op='ts_overnight_intraday_cov')
    expected=_ts_overnight_intraday_cov(c,o,p,20)
    np.testing.assert_allclose(actual,expected,atol=1e-13,equal_nan=True)
    assert actual.iloc[:4].isna().all().all()
    assert np.isfinite(actual.iloc[4,0])


@pytest.mark.parametrize('field',['close','open','pre_close'])
def test_opening_mispricing_uses_same_pairs_for_covariance_and_variance(field):
    c,o,p=sample()
    {'close':c,'open':o,'pre_close':p}[field].iloc[25:28,0]=np.nan
    actual=sql_result(c,o,p,op='ts_opening_mispricing_score')
    expected=_ts_opening_mispricing_score(c,o,p,20)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-13,equal_nan=True)
