"""Actual SQL/reference parity, including support masks and adversarial holes."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators import load_all
load_all()
from factor_engine.backend.sql_pushdown.emitter import _compile_layer, SqlDialect
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.cleaned_operators.robust_stats import TsTrimmedMean
from factor_engine.cleaned_operators.robust_tail import TsLowerPartialMoment, TsUpperPartialMoment
from factor_engine.cleaned_operators.semantic_hardening import TimeSeriesProductAudited
from factor_engine.cleaned_operators.filter_despike import TSRollingMedianCausal
from factor_engine.cleaned_operators.layer_topk_compat import _pd as topk_reference
from factor_engine.cleaned_operators.common.time_series import TSAutocorr, PriceSpreadDeviation
from factor_engine.cleaned_operators.technical.new_indicators import HMA
from factor_engine.cleaned_operators.overhaul.daily import pd_argmax, pd_argmin
from factor_engine.cleaned_operators.registry import OperatorRegistry


def registered_reference(name):
    obj=OperatorRegistry.get(name,mode='research')
    return getattr(obj,'_calculate_series',None) or obj._fn


CASES=[
    ('digital_count', registered_reference('digital_count'), {'d':20,'threshold':0.05,'run':2}),
    ('winsorize', registered_reference('winsorize'), {}),
    ('HMA', HMA()._calculate_series, {'window':20}),
    ('HMA', HMA()._calculate_series, {'window':15,'rounding':'round'}),
    ('ts_autocorr', TSAutocorr()._calculate_series, {'window':20,'lag':3,'min_periods':7}),
    ('price_spread_deviation', PriceSpreadDeviation()._calculate_series, {'d':20}),
    ('ts_argmax', pd_argmax, {'window':20,'min_periods':5}),
    ('ts_argmin', pd_argmin, {'window':20,'min_periods':5}),
    ('ts_rolling_median_causal', TSRollingMedianCausal()._calculate_series, {'window':20,'min_periods':3}),
    ('ts_trimmed_mean', TsTrimmedMean()._calculate_series, {'window':20}),
    ('ts_lower_partial_moment', TsLowerPartialMoment()._calculate_series, {'window':20,'threshold':1.0}),
    ('ts_upper_partial_moment', TsUpperPartialMoment()._calculate_series, {'window':20,'threshold':1.0}),
    ('ts_product', TimeSeriesProductAudited()._calculate_series, {'window':20}),
    ('ts_product', TimeSeriesProductAudited()._calculate_series, {'window':20,'min_periods':3,'skipna':False}),
]
for top in (True,False):
    for stat in ('mean','sum','std'):
        def reference(x,top=top,stat=stat,**kw):
            return topk_reference(x,top=top,stat=stat,**kw)
        CASES.append((f'ts_{"top" if top else "bottom"}k_{stat}',reference,{'window':20,'k':5,'min_periods':7}))

for name in ('ts_valid_count','ts_coverage_ratio','ts_argmax_age','ts_argmin_age',
             'ts_argmax_index_from_oldest','ts_argmin_index_from_oldest','ts_staleness',
             'ts_abs_entropy','ts_quantile_range','ts_quantile_skew','ts_quantile_kurtosis',
             'ts_tail_ratio','ts_expected_shortfall','ts_max_drawdown','ts_trend_tstat'):
    CASES.append((name,registered_reference(name),{'window':20}))


def sql_result(x,name,attrs,extra=None):
    duckdb=pytest.importorskip('duckdb')
    data={'x':x,**(extra or {})}
    node=PlanNode(op=name,inputs=[PlanNode(op='column',attrs={'name':c}) for c in data],attrs=attrs)
    layer=_compile_layer(node,dialect=SqlDialect.DUCKDB)
    assert layer is not None
    base=pd.concat([pd.DataFrame({'ts':x.index,'inst':c,**{k:v[c].values for k,v in data.items()}}) for c in x.columns]).sample(frac=1,random_state=9)
    with duckdb.connect(config={'threads':1}) as con:
        con.register('base',base)
        out=con.execute(layer.sql).df().pivot(index='ts',columns='inst',values='_v')
    return out.reindex(index=x.index,columns=x.columns).astype(float)


@pytest.mark.parametrize('name,reference,attrs',CASES)
@pytest.mark.parametrize('scenario',['normal','holes','infinity','constant','all_missing','future'])
def test_window_support(name,reference,attrs,scenario):
    rng=np.random.default_rng(914)
    x=pd.DataFrame(rng.uniform(.8,1.2,(65,2)),index=pd.date_range('2024-01-01',periods=65),columns=['A','B'])
    if scenario=='holes':x.iloc[25,0]=np.nan;x.iloc[:8,1]=np.nan
    if scenario=='infinity':x.iloc[25,0]=np.inf
    if scenario=='constant':x[:]=1.0
    if scenario=='all_missing':x[:]=np.nan
    actual=sql_result(x,name,attrs)
    expected=reference(x,**attrs)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12,equal_nan=True)
    if scenario=='future':
        x.iloc[40:]=42
        np.testing.assert_allclose(actual.iloc[:40],sql_result(x,name,attrs).iloc[:40],rtol=1e-10,atol=1e-12,equal_nan=True)


@pytest.mark.parametrize('name,ninputs',[
    ('rank_corr',2),('ts_corr_if',3),('ts_regression_slope',2),('ts_regression_tstat',2),
    ('ts_partial_corr',3),('ts_crossing_speed',2),('ts_crossing_acceleration',2),
    ('ATR_WILDER',3),('group_decay_linear',2),
])
@pytest.mark.parametrize('scenario',['normal','holes','constant','all_missing','future'])
def test_multi_input_windows(name,ninputs,scenario):
    rng=np.random.default_rng(45)
    data=[pd.DataFrame(100+rng.normal(0,2,(65,2)).cumsum(axis=0),index=pd.date_range('2024-01-01',periods=65),columns=['A','B']) for _ in range(ninputs)]
    if name=='ts_corr_if':data[2]=(data[2]>100).astype(float)
    if name=='group_decay_linear':data[1][:]=1.0
    if scenario=='holes':
        for j,x in enumerate(data):x.iloc[25+j,0]=np.nan;x.iloc[:4,1]=np.nan
    if scenario=='constant':
        for x in data:x[:]=1.0
    if scenario=='all_missing':
        for x in data:x[:]=np.nan
    attrs={'d':20} if name=='rank_corr' else {'window':20}
    actual=sql_result(data[0],name,attrs,{f'x{j}':v for j,v in enumerate(data[1:])})
    expected=registered_reference(name)(*data,**attrs)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8,equal_nan=True)
    if scenario=='future':
        data[0].iloc[40:]*=2
        after=sql_result(data[0],name,attrs,{f'x{j}':v for j,v in enumerate(data[1:])})
        np.testing.assert_allclose(actual.iloc[:40],after.iloc[:40],rtol=1e-6,atol=1e-8,equal_nan=True)


def test_semantic_gaps_are_not_advertised_as_sql_capable():
    from factor_engine.backend.sql_pushdown.emitter import _SQL_WINDOW_REPAIR_FALLBACKS, plan_is_sql_capable
    for name in _SQL_WINDOW_REPAIR_FALLBACKS:
        node=PlanNode(op=name,inputs=[PlanNode(op='column',attrs={'name':'x'})],attrs={'window':20})
        assert _compile_layer(node,dialect=SqlDialect.DUCKDB) is None
        assert not plan_is_sql_capable(node)
        assert not plan_is_sql_capable(PlanNode(op='neg',inputs=[node]))
