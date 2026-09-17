import ast
import numpy as np
import pandas as pd
from factor_engine.tools.catalog_r20_root_recipes import migrate_formula

def evaluate(formula, x):
    def divide(a,b):
        return a / (b.replace(0,np.nan) if hasattr(b,"replace") else b)
    env={"x":x,"safe_div_null":divide,"square":np.square,
         "add":lambda a,b:a+b,"subtract":lambda a,b:a-b,"multiply":lambda a,b:a*b,
         "ts_std":lambda x,w,ddof=1:x.rolling(w,min_periods=w).std(ddof=ddof),
         "ts_mean":lambda x,w:x.rolling(w,min_periods=w).mean(),
         "ts_time_slope":lambda x,w,min_periods:x.rolling(w,min_periods=min_periods).apply(lambda v:np.polyfit(np.arange(w),v,1)[0],raw=True),
         "log_positive_or_nan":lambda x:np.log(np.where(np.asarray(x)>0,x,np.nan))}
    return eval(compile(ast.parse(formula,mode="eval"),"<test>","eval"),{"__builtins__":{}},env)

def test_time_ols_expansion_against_direct_fit():
    x=pd.Series(np.random.default_rng(42).normal(size=80).cumsum())
    for kind in ("r2","resid"):
        formula,notes=migrate_formula(f"ts_regression_{kind}(x, 12)")
        actual=evaluate(formula,x)
        expected=[]
        for end in range(len(x)):
            if end<11:expected.append(np.nan);continue
            y=x.iloc[end-11:end+1].to_numpy()
            fitted=np.polyval(np.polyfit(np.arange(12),y,1),np.arange(12))
            expected.append(1-np.sum((y-fitted)**2)/np.sum((y-y.mean())**2) if kind=="r2" else y[-1]-fitted[-1])
        np.testing.assert_allclose(actual,expected,atol=1e-11,equal_nan=True)
        assert notes
        assert migrate_formula(formula)==(formula,[])

def test_entropy_keeps_all_ten_parts_and_nulls_invalid():
    formula,notes=migrate_formula("composition_normalized_entropy("+",".join("x*"+str(i) for i in range(1,11))+")")
    actual=evaluate(formula,np.array([1.,2.,-1.,0.]))
    p=np.arange(1.,11.);p/=p.sum()
    expected=-np.sum(p*np.log(p))/np.log(10)
    np.testing.assert_allclose(actual[:2],expected)
    assert np.isnan(actual[2:]).all()
    assert notes
    assert migrate_formula(formula)==(formula,[])

def test_named_parameters_are_not_discarded_and_clr_target_not_duplicated():
    src="composition_normalized_entropy("+",".join("x"+str(i) for i in range(10))+", zero_policy='reject')"
    assert migrate_formula(src)==(src,[])
    formula,notes=migrate_formula("composition_clr_component(x,x,a,b,c,d,e,f,g)")
    assert formula=="composition_clr_component(x, a, b, c, d, e, f, g)"
    assert notes
    assert migrate_formula(formula)==(formula,[])

def test_turnover_both_horizons_and_raw_suspend():
    formula,_=migrate_formula("turnover_shock(x,5,60)")
    assert formula=="turnover_shock(ts_mean(x, 5), 60)"
    formula,_=migrate_formula("ashare_suspension_episode_length(a,b,c,d,e)")
    assert "field('is_suspend', table='StockDailyBar')" in formula
    assert "volume" not in formula


def test_period_lag_redundant_flag_keeps_missing_quarter_null():
    from factor_engine.cleaned_operators.fiscal_strict import pd_period_lag
    x=pd.DataFrame({"A":[10.,30.,40.]})
    periods=pd.DataFrame({"A":[20200331,20200930,20201231]})
    out=pd_period_lag(x,periods,periods=1)
    assert np.isnan(out.iloc[0,0]) and np.isnan(out.iloc[1,0])
    assert out.iloc[2,0]==30.
    formula,notes=migrate_formula("period_lag(x,p,periods=1,require_consecutive=True,revision_policy=\'latest_available\')")
    assert "require_consecutive" not in formula and "revision_policy" in formula and notes
    untouched="period_lag(x,p,require_consecutive=False)"
    assert migrate_formula(untouched)==(untouched,[])


def test_keyword_second_feature_keeps_conditioning():
    formula,notes=migrate_formula("ts_conditional_mutual_information(x=x,source=y)")
    assert "y=y" in formula and "z=turnover_ratio" in formula
    assert len(notes)==2
