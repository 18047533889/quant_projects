import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge,ElasticNet
from factor_preprocess.neutralization.regularized import ridge_neutralize,lasso_neutralize,elastic_net_neutralize
from factor_preprocess.neutralization.diagnostics import compute_condition_number,check_residual_exposures

@pytest.mark.parametrize('kind,alpha,l1_ratio', [('ridge',.2,0.),('lasso',.03,1.),('elastic',.03,.35)])
@pytest.mark.parametrize('intercept',[False,True])
def test_t59_regularizers_match_declared_objective(kind,alpha,l1_ratio,intercept):
    rng=np.random.default_rng(8); n=80; X=rng.normal(size=(n,2))+np.array([4.,-3.]); y=2.5+X@np.array([1.2,-.7])+rng.normal(scale=.05,size=n)
    values=pd.DataFrame({'date':1,'asset_id':np.arange(n),'value':y},index=np.arange(n)*3+7); exposures=pd.DataFrame({'date':1,'asset_id':np.arange(n),'x1':X[:,0],'x2':X[:,1]}).sample(frac=1,random_state=2)
    kw=dict(alpha=alpha,min_observations=5,normalize=False,add_intercept=intercept)
    if kind=='ridge': got=ridge_neutralize(values,exposures,**kw); ref=Ridge(alpha=n*alpha,fit_intercept=intercept).fit(X,y)
    else:
        ratio=1. if kind=='lasso' else l1_ratio
        solve_kw={**kw,'max_iter':10000,'tol':1e-10}
        got=(lasso_neutralize(values,exposures,**solve_kw) if kind=='lasso' else elastic_net_neutralize(values,exposures,l1_ratio=ratio,**solve_kw))
        ref=ElasticNet(alpha=alpha,l1_ratio=ratio,fit_intercept=intercept,max_iter=10000,tol=1e-10,selection='cyclic').fit(X,y)
    np.testing.assert_allclose(got.to_numpy(),y-ref.predict(X),atol=2e-4)
    assert got.attrs['objective'].startswith('||y-Xb||^2/(2n)')

@pytest.mark.parametrize('name,value', [('alpha',np.nan),('alpha',True),('tol',0.),('max_iter',0),('l1_ratio',1.1)])
def test_t60_invalid_full_parameter_domain(name,value):
    v=pd.DataFrame({'date':[1,1],'asset_id':[1,2],'value':[1.,2.]}); e=pd.DataFrame({'date':[1,1],'asset_id':[1,2],'x':[1.,2.]}); kw={'alpha':.1,'tol':1e-5,'max_iter':10,'l1_ratio':.5,'min_observations':2}; kw[name]=value
    with pytest.raises((TypeError,ValueError)): elastic_net_neutralize(v,e,**kw)

def test_t61_budget_and_stationarity_evidence_are_distinct():
    rng=np.random.default_rng(3); n=60; X=rng.normal(size=(n,4)); y=X@np.array([1.,-.7,.3,.2])+rng.normal(size=n)*.01; v=pd.DataFrame({'date':1,'asset_id':range(n),'value':y}); e=pd.DataFrame({'date':1,'asset_id':range(n),**{f'x{i}':X[:,i] for i in range(4)}})
    short=lasso_neutralize(v,e,alpha=.001,min_observations=5,max_iter=1,tol=1e-20); mature=lasso_neutralize(v,e,alpha=.001,min_observations=5,max_iter=10000,tol=1e-10)
    assert short.attrs['fit_diagnostics'][0]['status']=='ITERATION_BUDGET_EXHAUSTED' and not short.attrs['fit_diagnostics'][0]['converged']
    assert mature.attrs['fit_diagnostics'][0]['status']=='CONVERGED' and np.isfinite(mature.attrs['fit_diagnostics'][0]['stationarity_gap'])

def test_t62_t63_rank_dof_and_same_sample_scope():
    exposure=pd.DataFrame({'date':[1,1],'asset_id':['a','b'],'x1':[1.,0.],'x2':[0.,1.],'x3':[1.,1.]})
    row=compute_condition_number(exposure,add_intercept=True).iloc[0]
    assert row['rank']<row['n_exposures'] and not row['full_column_rank'] and row['degrees_of_freedom']==0
    residual=pd.DataFrame({'date':[1]*5,'asset_id':range(5),'residual':[-2.,-1.,0.,1.,2.]}); exp=pd.DataFrame({'date':[1]*5,'asset_id':range(5),'x':[-2.,-1.,0.,1.,2.]})
    diag=check_residual_exposures(residual,exp)
    assert set(diag['diagnostic_scope'])=={'descriptive_same_sample_not_alpha_confidence'}

def test_t58_collision_and_duplicate_exposure_keys_reject():
    values=pd.DataFrame({'date':[1,1],'asset_id':[1,2],'value':[1.,2.]})
    duplicate=pd.DataFrame({'date':[1,1,1],'asset_id':[1,1,2],'x':[1.,1.,2.]})
    collision=pd.DataFrame({'date':[1,1],'asset_id':[1,2],'value':[3.,4.]})
    with pytest.raises(ValueError): ridge_neutralize(values,duplicate,min_observations=2)
    with pytest.raises(ValueError): ridge_neutralize(values,collision,min_observations=2)
