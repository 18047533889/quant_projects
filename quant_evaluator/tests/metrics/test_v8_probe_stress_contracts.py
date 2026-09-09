import numpy as np
import pytest

from quant_evaluator.metrics.probe_portfolio import compute_active_metrics
from quant_evaluator.metrics.probe_portfolio.costs import compute_transaction_costs,heuristic_cost_scan
from quant_evaluator.metrics.probe_portfolio_legacy import construct_long_short_portfolio
from quant_evaluator.metrics.risk.stress_testing import apply_target_correlation,apply_historical_scenario,compute_scenario_impact,evaluate_official_scenario_set
from quant_evaluator.metrics.risk.tail_risk import compute_finite_q_coexceedance

def test_active_basis_and_ir_do_not_subtract_cash_twice():
    rp=np.array([.2,-.1]); rb=np.array([.1,0.])
    got=compute_active_metrics(rp,rb,min_periods=2,risk_free_rate=.05)
    assert got['relative_nav'][-1]==pytest.approx(1.08/1.1)
    assert np.isnan(compute_active_metrics(rp,rp,min_periods=2,risk_free_rate=.05)['information_ratio'])

def test_cost_evidence_zero_trade_and_carry_are_distinct():
    got=compute_transaction_costs(np.zeros(2),100.,commission_rate=.1,financing_balance=[10,10],financing_rate=.01)
    assert np.all(got.components['commission']==0) and np.all(got.components['financing']>0)
    assert not heuristic_cost_scan(np.zeros(2)).formal_net_eligible

def test_legacy_mask_contract_accepts_2d_for_3d_and_rejects_nonboolean():
    values=np.arange(24.,dtype=float).reshape(2,4,3)
    long,short=construct_long_short_portfolio(values,validity_mask=np.ones((2,4),bool))
    assert long.shape==values.shape and short.shape==values.shape
    with pytest.raises(TypeError): construct_long_short_portfolio(values,validity_mask=np.ones((2,4)))

def test_named_correlation_mapping_is_permutation_equivariant_and_evidenced():
    rng=np.random.default_rng(2); x=rng.normal(size=(300,3))*.01; target=np.array([[1,.3,.1],[.3,1,.2],[.1,.2,1.]])
    a=apply_target_correlation(x,target,asset_ids=('a','b','c'),target_asset_ids=('a','b','c'),return_evidence=True)
    p=[2,0,1]; b=apply_target_correlation(x[:,p],target[np.ix_(p,p)],asset_ids=('c','a','b'),target_asset_ids=('c','a','b'))
    assert np.allclose(a['values'][:,p],b) and a['max_abs_error']<1e-10
    assert apply_historical_scenario(x,np.array([-.1,.2]),return_evidence=True)['formal_strategy_stress_eligible'] is False
    with pytest.raises(ValueError,match='naked returns'):
        compute_scenario_impact(x,x,evidence_type='POSITION_REPLAY')

def test_tail_direction_ties_and_weak_margin_status():
    x=np.array([0,0,0,1,2,3]*10,float); y=np.array([0,1,1,1,2,3]*10,float)
    xy=compute_finite_q_coexceedance(x,y,quantile=.2); yx=compute_finite_q_coexceedance(y,x,quantile=.2)
    assert xy['tails']['lower']['p_y_given_x']!=yx['tails']['lower']['p_y_given_x']
    constant=compute_finite_q_coexceedance(np.ones(50),np.arange(50.),quantile=.1)
    assert constant['tails']['lower']['status']=='DEGENERATE_MARGIN' and constant['tails']['lower']['marginal_quality']=='LOW'

def test_official_scenarios_ignore_candidate_set_and_duplicate_event_weight():
    baseline=np.linspace(-.01,.01,30); event=np.linspace(-.02,0,30)
    one=evaluate_official_scenario_set({'a':baseline},[{'event_id':'e1','returns':event},{'event_id':'copy','canonical_event_id':'e1','returns':event}])
    two=evaluate_official_scenario_set({'a':baseline,'irrelevant':baseline*0},[{'event_id':'e1','returns':event}])
    assert one['event_weight']==two['event_weight']==1 and one['results']['a']['e1']==two['results']['a']['e1']
    assert one['scenario_set_id']==two['scenario_set_id']
