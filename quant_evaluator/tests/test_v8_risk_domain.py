import numpy as np
import pytest
from quant_evaluator.metrics.risk.stress_testing import (
    apply_target_correlation, apply_hypothetical_scenario,
    evaluate_official_scenario_set, compute_position_replay_impact)
from quant_evaluator.metrics.risk.tail_risk import compute_finite_q_coexceedance


def test_t84_psd_domain_and_rank_deficient_target_are_explicit():
    x = np.random.default_rng(84).normal(size=(100,3))
    invalid = np.array([[1,.9,.9],[.9,1,-.9],[.9,-.9,1]])
    with pytest.raises(ValueError,match='PSD'):
        apply_target_correlation(x,invalid)
    with pytest.raises(ValueError,match='singular'):
        apply_target_correlation(np.column_stack([x[:,0]]*3),np.eye(3))
    missing=x.copy(); missing[4,0]=np.nan
    with pytest.raises(ValueError,match='complete'):
        apply_target_correlation(missing,np.eye(3))
    mapped=apply_target_correlation(x,np.ones((3,3)),return_evidence=True)
    np.testing.assert_allclose(mapped['achieved_correlation'],np.ones((3,3)),atol=1e-12)
    assert mapped['qualification']=='DIAGNOSTIC_ONLY'
    with pytest.raises(ValueError,match='axes'):
        apply_target_correlation(x,np.eye(2),asset_ids=('a','b','c'),target_asset_ids=('a','b','c'))


def test_stress_integer_storage_cannot_truncate_fractional_shocks():
    x=np.arange(6).reshape(3,2)
    np.testing.assert_allclose(apply_hypothetical_scenario(x,.1),x+.1)


def test_t86_event_identity_is_content_bound_order_invariant_not_value_dedup():
    x=np.linspace(-.01,.01,30)
    a={'event_id':'a|b','returns':x}
    b={'event_id':'c','returns':x}
    original=evaluate_official_scenario_set({'candidate':x},[a,b],metrics=['mean'])
    reordered=evaluate_official_scenario_set({'candidate':x,'irrelevant':x},[b,a],metrics=['mean'])
    assert original['scenario_set_id']==reordered['scenario_set_id']
    assert original['event_weight']==.5  # different economic events, same numerical result
    changed=evaluate_official_scenario_set({'candidate':x},[a,dict(b,returns=x+.01)],metrics=['mean'])
    assert changed['scenario_set_id']!=original['scenario_set_id']
    collision=evaluate_official_scenario_set({'candidate':x},[dict(a,event_id='a'),dict(b,event_id='b|c')],metrics=['mean'])
    assert collision['scenario_set_id']!=original['scenario_set_id']
    with pytest.raises(ValueError,match='conflicting'):
        evaluate_official_scenario_set({'candidate':x},[a,dict(a,returns=x+.01)])


def test_t85_research_trajectory_is_typed_but_not_executable_certification():
    from vectorbt_qs.contracts.trajectories import build_research_trajectory, TrajectoryRefs
    refs=TrajectoryRefs(('factor',),'snapshot','portfolio','cost','benchmark')
    def trajectory(scenario, offset):
        return build_research_trajectory(scenario_id=scenario,profile='LONG_ONLY_RESEARCH',
            dates=tuple(f'2024-01-{i:02d}' for i in range(1,31)),
            gross_return=np.linspace(-.01,.01,30)+offset, benchmark_return=np.zeros(30),
            cost_contributions={'commission':np.zeros(30)},refs=refs)
    base, stressed=trajectory('base',0),trajectory('stress',-.01)
    result=compute_position_replay_impact(base,stressed,metrics=['mean'])
    assert result['evidence_type']=='POSITION_REPLAY'
    assert not result['formal_strategy_stress_eligible']
    assert result['scenario_trajectory_ref']==stressed.artifact_id
    with pytest.raises(TypeError): compute_position_replay_impact(np.zeros(30),np.zeros(30))


def test_t110_tail_counts_and_small_samples_do_not_claim_confidence():
    x=np.arange(100.)
    result=compute_finite_q_coexceedance(x,x,min_tail_observations=10)
    assert result['tails']['lower']['status']=='INSUFFICIENT_TAIL_EVENTS'
    assert result['tails']['lower']['x_count']==5
    assert result['qualification']=='DESCRIPTIVE_ONLY'
    assert compute_finite_q_coexceedance(x[:2],x[:2])['tails']['lower']['status']=='INSUFFICIENT_DATA'
    with pytest.raises(ValueError): compute_finite_q_coexceedance(x,x,min_tail_observations=True)


def test_t80_upr_mar_translation_and_undefined_denominator():
    from quant_evaluator.metrics.risk.tail_risk import compute_upside_potential_ratio
    x=np.r_[np.full(90,.01),np.full(10,-.01)]
    expected=.009/np.sqrt(.00001)
    assert compute_upside_potential_ratio(x)==pytest.approx(expected)
    assert compute_upside_potential_ratio(x+.03,minimum_acceptable_return=.03)==pytest.approx(expected)
    assert np.isnan(compute_upside_potential_ratio(np.ones(30)))
    assert np.isnan(compute_upside_potential_ratio(np.zeros(30)))
    assert compute_upside_potential_ratio(-np.ones(30))==0


def test_t82_es_ratio_has_no_mechanical_annualization_or_absolute_epsilon():
    from quant_evaluator.metrics.risk.tail_risk import compute_expected_shortfall_ratio
    x=np.r_[-.04,np.full(99,.001)]
    # Lowest fixed five-percent mass: -.04 and four +.001 observations.
    expected=x.mean()/((.04-4*.001)/5)
    assert compute_expected_shortfall_ratio(x,periods_per_year=252)==pytest.approx(expected)
    assert compute_expected_shortfall_ratio(x,periods_per_year=12)==pytest.approx(expected)
    assert compute_expected_shortfall_ratio(x*1e-12)==pytest.approx(expected)


def test_t108_tiny_loss_peak_duration_and_underwater_statistics_share_events():
    from quant_evaluator.metrics.risk.drawdown_analysis import drawdown_events,compute_drawdown_statistics
    from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown
    from quant_evaluator.metrics.underwater import compute_max_underwater_duration
    x=np.r_[0.,-5e-13,np.zeros(18)]
    event,=drawdown_events(x)
    magnitude,path,peak=compute_maximum_drawdown(x)
    assert magnitude>0 and peak==event['peak_idx']==0
    assert event['trough_idx']==1 and event['duration']==19
    assert event['recovery_idx']==-1 and event['status']=='ACTIVE'
    assert compute_drawdown_statistics(x)['time_underwater_pct']==95.
    assert compute_max_underwater_duration(x)==19
    recovered,=drawdown_events(np.array([0.,-.1,1/9]))
    assert recovered['recovery_idx']==2 and recovered['peak_idx']==0
    defaulted,=drawdown_events(np.array([-1.,.5,0.]))
    assert defaulted['peak_idx']==-1 and defaulted['status']=='DEFAULTED'
