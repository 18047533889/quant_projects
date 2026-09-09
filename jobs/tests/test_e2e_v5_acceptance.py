from jobs.e2e_v5_acceptance import (accept_u_shape_chain, assert_cost_crush_chain,
    run_continuous_price_volume_chain, run_financial_event_maturation_chain,
    run_library_replacement_chain, run_long_only_bottom_exclusion_chain)
from factor_optimizer.search.paired_comparison import ComparisonStatus
import pytest


def test_v5_u_shape_runs_fe_fp_qe_and_fa_shape_policy():
    result, grade = accept_u_shape_chain()
    assert result.frozen_center is not None
    assert grade.score >= 0
    assert grade.dimension_id == "shape_quality"


def test_v5_cost_crush_final_decision_uses_public_scenario_outputs():
    decision = assert_cost_crush_chain(1.2, .2, -.1)
    assert decision["decision"] == "REJECT_NET_STRESS"


def test_v5_spec_9_5_continuous_price_volume_real_public_chain():
    upstream, routes, _, _, raw, candidate, comparison, frozen = run_continuous_price_volume_chain()
    assert upstream.dsl == "rank(close)"
    assert len(routes) == 5
    assert raw["base"] < raw["gross"] and candidate["base"] > raw["base"]
    assert comparison.status is ComparisonStatus.NON_INFERIOR_CHEAPER
    assert frozen.production_pointer_changed is False


def test_v5_spec_9_5_financial_event_real_maturity_chain(tmp_path):
    initial, stream, summaries, frozen = run_financial_event_maturation_chain(tmp_path)
    assert initial.instance_results
    assert stream and next(iter(summaries.values()))["count"] == 8
    assert frozen.use_case == "MODEL_FEATURE"


def test_v5_spec_9_5_long_only_bottom_exclusion_real_ledger_chain():
    trajectory, metrics, frozen = run_long_only_bottom_exclusion_chain()
    assert trajectory.profile == "LONG_ONLY_RESEARCH"
    assert trajectory.refs.borrow_ref is None
    assert .75 < metrics["mean_investment_fraction"] < 1.0
    assert metrics["capacity_utilization"] == pytest.approx(.04, rel=.01)
    assert metrics["tracking_error"] > 0
    assert metrics["turnover_cost"] > 0
    assert frozen.production_pointer_changed is False


def test_v5_spec_9_5_library_replacement_real_oof_and_rollback_chain(tmp_path):
    trial, compatibility, update, stopped = run_library_replacement_chain(tmp_path)
    assert trial.accepted and trial.paired_delta > 0
    assert compatibility.outcome == "RETRAINED"
    assert update.previous_version_ref == "features:v1"
    assert stopped is True
