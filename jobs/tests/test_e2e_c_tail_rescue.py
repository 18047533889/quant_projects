import pytest

from jobs.e2e_c_tail_rescue import run_e2e_c_tail_rescue


def test_q10_trigger_q20_confirmation_and_bounded_train_selection():
    result = run_e2e_c_tail_rescue("q20")
    assert result.policy_branch == "Q20_CONFIRMED_BOUNDED_REPAIR"
    assert result.selected_q == 20
    assert result.train_risk_evidence["q10_top_cliff"] < 0
    assert result.train_risk_evidence["q20_top_cliff"] < 0
    assert result.frozen_cutoff.value in (0.85, 0.90, 0.95)
    assert len(result.candidate_dsls) == 6
    assert set(result.oos_evaluations) == {"parent_oos", "selected_oos"}
    assert result.trace["selection_split"] == "TRAIN"
    assert result.trace["oos_role"] == "EVALUATION_ONLY"


def test_oos_poisoning_cannot_change_family_or_frozen_cutoff():
    normal = run_e2e_c_tail_rescue("q20")
    poisoned = run_e2e_c_tail_rescue("q20", poison_oos=True)
    assert normal.selected_family == poisoned.selected_family
    assert normal.frozen_cutoff.state_hash == poisoned.frozen_cutoff.state_hash
    assert normal.train_scores == poisoned.train_scores
    a = normal.oos_evaluations["selected_oos"].get_metric("rank_ic", "selected_oos").value
    b = poisoned.oos_evaluations["selected_oos"].get_metric("rank_ic", "selected_oos").value
    assert a == pytest.approx(-b, abs=.03)


@pytest.mark.parametrize("kind", ["small", "ties"])
def test_infeasible_q20_explicitly_degrades_without_tail_search(kind):
    result = run_e2e_c_tail_rescue(kind)
    assert result.policy_branch == "DEGRADED_NO_Q20_REPAIR"
    assert result.selected_q in {10, 5, None}
    assert result.frozen_cutoff is None
    assert result.candidate_dsls == {}
    assert result.oos_evaluations == {}
    assert result.trace["verdict"] == "Q20_INFEASIBLE_NO_REPAIR"


def test_missing_public_risk_evidence_fails_closed_before_candidate_search():
    result = run_e2e_c_tail_rescue("missing_risk")
    assert result.policy_branch == "Q20_RISK_EVIDENCE_INSUFFICIENT_NO_REPAIR"
    assert result.selected_q == 20
    assert result.train_risk_evidence["liquidity_exposure"] is None
    assert result.frozen_cutoff is None
    assert result.train_scores == {}
    assert result.candidate_dsls == {}
    assert result.oos_evaluations == {}
    assert result.trace["pipeline_status"] == "PARTIAL_NOT_PUBLISHED"
