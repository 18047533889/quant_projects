from factor_optimizer.search.pareto import ParetoPoint
from factor_optimizer.search.winner_selector import (
    WinnerPolicy, WinnerSetPolicy, select_complementary_winners,
)


UTILITY = WinnerPolicy(0.4, 0.4, 0.2, 0.1, "winner", "1")
SET = WinnerSetPolicy(max_winners=3, max_per_family=1, utility_epsilon=1e-4)


def _point(name, objectives, **metadata):
    return ParetoPoint(name, objectives, metadata)


def test_missing_core_or_gate_evidence_yields_legal_empty_set():
    points = [_point("unknown", (0.9, 0.9), eligible=True)]
    assert select_complementary_winners(points, {"unknown": 0.8}, {"unknown": 0.1}, UTILITY, SET) == []


def test_near_duplicate_family_and_value_keep_stable_simple_representative():
    points = [
        _point("complex", (0.80001, 0.8), eligible=True, incremental_value=0.1, family_id="parent"),
        _point("simple", (0.8, 0.8), eligible=True, incremental_value=0.1, family_id="parent"),
    ]
    selected = select_complementary_winners(
        reversed(points), {p.trial_id: 0.8 for p in points},
        {"complex": 0.8, "simple": 0.1}, UTILITY, SET,
    )
    assert [p.trial_id for p in selected] == ["simple"]


def test_complementary_purposes_can_both_survive_but_correlated_duplicate_cannot():
    points = [
        _point("predictive", (0.9, 0.5), eligible=True, incremental_value=0.2, family_id="alpha", purpose="prediction"),
        _point("low_turnover", (0.6, 0.9), eligible=True, incremental_value=0.15, family_id="cost", purpose="cost"),
        _point("alias", (0.89, 0.5), eligible=True, incremental_value=0.19, family_id="alias", purpose="prediction"),
    ]
    selected = select_complementary_winners(
        points, {p.trial_id: 0.8 for p in points}, {p.trial_id: 0.1 for p in points},
        UTILITY, SET, pairwise_correlations={("predictive", "alias"): 0.999},
    )
    assert {p.trial_id for p in selected} == {"predictive", "low_turnover"}
