"""A/B regressions: invalid evidence must never select a candidate."""
from dataclasses import replace
import numpy as np
import pytest
from factor_optimizer.search.paired_comparison import (
    ComparisonStatus, ComparisonThresholds, PairedDraws, compare_paired_draws,
)

def evidence():
    return PairedDraws("candidate", "raw", ("a", "b", "c"),
                       {"ic": (.2, .2, .2)}, {"ic": (.1, .1, .1)}, "train")

def compare(value, utility=lambda m: m["ic"]):
    return compare_paired_draws(value, ComparisonThresholds(0, .01, .01),
                               utility=utility, expected_context_identity="train")

def test_identical_metrics_are_not_an_improvement_at_zero_threshold():
    value = evidence()
    assert compare(replace(value, candidate_metrics=value.baseline_metrics)).status is ComparisonStatus.EQUIVALENT

def test_empty_metrics_cannot_prove_superiority():
    assert compare(replace(evidence(), candidate_metrics={}, baseline_metrics={}),
                   utility=lambda m: 1).status is ComparisonStatus.INVALID_CONTEXT

@pytest.mark.parametrize("bad", ["bad", None, float("nan"), float("inf")])
def test_bad_metric_is_rejected_even_when_utility_ignores_it(bad):
    value = replace(evidence(), candidate_metrics={"ic": (.2, bad, .2)})
    assert compare(value, utility=lambda m: 1).status is ComparisonStatus.INVALID_CONTEXT

@pytest.mark.parametrize("cost", [(1,), (1, float("nan"), 1), (1, float("inf"), 1), (1, "bad", 1)])
def test_superior_metrics_do_not_bypass_cost_validation(cost):
    value = replace(evidence(), candidate_cost=cost, baseline_cost=(2, 2, 2))
    assert compare(value).status is ComparisonStatus.INVALID_CONTEXT

def test_numpy_cost_draws_support_noninferior_cheaper_comparison():
    value = replace(evidence(), candidate_metrics={"ic": (.099, .099, .099)},
                    candidate_cost=np.array([1., 1., 1.]), baseline_cost=np.array([2., 2., 2.]))
    assert compare(value).status is ComparisonStatus.NON_INFERIOR_CHEAPER

def test_equal_costs_are_not_cheaper_at_zero_threshold():
    value = evidence()
    value = replace(value, candidate_metrics=value.baseline_metrics,
                    candidate_cost=(1, 1, 1), baseline_cost=(1, 1, 1))
    assert compare(value).status is ComparisonStatus.EQUIVALENT
