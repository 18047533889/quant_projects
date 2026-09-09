import pytest

from factor_optimizer.search.paired_comparison import (
    ComparisonStatus, ComparisonThresholds, PairedDraws, compare_paired_draws,
)


def _thresholds(**changes):
    values = dict(minimum_improvement=.001, maximum_noninferiority_loss=.002,
                  equivalence_bound=.001, minimum_cost_improvement=.1,
                  confidence_level=.90)
    values.update(changes)
    return ComparisonThresholds(**values)


def test_common_draw_nonlinear_utility_is_recomputed_and_improvement_passes():
    evidence = PairedDraws("new", "old", ("d1", "d2", "d3", "d4"),
        {"mean": (.03, .04, .05, .06), "risk": (.01, .01, .02, .02)},
        {"mean": (.02, .02, .03, .03), "risk": (.01, .01, .02, .02)}, "ctx")
    result = compare_paired_draws(evidence, _thresholds(),
        utility=lambda m: m["mean"] / (1 + m["risk"] ** 2),
        expected_context_identity="ctx")
    assert result.status is ComparisonStatus.SUPERIOR
    assert (result.wins, result.losses, result.ties) == (4, 0, 0)


def test_all_equal_draws_are_strict_ties_and_equivalent():
    evidence = PairedDraws("new", "old", ("a", "b", "c"),
                           {"x": (1., 1., 1.)}, {"x": (1., 1., 1.)}, "ctx")
    result = compare_paired_draws(evidence, _thresholds(), utility=lambda m: m["x"],
                                  expected_context_identity="ctx")
    assert result.status is ComparisonStatus.EQUIVALENT
    assert (result.wins, result.losses, result.ties) == (0, 0, 3)


def test_ci_overlap_is_not_equivalence_and_draw_context_fails_closed():
    evidence = PairedDraws("new", "old", ("a", "b", "c", "d"),
                           {"x": (-.02, .02, -.02, .02)}, {"x": (0, 0, 0, 0)}, "ctx")
    result = compare_paired_draws(evidence, _thresholds(), utility=lambda m: m["x"],
                                  expected_context_identity="ctx")
    assert result.status in {ComparisonStatus.TRADE_OFF, ComparisonStatus.INCONCLUSIVE}
    invalid = compare_paired_draws(evidence, _thresholds(), utility=lambda m: m["x"],
                                   expected_context_identity="other")
    assert invalid.status is ComparisonStatus.INVALID_CONTEXT


def test_signed_noninferior_cheaper():
    evidence = PairedDraws("new", "old", ("a", "b", "c"),
                           {"x": (.019, .019, .019)}, {"x": (.02, .02, .02)}, "ctx",
                           candidate_cost=(1., 1., 1.), baseline_cost=(2., 2., 2.))
    result = compare_paired_draws(evidence, _thresholds(), utility=lambda m: m["x"],
                                  expected_context_identity="ctx")
    assert result.status is ComparisonStatus.NON_INFERIOR_CHEAPER
