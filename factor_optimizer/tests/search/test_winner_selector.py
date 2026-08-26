"""Tests for the robust winner selector and Pareto frontier helpers."""

import math

import pytest

from factor_optimizer.search.pareto import ParetoPoint, ParetoFrontier
from factor_optimizer.search.winner_selector import (
    WinnerPolicy,
    RobustBalancedUtility,
    augmented_tchebycheff,
    select_winner,
)


# ---------------------------------------------------------------------------
# Pareto frontier helpers
# ---------------------------------------------------------------------------


def test_pareto_dominance():
    """A dominates C on every dim; A and B each dominate the other on some dim."""
    frontier = ParetoFrontier()

    a = ParetoPoint(trial_id="A", objectives=(0.9, 0.8, 0.7))
    b = ParetoPoint(trial_id="B", objectives=(0.7, 0.9, 0.8))
    c = ParetoPoint(trial_id="C", objectives=(0.5, 0.4, 0.3))

    assert frontier.add(a) is True
    assert frontier.add(b) is True
    # C is dominated by A (A higher on every dim) -> does not survive.
    assert frontier.add(c) is False

    # A and B both survive on the frontier (each beats the other on some dim).
    survivors = {p.trial_id for p in frontier.frontier()}
    assert survivors == {"A", "B"}

    # dominated() reports exactly the points that were excluded.
    dominated = frontier.dominated([a, b, c])
    assert dominated == {"C"}


# ---------------------------------------------------------------------------
# RobustBalancedUtility
# ---------------------------------------------------------------------------


def _policy(**overrides):
    defaults = dict(
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        lambda_=0.1,
        policy_id="test",
        policy_version="1.0.0",
    )
    defaults.update(overrides)
    return WinnerPolicy(**defaults)


def test_high_rank_ic_cannot_compensate_terrible_tradability():
    """A candidate with a terrible dimension loses to a balanced one.

    [0.98, 0.95, 0.15] has great predictive power but terrible tradability.
    Under RobustBalancedUtility the min-term drags it down, so the balanced
    [0.85, 0.82, 0.80] wins — RankIC high cannot compensate terrible Turnover.
    """
    policy = _policy()
    unbalanced = RobustBalancedUtility(
        dimension_desirabilities=[0.98, 0.95, 0.15],
        robustness_score=0.5,
        complexity_score=0.3,
        policy=policy,
    )
    balanced = RobustBalancedUtility(
        dimension_desirabilities=[0.85, 0.82, 0.80],
        robustness_score=0.5,
        complexity_score=0.3,
        policy=policy,
    )
    assert balanced > unbalanced


def test_complexity_tie_break():
    """Two statistically-indistinguishable candidates -> simpler one wins."""
    policy = _policy()
    # Same dimensions, same robustness; only complexity differs.
    complex_recipe = RobustBalancedUtility(
        dimension_desirabilities=[0.8, 0.8, 0.8],
        robustness_score=0.5,
        complexity_score=0.9,
        policy=policy,
    )
    simple_recipe = RobustBalancedUtility(
        dimension_desirabilities=[0.8, 0.8, 0.8],
        robustness_score=0.5,
        complexity_score=0.1,
        policy=policy,
    )
    # Utility differs only by the complexity penalty.
    assert simple_recipe > complex_recipe

    # End-to-end: select_winner picks the simpler one when utilities are within
    # tolerance.
    candidates = [
        ParetoPoint(trial_id="complex", objectives=(0.8, 0.8, 0.8)),
        ParetoPoint(trial_id="simple", objectives=(0.8, 0.8, 0.8)),
    ]
    robustness = {"complex": 0.5, "simple": 0.5}
    complexity = {"complex": 0.9, "simple": 0.1}
    winner = select_winner(candidates, robustness, complexity, policy)
    assert winner.trial_id == "simple"


def test_policy_versioned():
    """Changing a weight in a different policy version flips the winner."""
    # Two candidates trade off: X has a higher min-dimension (0.70 vs 0.50)
    # but Y has a higher geomean (0.788 vs 0.700).  So a min-heavy policy
    # (alpha) favors X, while a geomean-heavy policy (beta) favors Y.
    x = ParetoPoint(trial_id="X", objectives=(0.70, 0.70, 0.70))
    y = ParetoPoint(trial_id="Y", objectives=(0.99, 0.99, 0.50))
    candidates = [x, y]
    robustness = {"X": 0.5, "Y": 0.5}
    complexity = {"X": 0.3, "Y": 0.3}

    # Version 1.0.0: alpha (min-weight) dominates -> X's higher min wins.
    v1 = _policy(alpha=0.8, beta=0.1, gamma=0.05, lambda_=0.05,
                 policy_id="p", policy_version="1.0.0")
    # Version 2.0.0: beta (geomean-weight) dominates -> Y's higher geomean wins.
    v2 = _policy(alpha=0.1, beta=0.8, gamma=0.05, lambda_=0.05,
                 policy_id="p", policy_version="2.0.0")

    winner_v1 = select_winner(candidates, robustness, complexity, v1)
    winner_v2 = select_winner(candidates, robustness, complexity, v2)

    assert winner_v1.trial_id == "X"
    assert winner_v2.trial_id == "Y"
    assert v1.policy_version != v2.policy_version


def test_geomean_guards_nonpositive():
    """Geomean of a set containing a 0.0 dimension does not crash / return NaN."""
    policy = _policy()
    u = RobustBalancedUtility(
        dimension_desirabilities=[0.9, 0.0, 0.8],
        robustness_score=0.5,
        complexity_score=0.3,
        policy=policy,
    )
    assert math.isfinite(u)


# ---------------------------------------------------------------------------
# WinnerPolicy validation
# ---------------------------------------------------------------------------


def test_policy_rejects_negative_weights():
    with pytest.raises(ValueError):
        _policy(alpha=-0.1)


def test_policy_requires_version():
    with pytest.raises(ValueError):
        WinnerPolicy(
            alpha=0.4, beta=0.3, gamma=0.2, lambda_=0.1,
            policy_id="p", policy_version="",
        )


# ---------------------------------------------------------------------------
# Fail-closed input validation
# ---------------------------------------------------------------------------


def test_missing_robustness_score_fails_closed():
    """A frontier candidate with NO robustness entry must raise, not default 0."""
    policy = _policy()
    candidates = [ParetoPoint(trial_id="A", objectives=(0.8, 0.8, 0.8))]
    robustness = {}  # missing
    complexity = {"A": 0.3}
    with pytest.raises(KeyError, match="robustness"):
        select_winner(candidates, robustness, complexity, policy)


def test_missing_complexity_score_fails_closed():
    """A frontier candidate with NO complexity entry must raise, not default 0."""
    policy = _policy()
    candidates = [ParetoPoint(trial_id="A", objectives=(0.8, 0.8, 0.8))]
    robustness = {"A": 0.5}
    complexity = {}  # missing
    with pytest.raises(KeyError, match="complexity"):
        select_winner(candidates, robustness, complexity, policy)


def test_missing_score_fails_closed_before_sort_and_utility():
    """Missing score is rejected even when utility/tie-break would be reached.

    Regression guard: a candidate with a MISSING complexity must not reach the
    utility computation, sort key, or tie-break (which used to default to 0.0).
    """
    policy = _policy()
    candidates = [
        ParetoPoint(trial_id="A", objectives=(0.8, 0.8, 0.8)),
        ParetoPoint(trial_id="B", objectives=(0.9, 0.9, 0.9)),
    ]
    robustness = {"A": 0.5, "B": 0.5}
    complexity = {"A": 0.3}  # B missing
    with pytest.raises(KeyError, match="complexity"):
        select_winner(candidates, robustness, complexity, policy)


def test_nan_robustness_rejected():
    policy = _policy()
    with pytest.raises(ValueError, match="robustness"):
        RobustBalancedUtility(
            [0.8, 0.8, 0.8], math.nan, 0.3, policy
        )


def test_complexity_out_of_range_rejected():
    """complexity 1.2 is outside [0, 1] -> must raise, not be clamped."""
    policy = _policy()
    with pytest.raises(ValueError, match="complexity.*[0, 1]"):
        RobustBalancedUtility(
            [0.8, 0.8, 0.8], 0.5, 1.2, policy
        )


def test_dimension_desirability_out_of_range_rejected():
    """dimension -0.1 and 1.3 are outside [0, 1] -> must raise."""
    policy = _policy()
    with pytest.raises(ValueError, match="dimension_desirabilities"):
        RobustBalancedUtility(
            [-0.1, 0.8, 0.8], 0.5, 0.3, policy
        )
    with pytest.raises(ValueError, match="dimension_desirabilities"):
        RobustBalancedUtility(
            [1.3, 0.8, 0.8], 0.5, 0.3, policy
        )


def test_nan_dimension_rejected():
    """A NaN dimension must be rejected (would otherwise poison min/log)."""
    policy = _policy()
    with pytest.raises(ValueError, match="dimension_desirabilities"):
        RobustBalancedUtility(
            [0.8, math.nan, 0.8], 0.5, 0.3, policy
        )


def test_inf_robustness_rejected():
    policy = _policy()
    with pytest.raises(ValueError, match="robustness"):
        RobustBalancedUtility(
            [0.8, 0.8, 0.8], math.inf, 0.3, policy
        )


def test_augmented_tchebycheff_also_validates():
    """The alternative metric shares the same strict fail-closed validator."""
    policy = _policy()
    with pytest.raises(ValueError, match="dimension_desirabilities"):
        augmented_tchebycheff([0.8, -0.2, 0.8], 0.5, 0.3, policy)
    with pytest.raises(ValueError, match="complexity"):
        augmented_tchebycheff([0.8, 0.8, 0.8], 0.5, math.inf, policy)


def test_valid_happy_path_selects_correct_winner():
    """A fully-valid input still selects the intended winner (fail-open intact)."""
    policy = _policy()
    candidates = [
        ParetoPoint(trial_id="low", objectives=(0.5, 0.5, 0.5)),
        ParetoPoint(trial_id="high", objectives=(0.9, 0.9, 0.9)),
    ]
    robustness = {"low": 0.4, "high": 0.9}
    complexity = {"low": 0.2, "high": 0.1}
    winner = select_winner(candidates, robustness, complexity, policy)
    assert winner.trial_id == "high"
