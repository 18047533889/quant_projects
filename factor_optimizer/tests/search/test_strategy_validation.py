"""Strategy search-space validation tests (FO-P0-07).

Covers fail-closed validation of search spaces and strategy parameters:

- ``ParameterSpace`` / ``SearchSpace`` reject invalid definitions at
  construction (bounds ordering, empty ranges, non-numeric bounds, empty
  categorical choices, unknown kinds, duplicate names, log_scale misuse)
  instead of producing NaN / silently wrong results.
- Each strategy (RandomSearch / GridSearch / BayesianSearch / TPESearch)
  validates its own constructor parameters.
- BayesianSearch and TPESearch reject categorical (``choice``) parameters
  fail-closed (FO-P0-03) — a categorical param never produces a garbage float
  proposal.
- Valid spaces still work and remain seed-reproducible.
"""

import pytest

from factor_optimizer.search.strategies import (
    BayesianSearch,
    GridSearch,
    ParameterSpace,
    RandomSearch,
    SearchSpace,
    TPESearch,
)


def _space(*params):
    return SearchSpace(parameters=list(params))


FLOAT = ParameterSpace(name="x", kind="float", low=-5.0, high=5.0)
INT = ParameterSpace(name="n", kind="int", low=0, high=10)
CAT = ParameterSpace(name="c", kind="choice", choices=["a", "b", "c"])
NUMERIC_SPACE = _space(FLOAT, INT)


# ---------------------------------------------------------------------------
# ParameterSpace fail-closed validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["float", "int"])
def test_param_space_rejects_low_ge_high(kind):
    with pytest.raises(ValueError, match="low < high"):
        ParameterSpace(name="x", kind=kind, low=5.0, high=5.0)
    with pytest.raises(ValueError, match="low < high"):
        ParameterSpace(name="x", kind=kind, low=6.0, high=5.0)


@pytest.mark.parametrize("kind", ["float", "int"])
def test_param_space_rejects_non_numeric_bounds(kind):
    with pytest.raises(ValueError, match="low"):
        ParameterSpace(name="x", kind=kind, low="a", high=5.0)
    with pytest.raises(ValueError, match="high"):
        ParameterSpace(name="x", kind=kind, low=1.0, high="b")
    with pytest.raises(ValueError, match="low"):
        ParameterSpace(name="x", kind=kind, low=True, high=5.0)


def test_param_space_rejects_empty_int_range():
    with pytest.raises(ValueError, match="low < high"):
        ParameterSpace(name="n", kind="int", low=4, high=4)


def test_param_space_rejects_missing_bounds():
    with pytest.raises(ValueError, match="low"):
        ParameterSpace(name="x", kind="float", high=5.0)


def test_param_space_rejects_empty_choices():
    with pytest.raises(ValueError, match="choices"):
        ParameterSpace(name="c", kind="choice", choices=[])
    with pytest.raises(ValueError, match="choices"):
        ParameterSpace(name="c", kind="choice", choices=None)


def test_param_space_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown parameter kind"):
        ParameterSpace(name="x", kind="logistic", low=1.0, high=2.0)


def test_param_space_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        ParameterSpace(name="", kind="float", low=0.0, high=1.0)


def test_param_space_log_scale_requires_positive_bounds():
    with pytest.raises(ValueError, match="low > 0"):
        ParameterSpace(name="x", kind="float", low=0.0, high=5.0, log_scale=True)
    with pytest.raises(ValueError, match="low > 0"):
        ParameterSpace(name="x", kind="float", low=-1.0, high=5.0, log_scale=True)
    # A non-positive high is also rejected fail-closed (low < high implies a
    # positive high once low > 0, so the high guard is defense-in-depth).
    with pytest.raises(ValueError):
        ParameterSpace(name="x", kind="float", low=1.0, high=0.5, log_scale=True)
    with pytest.raises(ValueError, match="low > 0"):
        ParameterSpace(name="n", kind="int", low=0, high=10, log_scale=True)


def test_param_space_log_scale_not_applicable_to_choice():
    with pytest.raises(ValueError, match="log_scale"):
        ParameterSpace(
            name="c", kind="choice", choices=["a", "b"], log_scale=True
        )


# ---------------------------------------------------------------------------
# SearchSpace fail-closed validation
# ---------------------------------------------------------------------------

def test_search_space_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        _space(FLOAT, ParameterSpace(name="x", kind="int", low=0, high=5))


def test_search_space_rejects_non_parameter_elements():
    with pytest.raises(ValueError, match="ParameterSpace"):
        SearchSpace(parameters=[{"name": "x"}])
    with pytest.raises(ValueError, match="ParameterSpace"):
        SearchSpace(parameters="x")


def test_search_space_valid_space_samples():
    point = NUMERIC_SPACE.sample()
    assert set(point) == {"x", "n"}


# ---------------------------------------------------------------------------
# Strategy-constructor validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
def test_grid_search_rejects_invalid_grid_points(bad):
    with pytest.raises(ValueError, match="grid_points"):
        GridSearch(NUMERIC_SPACE, grid_points=bad)


@pytest.mark.parametrize("bad", [0, -3, 1.5, "4"])
def test_grid_search_rejects_invalid_max_combinations(bad):
    with pytest.raises(ValueError, match="max_combinations"):
        GridSearch(NUMERIC_SPACE, max_combinations=bad)


@pytest.mark.parametrize("bad", [-1, 1.5, "5", True])
def test_bayesian_rejects_invalid_n_initial(bad):
    with pytest.raises(ValueError, match="n_initial"):
        BayesianSearch(NUMERIC_SPACE, n_initial=bad)


@pytest.mark.parametrize("bad", ["maximize", "UCB", "", 1, None])
def test_bayesian_rejects_invalid_acquisition(bad):
    with pytest.raises(ValueError, match="acquisition"):
        BayesianSearch(NUMERIC_SPACE, acquisition=bad)


@pytest.mark.parametrize("bad", ["big", float("nan"), float("inf"), True])
def test_bayesian_rejects_invalid_ucb_kappa(bad):
    with pytest.raises(ValueError, match="ucb_kappa"):
        BayesianSearch(NUMERIC_SPACE, ucb_kappa=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, "0.25", True])
def test_tpe_rejects_invalid_gamma(bad):
    with pytest.raises(ValueError, match="gamma"):
        TPESearch(NUMERIC_SPACE, gamma=bad)


@pytest.mark.parametrize("bad", [0, -2, 1.5, "24"])
def test_tpe_rejects_invalid_n_candidates(bad):
    with pytest.raises(ValueError, match="n_candidates"):
        TPESearch(NUMERIC_SPACE, n_candidates=bad)


@pytest.mark.parametrize("bad", [-1, 1.5, "5", True])
def test_tpe_rejects_invalid_n_initial(bad):
    with pytest.raises(ValueError, match="n_initial"):
        TPESearch(NUMERIC_SPACE, n_initial=bad)


def test_tpe_rejects_invalid_maximize():
    with pytest.raises(ValueError, match="maximize"):
        TPESearch(NUMERIC_SPACE, maximize="yes")


# ---------------------------------------------------------------------------
# Categorical handling for Bayesian / TPE: fail-closed (FO-P0-03)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_cls", [BayesianSearch, TPESearch])
def test_numeric_kernel_strategies_reject_categorical_params(strategy_cls):
    space = _space(FLOAT, CAT)
    with pytest.raises(ValueError, match="categorical parameter 'c'"):
        strategy_cls(space, n_initial=2)


@pytest.mark.parametrize("strategy_cls", [BayesianSearch, TPESearch])
def test_numeric_kernel_strategies_reject_choice_only_space(strategy_cls):
    with pytest.raises(ValueError, match="categorical"):
        strategy_cls(_space(CAT))


@pytest.mark.parametrize("strategy_cls", [RandomSearch, GridSearch])
def test_non_numeric_strategies_still_accept_categorical(strategy_cls):
    strategy = strategy_cls(_space(FLOAT, CAT))
    trial = strategy.propose()
    assert trial.metadata["params"]["c"] in ("a", "b", "c")


def test_to_array_fails_closed_on_categorical_space():
    space = _space(FLOAT, CAT)
    with pytest.raises(ValueError, match="numeric-only"):
        space.to_array({"x": 1.0, "c": "a"})


def test_bayesian_record_with_categorical_space_cannot_happen():
    # Construction already fails; asserting the fail-closed gate exists.
    with pytest.raises(ValueError, match="categorical"):
        BayesianSearch(_space(FLOAT, CAT))
    # And a numeric-only space records fine.
    bayes = BayesianSearch(_space(FLOAT), n_initial=1)
    bayes.record({"x": 1.0}, 0.5)
    assert len(bayes._X) == 1


# ---------------------------------------------------------------------------
# Valid spaces still work and stay deterministic
# ---------------------------------------------------------------------------

def _trajectory(strategy_cls, space, n, **kwargs):
    strategy = strategy_cls(space, seed=7, **kwargs)
    return [strategy.propose().metadata["params"] for _ in range(n)]


def test_valid_spaces_still_propose():
    rng = RandomSearch(NUMERIC_SPACE, seed=3)
    for _ in range(5):
        params = rng.propose().metadata["params"]
        assert -5.0 <= params["x"] <= 5.0
        assert 0 <= params["n"] <= 10

    grid = GridSearch(NUMERIC_SPACE, grid_points=3)
    for _ in range(grid.total_combinations):
        params = grid.propose().metadata["params"]
        assert isinstance(params, dict) and "x" in params


@pytest.mark.parametrize(
    "strategy_cls",
    [RandomSearch, BayesianSearch, TPESearch],
)
def test_numeric_strategies_deterministic(strategy_cls):
    space = _space(FLOAT, INT)
    if strategy_cls is RandomSearch:
        left = _trajectory(strategy_cls, space, 6)
        right = _trajectory(strategy_cls, space, 6)
    else:
        left = _trajectory(strategy_cls, space, 6, n_initial=3)
        right = _trajectory(strategy_cls, space, 6, n_initial=3)
    assert left == right


def test_bayesian_proposals_are_finite_floats():
    bayes = BayesianSearch(_space(FLOAT, INT), n_initial=3, seed=5)
    for _ in range(4):
        bayes.record(bayes.propose().metadata["params"], 1.0)
    for _ in range(4):
        params = bayes.propose().metadata["params"]
        assert isinstance(params["x"], float)
        assert params["x"] == params["x"]  # not NaN
        assert isinstance(params["n"], int)
