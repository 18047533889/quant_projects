"""FO-P0-03 / FO-P1-01..04 regression tests.

Covers:
- FO-P0-03: ObjectiveSpec is the ONLY direction authority.  Strategies derive
  their direction from the spec/context; Bayesian/TPE maximize utility
  (direction_sign * raw_metric); runner/strategy direction mismatch is
  impossible.
- FO-P1-02: single central proposal counter owned by the base strategy.
- FO-P1-01: deterministic trial identity (no uuid4).
- FO-P1-03: int bounds must be integral; SearchSpace.validate_point.
- FO-P1-04: lazy GridSearch (no full Cartesian product materialization).
"""

import pytest

from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.search.runner import SearchConfig, SearchRunner
from factor_optimizer.search.strategies import (
    BayesianSearch,
    GridSearch,
    ParameterSpace,
    RandomSearch,
    SearchSpace,
    StrategyContext,
    TPESearch,
)


def _space(*params):
    return SearchSpace(parameters=list(params))


FLOAT = ParameterSpace(name="x", kind="float", low=-5.0, high=5.0)
INT = ParameterSpace(name="n", kind="int", low=0, high=10)
CAT = ParameterSpace(name="c", kind="choice", choices=["a", "b", "c"])
NUMERIC = _space(FLOAT, INT)
MIXED = _space(FLOAT, INT, CAT)


def _protocol(fn):
    return EvaluationProtocol(SplitPlan("search", [True], [False], [False], {}), fn)


def _config(**overrides):
    budget = SearchBudget(max_trials=50, max_evaluations=50, max_cost_units=500.0)
    return SearchConfig(budget=budget, **overrides)


# ---------------------------------------------------------------------------
# FO-P0-03: single direction authority
# ---------------------------------------------------------------------------

def test_strategy_direction_derived_from_spec():
    for cls in (RandomSearch, GridSearch, BayesianSearch, TPESearch):
        space = NUMERIC if cls in (BayesianSearch, TPESearch) else MIXED
        strat = cls(space, seed=1, objective_spec=ObjectiveSpec("loss", "minimize"))
        assert strat.direction == "minimize"
        assert strat.objective_spec == ObjectiveSpec("loss", "minimize")
        strat_max = cls(space, seed=1, objective_spec=ObjectiveSpec("rank_ic", "maximize"))
        assert strat_max.direction == "maximize"


def test_strategy_context_carries_spec():
    ctx = StrategyContext(objective_spec=ObjectiveSpec("loss", "minimize"))
    strat = RandomSearch(MIXED, seed=1, context=ctx)
    assert strat.direction == "minimize"
    assert strat._direction_sign == -1
    assert ctx.to_utility(0.5) == -0.5
    assert ctx.to_utility(-0.5) == 0.5


def test_bayesian_maximizes_utility():
    b = BayesianSearch(NUMERIC, n_initial=2, seed=1, objective_spec=ObjectiveSpec("loss", "minimize"))
    b.record({"x": 1.0, "n": 3}, 0.5)
    b.record({"x": 2.0, "n": 4}, 0.1)
    # Utility is direction_sign * raw_metric; the surrogate's "best" is the
    # max utility, so the lower raw loss (0.1) is the better utility (-0.1).
    assert b._y == [-0.5, -0.1]
    assert max(b._y) == -0.1


def test_tpe_maximizes_utility():
    t = TPESearch(NUMERIC, n_initial=2, seed=1, objective_spec=ObjectiveSpec("loss", "minimize"))
    t.record({"x": 1.0, "n": 3}, 0.5)
    t.record({"x": 2.0, "n": 4}, 0.1)
    # History stores utility; the "good" partition is always the highest
    # utility (lowest raw loss for minimize).
    assert t._history == [({"x": 1.0, "n": 3}, -0.5), ({"x": 2.0, "n": 4}, -0.1)]


def test_tpe_legacy_maximize_flag_backward_compatible():
    t = TPESearch(NUMERIC, n_initial=2, seed=1, maximize=True)
    assert t.direction == "maximize"
    t_min = TPESearch(NUMERIC, n_initial=2, seed=1, maximize=False)
    assert t_min.direction == "minimize"
    # Default (no flag) is maximize, matching the runner's default spec.
    t_default = TPESearch(NUMERIC, n_initial=2, seed=1)
    assert t_default.direction == "maximize"


def test_tpe_maximize_flag_cannot_disagree_with_spec():
    with pytest.raises(ValueError, match="disagree"):
        TPESearch(
            NUMERIC,
            n_initial=2,
            seed=1,
            maximize=True,
            objective_spec=ObjectiveSpec("loss", "minimize"),
        )


def test_runner_rejects_strategy_direction_mismatch():
    strat = RandomSearch(MIXED, seed=1, objective_spec=ObjectiveSpec("loss", "minimize"))
    config = _config(objective_spec=ObjectiveSpec("score", "maximize"))
    with pytest.raises(ValueError, match="single direction authority"):
        SearchRunner(
            config,
            strat.proposal_fn,
            _protocol(lambda trial, fidelity: {}),
            strategy=strat,
        )


def test_runner_accepts_matching_strategy_direction():
    strat = RandomSearch(MIXED, seed=1, objective_spec=ObjectiveSpec("score", "maximize"))
    config = _config(objective_spec=ObjectiveSpec("score", "maximize"))
    runner = SearchRunner(
        config,
        strat.proposal_fn,
        _protocol(lambda trial, fidelity: {}),
        strategy=strat,
    )
    assert runner.strategy.direction == "maximize"


def test_bayesian_minimize_and_maximize_both_run():
    for direction in ("maximize", "minimize"):
        strat = BayesianSearch(
            NUMERIC, n_initial=2, seed=3, objective_spec=ObjectiveSpec("score", direction)
        )
        config = _config(objective_spec=ObjectiveSpec("score", direction))
        runner = SearchRunner(
            config,
            strat.proposal_fn,
            _protocol(lambda trial, fidelity: {
                "evaluation_id": trial.trial_id,
                "score": 1.0,
                "cost": 1.0,
            }),
            strategy=strat,
        )
        session = runner.run("bayes-" + direction)
        assert session.is_finished()


def test_tpe_minimize_and_maximize_both_run():
    for direction in ("maximize", "minimize"):
        strat = TPESearch(
            NUMERIC, n_initial=2, seed=3, objective_spec=ObjectiveSpec("score", direction)
        )
        config = _config(objective_spec=ObjectiveSpec("score", direction))
        runner = SearchRunner(
            config,
            strat.proposal_fn,
            _protocol(lambda trial, fidelity: {
                "evaluation_id": trial.trial_id,
                "score": 1.0,
                "cost": 1.0,
            }),
            strategy=strat,
        )
        session = runner.run("tpe-" + direction)
        assert session.is_finished()


# ---------------------------------------------------------------------------
# FO-P1-02: single central proposal counter
# ---------------------------------------------------------------------------

def test_central_proposal_counter_increments_once_per_proposal():
    for cls in (RandomSearch, GridSearch, BayesianSearch, TPESearch):
        space = NUMERIC if cls in (BayesianSearch, TPESearch) else MIXED
        kwargs = {}
        if cls is GridSearch:
            kwargs = {"grid_points": 3, "max_combinations": 8}
        if cls in (BayesianSearch, TPESearch):
            kwargs = {"n_initial": 2}
        strat = cls(space, seed=1, **kwargs)
        assert strat.proposal_sequence == 0
        for expected in range(1, 5):
            strat.propose()
            assert strat.proposal_sequence == expected


def test_central_counter_restored_on_load_state():
    strat = RandomSearch(MIXED, seed=1)
    for _ in range(3):
        strat.propose()
    state = strat.save_state()
    assert state.iteration_count == 3
    restored = RandomSearch(MIXED, seed=1)
    restored.load_state(state)
    assert restored.proposal_sequence == 3


# ---------------------------------------------------------------------------
# FO-P1-01: deterministic trial identity
# ---------------------------------------------------------------------------

def test_trial_identity_is_deterministic():
    a = RandomSearch(MIXED, seed=5)
    b = RandomSearch(MIXED, seed=5)
    ta = [a.propose().trial_id for _ in range(4)]
    tb = [b.propose().trial_id for _ in range(4)]
    assert ta == tb
    assert all(len(tid) == 32 for tid in ta)  # sha256 hex prefix


def test_trial_identity_differs_across_seeds():
    a = RandomSearch(MIXED, seed=5)
    b = RandomSearch(MIXED, seed=6)
    ta = [a.propose().trial_id for _ in range(4)]
    tb = [b.propose().trial_id for _ in range(4)]
    assert ta != tb


def test_trial_identity_is_not_uuid4():
    strat = RandomSearch(MIXED, seed=5)
    tid = strat.propose().trial_id
    # uuid4 hex is 32 chars of lowercase hex but the identity is a sha256
    # digest; assert it is not a random uuid4 by checking determinism across
    # two identical strategies (already covered) and that it is hex.
    assert all(c in "0123456789abcdef" for c in tid)


# ---------------------------------------------------------------------------
# FO-P1-03: int bounds integral + validate_point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("low,high", [(0.5, 10), (0, 10.5), (1.2, 3.4)])
def test_int_param_rejects_non_integral_bounds(low, high):
    with pytest.raises(ValueError, match="integral"):
        ParameterSpace(name="n", kind="int", low=low, high=high)


def test_int_param_accepts_integral_float_bounds():
    p = ParameterSpace(name="n", kind="int", low=0.0, high=10.0)
    assert p.low == 0.0 and p.high == 10.0


def test_validate_point_accepts_valid_point():
    MIXED.validate_point({"x": 1.0, "n": 3, "c": "a"})


@pytest.mark.parametrize(
    "bad",
    [
        {"x": 1.0, "n": 3},  # missing key
        {"x": 1.0, "n": 3, "c": "a", "extra": 1},  # extra key
        {"x": 1.0, "n": 3.5, "c": "a"},  # non-integral int
        {"x": 100.0, "n": 3, "c": "a"},  # out of bounds
        {"x": 1.0, "n": 3, "c": "z"},  # bad choice
        {"x": float("nan"), "n": 3, "c": "a"},  # non-finite
    ],
)
def test_validate_point_rejects_invalid_points(bad):
    with pytest.raises(ValueError):
        MIXED.validate_point(bad)


def test_validate_point_rejects_non_dict():
    with pytest.raises(ValueError, match="dict"):
        MIXED.validate_point([1, 2, 3])


# ---------------------------------------------------------------------------
# FO-P1-04: lazy GridSearch
# ---------------------------------------------------------------------------

def test_grid_search_does_not_materialize_full_product():
    # 8 float params x 10 grid points = 10^8 combinations.
    space = _space(*[ParameterSpace(name=f"p{i}", kind="float", low=0.0, high=1.0) for i in range(8)])
    g = GridSearch(space, grid_points=10, max_combinations=1000, seed=1)
    assert g.total_combinations == 100_000_000
    assert len(g._indices) == 1000
    # The full grid is never materialized.
    assert not hasattr(g, "_grid")
    params = [g.propose().metadata["params"] for _ in range(1000)]
    assert len(params) == 1000
    assert all(set(p) == {f"p{i}" for i in range(8)} for p in params)


def test_grid_search_lazy_is_deterministic():
    space = _space(*[ParameterSpace(name=f"p{i}", kind="float", low=0.0, high=1.0) for i in range(8)])
    a = GridSearch(space, grid_points=10, max_combinations=1000, seed=1)
    b = GridSearch(space, grid_points=10, max_combinations=1000, seed=1)
    pa = [a.propose().metadata["params"] for _ in range(1000)]
    pb = [b.propose().metadata["params"] for _ in range(1000)]
    assert pa == pb


def test_grid_search_full_enumeration_still_works():
    space = _space(FLOAT, INT)
    g = GridSearch(space, grid_points=3)
    # float: 3 points, int: 11 points -> 33 combinations.
    assert g.total_combinations == 33
    seen = [g.propose().metadata["params"] for _ in range(33)]
    assert len(seen) == 33
    with pytest.raises(StopIteration):
        g.propose()
