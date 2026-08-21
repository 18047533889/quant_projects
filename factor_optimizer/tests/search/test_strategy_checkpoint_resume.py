"""Checkpoint/resume tests: RNG state preservation for search strategies.

A resumed search must reproduce the EXACT same proposal trajectory as an
uninterrupted one.  These tests verify that ``SearchSession.checkpoint`` /
``SearchSession.resume`` and the per-strategy ``save_state`` / ``load_state``
preserve the Python RNG state, the NumPy RNG state, recorded history, iteration
count, and strategy-specific fields.
"""

import os

import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import SearchConfig, SearchRunner, SearchSession
from factor_optimizer.search.strategies import (
    SearchStrategyState,
    SearchStrategy,
    RandomSearch,
    GridSearch,
    BayesianSearch,
    TPESearch,
    ParameterSpace,
    SearchSpace,
)


def _space() -> SearchSpace:
    return SearchSpace(
        parameters=[
            ParameterSpace(name="x", kind="float", low=-5.0, high=5.0),
            ParameterSpace(name="n", kind="int", low=0, high=10),
            ParameterSpace(name="c", kind="choice", choices=["a", "b", "c"]),
        ]
    )


def _numeric_space() -> SearchSpace:
    """Numeric-only space for BayesianSearch/TPESearch (categorical params
    are rejected fail-closed by the numeric-kernel strategies)."""
    return SearchSpace(
        parameters=[
            ParameterSpace(name="x", kind="float", low=-5.0, high=5.0),
            ParameterSpace(name="n", kind="int", low=0, high=10),
        ]
    )


def _protocol(fn):
    """Wrap a generic evaluator callback in the required safe boundary."""
    return EvaluationProtocol(
        SplitPlan("search", [True], [False], [False], {}), fn
    )


def _score_fn(params):
    """Deterministic pseudo-objective: rich enough for TPE to diverge paths."""
    x = params["x"]
    n = params["n"]
    return -(x - 1.0) ** 2 - (n - 3) ** 2


def _params_of(trial):
    return trial.metadata["params"]


def _drive(strategy, n_proposals, config, session_id, ckpt_path=None):
    """Run *n_proposals* full evaluations through the runner.

    Returns (session, params_trajectory).  When *ckpt_path* is given, the
    session is checkpointed mid-run (at the halfway point) and resumed from
    disk so the whole run exercises checkpoint/resume.
    """
    if ckpt_path is not None:
        # Run half the proposals, checkpoint, then resume for the rest.
        half = max(1, n_proposals // 2)
        cfg_half = SearchConfig(
            budget=SearchBudget(
                max_trials=half, max_evaluations=half, max_cost_units=float(half)
            ),
            plateau_window=50,
            plateau_threshold=0.0,
            enable_multifidelity=False,
        )
        runner = SearchRunner(
            cfg_half,
            strategy.proposal_fn,
            _protocol(lambda trial, fidelity: {
                "evaluation_id": trial.trial_id,
                "score": _score_fn(trial.metadata["params"]),
                "cost": 1.0,
            }),
            strategy=strategy,
        )
        session = runner.run(session_id)
        session.checkpoint(ckpt_path)
        # Restore from checkpoint and resume with remaining budget.
        restored = SearchSession.resume(ckpt_path)
        remaining = n_proposals - half
        cfg_rest = SearchConfig(
            budget=SearchBudget(
                max_trials=remaining, max_evaluations=remaining, max_cost_units=float(remaining)
            ),
            plateau_window=50,
            plateau_threshold=0.0,
            enable_multifidelity=False,
        )
        session = SearchRunner(
            cfg_rest,
            restored.strategy.proposal_fn,
            _protocol(lambda trial, fidelity: {
                "evaluation_id": trial.trial_id,
                "score": _score_fn(trial.metadata["params"]),
                "cost": 1.0,
            }),
            strategy=restored.strategy,
        ).resume(restored)
        assert len(session.trials) >= n_proposals
        return session, [_params_of(trial) for trial in session.trials[:n_proposals]]
    else:
        runner = SearchRunner(
            config,
            strategy.proposal_fn,
            _protocol(lambda trial, fidelity: {
                "evaluation_id": trial.trial_id,
                "score": _score_fn(trial.metadata["params"]),
                "cost": 1.0,
            }),
            strategy=strategy,
        )
        session = runner.run(session_id)
        assert len(session.trials) >= n_proposals
        return session, [_params_of(trial) for trial in session.trials[:n_proposals]]


def _config(max_trials):
    return SearchConfig(
        budget=SearchBudget(
            max_trials=max_trials, max_evaluations=max_trials, max_cost_units=float(max_trials)
        ),
        plateau_window=50,
        plateau_threshold=0.0,
        enable_multifidelity=False,
    )


def _full_run(strategy_cls, seed, ckpt_path):
    """Full uninterrupted run vs. checkpoint-then-resume run of the same length."""
    config = _config(24)

    # Trajectory A: uninterrupted (strategy constructed fresh).
    fresh = strategy_cls(_space(), seed=seed)
    if isinstance(fresh, (BayesianSearch, TPESearch)):
        # Numeric-kernel strategies reject categorical parameters (FO-P0-03);
        # use the numeric-only space for them.
        fresh = strategy_cls(_numeric_space(), seed=seed)
    _, trajectory_a = _drive(fresh, 12, config, "A")

    # Trajectory B: 6 proposals, checkpoint at 6, resume, 6 more.
    staged_space = _numeric_space() if isinstance(fresh, (BayesianSearch, TPESearch)) else _space()
    staged = strategy_cls(staged_space, seed=seed)
    _, _ = _drive(staged, 6, config, "B1", ckpt_path=ckpt_path)
    _, trajectory_b = _drive(
        SearchSession.resume(ckpt_path).strategy, 12, config, "B2"
    )
    return trajectory_a, trajectory_b


# ---------------------------------------------------------------------------
# Per-strategy save_state / load_state round trips
# ---------------------------------------------------------------------------

def test_strategy_state_capture_and_restore():
    space = _numeric_space()
    original = TPESearch(space, n_initial=4, seed=42)
    proposed_original = [_params_of(original.propose()) for _ in range(6)]
    state = original.save_state()
    assert isinstance(state, SearchStrategyState)
    assert state.strategy_type == "TPESearch"
    assert state.iteration_count == 6
    assert "state" in state.rng_state
    assert "bit_generator" in state.np_rng_state

    restored = TPESearch(space, n_initial=4, seed=42)
    restored.load_state(state)
    proposed_restored = [_params_of(restored.propose()) for _ in range(6)]
    assert proposed_restored == proposed_original


def test_strategy_state_type_mismatch_is_rejected():
    state = RandomSearch(_space(), seed=1).save_state()
    with pytest.raises(ValueError, match="RandomSearch"):
        TPESearch(_numeric_space(), seed=1).load_state(state)

def test_strategy_standalone_json_checkpoint_round_trip(tmp_path):
    original = BayesianSearch(_numeric_space(), n_initial=3, seed=7)
    for _ in range(4):
        original.propose()
    path = os.path.join(tmp_path, "strategy.json")
    original.save_checkpoint(path)
    restored = SearchStrategy.load_checkpoint(path)
    assert type(restored) is BayesianSearch
    assert restored.seed == 7
    assert restored.n_initial == 3
    left = [_params_of(original.propose()) for _ in range(3)]
    right = [_params_of(restored.propose()) for _ in range(3)]
    assert left == right


@pytest.mark.parametrize("strategy_cls", [RandomSearch, GridSearch, BayesianSearch, TPESearch])
def test_strategy_json_round_trip_exact(strategy_cls, tmp_path):
    space = _numeric_space()
    original = strategy_cls(space, seed=123)
    if isinstance(original, GridSearch):
        # Small grid so proposals do not exhaust it.
        original = strategy_cls(
            _numeric_space(),
            grid_points=3,
            max_combinations=8,
            seed=123,
        )
    for _ in range(4):
        original.propose()
    path = os.path.join(tmp_path, "strategy.json")
    original.save_checkpoint(path)
    restored = SearchStrategy.load_checkpoint(path)
    left = [_params_of(original.propose()) for _ in range(4)]
    right = [_params_of(restored.propose()) for _ in range(4)]
    assert left == right


# ---------------------------------------------------------------------------
# SearchSession.checkpoint / .resume exact-trajectory guarantees
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_cls", [RandomSearch, TPESearch, BayesianSearch])
def test_session_checkpoint_resume_exact_trajectory(strategy_cls, tmp_path):
    ckpt = os.path.join(tmp_path, "session.json")
    trajectory_a, trajectory_b = _full_run(strategy_cls, seed=11, ckpt_path=ckpt)
    assert trajectory_a == trajectory_b
