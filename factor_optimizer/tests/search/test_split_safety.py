"""Split-safety, objective-direction, and plateau-delegation regression tests.

Covers the FO-SPLIT-SAFETY fixes:
- purge/embargo/label-horizon leakage rejection in ``validate_split_plan``
- fail-closed rejection of leakage constraints without a time axis
- backward compatibility of pre-existing SplitPlan instances
- minimize-direction ``SearchSession.update_best``
- runner default plateau path delegating to the package ``PlateauDetector``
"""

import pytest

from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.splits import (
    EvaluationProtocol,
    SplitPlan,
    validate_split_plan,
)
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.plateau import PlateauConfig, PlateauDetector
from factor_optimizer.search.runner import (
    SearchConfig,
    SearchRunner,
    SearchSession,
)


def _plan(train, validation, test, **kwargs):
    return SplitPlan("split", train, validation, test, {}, **kwargs)


def _invalid_plan(train, validation, test, match, **kwargs):
    """Build a plan whose leakage constraints must reject it at construction."""
    with pytest.raises(ValueError, match=match):
        _plan(train, validation, test, **kwargs)


def _trial(trial_id):
    return Trial(trial_id=trial_id, mutation_id=f"m-{trial_id}", status=TrialStatus.PROPOSED)


def _protocol(fn, split_id="search", masks=([True], [False], [False])):
    return EvaluationProtocol(SplitPlan(split_id, *masks, {}), fn)


def _config(**overrides):
    budget = SearchBudget(max_trials=50, max_evaluations=50, max_cost_units=500.0)
    return SearchConfig(budget=budget, **overrides)


# ---------------------------------------------------------------------------
# Temporal leakage: purge / label horizon
# ---------------------------------------------------------------------------

def test_backward_compatible_plan_without_new_fields_still_validates():
    plan = _plan([True, True, False, False], [False, False, True, False], [False, False, False, True])
    result = validate_split_plan(plan)
    assert result["validated"] is True
    assert result["n_samples"] == 4
    # Defaults are inert.
    assert plan.time_index is None
    assert plan.label_horizon == 0
    assert plan.purge == 0
    assert plan.embargo == 0


def test_clean_plan_with_horizon_and_purge_passes():
    # Train [0..3], purge gap of 2 positions (4,5), test [6,7]; labels look
    # forward 1 position, so train label at 3 spans 3..4 — still before 6.
    time_index = tuple(range(8))
    plan = _plan(
        [True, True, True, True, False, False, False, False],
        [False] * 8,
        [False, False, False, False, False, False, True, True],
        time_index=time_index,
        label_horizon=1,
        purge=1,
    )
    assert validate_split_plan(plan)["validated"] is True


def test_purge_violation_rejected():
    # Train at position 3 with label_horizon=2 and purge=1: the label spans
    # up to position 6, and test starts at 5 — inside purge + horizon (3).
    time_index = tuple(range(8))
    _invalid_plan(
        [True, True, True, True, False, False, False, False],
        [False] * 8,
        [False, False, False, False, True, True, True, True],
        r"purge \+ label_horizon",
        time_index=time_index,
        label_horizon=2,
        purge=1,
    )


def test_purge_violation_via_validation_mask_rejected():
    # Same violation but the leaking position is a validation position.
    time_index = tuple(range(6))
    _invalid_plan(
        [False] * 6,
        [False, False, True, False, False, False],
        [False, False, False, True, False, False],
        r"purge \+ label_horizon",
        time_index=time_index,
        label_horizon=2,
        purge=1,
    )


# ---------------------------------------------------------------------------
# Temporal leakage: embargo
# ---------------------------------------------------------------------------

def test_embargo_violation_rejected():
    # Test at position 3; train at position 4 falls within embargo=2.
    time_index = tuple(range(6))
    _invalid_plan(
        [True, True, False, False, True, False],
        [False] * 6,
        [False, False, False, True, False, False],
        "embargo",
        time_index=time_index,
        embargo=2,
    )


def test_train_before_test_with_embargo_passes():
    # Train strictly before the test segment is never embargoed.
    time_index = tuple(range(6))
    plan = _plan(
        [True, True, False, False, False, False],
        [False] * 6,
        [False, False, False, True, True, True],
        time_index=time_index,
        embargo=2,
    )
    assert validate_split_plan(plan)["validated"] is True


def test_train_outside_embargo_span_passes():
    # Test at position 2; train at position 5 is beyond embargo=2.
    time_index = tuple(range(6))
    plan = _plan(
        [True, False, False, False, False, True],
        [False] * 6,
        [False, False, True, False, False, False],
        time_index=time_index,
        embargo=2,
    )
    assert validate_split_plan(plan)["validated"] is True


# ---------------------------------------------------------------------------
# Fail-closed without a time axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"purge": 1},
        {"embargo": 1},
        {"label_horizon": 1},
    ],
)
def test_leakage_constraints_without_time_index_rejected(kwargs):
    _invalid_plan([True, False], [False, True], [False, False], "time_index", **kwargs)


# ---------------------------------------------------------------------------
# New-field validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_index": (0, 1), "purge": -1},
        {"time_index": (0, 1), "embargo": -1},
        {"time_index": (0, 1), "label_horizon": -1},
        {"time_index": (0, 1), "purge": True},
        {"time_index": (0, 1), "embargo": 1.5},
        {"purge": 1, "time_index": "ab"},
        {"purge": 1, "time_index": (0,)},
        {"purge": 1, "time_index": (0, 1, 2)},
    ],
)
def test_malformed_leakage_fields_rejected(kwargs):
    _invalid_plan([True, False], [False, True], [False, False], ".", **kwargs)


def test_uncomparable_time_index_rejected():
    _invalid_plan(
        [True, False], [False, True], [False, False],
        "comparable",
        time_index=(0, "a"), purge=1,
    )


def test_time_axis_ordering_is_respected_not_mask_position():
    # Masks are positional; the leakage check must run on the time axis.
    # Mask position 0 is train, position 1 is test, but the time index says
    # the test sample happens FIRST (rank 0) and the train sample LAST
    # (rank 3) — no forward-label overlap and train is beyond the embargo
    # span, so the plan is legal despite the mask ordering.
    plan = _plan(
        [True, False, False, False],
        [False] * 4,
        [False, True, False, False],
        time_index=(3, 0, 1, 2),
        label_horizon=3,
        purge=2,
        embargo=2,
    )
    assert validate_split_plan(plan)["validated"] is True

    # Reversed time index: train sample is before the test sample with a
    # 4-position forward-label span — the test at +4 is inside the lookback.
    _invalid_plan(
        [False, True],
        [False, False],
        [True, False],
        r"purge \+ label_horizon",
        time_index=(5, 1),
        label_horizon=4,
        purge=0,
    )


# ---------------------------------------------------------------------------
# Objective direction
# ---------------------------------------------------------------------------

def test_minimize_direction_update_best_picks_lower_score():
    session = SearchSession(
        session_id="min-session",
        config=_config(objective_direction="minimize"),
        budget_tracker=BudgetTracker(_config().budget),
    )

    assert session.update_best("t1", 0.5)
    assert session.best_score == 0.5

    # A lower (better) loss replaces the incumbent.
    assert session.update_best("t2", 0.3)
    assert session.best_score == 0.3
    assert session.best_trial_id == "t2"

    # A higher (worse) loss does not.
    assert not session.update_best("t3", 0.4)
    assert session.best_score == 0.3
    assert session.best_trial_id == "t2"

    # Ties never replace the incumbent (strict comparison).
    assert not session.update_best("t4", 0.3)
    assert session.best_trial_id == "t2"


def test_maximize_direction_unchanged_semantics():
    session = SearchSession(
        session_id="max-session",
        config=_config(),
        budget_tracker=BudgetTracker(_config().budget),
    )
    assert session.config.objective_direction == "maximize"

    assert session.update_best("t1", 0.5)
    assert session.update_best("t2", 0.7)
    assert not session.update_best("t3", 0.6)
    assert (session.best_score, session.best_trial_id) == (0.7, "t2")


def test_invalid_objective_direction_rejected():
    with pytest.raises(ValueError, match="objective_direction"):
        _config(objective_direction="sideways")


def test_objective_direction_survives_checkpoint_roundtrip():
    session = SearchSession(
        session_id="ckpt",
        config=_config(objective_direction="minimize"),
        budget_tracker=BudgetTracker(_config(objective_direction="minimize").budget),
    )
    restored = SearchSession.from_dict(session.to_dict())
    assert restored.config.objective_direction == "minimize"
    assert restored.update_best("t1", 0.9)
    assert restored.update_best("t2", 0.2)
    assert restored.best_score == 0.2


def test_minimize_runner_selects_lowest_score_trial():
    scores = {"t1": 0.9, "t2": 0.2, "t3": 0.5}
    counter = [0]

    def proposal_fn():
        counter[0] += 1
        return _trial(f"t{counter[0]}")

    config = _config(
        objective_direction="minimize",
        plateau_window=10,
        enable_multifidelity=False,
    )

    def evaluation_fn(trial, fidelity):
        return {
            "evaluation_id": f"eval-{trial.trial_id}",
            "score": scores[trial.trial_id],
            "cost": 1.0,
        }

    session = SearchRunner(config, proposal_fn, _protocol(evaluation_fn)).run("min-run")
    assert session.best_trial_id == "t2"
    assert session.best_score == 0.2


# ---------------------------------------------------------------------------
# Plateau delegation
# ---------------------------------------------------------------------------

def test_default_plateau_path_delegates_to_package_detector():
    config = _config(plateau_window=5, plateau_threshold=0.001, enable_multifidelity=False)
    runner = SearchRunner(config, _trial, _protocol(lambda t, f: {"score": 1.0, "cost": 1.0}))

    # Fewer scores than the window: not a plateau, and no detector built.
    short_scores = [1.0, 0.5, 1.0]
    expected_short = PlateauDetector(
        PlateauConfig(window_size=5, min_relative_improvement=0.001)
    ).is_plateau(short_scores)
    assert runner._default_plateau_detector is None
    assert runner._check_plateau(short_scores) is expected_short is False
    assert runner._default_plateau_detector is None

    # Full window: the default path delegates to the package detector.
    scores = [1.0, 0.5, 1.0, 0.5, 1.0]
    expected = PlateauDetector(
        PlateauConfig(window_size=5, min_relative_improvement=0.001)
    ).is_plateau(scores)
    assert runner._check_plateau(scores) is expected
    assert isinstance(runner._default_plateau_detector, PlateauDetector)
    assert runner._default_plateau_detector.config.window_size == 5
    assert (
        runner._default_plateau_detector.config.min_relative_improvement == 0.001
    )


def test_default_plateau_matches_package_detector_on_split_half_semantics():
    # Split-half semantics differ from the retired max/min window heuristic:
    # [1.0, 0.5, 1.0] has a large max-min spread (never a plateau under the
    # old rule at threshold 0.001) but the split-half rule sees the second
    # half fail to beat the first half's best — a plateau.
    config = _config(plateau_window=3, plateau_threshold=0.001, enable_multifidelity=False)
    runner = SearchRunner(config, _trial, _protocol(lambda t, f: {"score": 1.0, "cost": 1.0}))

    scores = [1.0, 0.5, 1.0]
    reference = PlateauDetector(
        PlateauConfig(window_size=3, min_relative_improvement=0.001)
    )
    assert runner._check_plateau(scores) is reference.is_plateau(scores) is True


def test_injected_plateau_detector_still_takes_precedence():
    config = _config(plateau_window=3, enable_multifidelity=False)
    calls = []

    def injected(scores):
        calls.append(list(scores))
        return True

    runner = SearchRunner(config, _trial, _protocol(lambda t, f: {"score": 1.0, "cost": 1.0}), injected)
    assert runner._check_plateau([1.0, 1.0, 1.0]) is True
    assert calls == [[1.0, 1.0, 1.0]]
    assert runner._default_plateau_detector is None


def test_runner_stops_on_package_plateau_semantics():
    counter = [0]

    def proposal_fn():
        counter[0] += 1
        return _trial(f"t{counter[0]}")

    def evaluation_fn(trial, fidelity):
        # After a strong first half, the second half never improves on it.
        score = 1.0 if counter[0] <= 3 else 0.99
        return {
            "evaluation_id": f"eval-{trial.trial_id}",
            "score": score,
            "cost": 1.0,
        }

    config = _config(plateau_window=6, plateau_threshold=0.01, enable_multifidelity=False)
    session = SearchRunner(config, proposal_fn, _protocol(evaluation_fn)).run("plateau-run")

    assert session.is_finished()
    assert session.stop_reason == "plateau_detected"
