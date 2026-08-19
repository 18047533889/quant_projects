"""Adversarial regression tests for the 2026-08-19 sealed-test hardening.

Pins the audit repairs: consume-time mask equality (not just split_id),
zip-truncation in disjointness, boolean score rejection, NaN best-score
lockout, and proposal_fn failure visibility.
"""

import pytest

from factor_optimizer.contracts.search_budget import (
    BudgetTracker,
    SearchBudget,
)
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import (
    SearchConfig,
    SearchRunner,
    SearchSession,
    _sealed_test_disjoint,
)


def _protocol(fn, split_id="split", masks=([True], [False], [False])):
    return EvaluationProtocol(SplitPlan(split_id, *masks, {}), fn)


def _finished_session():
    config = SearchConfig(
        budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
        enable_multifidelity=False,
    )
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED)
    trial.update_status(
        TrialStatus.EVALUATED,
        evaluation_ref="eval-1",
        metadata={"score": 1.0, "fidelity": 4},
    )
    return SearchSession(
        session_id="s1",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
        trials=[trial],
        best_score=1.0,
        best_trial_id="t1",
        stop_reason="budget_exhausted",
    )


def _finish(session):
    if not session.is_finished():
        session.finish("manual_stop")
    return session


class TestSealedMaskEquality:
    def test_same_split_id_different_masks_is_rejected(self):
        # Pre-fix: only split_id was compared, so a plan that moved the
        # sealed segment but kept the ID would silently pass.
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        forged = SplitPlan("test", [False, False], [False, False], [True, False], {})
        with pytest.raises(ValueError, match="masks differ from the frozen plan"):
            session.consume_sealed_test(
                handle, forged, EvaluationProtocol(forged, lambda t, f: {})
            )

    def test_evaluator_bound_to_different_masks_is_rejected(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        other = SplitPlan("test", [False, False], [False, False], [False, True], {})
        with pytest.raises(ValueError, match="evaluator protocol masks must match"):
            session.consume_sealed_test(
                handle, frozen, EvaluationProtocol(other, lambda t, f: {})
            )

    def test_masks_survive_checkpoint_roundtrip(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        restored = SearchSession.from_dict(session.to_dict())
        assert dict(restored.sealed_test_masks) == {
            "train": (True, False),
            "validation": (False, False),
            "test": (False, True),
        }
        result = restored.consume_sealed_test(
            handle, frozen, EvaluationProtocol(frozen, lambda t, f: {"rank_ic": 0.25})
        )
        assert result.test_metrics == {"rank_ic": 0.25}

    def test_frozen_checkpoint_without_masks_is_rejected(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        session.freeze_for_sealed_test(frozen)
        data = session.to_dict()
        del data["sealed_test_masks"]
        with pytest.raises(ValueError, match="sealed_test_masks"):
            SearchSession.from_dict(data)

    def test_pinned_masks_are_immutable_in_place(self):
        # __setattr__ guards assignment, but an in-place dict mutation
        # would otherwise rewrite what the seal compares against.
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        session.freeze_for_sealed_test(frozen)
        with pytest.raises(TypeError):
            session.sealed_test_masks["test"] = (True, False)
        with pytest.raises((TypeError, AttributeError)):
            session.sealed_test_masks["test"].append(True)  # type: ignore[union-attr]
        assert session.sealed_test_masks["test"] == (False, True)

    def test_restored_session_pins_deep_copy_and_blocks_seal_reset(self):
        session = _finish(_finished_session())
        frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
        handle = session.freeze_for_sealed_test(frozen)
        data = session.to_dict()
        restored = SearchSession.from_dict(data)
        assert restored.sealed_test_masks["test"] == (False, True)
        # Mutating the caller-held checkpoint dict after restoration must
        # not reach the restored session's pin (from_dict deep-copies).
        data["sealed_test_masks"]["test"] = [True, False]
        assert restored.sealed_test_masks["test"] == (False, True)
        with pytest.raises(TypeError):
            restored.sealed_test_masks["test"] = (True, False)
        # Consume for real, then attempt the reset True -> False.  (A
        # False -> False no-op on an unconsumed session is correctly
        # permitted by the identity guard.)
        restored.consume_sealed_test(
            handle, frozen, EvaluationProtocol(frozen, lambda t, f: {"rank_ic": 0.5})
        )
        assert restored.sealed_test_consumed is True
        with pytest.raises(ValueError, match="immutable after freeze"):
            restored.sealed_test_consumed = False
        with pytest.raises(ValueError, match="already frozen"):
            restored.freeze_for_sealed_test(frozen)


class TestSealedDisjointnessLength:
    def test_unequal_mask_lengths_fail_closed(self):
        # Pre-fix: zip() truncated, so only the shared prefix was compared.
        # SplitPlan construction rejects unequal lengths itself, so build
        # the search plan via a bare object bypassing __post_init__.
        search = SplitPlan("search", [True, False], [False, True], [False, False], {})
        longer = object.__new__(SplitPlan)
        object.__setattr__(longer, "split_id", "test")
        object.__setattr__(longer, "train_mask", [False, False])
        object.__setattr__(longer, "validation_mask", [False, False])
        object.__setattr__(longer, "test_mask", [False, True, True])
        object.__setattr__(longer, "metadata", {})
        assert _sealed_test_disjoint(search, longer) is False

    def test_equal_length_disjoint_plan_passes(self):
        # The sealed test segment [True, False] never overlaps the search
        # plan's train [False, False] or validation [False, True] masks.
        search = SplitPlan("search", [False, False], [False, True], [True, False], {})
        sealed = SplitPlan("test", [False, False], [False, False], [True, False], {})
        assert _sealed_test_disjoint(search, sealed) is True

    def test_overlap_is_detected(self):
        search = SplitPlan("search", [True, False], [False, False], [False, True], {})
        sealed = SplitPlan("test", [False, True], [False, False], [True, False], {})
        assert _sealed_test_disjoint(search, sealed) is False


class TestUpdateBestValidation:
    def test_nan_score_is_rejected_not_silent_lockout(self):
        config = SearchConfig(budget=SearchBudget(max_trials=1, max_evaluations=1))
        session = SearchSession(
            session_id="s", config=config, budget_tracker=BudgetTracker(config.budget)
        )
        session.update_best("t1", 0.5)
        with pytest.raises(ValueError, match="finite non-boolean"):
            session.update_best("t2", float("nan"))
        assert session.best_score == 0.5
        assert session.best_trial_id == "t1"

    def test_bool_score_is_rejected(self):
        config = SearchConfig(budget=SearchBudget(max_trials=1, max_evaluations=1))
        session = SearchSession(
            session_id="s", config=config, budget_tracker=BudgetTracker(config.budget)
        )
        with pytest.raises(ValueError, match="finite non-boolean"):
            session.update_best("t1", True)

    def test_inf_score_is_rejected(self):
        config = SearchConfig(budget=SearchBudget(max_trials=1, max_evaluations=1))
        session = SearchSession(
            session_id="s", config=config, budget_tracker=BudgetTracker(config.budget)
        )
        with pytest.raises(ValueError, match="finite non-boolean"):
            session.update_best("t1", float("inf"))


class TestRunnerIngestion:
    def test_boolean_evaluation_score_fails_the_trial(self):
        # Pre-fix: float(True) == 1.0 passed isfinite and produced a
        # checkpoint that SearchSession.from_dict rejects.
        session = SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=10.0),
                enable_multifidelity=False,
            ),
            lambda: Trial(trial_id="t", mutation_id="m", status=TrialStatus.PROPOSED),
            _protocol(lambda t, f: {"evaluation_id": "e", "score": True, "cost": 1.0}),
        ).run("bool-score")
        trial = session.trials[0]
        assert trial.status is TrialStatus.FAILED
        assert "must not be boolean" in trial.failure_reason
        assert session.best_trial_id is None

    def test_proposal_fn_failure_is_recorded_not_silent(self):
        def boom():
            raise RuntimeError("generator exploded")

        session = SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=10.0),
                enable_multifidelity=False,
            ),
            boom,
            _protocol(lambda t, f: {"score": 1.0, "cost": 1.0}),
        ).run("proposal-failure")
        assert len(session.duplicate_trials) == 1
        failed = session.duplicate_trials[0]
        assert failed.status is TrialStatus.FAILED
        assert "generator exploded" in failed.failure_reason
        # Burned budget stays visible across a checkpoint roundtrip.
        restored = SearchSession.from_dict(session.to_dict())
        assert "generator exploded" in restored.duplicate_trials[0].failure_reason

    def test_nan_evaluation_score_never_reaches_update_best(self):
        session = SearchRunner(
            SearchConfig(
                budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=10.0),
                enable_multifidelity=False,
            ),
            lambda: Trial(trial_id="t", mutation_id="m", status=TrialStatus.PROPOSED),
            _protocol(lambda t, f: {"evaluation_id": "e", "score": float("nan"), "cost": 1.0}),
        ).run("nan-score")
        trial = session.trials[0]
        assert trial.status is TrialStatus.FAILED
        assert "must be finite" in trial.failure_reason
        assert session.best_score is None


# ---------------------------------------------------------------------------
# Checkpoint roundtrip corruption tests
# ---------------------------------------------------------------------------

def _frozen_checkpoint():
    """Return a clean frozen-session checkpoint dict."""
    session = _finish(_finished_session())
    frozen = SplitPlan("test", [True, False], [False, False], [False, True], {})
    session.freeze_for_sealed_test(frozen)
    return session.to_dict()


class TestCheckpointRoundtripCorruption:
    """Inject corrupted sealed_test_results / frozen_at / identity fields
    and verify from_dict rejects them."""

    def test_result_item_missing_required_keys(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{}]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="sealed_test_results"):
            SearchSession.from_dict(data)

    def test_result_item_bool_metric_value(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": True},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="test_metrics"):
            SearchSession.from_dict(data)

    def test_result_item_nan_metric_value(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": float("nan")},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="test_metrics"):
            SearchSession.from_dict(data)

    def test_result_item_inf_metric_value(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": float("inf")},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="test_metrics"):
            SearchSession.from_dict(data)

    def test_result_item_frozen_at_is_not_a_string(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": 42,
            "test_metrics": {"rank_ic": 0.5},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_result_item_frozen_at_is_bool(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "t1", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": True,
            "test_metrics": {"rank_ic": 0.5},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_result_item_empty_trial_id(self):
        data = _frozen_checkpoint()
        data["sealed_test_results"] = [{
            "trial_id": "", "split_id": "test", "evaluation_ref": "e1",
            "frozen_at": "2026-01-01T00:00:00",
            "test_metrics": {"rank_ic": 0.5},
        }]
        data["sealed_test_consumed"] = True
        with pytest.raises(ValueError, match="trial_id"):
            SearchSession.from_dict(data)

    def test_top_level_frozen_at_is_bool(self):
        data = _frozen_checkpoint()
        data["frozen_at"] = True
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_top_level_frozen_at_is_int(self):
        data = _frozen_checkpoint()
        data["frozen_at"] = 42
        with pytest.raises(ValueError, match="frozen_at"):
            SearchSession.from_dict(data)

    def test_frozen_with_empty_sealed_trial_id(self):
        data = _frozen_checkpoint()
        data["sealed_trial_id"] = ""
        with pytest.raises(ValueError, match="sealed_trial_id"):
            SearchSession.from_dict(data)

    def test_frozen_with_empty_sealed_split_id(self):
        data = _frozen_checkpoint()
        data["sealed_split_id"] = "  "
        with pytest.raises(ValueError, match="sealed_split_id"):
            SearchSession.from_dict(data)


class TestSearchSplitPlanSerialization:
    """The overlap guard's search-time plan must survive checkpointing."""

    def _search_session(self):
        # Search ran on train/validation masks that overlap the sealed test
        # segment [False, True] via validation [False, True].
        search_plan = SplitPlan(
            "search", [True, False], [False, True], [False, False], {}
        )
        config = SearchConfig(
            budget=SearchBudget(max_trials=1, max_evaluations=1, max_cost_units=1.0),
            enable_multifidelity=False,
        )
        object.__setattr__(config, "_search_split_plan", search_plan)
        trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED)
        trial.update_status(
            TrialStatus.EVALUATED,
            evaluation_ref="eval-1",
            metadata={"score": 1.0, "fidelity": 4},
        )
        session = SearchSession(
            session_id="s1",
            config=config,
            budget_tracker=BudgetTracker(config.budget),
            trials=[trial],
            best_score=1.0,
            best_trial_id="t1",
            stop_reason="budget_exhausted",
        )
        return _finish(session)

    def test_search_plan_roundtrips_through_checkpoint(self):
        session = self._search_session()
        data = session.to_dict()
        assert data["search_split_masks"]["split_id"] == "search"
        restored = SearchSession.from_dict(data)
        plan = getattr(restored.config, "_search_split_plan", None)
        assert plan is not None
        assert plan.split_id == "search"
        assert list(plan.train_mask) == [True, False]
        assert list(plan.validation_mask) == [False, True]
        assert list(plan.test_mask) == [False, False]

    def test_restored_session_still_rejects_overlapping_sealed_plan(self):
        # Pre-fix: from_dict dropped _search_split_plan, so a restored frozen
        # session silently skipped the sealed/search overlap check.
        session = self._search_session()
        restored = SearchSession.from_dict(session.to_dict())
        overlapping = SplitPlan(
            "test", [True, False], [False, False], [False, True], {}
        )
        with pytest.raises(ValueError, match="overlaps the search-time"):
            restored.freeze_for_sealed_test(overlapping)

    def test_restored_session_accepts_disjoint_sealed_plan(self):
        session = self._search_session()
        restored = SearchSession.from_dict(session.to_dict())
        disjoint = SplitPlan(
            "test", [True, False], [False, True], [False, False], {}
        )
        restored.freeze_for_sealed_test(disjoint)

    def test_malformed_search_split_masks_fail_closed(self):
        session = self._search_session()
        data = session.to_dict()
        data["search_split_masks"]["train"] = [True, 1]
        with pytest.raises(ValueError, match="search_split_masks"):
            SearchSession.from_dict(data)

    def test_unequal_search_split_mask_lengths_fail_closed(self):
        session = self._search_session()
        data = session.to_dict()
        data["search_split_masks"]["test"] = [False, False, False]
        with pytest.raises(ValueError, match="equal lengths"):
            SearchSession.from_dict(data)

    def test_non_bool_disjoint_mask_elements_fail_closed(self):
        # Pre-fix: `type(a) is bool and type(b) is bool` silently skipped
        # non-bool pairs, so numpy/int masks always reported disjoint.
        search = SplitPlan("search", [True, False], [False, False], [False, False], {})
        sealed = object.__new__(SplitPlan)
        object.__setattr__(sealed, "split_id", "test")
        object.__setattr__(sealed, "train_mask", [False, False])
        object.__setattr__(sealed, "validation_mask", [False, False])
        object.__setattr__(sealed, "test_mask", [1, 0])  # ints, not bools
        object.__setattr__(sealed, "metadata", {})
        assert _sealed_test_disjoint(search, sealed) is False
