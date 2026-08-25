"""SearchRunner: orchestrate mutation search with budget and plateau stopping."""

import json
import numbers
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional

from factor_optimizer.capabilities import ExecutionMode, require_production_capability
from factor_optimizer.contracts.objective import (
    OBJECTIVE_DIRECTIONS,
    ObjectiveDirection,
    ObjectiveSpec,
)
from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.budget_extension import BudgetExtensionAuthorization
from factor_optimizer.contracts.evaluation_artifact import (
    EvaluationStatus,
    TrialEvaluationArtifact,
    normalize_evaluation_result,
)
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.contracts.trial_ledger import TrialLedger
from factor_optimizer.contracts.splits import (
    EvaluationProtocol,
    LabelBundle,
    SealedTestHandle,
    SealedTestResult,
    SplitPlan,
    validate_split_plan,
)
from factor_optimizer.contracts.validator import TrialValidatorIdentity
from factor_optimizer.data_capabilities import (
    DataCapability,
    DataScope,
    TestAuthorityBroker,
    TestDataProvider,
    ScopedEvaluator,
    build_search_capabilities,
)
from factor_optimizer.search.multifidelity import FidelityTier, MultiFidelityScheduler
from factor_optimizer.search.plateau import PlateauConfig, PlateauDetector
from factor_optimizer.search.strategies import SearchStrategy


@dataclass
class SearchConfig:
    """Configuration for search execution."""

    budget: SearchBudget
    plateau_window: int = 20
    plateau_threshold: float = 0.001
    enable_multifidelity: bool = True
    # Search execution is currently serialized; retain the field for forward
    # compatibility but reject settings that the runner cannot enforce.
    max_concurrency: int = 1
    evaluation_cost_units: Optional[float] = None
    execution_mode: ExecutionMode = ExecutionMode.RESEARCH_ONLY
    # Fail-closed default: search must run through an EvaluationProtocol with a
    # validated SplitPlan. Generic evaluation callbacks are rejected outright
    # unless the caller explicitly opts out via
    # require_evaluation_protocol=False (deprecated escape hatch, research
    # mode only; always rejected under PRODUCTION).
    require_evaluation_protocol: bool = True
    # Direction of the search objective. "maximize" (default) keeps the
    # historical `score > best` incumbent rule; "minimize" inverts it
    # (strictly lower score wins, e.g. loss/error objectives).
    #
    # FO-P0-06: The authoritative description of the objective is the
    # ``objective_spec`` contract below.  This bare field is retained as a
    # backward-compatible convenience: callers that construct
    # ``SearchConfig(budget=..., objective_direction="minimize")`` keep
    # working, and the field is reconciled into the spec at construction.
    # The spec is the single authority — ``objective_spec.direction`` is what
    # the runner uses, and if both are supplied and disagree the config is
    # rejected rather than silently drifting.  ``None`` (the default) means
    # "not supplied"; it is resolved to the spec's direction during
    # ``__post_init__``.
    objective_direction: Optional[ObjectiveDirection] = None
    objective_spec: Optional[ObjectiveSpec] = None

    def __post_init__(self):
        if isinstance(self.execution_mode, str):
            try:
                self.execution_mode = ExecutionMode(self.execution_mode)
            except ValueError as exc:
                raise ValueError(f"unsupported execution_mode: {self.execution_mode}") from exc
        if self.execution_mode is ExecutionMode.PRODUCTION:
            require_production_capability()
        if self.plateau_window < 1:
            raise ValueError("plateau_window must be >= 1")
        if self.plateau_threshold < 0:
            raise ValueError("plateau_threshold must be >= 0")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.max_concurrency != 1:
            raise ValueError(
                "max_concurrency > 1 is unsupported while SearchRunner is serialized"
            )
        if self.evaluation_cost_units is not None and (
            not isfinite(self.evaluation_cost_units) or self.evaluation_cost_units < 0
        ):
            raise ValueError("evaluation_cost_units must be finite and >= 0")
        if self.objective_direction is not None:
            if self.objective_direction not in OBJECTIVE_DIRECTIONS:
                raise ValueError(
                    "objective_direction must be 'maximize' or 'minimize'"
                )
        # The spec is the authority for direction.  If both the spec and the
        # legacy field are given, they must agree; the field is then dropped
        # so the spec cannot drift from it.  If only the field is given, it is
        # promoted to a spec over the default metric ("score").
        if self.objective_spec is not None:
            if not isinstance(self.objective_spec, ObjectiveSpec):
                raise TypeError(
                    "objective_spec must be an ObjectiveSpec (or None)"
                )
            if (
                self.objective_direction is not None
                and self.objective_direction != self.objective_spec.direction
            ):
                raise ValueError(
                    "objective_spec.direction and objective_direction disagree; "
                    f"spec says {self.objective_spec.direction!r}, "
                    f"field says {self.objective_direction!r}"
                )
            # Normalize: the bare field is now derived from the authoritative
            # spec so the two can never drift after construction.
            self.objective_direction = self.objective_spec.direction
        else:
            if self.objective_direction is None:
                self.objective_direction = "maximize"
            self.objective_spec = ObjectiveSpec.from_direction(
                self.objective_direction
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize search configuration."""
        return {
            "budget": self.budget.to_dict(),
            "plateau_window": self.plateau_window,
            "plateau_threshold": self.plateau_threshold,
            "enable_multifidelity": self.enable_multifidelity,
            "max_concurrency": self.max_concurrency,
            "evaluation_cost_units": self.evaluation_cost_units,
            "execution_mode": self.execution_mode.value,
            "require_evaluation_protocol": self.require_evaluation_protocol,
            "objective_spec": self.objective_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchConfig":
        """Deserialize search configuration."""
        values = dict(data)
        values["budget"] = SearchBudget.from_dict(values["budget"])
        # Checkpoint round-trips (and any dict that carries the spec) restore
        # the authoritative spec.  The legacy bare field is reconstructed from
        # the spec so configs built via to_dict/from_dict remain comparable.
        if "objective_spec" in values:
            values["objective_spec"] = ObjectiveSpec.from_dict(
                values["objective_spec"]
            )
        return cls(**values)


@dataclass
class SearchSession:
    """Runtime state for an active search session."""

    session_id: str
    config: SearchConfig
    budget_tracker: BudgetTracker
    trials: List[Trial] = field(default_factory=list)
    duplicate_trials: List[Trial] = field(default_factory=list)
    best_score: Optional[float] = None
    best_trial_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    stop_reason: Optional[str] = None
    recent_scores: List[float] = field(default_factory=list)
    frozen_at: Optional[datetime] = None
    sealed_trial_id: Optional[str] = None
    sealed_split_id: Optional[str] = None
    sealed_test_masks: Optional[Dict[str, List[bool]]] = None
    sealed_test_consumed: bool = False
    sealed_test_results: List[Dict[str, Any]] = field(default_factory=list)
    strategy: Optional[SearchStrategy] = None
    # FO-P0-04: append-only ledger of every proposal attempt.  A burned-budget
    # proposal (raised / non-Trial) is recorded here as PROPOSAL_FAILED /
    # INVALID_PROPOSAL, NOT silently dropped and NOT mislabelled as DUPLICATE.
    ledger: TrialLedger = field(default_factory=TrialLedger)

    def __setattr__(self, name: str, value: Any) -> None:
        # Sealed-test state is append-only through freeze/consume; direct
        # assignment after freeze cannot reset the one-shot seal.
        if (
            getattr(self, "frozen_at", None) is not None
            and name in ("frozen_at", "sealed_trial_id", "sealed_split_id",
                         "sealed_test_masks", "sealed_test_consumed")
            and getattr(self, name, None) is not value
            and not getattr(self, "_sealing", False)
        ):
            raise ValueError(
                f"sealed test state '{name}' is immutable after freeze; "
                "use consume_sealed_test"
            )
        object.__setattr__(self, name, value)

    def _ensure_mutable(self) -> None:
        if self.frozen_at is not None:
            raise ValueError("search session is frozen")

    def add_trial(self, trial: Trial) -> None:
        """Add a trial to the session, rejecting reused trial IDs."""
        self._ensure_mutable()
        if any(existing.trial_id == trial.trial_id for existing in self.trials):
            raise ValueError(f"trial_id already recorded: {trial.trial_id}")
        self.trials.append(trial)

    def has_trial(self, trial_id: str) -> bool:
        """Return whether a trial ID has already been recorded."""
        return any(trial.trial_id == trial_id for trial in self.trials)

    def update_best(self, trial_id: str, score: float) -> bool:
        """Update best score if improved. Returns True if new best."""
        self._ensure_mutable()
        # A NaN comparison against any best_score is False, so accepting one
        # would lock the incumbent forever; reject non-finite scores outright
        # (mirroring SearchSession.from_dict's best_score contract).
        # Accept numpy scalars via numbers.Real but still reject bool (a
        # numbers.Real subclass would otherwise admit True as 1.0).
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float, numbers.Real))
        ):
            raise ValueError("best score must be a finite non-boolean number")
        try:
            finite = isfinite(score)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError("best score must be a finite non-boolean number")
        if self.best_score is None or _is_improvement(
            score, self.best_score, self.config.objective_spec.direction
        ):
            self.best_score = score
            self.best_trial_id = trial_id
            return True
        return False

    def successful_trials(self) -> List[Trial]:
        """Return all successfully evaluated trials."""
        return [t for t in self.trials if t.is_successful()]

    def finish(self, reason: str) -> None:
        """Mark session as finished."""
        self._ensure_mutable()
        self.finished_at = datetime.now()
        self.stop_reason = reason

    def freeze_for_sealed_test(self, split_plan: SplitPlan) -> SealedTestHandle:
        """Freeze the finished winner and issue its one-shot test authority."""
        validate_split_plan(split_plan)
        if not self.is_finished():
            raise ValueError("cannot freeze an unfinished search session")
        if self.frozen_at is not None:
            raise ValueError("search session is already frozen")
        if not self.best_trial_id:
            raise ValueError("cannot freeze a session without an evaluated winner")
        winner = next(
            (trial for trial in self.successful_trials() if trial.trial_id == self.best_trial_id),
            None,
        )
        if winner is None or not winner.evaluation_ref:
            raise ValueError("frozen winner must have evaluation evidence")
        search_plan = getattr(self.config, "_search_split_plan", None)
        if search_plan is not None and not _sealed_test_disjoint(search_plan, split_plan):
            raise ValueError(
                "sealed test_mask overlaps the search-time train/validation masks; "
                "the sealed segment must be disjoint from all search data"
            )
        self.frozen_at = datetime.now()
        # Allow the freeze itself to set the sealed identity exactly once.
        object.__setattr__(self, "_sealing", True)
        try:
            self.sealed_trial_id = winner.trial_id
            self.sealed_split_id = split_plan.split_id
            # Pin the frozen masks so consume_sealed_test can reject a
            # different plan that merely reuses the same split_id.  A
            # read-only proxy + tuples: in-place item mutation must not be
            # able to rewrite what the seal compares against.
            self.sealed_test_masks = MappingProxyType({
                "train": tuple(split_plan.train_mask),
                "validation": tuple(split_plan.validation_mask),
                "test": tuple(split_plan.test_mask),
            })
        finally:
            object.__setattr__(self, "_sealing", False)
        return SealedTestHandle(
            search_session_id=self.session_id,
            trial_id=winner.trial_id,
            split_id=split_plan.split_id,
            evaluation_ref=winner.evaluation_ref,
            frozen_at=self.frozen_at,
        )

    def consume_sealed_test(
        self,
        handle: SealedTestHandle,
        split_plan: SplitPlan,
        evaluator: Callable[[Trial, SplitPlan], Dict[str, float]],
    ) -> SealedTestResult:
        """Validate and consume a handle by evaluating only its bound test plan."""
        validate_split_plan(split_plan)
        if not isinstance(handle, SealedTestHandle):
            raise TypeError("handle must be a SealedTestHandle")
        if not isinstance(evaluator, EvaluationProtocol):
            raise TypeError(
                "sealed test evaluation requires an EvaluationProtocol whose "
                "split_plan is bound to the sealed test segment; bare callables "
                "decide their own data boundary and are not accepted"
            )
        if evaluator.split_plan.split_id != split_plan.split_id:
            raise ValueError(
                "evaluator protocol split_id must match the sealed split_plan"
            )
        if self.frozen_at is None:
            raise ValueError("search session is not frozen")
        if self.sealed_test_consumed:
            raise ValueError("sealed test handle has already been consumed")
        expected = (
            self.session_id,
            self.sealed_trial_id,
            self.sealed_split_id,
            self.frozen_at,
        )
        actual = (
            handle.search_session_id,
            handle.trial_id,
            handle.split_id,
            handle.frozen_at,
        )
        if actual != expected or split_plan.split_id != self.sealed_split_id:
            raise ValueError("sealed test handle does not match frozen session")
        # split_id equality alone is trivially forgeable; the plan presented
        # at consume time must be mask-identical to the one frozen earlier.
        frozen_masks = self.sealed_test_masks
        if frozen_masks is not None and (
            tuple(split_plan.train_mask) != frozen_masks["train"]
            or tuple(split_plan.validation_mask) != frozen_masks["validation"]
            or tuple(split_plan.test_mask) != frozen_masks["test"]
        ):
            raise ValueError(
                "sealed test split_plan masks differ from the frozen plan; "
                "a reused split_id cannot re-bind the sealed test segment"
            )
        # The evaluator must be bound to this exact plan, not just to a
        # plan carrying the same split_id.
        if (
            tuple(evaluator.split_plan.train_mask) != tuple(split_plan.train_mask)
            or tuple(evaluator.split_plan.validation_mask) != tuple(split_plan.validation_mask)
            or tuple(evaluator.split_plan.test_mask) != tuple(split_plan.test_mask)
        ):
            raise ValueError(
                "evaluator protocol masks must match the sealed split_plan masks"
            )
        winner = next(
            (trial for trial in self.successful_trials() if trial.trial_id == handle.trial_id),
            None,
        )
        if winner is None or winner.evaluation_ref != handle.evaluation_ref:
            raise ValueError("sealed test handle evidence does not match frozen winner")
        metrics = evaluator.evaluate(winner, 0)
        if not isinstance(metrics, dict) or not all(
            isinstance(name, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            for name, value in metrics.items()
        ):
            raise ValueError("sealed test evaluator must return finite numeric metrics")
        # Consume the seal via the guarded path (False→True is the one legal
        # transition; any reset attempt stays blocked by __setattr__).
        object.__setattr__(self, "sealed_test_consumed", True)
        result = SealedTestResult(
            trial_id=winner.trial_id,
            test_metrics=dict(metrics),
            frozen_at=self.frozen_at,
            search_session_id=self.session_id,
            split_id=split_plan.split_id,
            evaluation_ref=winner.evaluation_ref,
        )
        self.sealed_test_results.append(
            {
                "trial_id": result.trial_id,
                "split_id": result.split_id,
                "evaluation_ref": result.evaluation_ref,
                "frozen_at": result.frozen_at.isoformat(),
                "test_metrics": dict(result.test_metrics),
            }
        )
        return result

    def is_finished(self) -> bool:
        """Check if session is complete."""
        return self.finished_at is not None

    def duration_seconds(self) -> float:
        """Return session duration in seconds."""
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

    # -- checkpoint / resume ---------------------------------------------------

    def checkpoint(self, path: str) -> None:
        """Write a full checkpoint (strategy + RNG + budget) to *path*.

        The payload embeds the session's config, budget tracker, trials, and the
        complete stochastic state of its ``strategy`` (Python RNG, NumPy RNG,
        history, iteration count, and strategy-specific fields).  A session
        resumed from this checkpoint reproduces the exact same proposal
        trajectory as an uninterrupted one.
        """
        payload = self.to_dict()
        if self.strategy is not None:
            payload["strategy"] = self.strategy.to_checkpoint_dict()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    @classmethod
    def resume(cls, path: str) -> "SearchSession":
        """Load and restore a session checkpoint written by ``checkpoint``.

        Returns a live, quiescent ``SearchSession`` (unfinished) with its
        strategy's RNG state fully restored, ready for ``SearchRunner.resume``.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        session = cls.from_dict(payload)
        if "strategy" in payload:
            session.strategy = SearchStrategy.from_checkpoint_dict(payload["strategy"])
        return session

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state for a quiescent checkpoint."""
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "budget_tracker": self.budget_tracker.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "duplicate_trials": [trial.to_dict() for trial in self.duplicate_trials],
            "best_score": self.best_score,
            "best_trial_id": self.best_trial_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "stop_reason": self.stop_reason,
            "recent_scores": list(self.recent_scores),
            # Checkpoint format: 2 serializes sealed_test_masks alongside
            # frozen_at; format 1 checkpoints (no key) predate mask pinning.
            "checkpoint_format": 2,
            # The search-time data boundary must survive a checkpoint
            # roundtrip: freeze_for_sealed_test's overlap guard reads it
            # from the config, so a restored session without it would
            # silently skip the sealed/search disjointness check.
            "search_split_masks": _serialize_search_split_plan(self.config),
            "frozen_at": self.frozen_at.isoformat() if self.frozen_at else None,
            "sealed_trial_id": self.sealed_trial_id,
            "sealed_split_id": self.sealed_split_id,
            "sealed_test_masks": (
                {
                    "train": list(self.sealed_test_masks["train"]),
                    "validation": list(self.sealed_test_masks["validation"]),
                    "test": list(self.sealed_test_masks["test"]),
                }
                if self.sealed_test_masks is not None
                else None
            ),
            "sealed_test_consumed": self.sealed_test_consumed,
            "sealed_test_results": [dict(item) for item in self.sealed_test_results],
            "ledger": self.ledger.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSession":
        """Deserialize and validate a session checkpoint."""
        values = dict(data)
        required = {
            "session_id", "config", "budget_tracker", "trials", "duplicate_trials",
            "best_score", "best_trial_id", "started_at", "finished_at", "stop_reason",
            "recent_scores", "frozen_at", "sealed_trial_id", "sealed_split_id",
            "sealed_test_consumed",
        }
        missing = required.difference(values)
        if missing:
            raise ValueError(f"session checkpoint missing fields: {sorted(missing)}")
        started_at = values["started_at"]
        finished_at = values["finished_at"]
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        if isinstance(finished_at, str):
            finished_at = datetime.fromisoformat(finished_at)
        frozen_at = values["frozen_at"]
        if frozen_at is not None and not isinstance(frozen_at, str):
            raise ValueError(
                "checkpoint frozen_at must be null or an ISO timestamp string"
            )
        if isinstance(frozen_at, str):
            frozen_at = datetime.fromisoformat(frozen_at)
        budget_tracker = BudgetTracker.from_dict(values["budget_tracker"])
        config = SearchConfig.from_dict(values["config"])
        if budget_tracker.budget.to_dict() != config.budget.to_dict():
            raise ValueError("checkpoint configuration and budget do not match")
        counters = (
            budget_tracker.trials_used,
            budget_tracker.evaluations_used,
            budget_tracker.cost_used,
            budget_tracker.llm_calls_used,
            budget_tracker.evaluations_reserved,
            budget_tracker.cost_reserved,
        )
        if (
            not isinstance(budget_tracker.trials_used, int)
            or isinstance(budget_tracker.trials_used, bool)
            or not isinstance(budget_tracker.evaluations_used, int)
            or isinstance(budget_tracker.evaluations_used, bool)
            or not isinstance(budget_tracker.llm_calls_used, int)
            or isinstance(budget_tracker.llm_calls_used, bool)
            or not isinstance(budget_tracker.evaluations_reserved, int)
            or isinstance(budget_tracker.evaluations_reserved, bool)
            or any(not isfinite(value) for value in counters[2:])
            or any(value < 0 for value in counters)
            or budget_tracker.trials_used > config.budget.max_trials
            or budget_tracker.evaluations_used + budget_tracker.evaluations_reserved > config.budget.max_evaluations
            or budget_tracker.cost_used + budget_tracker.cost_reserved > config.budget.max_cost_units
            or (
                config.budget.max_llm_calls is not None
                and budget_tracker.llm_calls_used > config.budget.max_llm_calls
            )
        ):
            raise ValueError("checkpoint budget counters are invalid")
        trials = [Trial.from_dict(item) for item in values["trials"]]
        duplicates = [Trial.from_dict(item) for item in values["duplicate_trials"]]
        if budget_tracker.evaluations_reserved or budget_tracker.cost_reserved:
            raise ValueError("cannot restore checkpoint with active evaluation reservations")
        if any(not trial.is_terminal() for trial in trials + duplicates):
            raise ValueError("cannot restore checkpoint with non-terminal trials")
        best_score = values["best_score"]
        best_trial_id = values["best_trial_id"]
        if best_score is not None and (
            isinstance(best_score, bool)
            or not isinstance(best_score, (int, float))
            or not isfinite(best_score)
        ):
            raise ValueError("checkpoint best_score must be null or a finite number")
        if (best_score is None) != (best_trial_id is None):
            raise ValueError("checkpoint best_score and best_trial_id must be paired")
        if best_trial_id is not None:
            best_trials = [trial for trial in trials if trial.trial_id == best_trial_id]
            if len(best_trials) != 1 or not best_trials[0].is_successful():
                raise ValueError("checkpoint best_trial_id must reference one evaluated trial")
            trial_score = best_trials[0].metadata.get("score")
            if (
                isinstance(trial_score, bool)
                or not isinstance(trial_score, (int, float))
                or not isfinite(trial_score)
                or trial_score != best_score
            ):
                raise ValueError("checkpoint best_score does not match best trial")
        recent_scores = values["recent_scores"]
        if not isinstance(recent_scores, list) or not all(
            isinstance(score, (int, float)) and not isinstance(score, bool) and isfinite(score)
            for score in recent_scores
        ):
            raise ValueError("checkpoint recent_scores must contain finite numbers")
        if len(recent_scores) > config.plateau_window:
            raise ValueError("checkpoint recent_scores exceeds plateau_window")
        evaluated_scores = []
        for trial in trials:
            if not trial.is_successful():
                continue
            score = trial.metadata.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(score)
            ):
                raise ValueError("checkpoint evaluated trial score must be finite")
            evaluated_scores.append(float(score))
        if recent_scores and evaluated_scores[-len(recent_scores):] != recent_scores:
            raise ValueError("checkpoint recent_scores do not match evaluated trial history")
        sealed_test_consumed = values["sealed_test_consumed"]
        if not isinstance(sealed_test_consumed, bool):
            raise ValueError("checkpoint sealed_test_consumed must be a boolean")
        sealed_test_masks = values.get("sealed_test_masks")
        if sealed_test_masks is not None:
            if not isinstance(sealed_test_masks, dict) or set(sealed_test_masks) != {
                "train", "validation", "test"
            }:
                raise ValueError("checkpoint sealed_test_masks must contain train/validation/test")
            if not all(
                isinstance(sealed_test_masks[name], list)
                and sealed_test_masks[name]
                and all(type(mask) is bool for mask in sealed_test_masks[name])
                for name in ("train", "validation", "test")
            ):
                raise ValueError("checkpoint sealed_test_masks must be non-empty boolean lists")
            mask_lengths = {len(sealed_test_masks[name]) for name in ("train", "validation", "test")}
            if len(mask_lengths) != 1:
                raise ValueError("checkpoint sealed_test_masks must have equal lengths")
        # Format 2 checkpoints serialize sealed_test_masks alongside
        # frozen_at, so a frozen session without masks is tampered/corrupt.
        # Format 1 (pre-mask) checkpoints predate the key entirely and must
        # still restore: their seal identity survives via frozen_at /
        # sealed_trial_id / sealed_split_id, and the consume-time mask
        # comparison degrades to skip rather than reject.
        _format = values.get("checkpoint_format", 1)
        if _format >= 2 and (frozen_at is None) != (sealed_test_masks is None):
            raise ValueError("checkpoint sealed_test_masks must be present exactly when frozen")
        if frozen_at is not None and sealed_test_masks is None:
            warnings.warn(
                "legacy checkpoint predates sealed_test_masks serialization; "
                "restoring without frozen mask pinning (consume-time mask "
                "comparison will be skipped)",
                UserWarning,
                stacklevel=2,
            )
        sealed_test_results = values.get("sealed_test_results", [])
        if not isinstance(sealed_test_results, list):
            raise ValueError("checkpoint sealed_test_results must be a list")
        _REQUIRED_RESULT_KEYS = {"trial_id", "split_id", "evaluation_ref", "frozen_at", "test_metrics"}
        for _idx, _item in enumerate(sealed_test_results):
            if not isinstance(_item, dict):
                raise ValueError(
                    f"checkpoint sealed_test_results[{_idx}] must be a dict"
                )
            _missing_keys = _REQUIRED_RESULT_KEYS - _item.keys()
            if _missing_keys:
                raise ValueError(
                    f"checkpoint sealed_test_results[{_idx}] missing keys: {sorted(_missing_keys)}"
                )
            # Validate string identity fields (non-empty, non-whitespace).
            for _key in ("trial_id", "split_id", "evaluation_ref"):
                _val = _item[_key]
                if not isinstance(_val, str) or not _val.strip():
                    raise ValueError(
                        f"checkpoint sealed_test_results[{_idx}].{_key} must be a non-empty string"
                    )
            # frozen_at must be a parseable ISO timestamp string.
            _frozen_at_raw = _item["frozen_at"]
            if not isinstance(_frozen_at_raw, str):
                raise ValueError(
                    f"checkpoint sealed_test_results[{_idx}].frozen_at must be an ISO timestamp string"
                )
            try:
                datetime.fromisoformat(_frozen_at_raw)
            except (ValueError, TypeError) as _exc:
                raise ValueError(
                    f"checkpoint sealed_test_results[{_idx}].frozen_at is not a valid ISO timestamp"
                ) from _exc
            # test_metrics must be a dict of finite non-boolean numbers.
            _metrics = _item["test_metrics"]
            if not isinstance(_metrics, dict):
                raise ValueError(
                    f"checkpoint sealed_test_results[{_idx}].test_metrics must be a dict"
                )
            for _mkey, _mval in _metrics.items():
                if (
                    isinstance(_mval, bool)
                    or not isinstance(_mval, (int, float))
                    or not isfinite(_mval)
                ):
                    raise ValueError(
                        f"checkpoint sealed_test_results[{_idx}].test_metrics[{_mkey!r}] "
                        f"must be a finite non-boolean number, got {type(_mval).__name__}"
                    )
        if sealed_test_consumed and not sealed_test_results and sealed_test_masks is not None:
            raise ValueError(
                "checkpoint claims a consumed sealed test without recorded evidence"
            )
        if sealed_test_results and not sealed_test_consumed:
            raise ValueError(
                "checkpoint records sealed test evidence but the seal is unconsumed"
            )
        # Reattach the serialized search-time data boundary so the frozen
        # session's overlap guard survives the checkpoint roundtrip.
        _restore_search_split_plan(config, values.get("search_split_masks"))
        # FO-P0-04: restore the append-only proposal ledger (a checkpoint with
        # a malformed/omitted ledger is rejected so burned-budget records can
        # never be silently dropped).
        ledger_raw = values.get("ledger")
        if ledger_raw is None:
            raise ValueError(
                "session checkpoint missing the FO-P0-04 trial ledger; "
                "refusing to restore a checkpoint that could hide burned "
                "proposal budget"
            )
        ledger = TrialLedger.from_dict(ledger_raw)
        session = cls(
            session_id=values["session_id"],
            config=config,
            budget_tracker=budget_tracker,
            trials=trials,
            duplicate_trials=duplicates,
            best_score=values["best_score"],
            best_trial_id=values["best_trial_id"],
            started_at=started_at,
            finished_at=finished_at,
            stop_reason=values["stop_reason"],
            recent_scores=list(values["recent_scores"]),
            frozen_at=None,
            sealed_trial_id=None,
            sealed_split_id=None,
            sealed_test_masks=None,
            sealed_test_consumed=False,
            sealed_test_results=list(sealed_test_results),
            ledger=ledger,
        )
        if frozen_at is not None:
            # Replay the freeze through the guarded path so the restored
            # session carries the same immutability guarantees as a live one.
            object.__setattr__(session, "_sealing", True)
            try:
                session.frozen_at = frozen_at
                # Validate identity fields for frozen sessions.
                _stid = values["sealed_trial_id"]
                if not isinstance(_stid, str) or not _stid.strip():
                    raise ValueError(
                        "checkpoint sealed_trial_id must be a non-empty string for a frozen session"
                    )
                session.sealed_trial_id = _stid
                _ssid = values["sealed_split_id"]
                if not isinstance(_ssid, str) or not _ssid.strip():
                    raise ValueError(
                        "checkpoint sealed_split_id must be a non-empty string for a frozen session"
                    )
                session.sealed_split_id = _ssid
                if sealed_test_masks is not None:
                    session.sealed_test_masks = MappingProxyType({
                        "train": tuple(sealed_test_masks["train"]),
                        "validation": tuple(sealed_test_masks["validation"]),
                        "test": tuple(sealed_test_masks["test"]),
                    })
                session.sealed_test_consumed = sealed_test_consumed
            finally:
                object.__setattr__(session, "_sealing", False)
        return session


def _is_improvement(
    score: float, best_score: float, direction: str
) -> bool:
    """Strict improvement under the session's objective direction."""
    if direction == "minimize":
        return score < best_score
    return score > best_score


def _sealed_test_disjoint(search_plan: SplitPlan, sealed_plan: SplitPlan) -> bool:
    """Check the sealed test segment is disjoint from all search-time data."""
    try:
        for search_mask in (
            search_plan.train_mask,
            search_plan.validation_mask,
        ):
            # Unequal lengths make zip() silently truncate the comparison,
            # so a longer sealed mask would only be checked on its prefix.
            if len(sealed_plan.test_mask) != len(search_mask):
                return False
            for a, b in zip(sealed_plan.test_mask, search_mask):
                # Fail closed on non-bool elements: a non-bool pair must be
                # treated as overlap, not silently skipped.
                if type(a) is not bool or type(b) is not bool:
                    return False
                if a and b:
                    return False
    except TypeError:
        return False
    return True


def _serialize_search_split_plan(config: SearchConfig) -> Optional[Dict[str, Any]]:
    """Serialize the search-time SplitPlan stashed on the config, if any."""
    plan = getattr(config, "_search_split_plan", None)
    if plan is None:
        return None
    return {
        "split_id": plan.split_id,
        "train": [bool(v) for v in plan.train_mask],
        "validation": [bool(v) for v in plan.validation_mask],
        "test": [bool(v) for v in plan.test_mask],
    }


def _restore_search_split_plan(
    config: SearchConfig, payload: Optional[Dict[str, Any]]
) -> None:
    """Reattach a serialized search-time SplitPlan to a restored config.

    Fail-closed: a malformed payload raises rather than leaving the overlap
    guard silently disarmed.
    """
    if payload is None:
        return
    if not isinstance(payload, dict) or set(payload) != {
        "split_id", "train", "validation", "test"
    }:
        raise ValueError("checkpoint search_split_masks must contain split_id/train/validation/test")
    split_id = payload["split_id"]
    if not isinstance(split_id, str) or not split_id.strip():
        raise ValueError("checkpoint search_split_masks split_id must be a non-empty string")
    masks = {}
    for name in ("train", "validation", "test"):
        mask = payload[name]
        if not isinstance(mask, list) or not mask or any(
            type(v) is not bool for v in mask
        ):
            raise ValueError(
                "checkpoint search_split_masks must be non-empty boolean lists"
            )
        masks[name] = list(mask)
    if len({len(masks[name]) for name in masks}) != 1:
        raise ValueError("checkpoint search_split_masks must have equal lengths")
    object.__setattr__(
        config,
        "_search_split_plan",
        SplitPlan(
            split_id,
            masks["train"],
            masks["validation"],
            masks["test"],
            {"restored_from_checkpoint": True},
        ),
    )


class SearchRunner:
    """Orchestrates mutation search with budget and stopping criteria."""

    def __init__(
        self,
        config: SearchConfig,
        proposal_fn: Callable[[], Trial],
        evaluation_fn: Callable[[Trial, int], Dict[str, Any]],
        plateau_detector: Optional[Callable[[List[float]], bool]] = None,
        trial_validator: Optional[Callable[[Trial], Dict[str, Any]]] = None,
        strategy: Optional[SearchStrategy] = None,
    ):
        if config.execution_mode is ExecutionMode.PRODUCTION:
            require_production_capability()
            if trial_validator is None:
                raise TypeError(
                    "production search requires a trial_validator with a "
                    "TrialValidatorIdentity; refusing to evaluate without a "
                    "grammar/legality validator"
                )
            identity = getattr(trial_validator, "identity", None)
            if not isinstance(identity, TrialValidatorIdentity):
                raise TypeError(
                    "production search requires a trial_validator exposing an "
                    "identity that is a TrialValidatorIdentity; got "
                    f"{type(identity).__name__ if identity is not None else None!r}"
                )
        if not isinstance(evaluation_fn, EvaluationProtocol):
            if config.execution_mode is ExecutionMode.PRODUCTION:
                raise TypeError(
                    "production search requires an EvaluationProtocol with a "
                    "validated SplitPlan; generic evaluation callbacks are "
                    "not accepted"
                )
            if config.require_evaluation_protocol:
                raise TypeError(
                    "safe search requires an EvaluationProtocol with a validated SplitPlan"
                )
            warnings.warn(
                "SearchRunner received a generic evaluation_fn; this escape "
                "hatch is deprecated and will be removed. Wrap the callback "
                "in an EvaluationProtocol with a validated SplitPlan.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.config = config
        self.proposal_fn = proposal_fn
        self.evaluation_fn = evaluation_fn
        self.plateau_detector = plateau_detector
        self.trial_validator = trial_validator
        self.strategy = strategy
        # FO-P0-03: the strategy's direction must be derived from the
        # authoritative ObjectiveSpec.  A strategy whose direction disagrees
        # with the config spec is rejected outright — the strategy can never
        # drift from the session's objective.
        if strategy is not None:
            strategy_dir = getattr(strategy, "direction", None)
            if strategy_dir is not None and strategy_dir != config.objective_spec.direction:
                raise ValueError(
                    "strategy direction and objective_spec disagree; the "
                    f"objective spec is the single direction authority "
                    f"(strategy={strategy_dir!r}, spec={config.objective_spec.direction!r})"
                )
        # Lazily-built default PlateauDetector (package semantics).
        self._default_plateau_detector: Optional[PlateauDetector] = None
        # R46 P0-R: capability-only data boundary.  The runner builds ONLY the
        # train and validation capabilities from the protocol's split plan.
        # The search-process object graph NEVER contains a TestDataCapability
        # or any test payload: obtaining a TEST capability requires a
        # TestAuthorityBroker, which the search worker does not hold.
        self._data_capabilities = None
        # R46 P0-S: the search worker never has a test provider.  This broker
        # is always None inside a search session; sealed test evaluation
        # happens only through a separate test authority.
        self._test_authority_broker: Optional[TestAuthorityBroker] = None
        # FO-P0-02: the real search-session identity used for capability
        # provenance.  Defaults to None (a placeholder that fails closed if a
        # capability is demanded before a session id is established).
        self._search_session_id: Optional[str] = None
        # The search-time split plan.  Every capability check in the runner's
        # evaluation path compares the requested coordinates against this plan
        # (R46 P0-Q exact-subset authorization).
        self._search_split_plan: Optional[SplitPlan] = None
        if isinstance(evaluation_fn, EvaluationProtocol):
            # Record the search-time data boundary so freeze_for_sealed_test
            # can reject sealed plans that overlap search data, and so every
            # capability evaluation can be authorized against the plan.
            self._search_split_plan = evaluation_fn.split_plan
            object.__setattr__(
                self.config, "_search_split_plan", evaluation_fn.split_plan
            )
            # FO-P0-02: capabilities are bound placeholder-free and re-issued
            # with the REAL session id once ``run``/``resume`` names it, so a
            # capability is always attributable to a real session/artifact and
            # never carries the placeholder strings "search"/"search_dataset".
            self._bind_capabilities(session_id="<pending>")

    def _bind_capabilities(self, session_id: str) -> None:
        """Build/re-issue the search data capabilities bound to a REAL session id.

        FO-P0-02: the capability provenance uses the real ``session_id`` and the
        split artifact, never the placeholder strings ``"search"`` /
        ``"search_dataset"``.  The search worker still holds ONLY train and
        validation capabilities (no TEST capability / no test provider).
        """
        if self._search_split_plan is None:
            self._data_capabilities = None
            return
        self._search_session_id = session_id
        self._data_capabilities = build_search_capabilities(
            self._search_split_plan,
            search_session_id=session_id,
            dataset_identity=f"dataset:{self._search_split_plan.split_id}",
            provider_identity=f"provider:{self._search_split_plan.split_id}",
        )

    def run(self, session_id: str) -> SearchSession:
        """Execute a new search until budget exhausted or plateau reached."""
        self._bind_capabilities(session_id=session_id)
        session = SearchSession(
            session_id=session_id,
            config=self.config,
            budget_tracker=BudgetTracker(budget=self.config.budget),
            strategy=self.strategy,
        )
        return self._run_session(session)

    def resume(
        self,
        session: SearchSession,
        *,
        resume_same_budget: bool = False,
        budget_extension: Optional["BudgetExtensionAuthorization"] = None,
    ) -> SearchSession:
        """Resume a quiescent, unfinished session with matching configuration.

        A session that was checkpointed after its budget was exhausted is
        resumable: the checkpoint/resume flow finishes a session when the
        budget runs out, then resumes it with a fresh runner.  Only a
        finished session with no strategy on either side is rejected (a
        terminal session with nothing to continue is meaningless).

        FO-P1-24 budget hardening.  The default (legacy) ``resume`` keeps the
        historical "fresh budget on resume" behavior for backward
        compatibility.  Two explicit opt-in modes prevent silently re-issuing
        the whole budget every resume:

        - ``resume_same_budget=True``: continue an UNFINISHED session with the
          SAME budget tracker (the partially-consumed budget is preserved).  A
          finished session is rejected under this mode — you cannot re-arm the
          entire budget by merely resuming.
        - ``budget_extension=<BudgetExtensionAuthorization>``: extend the
          budget of a FINISHED session to a new (larger) budget.  The extension
          must be authorized (signed content hash) and the new budget must not
          shrink the session's consumption limits.
        """
        if not isinstance(session, SearchSession):
            raise TypeError("session must be a SearchSession")
        if (
            session.is_finished()
            and session.strategy is None
            and self.strategy is None
            and budget_extension is None
        ):
            raise ValueError("cannot resume a finished session")
        if session.budget_tracker.evaluations_reserved or session.budget_tracker.cost_reserved:
            raise ValueError("cannot resume with active evaluation reservations")
        if any(not trial.is_terminal() for trial in session.trials + session.duplicate_trials):
            raise ValueError("cannot resume with non-terminal trials")
        if session.config.to_dict() != self.config.to_dict():
            raise ValueError("session configuration does not match runner configuration")
        if session.strategy is None and self.strategy is not None:
            # Legacy checkpoint (SearchSession.from_dict) restored without a
            # strategy: keep working by adopting the runner's strategy.
            session.strategy = self.strategy
        # FO-P0-02: re-issue the capabilities bound to the REAL session id being
        # resumed (never a placeholder).
        self._bind_capabilities(session_id=session.session_id)

        if resume_same_budget and budget_extension is not None:
            raise ValueError(
                "resume_same_budget and budget_extension are mutually exclusive"
            )
        if budget_extension is not None:
            # FO-P1-24: extending a budget requires an authorized extension.
            # A finished session is extended to the extension's new budget;
            # an unfinished session is NOT re-armed.
            budget_extension.verify()
            if budget_extension.search_session_id != session.session_id:
                raise ValueError(
                    "budget extension search_session_id does not match the "
                    "session being resumed"
                )
            if budget_extension.old_budget.to_dict() != session.budget_tracker.budget.to_dict():
                raise ValueError(
                    "budget extension old_budget does not match the session's "
                    "current budget"
                )
            self._validate_extension_growth(budget_extension)
            session.budget_tracker = BudgetTracker(budget_extension.new_budget)
            session.finished_at = None
            session.stop_reason = None
            return self._run_session(session)
        if resume_same_budget:
            # FO-P1-24: continuing an UNFINISHED session keeps its budget.  A
            # finished session cannot re-arm its entire budget this way.
            if session.is_finished():
                raise ValueError(
                    "cannot resume a finished session with resume_same_budget=True; "
                    "re-issuing the whole budget requires a BudgetExtensionAuthorization"
                )
            return self._run_session(session)

        # Legacy backward-compatible path: a checkpointed session that
        # exhausted its budget is resumed with a fresh budget.  The
        # already-evaluated trials are preserved; only the consumption counters
        # and the terminal flags are reset so the runner can propose again.
        if session.is_finished():
            session.budget_tracker = BudgetTracker(self.config.budget)
            session.finished_at = None
            session.stop_reason = None
        return self._run_session(session)

    def _validate_extension_growth(self, extension: "BudgetExtensionAuthorization") -> None:
        """Reject an extension that shrinks the budget (fail-closed)."""
        old = extension.old_budget
        new = extension.new_budget
        if (
            new.max_trials < old.max_trials
            or new.max_evaluations < old.max_evaluations
            or new.max_cost_units < old.max_cost_units
        ):
            raise ValueError(
                "budget extension must not shrink max_trials/max_evaluations/"
                "max_cost_units below the current budget"
            )

    def create_train_evaluation_context(self) -> "TrainEvaluationContext":
        """Create an isolated evaluation context for training data."""
        return TrainEvaluationContext(self)

    def create_validation_evaluation_context(self) -> "ValidationEvaluationContext":
        """Create an isolated evaluation context for validation data."""
        return ValidationEvaluationContext(self)

    def create_sealed_test_executor(self) -> "SealedTestExecutor":
        """Create an isolated executor for sealed test data.

        R46 P0-S: the search runner never holds a test authority broker, so a
        sealed-test executor created from the search runner is inert — it has
        no test provider and cannot resolve test data.  Sealed test evaluation
        happens only through a separate test authority that supplies a
        ``TestAuthorityBroker``.
        """
        return SealedTestExecutor(self, self._test_authority_broker)

    def _run_session(self, session: SearchSession) -> SearchSession:
        """Execute a session from its current quiescent state."""
        budget_tracker = session.budget_tracker
        recent_scores = session.recent_scores

        while not session.is_finished():
            if budget_tracker.is_exhausted():
                session.finish(reason="budget_exhausted")
                break
            if self._check_plateau(recent_scores):
                session.finish(reason="plateau_detected")
                break
            if not budget_tracker.can_propose_trial():
                session.finish(reason="max_trials_reached")
                break

            try:
                trial = self.proposal_fn()
            except Exception as exc:
                budget_tracker.record_trial()
                # FO-P0-04: a raised proposal is a distinct burned-budget
                # outcome, recorded in the append-only ledger as
                # PROPOSAL_FAILED (NOT a duplicate).  The historical
                # "proposal-failure" duplicate-slot Trial is replaced by the
                # typed ledger entry so the burned budget is visible and
                # attributable without mislabelling it as a duplicate.
                session.ledger.append_proposal_failed(
                    f"proposal_fn error: {exc}"
                )
                continue
            budget_tracker.record_trial()
            if not isinstance(trial, Trial):
                # FO-P0-04: a proposal that returned a non-Trial is an
                # INVALID_PROPOSAL outcome, recorded in the ledger rather than
                # silently dropped.
                session.ledger.append_invalid_proposal(
                    f"proposal_fn returned {type(trial).__name__}, not a Trial"
                )
                continue
            if session.has_trial(trial.trial_id):
                duplicate = Trial.from_dict(trial.to_dict())
                duplicate.update_status(
                    TrialStatus.DUPLICATE,
                    failure_reason=f"trial_id already recorded: {trial.trial_id}",
                )
                session.duplicate_trials.append(duplicate)
                session.ledger.append_trial("DUPLICATE", trial.trial_id)
                continue
            session.add_trial(trial)
            session.ledger.append_trial("PROPOSED", trial.trial_id)

            trial.update_status(TrialStatus.VALIDATING)
            legality = self._validate_trial(trial)
            if not legality["is_legal"]:
                trial.update_status(TrialStatus.ILLEGAL, legality_check=legality)
                session.ledger.append_trial("ILLEGAL", trial.trial_id)
                continue
            trial.update_status(TrialStatus.LEGAL, legality_check=legality)

            scheduler = MultiFidelityScheduler() if self.config.enable_multifidelity else None
            fidelity = 0 if scheduler is not None else 4
            result = None
            artifact = None
            while True:
                reserved_cost = self.config.evaluation_cost_units
                if reserved_cost is None:
                    reserved_cost = budget_tracker.remaining_cost()
                if not budget_tracker.reserve_evaluation(reserved_cost):
                    trial.update_status(TrialStatus.FAILED, failure_reason="evaluation budget unavailable")
                    session.ledger.append_trial(
                        "EVALUATION_FAILED", trial.trial_id,
                        failure_reason="evaluation budget unavailable",
                    )
                    session.finish(reason="evaluation_budget_unavailable")
                    break

                trial.update_status(TrialStatus.EVALUATING)
                try:
                    train_capability = None
                    if self._data_capabilities is not None:
                        train_capability = self._data_capabilities.get(
                            DataScope.TRAIN
                        )
                    if train_capability is None:
                        raise ValueError(
                            "no TRAIN data capability available; the search "
                            "runner cannot evaluate without a "
                            "capability-authorized train boundary"
                        )
                    raw_result = self._evaluate_with_capability(
                        train_capability, trial, fidelity
                    )
                    # FO-P1-25: every evaluation output is normalized into a
                    # typed TrialEvaluationArtifact.  The runner reads ONLY the
                    # artifact's fields (never bare magic dict keys), so an
                    # evaluator that renames/removes a key is caught loudly,
                    # never silently breaking the run.
                    artifact = normalize_evaluation_result(
                        trial.trial_id,
                        raw_result,
                        objective_name=self.config.objective_spec.metric_name,
                        objective_spec_ref=self.config.objective_spec.metric_name,
                        split_ref=getattr(self._search_split_plan, "split_id", None),
                        fidelity=fidelity,
                    )
                    actual_cost = artifact.compute_cost
                    budget_tracker.commit_evaluation(reserved_cost, float(actual_cost))
                except Exception as exc:
                    if budget_tracker.evaluations_reserved:
                        budget_tracker.release_evaluation(reserved_cost)
                    trial.update_status(TrialStatus.FAILED, failure_reason=str(exc))
                    session.ledger.append_trial(
                        "EVALUATION_FAILED", trial.trial_id, failure_reason=str(exc)
                    )
                    break

                try:
                    raw_score = artifact.primary_objective_value
                    # bool is an int subclass; True silently coerces to 1.0
                    # and produces a checkpoint SearchSession.from_dict
                    # rejects, making the session un-deserializable.
                    # np.bool_ does not subclass Python bool but also coerces.
                    if isinstance(raw_score, bool) or (
                        hasattr(raw_score, "dtype")
                        and getattr(raw_score, "dtype", None) is not None
                        and raw_score.dtype.kind == "b"
                    ):
                        raise ValueError("evaluation result score must not be boolean")
                    score = float(raw_score)
                    if not isfinite(score):
                        raise ValueError("evaluation result score must be finite")
                    promotion = artifact.promotion_evidence or {}
                    rank = promotion.get("rank")
                    total = promotion.get("total")
                    peer_evidence_valid = (
                        isinstance(rank, int)
                        and not isinstance(rank, bool)
                        and isinstance(total, int)
                        and not isinstance(total, bool)
                        and total > 0
                        and 0 <= rank < total
                    )
                except (TypeError, ValueError, KeyError, AttributeError,
                        OverflowError) as exc:
                    # A lazy score object raising anything else (e.g. a
                    # RuntimeError from a user evaluator) escapes to the
                    # caller instead of silently killing the whole run —
                    # mark FAILED only for the expected coercion failures.
                    trial.update_status(TrialStatus.FAILED, failure_reason=str(exc))
                    session.ledger.append_trial(
                        "EVALUATION_FAILED", trial.trial_id, failure_reason=str(exc)
                    )
                    break

                # Promotion is opt-in and requires coherent peer rank evidence.
                next_tier = scheduler.next_tier(FidelityTier(fidelity)) if scheduler else None
                can_promote = (
                    scheduler is not None
                    and bool(promotion.get("promote", False))
                    and next_tier is not None
                    and peer_evidence_valid
                    and scheduler.should_promote(
                        FidelityTier(fidelity), score, rank, total,
                        promotion.get("baseline_score")
                    )
                )
                if can_promote:
                    fidelity = next_tier.value
                    continue
                break

            if session.is_finished() and (artifact is None or trial.status == TrialStatus.FAILED):
                break
            if artifact is None or trial.status == TrialStatus.FAILED:
                continue

            evaluation_ref = artifact.evidence_ref
            if not isinstance(evaluation_ref, str) or not evaluation_ref.strip():
                trial.update_status(
                    TrialStatus.FAILED,
                    failure_reason=(
                        "evaluation result must include a non-empty evidence_ref "
                        "or evaluation_id"
                    ),
                )
                session.ledger.append_trial(
                    "EVALUATION_FAILED", trial.trial_id,
                    failure_reason="missing evidence_ref",
                )
                continue

            trial.update_status(
                TrialStatus.EVALUATED,
                evaluation_ref=evaluation_ref,
                metadata={
                    "score": artifact.primary_objective_value,
                    "fidelity": fidelity,
                    "evaluation_artifact": artifact.to_dict(),
                },
            )
            session.ledger.append_trial("EVALUATED", trial.trial_id)
            score = artifact.primary_objective_value
            # FO-P1-25: the trial's metadata.score is authoritative — the
            # runner no longer reads a bare "score" magic key.
            trial.metadata["score"] = score
            recent_scores.append(score)
            if len(recent_scores) > self.config.plateau_window:
                recent_scores.pop(0)
            improved = session.update_best(trial.trial_id, score)
            if improved:
                session.ledger.append_trial("SELECTED", trial.trial_id)
            if session.strategy is not None:
                session.strategy.record(
                    trial.metadata.get("params", {}), float(score)
                )

        if not session.is_finished():
            session.finish(reason="manual_stop")
        return session

    def _validate_trial(self, trial: Trial) -> Dict[str, Any]:
        """Validate trial structure and, when supplied, the legality contract."""
        errors: List[str] = []
        if not isinstance(trial, Trial):
            return {"is_legal": False, "errors": ["trial must be a Trial"]}
        if not isinstance(trial.trial_id, str) or not trial.trial_id.strip():
            errors.append("trial_id is required")
        if not isinstance(trial.mutation_id, str) or not trial.mutation_id.strip():
            errors.append("mutation_id is required")
        if not isinstance(trial.parent_factor_ids, list) or not all(
            isinstance(value, str) for value in trial.parent_factor_ids
        ):
            errors.append("parent_factor_ids must be a list of strings")
        if not isinstance(trial.metadata, dict):
            errors.append("metadata must be a dictionary")
        if trial.status is not TrialStatus.VALIDATING:
            errors.append("trial must be validating before legality check")
        if self.trial_validator is not None and not errors:
            try:
                result = self.trial_validator(trial)
            except Exception as exc:
                return {"is_legal": False, "errors": [f"validator error: {exc}"]}
            if not isinstance(result, dict) or not isinstance(result.get("is_legal"), bool):
                return {"is_legal": False, "errors": ["validator must return boolean is_legal"]}
            # P0-11 audit trail: pin the validator identity onto the legality
            # result so every admission decision is attributable.
            identity = getattr(self.trial_validator, "identity", None)
            if isinstance(identity, TrialValidatorIdentity):
                result["validator_id"] = identity.validator_id
                result["validator_version"] = identity.validator_version
                result["implementation_hash"] = identity.implementation_hash
            # P0-11: a research_only legality verdict is not acceptable in
            # production; the trial must not proceed.
            if (
                self.config.execution_mode is ExecutionMode.PRODUCTION
                and result.get("research_only") is True
            ):
                return {
                    "is_legal": False,
                    "errors": [
                        "research_only legality is not acceptable in production"
                    ],
                    "research_only": False,
                    "validator_id": result.get("validator_id"),
                    "validator_version": result.get("validator_version"),
                    "implementation_hash": result.get("implementation_hash"),
                }
            return result
        if self.trial_validator is None:
            # Structural validation is the only honest fallback; it is not a
            # production legality claim and callers should supply the grammar validator.
            if self.config.execution_mode is ExecutionMode.PRODUCTION:
                return {
                    "is_legal": False,
                    "errors": ["production requires a trial_validator"],
                    "research_only": False,
                }
            return {
                "is_legal": not errors,
                "checks_passed": ["trial_structure"] if not errors else [],
                "errors": errors,
                "research_only": True,
            }
        return {"is_legal": not errors, "errors": errors}

    def _check_plateau(self, recent_scores: List[float]) -> bool:
        """Check if search has plateaued.

        Canonical semantics: delegate to the package ``PlateauDetector``
        (split-half comparison) with this config's window/threshold mapped
        onto ``PlateauConfig``. The package detector is now the single
        source of truth; the pre-duplication max/min window heuristic no
        longer runs here.
        """
        if len(recent_scores) < self.config.plateau_window:
            return False
        if self.plateau_detector is not None:
            return self.plateau_detector(recent_scores)
        if not recent_scores:
            return False
        if self._default_plateau_detector is None:
            self._default_plateau_detector = PlateauDetector(
                PlateauConfig(
                    window_size=max(self.config.plateau_window, 2),
                    min_relative_improvement=self.config.plateau_threshold,
                )
            )
        return self._default_plateau_detector.is_plateau(recent_scores)

    def _evaluate_with_capability(
        self, capability: DataCapability, trial: Trial, fidelity: int
    ) -> Dict[str, Any]:
        """Evaluate ``trial`` only after the capability authorizes it.

        R46 P0-Q/P0-R/P0-S: every evaluation path in the runner funnels
        through this method.  The capability's real identity is verified
        (rejecting forgery) and the requested split plan must be an EXACT
        SUBSET of the authorized coordinates (rejecting wrong-scope or
        out-of-bound rows) BEFORE the underlying evaluation callback is
        invoked.  This closes the isolation gap where a context/executor
        called ``evaluation_fn.evaluate`` directly with no capability check.
        """
        if not isinstance(capability, DataCapability):
            raise TypeError(
                f"capability must be a DataCapability, got "
                f"{type(capability).__name__}"
            )
        # Fail closed on a forged/altered identity (mask/scope tampering).
        capability.verify_identity()
        plan = self._search_split_plan
        if not isinstance(plan, SplitPlan):
            raise ValueError(
                "no search SplitPlan available to authorize capability "
                f"{capability.scope.value!r} evaluation"
            )
        if not capability.can_evaluate(plan):
            raise ValueError(
                f"capability scope={capability.scope.value!r} does not "
                "authorize the current search split plan (exact-subset "
                "authorization failed)"
            )
        if self._data_capabilities is None:
            raise ValueError(
                "no data capabilities available; refusing to evaluate without "
                "a capability-authorized boundary"
            )
        if isinstance(self.evaluation_fn, EvaluationProtocol):
            return self.evaluation_fn.evaluate(trial, fidelity)
        return self.evaluation_fn(trial, fidelity)


class TrainEvaluationContext:
    """Isolated evaluation context for training data.

    Ensures that training data cannot be used for validation or test evaluation.
    """

    def __init__(self, runner: SearchRunner):
        self._runner = runner
        self._split_plan = None
        self._capability: Optional[DataCapability] = None
        if isinstance(runner.evaluation_fn, EvaluationProtocol):
            self._split_plan = runner.evaluation_fn.split_plan
            if runner._data_capabilities is not None:
                self._capability = runner._data_capabilities.get(DataScope.TRAIN)

    def evaluate(self, trial: Trial, fidelity: int) -> Dict[str, Any]:
        """Evaluate a trial using only training data.

        R46 P0-Q/P0-R: goes through the runner's capability-authorized
        evaluation path using the TRAIN capability.  A caller that swaps in a
        VALIDATION/TEST capability is rejected by the capability's exact-subset
        authorization.
        """
        if self._split_plan is None:
            raise ValueError("TrainEvaluationContext requires a SplitPlan")
        # Ensure we're only using train_mask
        if not any(self._split_plan.train_mask):
            raise ValueError("SplitPlan has no training data")
        if self._capability is None:
            raise ValueError(
                "TrainEvaluationContext requires a TRAIN data capability"
            )
        if self._capability.scope is not DataScope.TRAIN:
            raise ValueError(
                "TrainEvaluationContext requires a TRAIN capability; got "
                f"scope={self._capability.scope.value!r}"
            )
        return self._runner._evaluate_with_capability(
            self._capability, trial, fidelity
        )


class ValidationEvaluationContext:
    """Isolated evaluation context for validation data.

    Ensures that validation data cannot be used for training or test evaluation.
    """

    def __init__(self, runner: SearchRunner):
        self._runner = runner
        self._split_plan = None
        self._capability: Optional[DataCapability] = None
        if isinstance(runner.evaluation_fn, EvaluationProtocol):
            self._split_plan = runner.evaluation_fn.split_plan
            if runner._data_capabilities is not None:
                self._capability = runner._data_capabilities.get(
                    DataScope.VALIDATION
                )

    def evaluate(self, trial: Trial, fidelity: int) -> Dict[str, Any]:
        """Evaluate a trial using only validation data.

        R46 P0-Q/P0-R: goes through the runner's capability-authorized
        evaluation path using the VALIDATION capability.
        """
        if self._split_plan is None:
            raise ValueError("ValidationEvaluationContext requires a SplitPlan")
        # Ensure we're only using validation_mask
        if not any(self._split_plan.validation_mask):
            raise ValueError("SplitPlan has no validation data")
        if self._capability is None:
            raise ValueError(
                "ValidationEvaluationContext requires a VALIDATION data capability"
            )
        if self._capability.scope is not DataScope.VALIDATION:
            raise ValueError(
                "ValidationEvaluationContext requires a VALIDATION capability; "
                f"got scope={self._capability.scope.value!r}"
            )
        return self._runner._evaluate_with_capability(
            self._capability, trial, fidelity
        )


class SealedTestExecutor:
    """Isolated executor for sealed test data.

    R46 P0-S: sealed test evaluation happens only through a separate test
    authority.  A ``SealedTestExecutor`` created from a search runner carries
    no ``TestAuthorityBroker`` and therefore has no test provider — it cannot
    resolve test data.  A test authority constructs a ``SealedTestExecutor``
    with a broker, which supplies the physical test-data path.
    """

    def __init__(
        self,
        runner: SearchRunner,
        test_authority_broker: Optional[TestAuthorityBroker] = None,
    ):
        self._runner = runner
        self._split_plan = None
        if isinstance(runner.evaluation_fn, EvaluationProtocol):
            self._split_plan = runner.evaluation_fn.split_plan
        self._test_authority_broker = test_authority_broker

    @property
    def has_test_authority(self) -> bool:
        """True only when a test authority broker is attached."""
        return self._test_authority_broker is not None

    def evaluate_sealed_test(
        self,
        session: SearchSession,
        handle: SealedTestHandle,
        split_plan: SplitPlan,
    ) -> SealedTestResult:
        """Evaluate the sealed test using only test data.

        Requires a test authority broker (the physical test-data path).  A
        search-runner-created executor without a broker raises rather than
        silently evaluating against search data.

        R46 P0-S: with a broker, evaluation is fully capability-authorized.
        The broker issues a TEST capability, resolves the test payload through
        a provider gated on that capability, and only then runs the underlying
        evaluation callback against the resolved test data.  This replaces the
        old direct ``session.consume_sealed_test(..., evaluation_fn)`` call
        which bypassed the capability boundary entirely.
        """
        if self._test_authority_broker is None:
            raise ValueError(
                "SealedTestExecutor has no test authority broker; sealed test "
                "evaluation requires a separate test authority, not the search "
                "runner"
            )
        if self._split_plan is None:
            raise ValueError("SealedTestExecutor requires a SplitPlan")
        # Ensure we're only using test_mask
        if not any(split_plan.test_mask):
            raise ValueError("SplitPlan has no test data")
        validate_split_plan(split_plan)
        if not isinstance(handle, SealedTestHandle):
            raise TypeError("handle must be a SealedTestHandle")
        if session.frozen_at is None:
            raise ValueError("search session is not frozen")
        if session.sealed_test_consumed:
            raise ValueError("sealed test handle has already been consumed")
        if session.sealed_trial_id != handle.trial_id:
            raise ValueError("sealed test handle does not match frozen session")
        winner = next(
            (
                trial
                for trial in session.successful_trials()
                if trial.trial_id == handle.trial_id
            ),
            None,
        )
        if winner is None or winner.evaluation_ref != handle.evaluation_ref:
            raise ValueError(
                "sealed test handle evidence does not match frozen winner"
            )
        # R46 P0-S: route the physical test-data path through the broker.
        broker = self._test_authority_broker
        test_capability = broker.issue_test_capability(split_plan.test_mask)
        provider = broker.create_test_provider({"test_segment": handle.split_id})
        data = provider.resolve(test_capability)
        scoped = ScopedEvaluator(provider, self._runner.evaluation_fn.evaluator)
        metrics = scoped.evaluate(test_capability, winner, 0)
        if not isinstance(metrics, dict) or not all(
            isinstance(name, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            for name, value in metrics.items()
        ):
            raise ValueError(
                "sealed test evaluator must return finite numeric metrics"
            )
        # Consume the seal via the guarded path (False->True is the one legal
        # transition; any reset attempt stays blocked by __setattr__).
        object.__setattr__(session, "sealed_test_consumed", True)
        result = SealedTestResult(
            trial_id=winner.trial_id,
            test_metrics=dict(metrics),
            frozen_at=session.frozen_at,
            search_session_id=session.session_id,
            split_id=split_plan.split_id,
            evaluation_ref=winner.evaluation_ref,
        )
        session.sealed_test_results.append(
            {
                "trial_id": result.trial_id,
                "split_id": result.split_id,
                "evaluation_ref": result.evaluation_ref,
                "frozen_at": result.frozen_at.isoformat(),
                "test_metrics": dict(result.test_metrics),
            }
        )
        return result
