"""SearchRunner: orchestrate mutation search with budget and plateau stopping."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Callable, Dict, List, Optional

from factor_optimizer.capabilities import ExecutionMode, require_production_capability
from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.multifidelity import FidelityTier, MultiFidelityScheduler


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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchConfig":
        """Deserialize search configuration."""
        values = dict(data)
        values["budget"] = SearchBudget.from_dict(values["budget"])
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

    def add_trial(self, trial: Trial) -> None:
        """Add a trial to the session, rejecting reused trial IDs."""
        if any(existing.trial_id == trial.trial_id for existing in self.trials):
            raise ValueError(f"trial_id already recorded: {trial.trial_id}")
        self.trials.append(trial)

    def has_trial(self, trial_id: str) -> bool:
        """Return whether a trial ID has already been recorded."""
        return any(trial.trial_id == trial_id for trial in self.trials)

    def update_best(self, trial_id: str, score: float) -> bool:
        """Update best score if improved. Returns True if new best."""
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_trial_id = trial_id
            return True
        return False

    def successful_trials(self) -> List[Trial]:
        """Return all successfully evaluated trials."""
        return [t for t in self.trials if t.is_successful()]

    def finish(self, reason: str) -> None:
        """Mark session as finished."""
        self.finished_at = datetime.now()
        self.stop_reason = reason

    def is_finished(self) -> bool:
        """Check if session is complete."""
        return self.finished_at is not None

    def duration_seconds(self) -> float:
        """Return session duration in seconds."""
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSession":
        """Deserialize and validate a session checkpoint."""
        values = dict(data)
        required = {"session_id", "config", "budget_tracker", "trials", "duplicate_trials",
                    "best_score", "best_trial_id", "started_at", "finished_at",
                    "stop_reason", "recent_scores"}
        missing = required.difference(values)
        if missing:
            raise ValueError(f"session checkpoint missing fields: {sorted(missing)}")
        started_at = values["started_at"]
        finished_at = values["finished_at"]
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        if isinstance(finished_at, str):
            finished_at = datetime.fromisoformat(finished_at)
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
        )
        return session


class SearchRunner:
    """Orchestrates mutation search with budget and stopping criteria."""

    def __init__(
        self,
        config: SearchConfig,
        proposal_fn: Callable[[], Trial],
        evaluation_fn: Callable[[Trial, int], Dict[str, Any]],
        plateau_detector: Optional[Callable[[List[float]], bool]] = None,
        trial_validator: Optional[Callable[[Trial], Dict[str, Any]]] = None,
    ):
        if config.execution_mode is ExecutionMode.PRODUCTION:
            require_production_capability()
        self.config = config
        self.proposal_fn = proposal_fn
        self.evaluation_fn = evaluation_fn
        self.plateau_detector = plateau_detector
        self.trial_validator = trial_validator

    def run(self, session_id: str) -> SearchSession:
        """Execute a new search until budget exhausted or plateau reached."""
        session = SearchSession(
            session_id=session_id,
            config=self.config,
            budget_tracker=BudgetTracker(budget=self.config.budget),
        )
        return self._run_session(session)

    def resume(self, session: SearchSession) -> SearchSession:
        """Resume a quiescent, unfinished session with matching configuration."""
        if not isinstance(session, SearchSession):
            raise TypeError("session must be a SearchSession")
        if session.is_finished():
            raise ValueError("cannot resume a finished session")
        if session.budget_tracker.evaluations_reserved or session.budget_tracker.cost_reserved:
            raise ValueError("cannot resume with active evaluation reservations")
        if any(not trial.is_terminal() for trial in session.trials + session.duplicate_trials):
            raise ValueError("cannot resume with non-terminal trials")
        if session.config.to_dict() != self.config.to_dict():
            raise ValueError("session configuration does not match runner configuration")
        return self._run_session(session)

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
            except Exception:
                budget_tracker.record_trial()
                continue
            budget_tracker.record_trial()
            if not isinstance(trial, Trial):
                continue
            if session.has_trial(trial.trial_id):
                duplicate = Trial.from_dict(trial.to_dict())
                duplicate.update_status(
                    TrialStatus.DUPLICATE,
                    failure_reason=f"trial_id already recorded: {trial.trial_id}",
                )
                session.duplicate_trials.append(duplicate)
                continue
            session.add_trial(trial)

            trial.update_status(TrialStatus.VALIDATING)
            legality = self._validate_trial(trial)
            if not legality["is_legal"]:
                trial.update_status(TrialStatus.ILLEGAL, legality_check=legality)
                continue
            trial.update_status(TrialStatus.LEGAL, legality_check=legality)

            scheduler = MultiFidelityScheduler() if self.config.enable_multifidelity else None
            fidelity = 0 if scheduler is not None else 4
            result = None
            while True:
                reserved_cost = self.config.evaluation_cost_units
                if reserved_cost is None:
                    reserved_cost = budget_tracker.remaining_cost()
                if not budget_tracker.reserve_evaluation(reserved_cost):
                    trial.update_status(TrialStatus.FAILED, failure_reason="evaluation budget unavailable")
                    session.finish(reason="evaluation_budget_unavailable")
                    break

                trial.update_status(TrialStatus.EVALUATING)
                try:
                    result = self.evaluation_fn(trial, fidelity)
                    actual_cost = result.get("cost")
                    if actual_cost is None:
                        raise ValueError("evaluation result cost is unknown")
                    budget_tracker.commit_evaluation(reserved_cost, float(actual_cost))
                except Exception as exc:
                    if budget_tracker.evaluations_reserved:
                        budget_tracker.release_evaluation(reserved_cost)
                    trial.update_status(TrialStatus.FAILED, failure_reason=str(exc))
                    break

                try:
                    score = float(result["score"])
                    if not isfinite(score):
                        raise ValueError("evaluation result score must be finite")
                    rank = result.get("rank")
                    total = result.get("total")
                    peer_evidence_valid = (
                        isinstance(rank, int)
                        and not isinstance(rank, bool)
                        and isinstance(total, int)
                        and not isinstance(total, bool)
                        and total > 0
                        and 0 <= rank < total
                    )
                except Exception as exc:
                    trial.update_status(TrialStatus.FAILED, failure_reason=str(exc))
                    break

                # Promotion is opt-in and requires coherent peer rank evidence.
                next_tier = scheduler.next_tier(FidelityTier(fidelity)) if scheduler else None
                can_promote = (
                    scheduler is not None
                    and bool(result.get("promote", False))
                    and next_tier is not None
                    and peer_evidence_valid
                    and scheduler.should_promote(
                        FidelityTier(fidelity), score, rank, total,
                        result.get("baseline_score")
                    )
                )
                if can_promote:
                    fidelity = next_tier.value
                    continue
                break

            if session.is_finished() and (result is None or trial.status == TrialStatus.FAILED):
                break
            if result is None or trial.status == TrialStatus.FAILED:
                continue

            evaluation_ref = result.get("evidence_ref", result.get("evaluation_id"))
            if not isinstance(evaluation_ref, str) or not evaluation_ref.strip():
                trial.update_status(
                    TrialStatus.FAILED,
                    failure_reason=(
                        "evaluation result must include a non-empty evidence_ref "
                        "or evaluation_id"
                    ),
                )
                continue

            trial.update_status(
                TrialStatus.EVALUATED,
                evaluation_ref=evaluation_ref,
                metadata={"score": result.get("score"), "fidelity": fidelity},
            )
            score = result.get("score")
            if score is not None:
                recent_scores.append(score)
                if len(recent_scores) > self.config.plateau_window:
                    recent_scores.pop(0)
                session.update_best(trial.trial_id, score)

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
            return result
        if self.trial_validator is None:
            # Structural validation is the only honest fallback; it is not a
            # production legality claim and callers should supply the grammar validator.
            return {
                "is_legal": not errors,
                "checks_passed": ["trial_structure"] if not errors else [],
                "errors": errors,
                "research_only": True,
            }
        return {"is_legal": not errors, "errors": errors}

    def _check_plateau(self, recent_scores: List[float]) -> bool:
        """Check if search has plateaued."""
        if len(recent_scores) < self.config.plateau_window:
            return False
        if self.plateau_detector is not None:
            return self.plateau_detector(recent_scores)
        if not recent_scores:
            return False

        max_score = max(recent_scores)
        min_score = min(recent_scores)
        if max_score == 0:
            return min_score == 0
        relative_improvement = (max_score - min_score) / abs(max_score)
        return relative_improvement < self.config.plateau_threshold
