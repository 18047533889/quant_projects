"""SearchRunner: orchestrate mutation search with budget and plateau stopping."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Callable, Dict, List, Optional

from factor_optimizer.capabilities import ExecutionMode, require_production_capability
from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.trial import Trial, TrialStatus


@dataclass
class SearchConfig:
    """Configuration for search execution."""

    budget: SearchBudget
    plateau_window: int = 20
    plateau_threshold: float = 0.001
    enable_multifidelity: bool = True
    max_concurrency: int = 4
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
        if self.evaluation_cost_units is not None and (
            not isfinite(self.evaluation_cost_units) or self.evaluation_cost_units < 0
        ):
            raise ValueError("evaluation_cost_units must be finite and >= 0")


@dataclass
class SearchSession:
    """Runtime state for an active search session."""

    session_id: str
    config: SearchConfig
    budget_tracker: BudgetTracker
    trials: List[Trial] = field(default_factory=list)
    best_score: Optional[float] = None
    best_trial_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    stop_reason: Optional[str] = None

    def add_trial(self, trial: Trial) -> None:
        """Add a trial to the session."""
        self.trials.append(trial)

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


class SearchRunner:
    """Orchestrates mutation search with budget and stopping criteria."""

    def __init__(
        self,
        config: SearchConfig,
        proposal_fn: Callable[[], Trial],
        evaluation_fn: Callable[[Trial, int], Dict[str, Any]],
        plateau_detector: Optional[Callable[[List[float]], bool]] = None,
    ):
        if config.execution_mode is ExecutionMode.PRODUCTION:
            require_production_capability()
        self.config = config
        self.proposal_fn = proposal_fn
        self.evaluation_fn = evaluation_fn
        self.plateau_detector = plateau_detector

    def run(self, session_id: str) -> SearchSession:
        """Execute search until budget exhausted or plateau reached."""
        budget_tracker = BudgetTracker(budget=self.config.budget)
        session = SearchSession(
            session_id=session_id,
            config=self.config,
            budget_tracker=budget_tracker,
        )
        recent_scores: List[float] = []

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

            trial = self.proposal_fn()
            budget_tracker.record_trial()
            session.add_trial(trial)

            trial.update_status(TrialStatus.VALIDATING)
            legality = self._validate_trial(trial)
            if not legality["is_legal"]:
                trial.update_status(TrialStatus.ILLEGAL, legality_check=legality)
                continue
            trial.update_status(TrialStatus.LEGAL, legality_check=legality)

            reserved_cost = self.config.evaluation_cost_units
            if reserved_cost is None:
                reserved_cost = budget_tracker.remaining_cost()
            if not budget_tracker.reserve_evaluation(reserved_cost):
                trial.update_status(TrialStatus.FAILED, failure_reason="evaluation budget unavailable")
                session.finish(reason="evaluation_budget_unavailable")
                break

            trial.update_status(TrialStatus.EVALUATING)
            fidelity = 0 if self.config.enable_multifidelity else 4
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
                continue

            trial.update_status(
                TrialStatus.EVALUATED,
                evaluation_ref=result.get("evaluation_id"),
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
        """Validate trial legality (stub for now)."""
        return {
            "is_legal": True,
            "checks_passed": ["syntax", "grammar", "complexity"],
        }

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
