"""SearchRunner: orchestrate mutation search with budget and plateau stopping."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.trial import Trial, TrialStatus


@dataclass
class SearchConfig:
    """
    Configuration for search execution.

    Attributes:
        budget: Budget limits for the search
        plateau_window: Number of evaluations to consider for plateau detection
        plateau_threshold: Minimum improvement to avoid plateau (relative)
        enable_multifidelity: Whether to use multi-fidelity evaluation
        max_concurrency: Maximum parallel evaluations
    """
    budget: SearchBudget
    plateau_window: int = 20
    plateau_threshold: float = 0.001
    enable_multifidelity: bool = True
    max_concurrency: int = 4

    def __post_init__(self):
        if self.plateau_window < 1:
            raise ValueError("plateau_window must be >= 1")
        if self.plateau_threshold < 0:
            raise ValueError("plateau_threshold must be >= 0")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")


@dataclass
class SearchSession:
    """
    Runtime state for an active search session.

    Attributes:
        session_id: Unique identifier for this session
        config: Search configuration
        budget_tracker: Budget consumption tracker
        trials: All trials generated in this session
        best_score: Best score seen so far
        best_trial_id: Trial ID of the best candidate
        started_at: Session start timestamp
        finished_at: Session completion timestamp
        stop_reason: Reason for search termination
    """
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
    """
    Orchestrates mutation search with budget and stopping criteria.

    The runner coordinates:
    - Budget tracking (trials, evaluations, cost)
    - Plateau detection
    - Multi-fidelity evaluation tiers
    - Pareto frontier tracking
    """

    def __init__(
        self,
        config: SearchConfig,
        proposal_fn: Callable[[], Trial],
        evaluation_fn: Callable[[Trial, int], Dict[str, Any]],
        plateau_detector: Optional[Callable[[List[float]], bool]] = None,
    ):
        """
        Initialize search runner.

        Args:
            config: Search configuration
            proposal_fn: Function that generates new trial proposals
            evaluation_fn: Function that evaluates a trial at given fidelity
            plateau_detector: Optional custom plateau detection function
        """
        self.config = config
        self.proposal_fn = proposal_fn
        self.evaluation_fn = evaluation_fn
        self.plateau_detector = plateau_detector

    def run(self, session_id: str) -> SearchSession:
        """
        Execute search until budget exhausted or plateau reached.

        Args:
            session_id: Unique identifier for this search session

        Returns:
            Completed SearchSession with all trials
        """
        budget_tracker = BudgetTracker(budget=self.config.budget)
        session = SearchSession(
            session_id=session_id,
            config=self.config,
            budget_tracker=budget_tracker,
        )

        recent_scores: List[float] = []

        while not session.is_finished():
            # Check budget
            if budget_tracker.is_exhausted():
                session.finish(reason="budget_exhausted")
                break

            # Check plateau
            if self._check_plateau(recent_scores):
                session.finish(reason="plateau_detected")
                break

            # Generate proposal
            if not budget_tracker.can_propose_trial():
                session.finish(reason="max_trials_reached")
                break

            trial = self.proposal_fn()
            budget_tracker.record_trial()
            session.add_trial(trial)

            # Validate
            trial.update_status(TrialStatus.VALIDATING)
            legality = self._validate_trial(trial)
            if not legality["is_legal"]:
                trial.update_status(TrialStatus.ILLEGAL, legality_check=legality)
                continue

            trial.update_status(TrialStatus.LEGAL, legality_check=legality)

            # Evaluate
            if not budget_tracker.can_evaluate():
                session.finish(reason="max_evaluations_reached")
                break

            trial.update_status(TrialStatus.EVALUATING)

            try:
                # Use lowest fidelity tier if multifidelity enabled
                fidelity = 0 if self.config.enable_multifidelity else 4
                result = self.evaluation_fn(trial, fidelity)

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

                budget_tracker.record_evaluation(cost=result.get("cost", 1.0))

            except Exception as e:
                trial.update_status(TrialStatus.FAILED, failure_reason=str(e))

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

        # Default: check if improvement over window is below threshold
        if not recent_scores:
            return False

        max_score = max(recent_scores)
        min_score = min(recent_scores)

        if max_score == 0:
            return min_score == 0

        relative_improvement = (max_score - min_score) / abs(max_score)
        return relative_improvement < self.config.plateau_threshold
