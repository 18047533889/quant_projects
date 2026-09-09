"""Tiered (funnel) evaluation for factor auto-treatment optimization.

Factor auto-treatment search can propose hundreds of thousands of candidate
treatment recipes.  Running a full backtest on every candidate is
prohibitively expensive.  This module implements a *tiered funnel*: cheap
screening tiers early (small subset / higher noise), increasingly expensive
tiers later, and a full backtest ONLY for the survivors that pass every
prior tier.

STRICT-OOS note (model selection): treatment search is model selection.  The
chosen treatment is fit on TRAIN, the winner is SELECTED on VALIDATION, and
only the frozen winner is evaluated once on the SEALED TEST segment.  The
tiered funnel operates purely on the train/validation search data.  It must
NEVER let a sealed-test observation influence which tier a candidate is
promoted to (and the runner's sealed-test machinery enforces that boundary
separately).  Tiered evaluation therefore routes candidates to progressively
more expensive fidelity tiers based ONLY on their earlier-tier performance on
search data.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class IssuedTierJob:
    job_id: str
    candidate_id: str
    recipe_version: str
    tier_name: str
    attempt: int

    def __post_init__(self):
        for name in ("job_id", "candidate_id", "recipe_version", "tier_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")


@dataclass(frozen=True)
class TierEvaluationOutcome:
    outcome_id: str
    job_id: str
    candidate_id: str
    recipe_version: str
    tier_name: str
    attempt: int
    evaluation_ref: str
    passed: bool

    def __post_init__(self):
        for name in ("outcome_id", "job_id", "candidate_id", "recipe_version",
                     "tier_name", "evaluation_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a strict bool")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")


@dataclass(frozen=True)
class EvaluationTier:
    """A single stage in the evaluation funnel.

    Attributes:
        name: Human-readable tier name (e.g. ``"Tier1"``..``"Tier4"``).
        cost_multiplier: Relative evaluation cost of this tier (> 0).  Each
            successive tier should cost strictly more than the last so the
            funnel's point — cheap screening → expensive confirmation — holds.
        fidelity_threshold: The evaluation fidelity level this tier runs at.
            Maps onto :class:`~factor_optimizer.search.multifidelity.FidelityTier`
            (0..4).  Tier4 / full backtest corresponds to the highest fidelity.
        promote_after: Number of *consecutive successful* evaluations required
            at this tier before the candidate is promoted to the next tier.
            ``0`` promotes immediately on the first success (default).
    """

    name: str
    cost_multiplier: float
    fidelity_threshold: int
    promote_after: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tier name must be a non-empty string")
        if isinstance(self.cost_multiplier, bool) or not isinstance(
            self.cost_multiplier, (int, float)
        ):
            raise ValueError(
                f"tier '{self.name}' cost_multiplier must be a number"
            )
        if not self.cost_multiplier > 0:
            raise ValueError(
                f"tier '{self.name}' cost_multiplier must be > 0, got "
                f"{self.cost_multiplier!r}"
            )
        if (
            isinstance(self.fidelity_threshold, bool)
            or not isinstance(self.fidelity_threshold, int)
            or not 0 <= self.fidelity_threshold <= 4
        ):
            raise ValueError(
                f"tier '{self.name}' fidelity_threshold must be an int in "
                "[0, 4], got {self.fidelity_threshold!r}"
            )
        if (
            isinstance(self.promote_after, bool)
            or not isinstance(self.promote_after, int)
            or self.promote_after < 0
        ):
            raise ValueError(
                f"tier '{self.name}' promote_after must be a non-negative int"
            )


@dataclass(frozen=True)
class TieredEvaluationPolicy:
    """Frozen policy describing an ordered evaluation funnel.

    Attributes:
        tiers: Ordered ``EvaluationTier`` list, from cheapest/cheapest-noise
            screening to the full backtest.  The order MUST be strictly
            increasing in both cost and fidelity (validated at construction),
            so a candidate cannot skip the funnel or run the full backtest
            before surviving the cheaper tiers.
    """

    tiers: Tuple[EvaluationTier, ...] = field(
        default_factory=lambda: (
            EvaluationTier("Tier1", 1.0, 0),
            EvaluationTier("Tier2", 5.0, 1),
            EvaluationTier("Tier3", 25.0, 3),
            EvaluationTier("Tier4", 100.0, 4),
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tiers, (tuple, list)) or not self.tiers:
            raise ValueError(
                "tiered_evaluation requires a non-empty, ordered list of tiers"
            )
        tiers = tuple(self.tiers)
        if not all(isinstance(t, EvaluationTier) for t in tiers):
            raise TypeError(
                "tiered_evaluation tiers must be EvaluationTier instances"
            )
        names = [t.name for t in tiers]
        if len(names) != len(set(names)):
            raise ValueError("tier names must be unique")
        # Fail-closed ordering: cost AND fidelity must be strictly increasing
        # so a candidate physically cannot reach a more expensive tier without
        # passing every cheaper one first.
        for prev, nxt in zip(tiers, tiers[1:]):
            if nxt.cost_multiplier <= prev.cost_multiplier:
                raise ValueError(
                    "tiered_evaluation tiers must be strictly increasing in "
                    f"cost: {prev.name}({prev.cost_multiplier}) -> "
                    f"{nxt.name}({nxt.cost_multiplier})"
                )
            if nxt.fidelity_threshold <= prev.fidelity_threshold:
                raise ValueError(
                    "tiered_evaluation tiers must be strictly increasing in "
                    f"fidelity: {prev.name}({prev.fidelity_threshold}) -> "
                    f"{nxt.name}({nxt.fidelity_threshold})"
                )
        object.__setattr__(self, "tiers", tiers)

    @property
    def first_tier(self) -> EvaluationTier:
        """The cheapest screening tier every candidate enters at."""
        return self.tiers[0]

    @property
    def full_tier(self) -> EvaluationTier:
        """The final tier (full backtest), reached only by survivors."""
        return self.tiers[-1]

    @property
    def tier_names(self) -> List[str]:
        return [t.name for t in self.tiers]

    def tier_index(self, name: str) -> int:
        for idx, t in enumerate(self.tiers):
            if t.name == name:
                return idx
        raise ValueError(f"unknown tier: {name!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tiers": [asdict(t) for t in self.tiers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TieredEvaluationPolicy":
        if not isinstance(data, dict) or "tiers" not in data:
            raise ValueError("TieredEvaluationPolicy checkpoint missing 'tiers'")
        tiers = data["tiers"]
        if not isinstance(tiers, list):
            raise ValueError("TieredEvaluationPolicy tiers must be a list")
        return cls(
            tiers=tuple(EvaluationTier(**t) for t in tiers),
        )


class TieredEvaluationScheduler:
    """Routes each candidate through the funnel, pruning non-survivors.

    Semantics:
      - Every candidate enters at the first (cheapest) tier.
      - A candidate that PASSES its current tier may advance to the next,
        more expensive tier (after the tier's ``promote_after`` consecutive
        successful evaluations).
      - A candidate that FAILS its current tier is PRUNED: it never reaches
        any later (more expensive) tier, in particular never the full
        backtest tier.

    This is what prevents running a full backtest on all 100k factors: only
    the funnel survivors ever see the expensive/full tier.
    """

    def __init__(self, policy: TieredEvaluationPolicy) -> None:
        if not isinstance(policy, TieredEvaluationPolicy):
            raise TypeError("policy must be a TieredEvaluationPolicy")
        # Validation happens in the frozen dataclass; this is just a guard.
        self._policy = policy
        # candidate_id -> (tier_name, consecutive_passes) that it is currently
        # being evaluated at.
        self._at: Dict[str, Tuple[str, int]] = {}
        # candidate_id -> True when it has been pruned (failed a tier).
        self._pruned: set = set()
        # candidate_id -> True when it has completed the full funnel.
        self._completed: set = set()
        self._issued: Dict[str, IssuedTierJob] = {}
        self._active_job: Dict[str, str] = {}
        self._attempts: Dict[str, int] = {}
        self._outcomes: Dict[str, Tuple[Dict[str, Any], Optional[str]]] = {}
        self._evaluation_refs: set = set()

    @property
    def policy(self) -> TieredEvaluationPolicy:
        return self._policy

    def initial_tier(self) -> str:
        """The tier every candidate starts at (the cheapest)."""
        return self._policy.first_tier.name

    def fidelity_for(self, candidate_id: str) -> int:
        """The fidelity threshold of the tier *candidate_id* currently runs at."""
        name = self.tier_for(candidate_id)
        return self._policy.tiers[self._policy.tier_index(name)].fidelity_threshold

    def tier_name_for(self, candidate_id: str) -> str:
        """The name of the tier *candidate_id* currently runs at."""
        return self.tier_for(candidate_id)

    def tier_for(self, candidate_id: str) -> str:
        """The tier *candidate_id* currently occupies.

        A brand-new candidate starts at the first tier; a promoted one resumes
        at the tier the scheduler last routed it to.
        """
        if candidate_id in self._pruned:
            raise ValueError(
                f"candidate {candidate_id!r} was pruned and has no active tier"
            )
        if candidate_id in self._completed:
            raise ValueError(
                f"candidate {candidate_id!r} completed and has no active tier"
            )
        if candidate_id in self._at:
            return self._at[candidate_id][0]
        self._at[candidate_id] = (self.initial_tier(), 0)
        return self.initial_tier()

    def is_pruned(self, candidate_id: str) -> bool:
        return candidate_id in self._pruned

    def is_completed(self, candidate_id: str) -> bool:
        return candidate_id in self._completed

    def issue(self, candidate_id: str, recipe_version: str) -> IssuedTierJob:
        """Issue the only outcome-capable job for a candidate's current tier."""
        tier_name = self.tier_for(candidate_id)
        if candidate_id in self._active_job:
            active = self._issued[self._active_job[candidate_id]]
            if active.recipe_version != recipe_version:
                raise ValueError("active issued job is bound to another recipe version")
            return active
        attempt = self._attempts.get(candidate_id, 0) + 1
        job_id = f"tier-job:{candidate_id}:{attempt}"
        job = IssuedTierJob(job_id, candidate_id, recipe_version, tier_name, attempt)
        self._issued[job_id] = job
        self._active_job[candidate_id] = job_id
        self._attempts[candidate_id] = attempt
        return job

    def advance(self, outcome: TierEvaluationOutcome) -> Optional[str]:
        """Apply an outcome bound to an actually issued job and attempt.

        Returns the tier the candidate should run next:
          - the same tier when it passed but has not yet met ``promote_after``;
          - the next (more expensive) tier when it passed and is promoted;
          - ``None`` when it was pruned (failed) or completed the funnel.
        """
        if not isinstance(outcome, TierEvaluationOutcome):
            raise TypeError("advance requires a TierEvaluationOutcome")
        payload = asdict(outcome)
        if outcome.outcome_id in self._outcomes:
            prior_payload, prior_result = self._outcomes[outcome.outcome_id]
            if prior_payload != payload:
                raise ValueError("outcome_id was already used for a different payload")
            return prior_result
        job = self._issued.get(outcome.job_id)
        if job is None:
            raise ValueError("outcome references an unknown issued job")
        for name in ("candidate_id", "recipe_version", "tier_name", "attempt"):
            if getattr(outcome, name) != getattr(job, name):
                raise ValueError(f"outcome {name} does not match issued job")
        candidate_id = outcome.candidate_id
        if self._active_job.get(candidate_id) != outcome.job_id:
            raise ValueError("outcome references a stale or already-consumed attempt")
        evidence_key = (candidate_id, outcome.tier_name, outcome.evaluation_ref)
        if evidence_key in self._evaluation_refs:
            raise ValueError("evaluation_ref was already consumed for this candidate and tier")
        if candidate_id in self._pruned or candidate_id in self._completed:
            raise ValueError("terminal candidate cannot accept another outcome")
        expected_tier = self.tier_for(candidate_id)
        if outcome.tier_name != expected_tier:
            raise ValueError("outcome tier does not match candidate current tier")
        del self._active_job[candidate_id]
        self._evaluation_refs.add(evidence_key)
        idx = self._policy.tier_index(outcome.tier_name)
        current_tier = self._policy.tiers[idx]

        if not outcome.passed:
            self._pruned.add(candidate_id)
            if candidate_id in self._at:
                del self._at[candidate_id]
            result = None
            self._outcomes[outcome.outcome_id] = (payload, result)
            return result

        consecutive = self._at.get(candidate_id, (outcome.tier_name, 0))[1] + 1
        self._at[candidate_id] = (outcome.tier_name, consecutive)

        if consecutive < current_tier.promote_after:
            # Not yet promoted: stay at the same (cheap) tier to re-confirm.
            result = outcome.tier_name
            self._outcomes[outcome.outcome_id] = (payload, result)
            return result

        if idx + 1 < len(self._policy.tiers):
            next_name = self._policy.tiers[idx + 1].name
            self._at[candidate_id] = (next_name, 0)
            result = next_name
            self._outcomes[outcome.outcome_id] = (payload, result)
            return result

        # Survived every tier: full backtest done.
        self._completed.add(candidate_id)
        del self._at[candidate_id]
        result = None
        self._outcomes[outcome.outcome_id] = (payload, result)
        return result

    def promote_count(self, candidate_id: str, tier_name: str) -> int:
        """Number of consecutive successful evaluations recorded at a tier."""
        if candidate_id in self._at and self._at[candidate_id][0] == tier_name:
            return self._at[candidate_id][1]
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self._policy.to_dict(),
            "at": {k: [v[0], v[1]] for k, v in self._at.items()},
            "pruned": sorted(self._pruned),
            "completed": sorted(self._completed),
            "issued": {k: asdict(v) for k, v in self._issued.items()},
            "active_job": dict(self._active_job),
            "attempts": dict(self._attempts),
            "outcomes": {k: [payload, result] for k, (payload, result) in self._outcomes.items()},
            "evaluation_refs": [list(v) for v in sorted(self._evaluation_refs)],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TieredEvaluationScheduler":
        policy = TieredEvaluationPolicy.from_dict(data["policy"])
        sched = cls(policy)
        sched._at = {
            k: (v[0], int(v[1])) for k, v in data.get("at", {}).items()
        }
        sched._pruned = set(data.get("pruned", []))
        sched._completed = set(data.get("completed", []))
        sched._issued = {k: IssuedTierJob(**v) for k, v in data.get("issued", {}).items()}
        sched._active_job = dict(data.get("active_job", {}))
        sched._attempts = {k: int(v) for k, v in data.get("attempts", {}).items()}
        sched._outcomes = {k: (v[0], v[1]) for k, v in data.get("outcomes", {}).items()}
        sched._evaluation_refs = {tuple(v) for v in data.get("evaluation_refs", [])}
        return sched
