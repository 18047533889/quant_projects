"""Diagnosis-to-mutation mapping for failure repair strategies."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DiagnosisKind(Enum):
    """Kind of failure diagnosis.

    Legacy members (kept for migration compatibility, R61-D09/D19) are listed
    first with their historical values:

    - HIGH_COMPLEXITY, POOR_COVERAGE, HIGH_VARIANCE, LOW_SIGNAL,
      HIGH_TURNOVER, TIMING_VIOLATION, DOMAIN_MISMATCH,
      NUMERICAL_INSTABILITY, SEMANTIC_DUPLICATE, OVERFITTING.

    R61-FI-034 extends the taxonomy to the full plan E2 vocabulary.  Newer
    members reuse the historical value where one already carried the semantic
    (POOR_COVERAGE/LOW_SIGNAL/HIGH_TURNOVER/NUMERICAL_INSTABILITY/
    SEMANTIC_DUPLICATE/HIGH_COMPLEXITY are kept as the canonical E2 name) and
    add the remaining 23 plan E2 kinds with snake_case values derived from
    their upper-snake canonical name.
    """

    # -- legacy vocabulary (migration compatibility) -----------------------
    HIGH_COMPLEXITY = "high_complexity"  # Exceeds compute/memory budget
    POOR_COVERAGE = "poor_coverage"  # Insufficient valid data points
    HIGH_VARIANCE = "high_variance"  # Unstable across periods
    LOW_SIGNAL = "low_signal"  # Weak predictive power (low IC)
    HIGH_TURNOVER = "high_turnover"  # Excessive portfolio churn
    TIMING_VIOLATION = "timing_violation"  # Look-ahead bias or PIT violation
    DOMAIN_MISMATCH = "domain_mismatch"  # Wrong data domain
    NUMERICAL_INSTABILITY = "numerical_instability"  # NaN/Inf production
    SEMANTIC_DUPLICATE = "semantic_duplicate"  # Equivalent to existing factor
    OVERFITTING = "overfitting"  # Train/test performance gap

    # -- R61-FI-034 plan E2 extension (canonical E2 vocabulary) -------------
    # Integrity / data quality
    INTEGRITY_FAILURE = "integrity_failure"
    PIT_VIOLATION = "pit_violation"
    LABEL_TIMING_VIOLATION = "label_timing_violation"
    DATA_QUALITY_FAILURE = "data_quality_failure"
    STALE_DATA = "stale_data"
    HIGH_TIE_RATIO = "high_tie_ratio"
    SPARSE_FACTOR = "sparse_factor"
    # Predictive / stability / generalization
    LOW_PREDICTIVE = "low_predictive"
    UNSTABLE_IC = "unstable_ic"
    RECENT_DEGRADATION = "recent_degradation"
    OVERFIT_GENERALIZATION = "overfit_generalization"
    LOW_STATISTICAL_CONFIDENCE = "low_statistical_confidence"
    # Shape
    U_SHAPE = "u_shape"
    INVERTED_U = "inverted_u"
    TOP_TAIL_COLLAPSE = "top_tail_collapse"
    BOTTOM_TAIL_COLLAPSE = "bottom_tail_collapse"
    TAIL_ONLY = "tail_only"
    NONSTATIONARY_SHAPE = "nonstationary_shape"
    # Portfolio economics
    HIGH_COST_DRAG = "high_cost_drag"
    LOW_CAPACITY = "low_capacity"
    HIGH_DRAWDOWN = "high_drawdown"
    LONG_UNDERWATER = "long_underwater"
    NEGATIVE_TAIL_RISK = "negative_tail_risk"
    REGIME_DEPENDENT = "regime_dependent"
    # Exposure / purity
    SIZE_EXPOSURE = "size_exposure"
    INDUSTRY_EXPOSURE = "industry_exposure"
    BETA_EXPOSURE = "beta_exposure"
    LIQUIDITY_EXPOSURE = "liquidity_exposure"
    VOLATILITY_EXPOSURE = "volatility_exposure"
    MULTI_STYLE_EXPOSURE = "multi_style_exposure"
    # Novelty / redundancy / complexity
    VALUE_NEAR_DUPLICATE = "value_near_duplicate"
    LOW_NOVELTY = "low_novelty"

    @classmethod
    def canonical_names(cls) -> tuple[str, ...]:
        """Upper-snake canonical E2 diagnosis kind names (plan E2/FA C7).

        These are the names FO consumes from the FA diagnosis engine
        (``factor_assets.profiling.diagnosis.DIAGNOSIS_TAGS`` upper-snake
        tokens).  Every member's ``canonical_name`` is a member of this set.
        """
        return tuple(sorted(m.canonical_name() for m in cls))

    def canonical_name(self) -> str:
        """Upper-snake canonical name (e.g. ``HIGH_TURNOVER``)."""
        return self.name


class DiagnosisCategory(Enum):
    """Functional grouping of diagnosis kinds (plan E2 taxonomy families)."""

    INTEGRITY = "integrity"
    DATA_QUALITY = "data_quality"
    PREDICTIVE = "predictive"
    STABILITY = "stability"
    GENERALIZATION = "generalization"
    SHAPE = "shape"
    TRADABILITY = "tradability"
    DRAWDOWN_RISK = "drawdown_risk"
    EXPOSURE = "exposure"
    NOVELTY_REDUNDANCY = "novelty_redundancy"
    COMPLEXITY = "complexity"


#: Canonical category assignment for every DiagnosisKind member.
DIAGNOSIS_CATEGORY: Dict[DiagnosisKind, DiagnosisCategory] = {
    DiagnosisKind.INTEGRITY_FAILURE: DiagnosisCategory.INTEGRITY,
    DiagnosisKind.PIT_VIOLATION: DiagnosisCategory.INTEGRITY,
    DiagnosisKind.LABEL_TIMING_VIOLATION: DiagnosisCategory.INTEGRITY,
    DiagnosisKind.DATA_QUALITY_FAILURE: DiagnosisCategory.DATA_QUALITY,
    DiagnosisKind.POOR_COVERAGE: DiagnosisCategory.DATA_QUALITY,
    DiagnosisKind.STALE_DATA: DiagnosisCategory.DATA_QUALITY,
    DiagnosisKind.HIGH_TIE_RATIO: DiagnosisCategory.DATA_QUALITY,
    DiagnosisKind.SPARSE_FACTOR: DiagnosisCategory.DATA_QUALITY,
    DiagnosisKind.NUMERICAL_INSTABILITY: DiagnosisCategory.DATA_QUALITY,
    DiagnosisKind.LOW_SIGNAL: DiagnosisCategory.PREDICTIVE,
    DiagnosisKind.LOW_PREDICTIVE: DiagnosisCategory.PREDICTIVE,
    DiagnosisKind.UNSTABLE_IC: DiagnosisCategory.STABILITY,
    DiagnosisKind.HIGH_VARIANCE: DiagnosisCategory.STABILITY,
    DiagnosisKind.RECENT_DEGRADATION: DiagnosisCategory.STABILITY,
    DiagnosisKind.OVERFITTING: DiagnosisCategory.GENERALIZATION,
    DiagnosisKind.OVERFIT_GENERALIZATION: DiagnosisCategory.GENERALIZATION,
    DiagnosisKind.LOW_STATISTICAL_CONFIDENCE: DiagnosisCategory.STABILITY,
    DiagnosisKind.U_SHAPE: DiagnosisCategory.SHAPE,
    DiagnosisKind.INVERTED_U: DiagnosisCategory.SHAPE,
    DiagnosisKind.TOP_TAIL_COLLAPSE: DiagnosisCategory.SHAPE,
    DiagnosisKind.BOTTOM_TAIL_COLLAPSE: DiagnosisCategory.SHAPE,
    DiagnosisKind.TAIL_ONLY: DiagnosisCategory.SHAPE,
    DiagnosisKind.NONSTATIONARY_SHAPE: DiagnosisCategory.SHAPE,
    DiagnosisKind.HIGH_TURNOVER: DiagnosisCategory.TRADABILITY,
    DiagnosisKind.HIGH_COST_DRAG: DiagnosisCategory.TRADABILITY,
    DiagnosisKind.LOW_CAPACITY: DiagnosisCategory.TRADABILITY,
    DiagnosisKind.HIGH_DRAWDOWN: DiagnosisCategory.DRAWDOWN_RISK,
    DiagnosisKind.LONG_UNDERWATER: DiagnosisCategory.DRAWDOWN_RISK,
    DiagnosisKind.NEGATIVE_TAIL_RISK: DiagnosisCategory.DRAWDOWN_RISK,
    DiagnosisKind.REGIME_DEPENDENT: DiagnosisCategory.STABILITY,
    DiagnosisKind.SIZE_EXPOSURE: DiagnosisCategory.EXPOSURE,
    DiagnosisKind.INDUSTRY_EXPOSURE: DiagnosisCategory.EXPOSURE,
    DiagnosisKind.BETA_EXPOSURE: DiagnosisCategory.EXPOSURE,
    DiagnosisKind.LIQUIDITY_EXPOSURE: DiagnosisCategory.EXPOSURE,
    DiagnosisKind.VOLATILITY_EXPOSURE: DiagnosisCategory.EXPOSURE,
    DiagnosisKind.MULTI_STYLE_EXPOSURE: DiagnosisCategory.EXPOSURE,
    DiagnosisKind.DOMAIN_MISMATCH: DiagnosisCategory.INTEGRITY,
    DiagnosisKind.SEMANTIC_DUPLICATE: DiagnosisCategory.NOVELTY_REDUNDANCY,
    DiagnosisKind.VALUE_NEAR_DUPLICATE: DiagnosisCategory.NOVELTY_REDUNDANCY,
    DiagnosisKind.LOW_NOVELTY: DiagnosisCategory.NOVELTY_REDUNDANCY,
    DiagnosisKind.HIGH_COMPLEXITY: DiagnosisCategory.COMPLEXITY,
    DiagnosisKind.TIMING_VIOLATION: DiagnosisCategory.INTEGRITY,
}


class RepairStrategy(Enum):
    """Repair mutation strategy."""

    REDUCE_WINDOW = "reduce_window"  # Shorten lookback period
    INCREASE_WINDOW = "increase_window"  # Lengthen lookback period
    INCREASE_HORIZON = "increase_horizon"  # Lengthen prediction horizon (for turnover)
    ADJUST_DECAY = "adjust_decay"  # Modify decay/smoothing parameter
    THRESHOLD_TUNE = "threshold_tune"  # Adjust threshold cutoff
    OPERATOR_SWAP = "operator_swap"  # Replace operator with similar
    ADD_INTERACTION = "add_interaction"  # Add interaction term (for low IC)
    ADD_REGULARIZATION = "add_regularization"  # Add smoothing/regularization
    CHANGE_NORMALIZATION = "change_normalization"  # Modify normalization method
    FILTER_UNIVERSE = "filter_universe"  # Restrict coverage universe
    WINSORIZE = "winsorize"  # Add outlier capping
    LAG_CORRECTION = "lag_correction"  # Adjust timing/alignment
    ABANDON = "abandon"  # No repair available


#: Canonical order-independent set of the 33 plan-E2 upper-snake diagnosis
#: kind names.  FO consumes upper-snake tag strings from the FA diagnosis
#: engine (``factor_assets.profiling.diagnosis.DIAGNOSIS_TAGS``) and must not
#: need a QE/FA import to validate/route them.
PLAN_E2_DIAGNOSIS_NAMES: tuple[str, ...] = (
    # integrity / data quality
    "INTEGRITY_FAILURE",
    "PIT_VIOLATION",
    "LABEL_TIMING_VIOLATION",
    "DATA_QUALITY_FAILURE",
    "POOR_COVERAGE",
    "STALE_DATA",
    "HIGH_TIE_RATIO",
    "SPARSE_FACTOR",
    "NUMERICAL_INSTABILITY",
    # predictive / stability / generalization
    "LOW_PREDICTIVE",
    "UNSTABLE_IC",
    "RECENT_DEGRADATION",
    "OVERFIT_GENERALIZATION",
    "LOW_STATISTICAL_CONFIDENCE",
    # shape
    "U_SHAPE",
    "INVERTED_U",
    "TOP_TAIL_COLLAPSE",
    "BOTTOM_TAIL_COLLAPSE",
    "TAIL_ONLY",
    "NONSTATIONARY_SHAPE",
    # portfolio economics
    "HIGH_TURNOVER",
    "HIGH_COST_DRAG",
    "LOW_CAPACITY",
    "HIGH_DRAWDOWN",
    "LONG_UNDERWATER",
    "NEGATIVE_TAIL_RISK",
    "REGIME_DEPENDENT",
    # exposure / purity
    "SIZE_EXPOSURE",
    "INDUSTRY_EXPOSURE",
    "BETA_EXPOSURE",
    "LIQUIDITY_EXPOSURE",
    "VOLATILITY_EXPOSURE",
    "MULTI_STYLE_EXPOSURE",
    # novelty / redundancy / complexity
    "SEMANTIC_DUPLICATE",
    "VALUE_NEAR_DUPLICATE",
    "LOW_NOVELTY",
    "HIGH_COMPLEXITY",
)


def canonical_diagnosis_name(kind: DiagnosisKind) -> str:
    """Upper-snake canonical E2 name of a DiagnosisKind."""
    return kind.canonical_name()


@dataclass
class DiagnosisRecord:
    """
    Record of a failure diagnosis.

    Attributes:
        trial_id: Trial that failed
        diagnosis_kind: Primary failure reason
        severity: Severity score [0.0, 1.0] where 1.0 is most severe
        evidence: Supporting evidence for diagnosis
        secondary_diagnoses: Additional contributing factors
        metadata: Additional diagnostic metadata
    """

    trial_id: str
    diagnosis_kind: DiagnosisKind
    severity: float = 0.5
    evidence: Dict[str, Any] = field(default_factory=dict)
    secondary_diagnoses: List[DiagnosisKind] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate diagnosis record."""
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"Severity must be in [0.0, 1.0], got {self.severity}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trial_id": self.trial_id,
            "diagnosis_kind": self.diagnosis_kind.value,
            "severity": self.severity,
            "evidence": dict(self.evidence),
            "secondary_diagnoses": [d.value for d in self.secondary_diagnoses],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisRecord":
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("diagnosis_kind"), str):
            data["diagnosis_kind"] = DiagnosisKind(data["diagnosis_kind"])
        if "secondary_diagnoses" in data:
            data["secondary_diagnoses"] = [
                DiagnosisKind(d) if isinstance(d, str) else d for d in data["secondary_diagnoses"]
            ]
        return cls(**data)


@dataclass
class RepairProposal:
    """
    Proposed repair mutation for a diagnosed failure.

    Attributes:
        diagnosis_record: Original diagnosis
        strategy: Repair strategy to apply
        mutation_type: Mutation operation type
        parameters: Mutation parameters
        expected_improvement: Expected improvement description
        confidence: Confidence in repair success [0.0, 1.0]
        fallback_strategies: Alternative strategies if primary fails
    """

    diagnosis_record: DiagnosisRecord
    strategy: RepairStrategy
    mutation_type: str
    parameters: Dict[str, Any]
    expected_improvement: str = ""
    confidence: float = 0.5
    fallback_strategies: List[RepairStrategy] = field(default_factory=list)

    def __post_init__(self):
        """Validate repair proposal."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "diagnosis_record": self.diagnosis_record.to_dict(),
            "strategy": self.strategy.value,
            "mutation_type": self.mutation_type,
            "parameters": dict(self.parameters),
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "fallback_strategies": [s.value for s in self.fallback_strategies],
        }


class RepairMapper:
    """
    Maps diagnosis to repair mutation proposals.

    This encodes domain knowledge about which mutations address which failures.
    """

    def __init__(self):
        """Initialize repair mapper with default rules."""
        self._rules: Dict[DiagnosisKind, List[RepairStrategy]] = {
            DiagnosisKind.HIGH_COMPLEXITY: [
                RepairStrategy.REDUCE_WINDOW,
                RepairStrategy.OPERATOR_SWAP,
                RepairStrategy.ABANDON,
            ],
            DiagnosisKind.POOR_COVERAGE: [
                RepairStrategy.REDUCE_WINDOW,
                RepairStrategy.FILTER_UNIVERSE,
                RepairStrategy.ABANDON,
            ],
            DiagnosisKind.HIGH_VARIANCE: [
                RepairStrategy.ADD_REGULARIZATION,
                RepairStrategy.WINSORIZE,
                RepairStrategy.ADJUST_DECAY,
            ],
            DiagnosisKind.LOW_SIGNAL: [
                RepairStrategy.ADD_INTERACTION,
                RepairStrategy.INCREASE_WINDOW,
                RepairStrategy.ADJUST_DECAY,
                RepairStrategy.ABANDON,
            ],
            DiagnosisKind.HIGH_TURNOVER: [
                RepairStrategy.INCREASE_HORIZON,
                RepairStrategy.ADD_REGULARIZATION,
                RepairStrategy.ADJUST_DECAY,
            ],
            DiagnosisKind.TIMING_VIOLATION: [RepairStrategy.LAG_CORRECTION, RepairStrategy.ABANDON],
            DiagnosisKind.DOMAIN_MISMATCH: [RepairStrategy.ABANDON],
            DiagnosisKind.NUMERICAL_INSTABILITY: [
                RepairStrategy.WINSORIZE,
                RepairStrategy.OPERATOR_SWAP,
                RepairStrategy.ABANDON,
            ],
            DiagnosisKind.SEMANTIC_DUPLICATE: [RepairStrategy.ABANDON],
            DiagnosisKind.OVERFITTING: [
                RepairStrategy.ADD_REGULARIZATION,
                RepairStrategy.REDUCE_WINDOW,
                RepairStrategy.ABANDON,
            ],
        }
        self._priority_weights: Dict[RepairStrategy, float] = {
            RepairStrategy.REDUCE_WINDOW: 0.8,
            RepairStrategy.INCREASE_WINDOW: 0.7,
            RepairStrategy.INCREASE_HORIZON: 0.75,
            RepairStrategy.ADJUST_DECAY: 0.7,
            RepairStrategy.THRESHOLD_TUNE: 0.6,
            RepairStrategy.OPERATOR_SWAP: 0.75,
            RepairStrategy.ADD_INTERACTION: 0.7,
            RepairStrategy.ADD_REGULARIZATION: 0.8,
            RepairStrategy.CHANGE_NORMALIZATION: 0.6,
            RepairStrategy.FILTER_UNIVERSE: 0.7,
            RepairStrategy.WINSORIZE: 0.75,
            RepairStrategy.LAG_CORRECTION: 0.9,
            RepairStrategy.ABANDON: 0.0,
        }

    def propose_repairs(self, diagnosis: DiagnosisRecord, max_proposals: int = 3) -> List[RepairProposal]:
        """
        Generate repair proposals for a diagnosis.

        Args:
            diagnosis: Failure diagnosis
            max_proposals: Maximum number of proposals to generate

        Returns:
            List of repair proposals, ordered by confidence
        """
        strategies = self._rules.get(diagnosis.diagnosis_kind, [RepairStrategy.ABANDON])
        proposals = []

        for i, strategy in enumerate(strategies[:max_proposals]):
            if strategy == RepairStrategy.ABANDON:
                continue

            mutation_type, parameters = self._strategy_to_mutation(strategy, diagnosis)
            confidence = self._estimate_confidence(strategy, diagnosis, rank=i)

            proposal = RepairProposal(
                diagnosis_record=diagnosis,
                strategy=strategy,
                mutation_type=mutation_type,
                parameters=parameters,
                expected_improvement=self._describe_improvement(strategy, diagnosis),
                confidence=confidence,
                fallback_strategies=strategies[i + 1 : max_proposals],
            )
            proposals.append(proposal)

        return proposals

    def _strategy_to_mutation(
        self, strategy: RepairStrategy, diagnosis: DiagnosisRecord
    ) -> tuple[str, Dict[str, Any]]:
        """
        Convert repair strategy to mutation type and parameters.

        Args:
            strategy: Repair strategy
            diagnosis: Diagnosis with evidence

        Returns:
            (mutation_type, parameters)
        """
        if strategy == RepairStrategy.REDUCE_WINDOW:
            current_window = diagnosis.evidence.get("lookback_periods", 20)
            new_window = max(5, int(current_window * 0.7))
            return "window_adjust", {"new_window": new_window}

        elif strategy == RepairStrategy.INCREASE_WINDOW:
            current_window = diagnosis.evidence.get("lookback_periods", 20)
            new_window = min(252, int(current_window * 1.5))
            return "window_adjust", {"new_window": new_window}

        elif strategy == RepairStrategy.INCREASE_HORIZON:
            current_horizon = diagnosis.evidence.get("prediction_horizon", 1)
            new_horizon = min(20, current_horizon + 5)
            return "horizon_adjust", {"new_horizon": new_horizon}

        elif strategy == RepairStrategy.ADD_INTERACTION:
            return "add_interaction", {"interaction_type": "cross_sectional"}

        elif strategy == RepairStrategy.ADJUST_DECAY:
            current_decay = diagnosis.evidence.get("decay_param", 0.5)
            new_decay = max(0.1, min(0.9, current_decay * 0.8))
            return "decay_adjust", {"new_decay": new_decay}

        elif strategy == RepairStrategy.THRESHOLD_TUNE:
            current_threshold = diagnosis.evidence.get("threshold", 0.5)
            new_threshold = current_threshold * 0.9
            return "threshold_adjust", {"new_threshold": new_threshold}

        elif strategy == RepairStrategy.OPERATOR_SWAP:
            current_op = diagnosis.evidence.get("operator", "unknown")
            return "operator_swap", {"target_operator": "alternative", "current_operator": current_op}

        elif strategy == RepairStrategy.ADD_REGULARIZATION:
            return "add_regularization", {"regularization_strength": 0.1}

        elif strategy == RepairStrategy.CHANGE_NORMALIZATION:
            return "normalization_change", {"normalization_method": "robust"}

        elif strategy == RepairStrategy.FILTER_UNIVERSE:
            min_coverage = diagnosis.evidence.get("coverage_ratio", 0.5)
            return "filter_universe", {"min_coverage": max(0.7, min_coverage)}

        elif strategy == RepairStrategy.WINSORIZE:
            return "add_winsorization", {"lower_quantile": 0.01, "upper_quantile": 0.99}

        elif strategy == RepairStrategy.LAG_CORRECTION:
            return "lag_adjustment", {"additional_lag_days": 1}

        else:
            return "unknown", {}

    def _estimate_confidence(self, strategy: RepairStrategy, diagnosis: DiagnosisRecord, rank: int) -> float:
        """
        Estimate confidence in repair success.

        Args:
            strategy: Repair strategy
            diagnosis: Diagnosis record
            rank: Strategy rank (0 = primary)

        Returns:
            Confidence score [0.0, 1.0]
        """
        base_confidence = 0.7 - (rank * 0.15)

        if diagnosis.severity > 0.8:
            base_confidence *= 0.8

        if len(diagnosis.secondary_diagnoses) > 2:
            base_confidence *= 0.85

        return max(0.1, min(0.95, base_confidence))

    def _describe_improvement(self, strategy: RepairStrategy, diagnosis: DiagnosisRecord) -> str:
        """Generate expected improvement description."""
        descriptions = {
            RepairStrategy.REDUCE_WINDOW: "Reduce complexity and improve stability",
            RepairStrategy.INCREASE_WINDOW: "Increase signal strength with longer history",
            RepairStrategy.INCREASE_HORIZON: "Reduce turnover by lengthening prediction horizon",
            RepairStrategy.ADJUST_DECAY: "Improve temporal weighting balance",
            RepairStrategy.THRESHOLD_TUNE: "Optimize decision boundary",
            RepairStrategy.OPERATOR_SWAP: "Replace with more robust operator",
            RepairStrategy.ADD_INTERACTION: "Enhance predictive power with interaction terms",
            RepairStrategy.ADD_REGULARIZATION: "Reduce overfitting via regularization",
            RepairStrategy.CHANGE_NORMALIZATION: "Improve outlier handling",
            RepairStrategy.FILTER_UNIVERSE: "Focus on higher-quality coverage",
            RepairStrategy.WINSORIZE: "Cap extreme outliers",
            RepairStrategy.LAG_CORRECTION: "Eliminate look-ahead bias",
        }
        return descriptions.get(strategy, "Unknown improvement")

    def rank_repairs(self, proposals: List[RepairProposal]) -> List[RepairProposal]:
        """
        Rank repair proposals by priority.

        Args:
            proposals: List of repair proposals

        Returns:
            Sorted list with highest priority first
        """
        def priority_score(proposal: RepairProposal) -> float:
            base_weight = self._priority_weights.get(proposal.strategy, 0.5)
            confidence_factor = proposal.confidence
            severity_factor = 1.0 - (proposal.diagnosis_record.severity * 0.3)
            return base_weight * confidence_factor * severity_factor

        return sorted(proposals, key=priority_score, reverse=True)

    def generate_mutation_spec(self, proposal: RepairProposal) -> Dict[str, Any]:
        """
        Generate MutationSpec from repair proposal.

        Args:
            proposal: Repair proposal

        Returns:
            MutationSpec dictionary compatible with CandidateMutation
        """
        mutation_spec = {
            "mutation_type": proposal.mutation_type,
            "parameters": dict(proposal.parameters),
            "provenance": {
                "source": "repair_mapper",
                "diagnosis_kind": proposal.diagnosis_record.diagnosis_kind.value,
                "repair_strategy": proposal.strategy.value,
                "confidence": proposal.confidence,
            },
            "expected_improvement": proposal.expected_improvement,
        }

        if proposal.diagnosis_record.evidence:
            mutation_spec["evidence_context"] = dict(proposal.diagnosis_record.evidence)

        return mutation_spec
