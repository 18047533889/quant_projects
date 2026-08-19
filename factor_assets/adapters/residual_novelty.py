"""
Residual-IC conditional-novelty producer (QE wiring).

This is the producer behind ``SelectionPolicy.make_decision``'s
novelty_score / residual_ic parameters (the SHADOW / SHADOWED_COEXIST
chain).  It computes the incremental IC of a candidate factor on top of
its most-similar admitted factor(s) via
``quant_evaluator.metrics.compute_incremental_ic`` and maps it to a
bounded conditional-novelty score in [0, 1].

Signals can be carried on ``EvidenceResult.secondary_metrics`` (see
``evidence_result_from_signals`` / ``novelty_inputs_from_evidence``) so
the policy boundary keeps consuming evidence, not raw factor values.

This is an OPTIONAL adapter — FA core does not depend on QE.
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from quant_evaluator.metrics import (
        compute_incremental_ic as _qe_compute_incremental_ic,
    )
    QE_AVAILABLE = True
except ImportError:
    QE_AVAILABLE = False
    _qe_compute_incremental_ic = None

from factor_assets.adapters import OptionalDependencyMissing
from factor_assets.novelty.provider import EvidenceResult


RESIDUAL_IC_KEY = "residual_ic"
INCREMENTAL_IC_MEAN_KEY = "incremental_ic_mean"
TOTAL_IC_MEAN_KEY = "total_ic_mean"
NOVELTY_SCORE_KEY = "novelty_score"
NUM_VALID_PERIODS_KEY = "num_valid_periods"

DEFAULT_MIN_PERIODS = 20

# Below this |total_ic| the ratio is numerically meaningless; the score
# then depends only on whether any residual signal exists at all.
_TOTAL_IC_EPS = 1e-12


class ResidualICNoveltyProducer:
    """
    Produces residual-IC and conditional-novelty signals from QE.

    The novelty score is the fraction of the test factor's total IC that
    survives residualization against the base (most-similar admitted)
    factors: ``min(1, |incremental_ic| / |total_ic|)``.  A factor whose
    predictive power is fully explained by an admitted factor scores ~0
    (SHADOW); a factor carrying independent signal scores toward 1
    (eligible for SHADOWED_COEXIST).

    Fail-closed: if fewer than ``min_periods`` periods produce a finite
    incremental IC, ``compute_signals`` returns None — the novelty signal
    is absent and the policy rejects on similarity, never invents a score.
    """

    def __init__(
        self,
        min_periods: int = DEFAULT_MIN_PERIODS,
        method: str = "pearson",
        min_assets: int = 30,
    ):
        """
        Initialize producer.

        Args:
            min_periods: Minimum finite incremental-IC periods required
                to emit a signal (fail-closed below this).
            method: IC method passed to QE ("pearson" or "spearman").
            min_assets: Minimum valid assets per period passed to QE.
        """
        if min_periods < 1:
            raise ValueError("min_periods must be >= 1")
        if method not in ("pearson", "spearman"):
            raise ValueError(f"Unknown method: {method!r}. Must be 'pearson' or 'spearman'")
        if min_assets < 2:
            raise ValueError("min_assets must be >= 2")
        self._min_periods = min_periods
        self._method = method
        self._min_assets = min_assets

    @property
    def min_periods(self) -> int:
        """Minimum finite-period count required to emit a signal."""
        return self._min_periods

    def compute_signals(
        self,
        factor_batch: Any,
        label_bundle: Any,
        base_factor_indices: Tuple[int, ...],
        test_factor_idx: int,
    ) -> Optional[Dict[str, float]]:
        """
        Compute residual-IC novelty signals for a candidate factor.

        Args:
            factor_batch: QE FactorBatch (T, N, F) containing the base
                factors and the test factor.
            label_bundle: QE LabelBundle aligned with the batch.
            base_factor_indices: Factor indices to control for (the most
                similar admitted factor(s)).
            test_factor_idx: Index of the candidate factor being assessed.

        Returns:
            Dict with residual_ic, incremental_ic_mean, total_ic_mean,
            novelty_score (in [0, 1]) and num_valid_periods; or None when
            insufficient periods produce a finite signal (fail-closed).

        Raises:
            OptionalDependencyMissing: If QE is not installed.
            ValueError: If base_factor_indices is empty or contains
                test_factor_idx (self-residualization is a caller error).
        """
        if not QE_AVAILABLE:
            raise OptionalDependencyMissing(
                "quant_evaluator", "ResidualICNoveltyProducer"
            )
        if not base_factor_indices:
            raise ValueError(
                "base_factor_indices must be non-empty — residual novelty "
                "is assessed against at least one admitted factor"
            )
        if test_factor_idx in tuple(base_factor_indices):
            raise ValueError(
                "test_factor_idx cannot appear in base_factor_indices "
                "(self-residualization)"
            )

        incremental_ic, _base_ic, total_ic = _qe_compute_incremental_ic(
            factor_batch,
            label_bundle,
            base_factor_indices=base_factor_indices,
            test_factor_idx=test_factor_idx,
            method=self._method,
            min_assets=self._min_assets,
        )

        finite_incr = np.isfinite(incremental_ic)
        # Count periods usable for the ratio (both series finite), not
        # just finite-increment periods: num_valid_periods is the count
        # behind the means below, so it must use the same mask.
        both_finite = finite_incr & np.isfinite(total_ic)
        num_valid = int(both_finite.sum())
        if int(finite_incr.sum()) < self._min_periods:
            return None

        # Ratio consistency: numerator and denominator must be averaged
        # over the SAME periods — a period finite in one series but not
        # the other would bias the ratio in either direction.
        if num_valid < self._min_periods:
            # Absent denominator evidence fails closed: no signal at all,
            # never a maximal novelty score (which would admit a
            # redundant factor on a corrupt record).
            return None

        residual_ic = float(np.mean(incremental_ic[both_finite]))
        total_ic_mean = float(np.mean(total_ic[both_finite]))

        novelty_score = self.map_novelty_score(residual_ic, total_ic_mean)

        return {
            RESIDUAL_IC_KEY: residual_ic,
            INCREMENTAL_IC_MEAN_KEY: residual_ic,  # identical by definition
            TOTAL_IC_MEAN_KEY: total_ic_mean,
            NOVELTY_SCORE_KEY: novelty_score,
            NUM_VALID_PERIODS_KEY: float(num_valid),
        }

    @staticmethod
    def map_novelty_score(
        residual_ic: Optional[float],
        total_ic: Optional[float],
    ) -> float:
        """
        Map a residual IC to a conditional-novelty score in [0, 1].

        The score is the share of the factor's total IC that survives
        residualization against the base factors.  When the total IC is
        absent or numerically zero, any nonzero residual signal counts as
        fully novel (1.0); no residual signal counts as fully redundant
        (0.0).

        Args:
            residual_ic: Mean incremental IC (finite, or None/NaN → 0.0).
            total_ic: Mean unresidualized IC (may be None/NaN).

        Returns:
            Novelty score in [0, 1].
        """
        if residual_ic is None or not math.isfinite(residual_ic):
            return 0.0
        if total_ic is None or not math.isfinite(total_ic):
            total = None
        else:
            total = abs(total_ic)
        if total is None or total <= _TOTAL_IC_EPS:
            return 1.0 if abs(residual_ic) > _TOTAL_IC_EPS else 0.0
        return min(1.0, abs(residual_ic) / total)


def evidence_result_from_signals(
    signals: Dict[str, float],
    factor_id: str,
    evaluation_run_id: str,
    timestamp: str,
    universe_ref: Optional[str] = None,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    qe_version: Optional[str] = None,
    warnings: Tuple[str, ...] = (),
) -> EvidenceResult:
    """
    Wrap producer signals as an EvidenceResult for storage/retrieval.

    The residual IC is the primary metric; every signal (including the
    novelty score) is mirrored into secondary_metrics so consumers only
    need the standard evidence boundary.

    Args:
        signals: Output of ResidualICNoveltyProducer.compute_signals.
        factor_id: Candidate factor identifier.
        evaluation_run_id: Evaluation run identifier.
        timestamp: ISO 8601 timestamp.
        universe_ref / period_start / period_end: Evaluation context.
        qe_version: QE version provenance.
        warnings: Evidence warnings.

    Returns:
        EvidenceResult with the signals in secondary_metrics.

    Raises:
        ValueError: If signals is missing the residual-IC key.
    """
    if RESIDUAL_IC_KEY not in signals:
        raise ValueError(
            f"signals must contain {RESIDUAL_IC_KEY!r} (was it produced by "
            "ResidualICNoveltyProducer.compute_signals?)"
        )
    secondary = dict(signals)
    return EvidenceResult(
        factor_id=factor_id,
        evaluation_run_id=evaluation_run_id,
        evidence_id=f"{evaluation_run_id}:{factor_id}:residual_ic",
        timestamp=timestamp,
        available=True,
        primary_metric_name=RESIDUAL_IC_KEY,
        primary_metric_value=float(signals[RESIDUAL_IC_KEY]),
        secondary_metrics=secondary,
        universe_ref=universe_ref,
        period_start=period_start,
        period_end=period_end,
        qe_version=qe_version,
        warnings=warnings,
    )


def novelty_inputs_from_evidence(
    evidence: Optional[EvidenceResult],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract (novelty_score, residual_ic) for make_decision from evidence.

    This is the read side of the residual-IC wiring: it pulls the signals
    that ResidualICNoveltyProducer stored in secondary_metrics and
    validates them so SelectionPolicy.make_decision never sees a
    nonfinite or out-of-range value from a corrupted evidence record
    (nonfinite values are reported as absent, fail-closed).

    Args:
        evidence: EvidenceResult from storage (may be None).

    Returns:
        Tuple (novelty_score, residual_ic); either entry is None when the
        signal is absent or not finite.
    """
    if evidence is None or not evidence.available:
        return (None, None)
    secondary = evidence.secondary_metrics or {}

    def _finite(key: str) -> Optional[float]:
        value = secondary.get(key)
        try:
            if value is None or not math.isfinite(value):
                return None
        except TypeError:
            # Non-numeric entries in a corrupt record are absent, not a
            # crash — the policy fails closed toward rejection.
            return None
        return float(value)

    novelty_score = _finite(NOVELTY_SCORE_KEY)
    if novelty_score is not None and not 0.0 <= novelty_score <= 1.0:
        # Out-of-range scores are corrupt — treat as absent (the policy
        # fails closed toward REJECTED_SIMILARITY, never guesses).
        novelty_score = None
    residual_ic = _finite(RESIDUAL_IC_KEY)
    return (novelty_score, residual_ic)


__all__ = [
    "RESIDUAL_IC_KEY",
    "INCREMENTAL_IC_MEAN_KEY",
    "TOTAL_IC_MEAN_KEY",
    "NOVELTY_SCORE_KEY",
    "NUM_VALID_PERIODS_KEY",
    "DEFAULT_MIN_PERIODS",
    "ResidualICNoveltyProducer",
    "evidence_result_from_signals",
    "novelty_inputs_from_evidence",
]
