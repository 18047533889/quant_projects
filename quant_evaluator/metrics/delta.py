"""Delta-vs-raw computation for the factor auto-treatment optimizer.

Computes, for each metric, how a *treatment* candidate moved the value
relative to the *raw* / no-op baseline: ``delta = treatment - raw``.

Fail-closed semantics (never fabricate a delta): when the RAW baseline metric
is absent from ``raw_metrics``, the corresponding delta field is ``None`` —
it is *unknown*, never a fabricated ``0.0``.  A metric present in treatment
but absent in raw therefore yields ``None``, not zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from quant_evaluator.contracts.treatment_evaluation import (
    DeltaMetrics,
    _DELTA_FIELDS,
)

__all__ = [
    "compute_delta_vs_raw",
    "assert_raw_candidate_first",
    "RAW_TREATMENT_IDS",
]

#: Canonical treatment identifiers denoting the raw / no-op baseline.
#: RAW is ALWAYS candidate 0 in any treatment candidate list.
RAW_TREATMENT_IDS = ("RAW", "NO_OP")


def compute_delta_vs_raw(
    treatment_metrics: Dict[str, float],
    raw_metrics: Dict[str, float],
) -> DeltaMetrics:
    """Compute ``treatment - raw`` for each delta field.

    For each :class:`DeltaMetrics` field, the raw metric is looked up by the
    field's metric-name key (e.g. ``delta_rank_ic`` -> ``"rank_ic"``) in
    ``raw_metrics``.  When the raw metric is ABSENT from ``raw_metrics``, that
    delta is ``None`` (fail-closed: unknown, never ``0``).

    Args:
        treatment_metrics: Metric name -> value for the treatment candidate.
        raw_metrics: Metric name -> value for the RAW / no-op baseline.

    Returns:
        :class:`DeltaMetrics` with each field = ``treatment - raw``, or ``None``
        when the raw baseline metric is unavailable.
    """
    kwargs: Dict[str, Optional[float]] = {}
    for field in _DELTA_FIELDS:
        # Map delta field -> base metric key ("delta_rank_ic" -> "rank_ic").
        metric_key = field[len("delta_"):]
        raw_value = raw_metrics.get(metric_key)
        if raw_value is None:
            # Fail-closed: a missing raw baseline metric means the delta is
            # unknown, never fabricated as 0.
            kwargs[field] = None
            continue
        treatment_value = treatment_metrics.get(metric_key)
        # If the treatment itself lacks the metric, its delta is also unknown.
        if treatment_value is None:
            kwargs[field] = None
            continue
        kwargs[field] = float(treatment_value) - float(raw_value)
    return DeltaMetrics(**kwargs)


def assert_raw_candidate_first(candidates: List[Any]) -> None:
    """Assert the candidate list's first entry is the RAW / NO_OP treatment.

    Enforces the invariant that RAW is candidate 0: every treatment candidate
    is compared against a raw baseline, so the raw / no-op treatment must
    occupy the first position of the candidate list.  Candidates may be
    :class:`TreatmentEvaluationArtifact` objects (whose ``treatment_id`` is
    inspected) or plain objects carrying a ``treatment_id`` attribute.

    Args:
        candidates: The ordered treatment candidate list.

    Raises:
        ValueError: If ``candidates`` is empty or its first element is not the
            RAW / NO_OP treatment.
    """
    if not candidates:
        raise ValueError(
            "assert_raw_candidate_first: candidate list is empty; RAW/NO_OP must "
            "be candidate 0"
        )
    first = candidates[0]
    tid = getattr(first, "treatment_id", None)
    if not isinstance(tid, str) or tid.strip().upper() not in RAW_TREATMENT_IDS:
        raise ValueError(
            "assert_raw_candidate_first: the first candidate must be the RAW/NO_OP "
            f"treatment, got treatment_id={tid!r}; RAW is always candidate 0"
        )
