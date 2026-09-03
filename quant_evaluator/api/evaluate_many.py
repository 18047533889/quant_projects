"""`evaluate_many` — batch evaluation facade (spec §26, §43).

Evaluates one FactorBatch against multiple LabelBundles (e.g. different
horizons H01/H05/H10/H20) without re-staging factors per horizon on GPU.
Returns a dict keyed by label_id → BatchEvaluationBundle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.contracts.label_bundle import LabelBundle


def evaluate_many(
    factor_batch,
    labels: Sequence[LabelBundle],
    *,
    metrics: Optional[Sequence[str]] = None,
    context=None,
    backend=None,
    gpu_policy=None,
    split_ref=None,
) -> Dict[str, Any]:
    """Evaluate factor_batch against each LabelBundle.

    Args:
        factor_batch: FactorBatch evaluated against every label.
        labels: LabelBundles (typically one per horizon).
        metrics: metric ids requested for every horizon.
        context / backend / gpu_policy: forwarded to ``evaluate``.
        split_ref: optional sealed split ref.

    Returns:
        dict {label.target_id: bundle} where bundle is the result of
        ``evaluate(..., backend=backend)`` for that label.
    """
    out: Dict[str, Any] = {}
    for lb in labels:
        out[lb.target_id] = evaluate(
            factor_batch,
            lb,
            context=context,
            metrics=metrics or ("rank_ic", "ic_ir"),
            backend=backend,
            gpu_policy=gpu_policy,
            split_ref=split_ref,
        )
    return out
