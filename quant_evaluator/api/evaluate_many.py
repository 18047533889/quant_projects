"""`evaluate_many` — batch evaluation facade (spec §26, §43).

Evaluates one FactorBatch against multiple LabelBundles (e.g. different
horizons H01/H05/H10/H20). CUDA IC requests reuse each uploaded factor tile
across labels. Returns a dict keyed by label_id → EvaluationBundle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.diagnosis.factor import diagnose_all_factors
from quant_evaluator.registry.metrics import resolve_alias


_PANEL_GPU_METRICS = frozenset({
    "rank_ic", "ic_ir", "ic_std", "ic_median", "rank_ic_series",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_returns_full", "quantile_returns_daily",
    "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_series",
    "daily_quantile_monotonicity_rate", "turnover",
    "factor_turnover_rate", "rank_ic_positive_ratio",
})


def _evaluate_ic_labels_shared_cuda(
    factor_batch, labels, *, metrics, backend, gpu_policy=None,
    split_ref=None, metric_parameters=None, quantile_builder_parameters=None,
):
    """Return CUDA evaluation bundles with one session and one factor upload per tile.

    The ordinary evaluator performs every label's contract, split, and backend
    admission checks before device allocation. Other requests use its existing
    path. GPUExecutor.run() keeps pairwise rank caches local to each label.
    """
    selected_metrics = tuple(metrics)
    canonical_metrics = tuple(
        dict.fromkeys(resolve_alias(mid) for mid in selected_metrics))
    from quant_evaluator.runtime.gpu_quantile_shape_adapter import (
        GPU_PROFILE_QUANTILE_METRICS,
    )
    if not labels or not set(canonical_metrics) <= (_PANEL_GPU_METRICS | GPU_PROFILE_QUANTILE_METRICS):
        return None
    route_decisions = []
    for label in labels:
        selected, reason = evaluate(
            factor_batch, label, metrics=selected_metrics, backend=backend,
            gpu_policy=gpu_policy, split_ref=split_ref,
            metric_parameters=metric_parameters, _prepare_only=True,
            quantile_builder_parameters=quantile_builder_parameters,
            _include_auto_route_reason=True,
        )
        if selected not in {"cuda", "cuda_strict", "gpu"}:
            return None
        route_decisions.append((selected, reason))
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    from quant_evaluator.runtime.gpu_executor import GPUExecutor
    from quant_evaluator.runtime.gpu_quantile_builder_contract import (
        normalize_gpu_quantile_builder_parameters,
    )
    builder_parameters = normalize_gpu_quantile_builder_parameters(
        quantile_builder_parameters, canonical_metrics,
    )

    with DeviceEvaluationSession(gpu_policy or GPUExecutionPolicy()) as session:
        executor = GPUExecutor(session)
        executor.quantile_builder_parameters = dict(builder_parameters)
        executor.metric_parameters = {
            resolve_alias(mid): dict(parameters)
            for mid, parameters in (metric_parameters or {}).items()
        }
        gpu_bundles = executor.run_tiled_many(factor_batch, labels, canonical_metrics)
    # GPUExecutor works with registry ids, while the public request and result
    # retain the caller's requested names. Mirror evaluate()'s direct-CUDA
    # alias projection before handing each bundle back through the facade.
    for gpu_bundle in gpu_bundles:
        for requested_id in selected_metrics:
            canonical = resolve_alias(requested_id)
            if canonical == requested_id:
                continue
            for field in ("scalar_metrics", "series_metrics", "vector_metrics",
                          "observation_counts"):
                group = getattr(gpu_bundle, field)
                if canonical in group:
                    group[requested_id] = group[canonical]
    shared_diagnostics = diagnose_all_factors(factor_batch)
    return [
        evaluate(
            factor_batch, label, metrics=selected_metrics, backend=backend,
            gpu_policy=gpu_policy, split_ref=split_ref,
            metric_parameters=metric_parameters,
            quantile_builder_parameters=quantile_builder_parameters,
            _auto_route_override=route_decision, _gpu_result_override=gpu_bundle,
            _diagnostics_override=(factor_batch, shared_diagnostics),
        )
        for label, gpu_bundle, route_decision in zip(labels, gpu_bundles, route_decisions)
    ]

def evaluate_many(
    factor_batch,
    labels: Sequence[LabelBundle],
    *,
    metrics: Optional[Sequence[str]] = None,
    context=None,
    backend=None,
    gpu_policy=None,
    split_ref=None,
    metric_parameters=None,
    quantile_builder_parameters=None,
) -> Dict[str, Any]:
    """Evaluate factor_batch against each LabelBundle.

    Args:
        factor_batch: FactorBatch evaluated against every label.
        labels: LabelBundles (typically one per horizon), with unique target_id values.
        metrics: metric ids requested for every horizon.
        context / backend / gpu_policy: forwarded to ``evaluate``.
        split_ref: optional sealed split ref.
        metric_parameters: per-metric options forwarded to each evaluation.
        quantile_builder_parameters: shared shape Q/min-assets options; CUDA rejects window_size.

    Returns:
        dict {label.target_id: bundle} where bundle is the result of
        ``evaluate(..., backend=backend)`` for that label.
    """
    selected_metrics = tuple(metrics or ("rank_ic", "ic_ir"))
    label_list = tuple(labels)
    target_ids = tuple(label.target_id for label in label_list)
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("evaluate_many labels must have unique target_id values")
    if context is None:
        shared = _evaluate_ic_labels_shared_cuda(
            factor_batch, label_list, metrics=selected_metrics, backend=backend,
            gpu_policy=gpu_policy, split_ref=split_ref,
            metric_parameters=metric_parameters,
            quantile_builder_parameters=quantile_builder_parameters,
        )
        if shared is not None:
            return {label.target_id: result for label, result in zip(label_list, shared)}
    out: Dict[str, Any] = {}
    for lb in label_list:
        out[lb.target_id] = evaluate(
            factor_batch,
            lb,
            context=context,
            metrics=selected_metrics,
            backend=backend,
            gpu_policy=gpu_policy,
            split_ref=split_ref,
            metric_parameters=metric_parameters,
            quantile_builder_parameters=quantile_builder_parameters,
        )
    return out
