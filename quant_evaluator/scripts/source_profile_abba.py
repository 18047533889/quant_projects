"""Caller-trusted v2 source-route profiles from a real CPU/CUDA ABBA run.

The default observer captures the live source/runtime context around each
measured API invocation. A separately supplied oracle must provide expected
metric arrays and observation counts for every run; CPU/CUDA parity alone is
not accepted as correctness evidence. Receipts are not signed attestations.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_tile_source import capture_factor_tile_source
from quant_evaluator.runtime import source_profile_router
from quant_evaluator.runtime.source_route_profiles import (
    CounterbalancedRouteProfileRecord, SourceRouteProfileContext,
    validate_source_route_profile_qualification,
)
from quant_evaluator.scripts.source_profile_measurement import (
    _counts_array, _metric_array, build_counterbalanced_profile_record,
)

RUN_ORDER = ("cpu", "cuda_strict", "cuda_strict", "cpu")


class IndependentOracleMismatchError(ValueError):
    """A run's complete output failed comparison with its independent oracle."""

    def __init__(self, message: str, *, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SourceRouteProfileABBAResult:
    records: tuple[CounterbalancedRouteProfileRecord,
                   CounterbalancedRouteProfileRecord]
    run_order: tuple[str, str, str, str]
    oracle_reports: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]


def live_source_profile_context_observer(
    *, phase: str, source, labels, metrics, requested_tile_size: int,
    policy: GPUExecutionPolicy,
) -> SourceRouteProfileContext:
    """Capture the exact live request with the production v2 router helper."""
    if phase not in ("before", "after"):
        raise ValueError("context observer phase must be before or after")
    if not isinstance(policy, GPUExecutionPolicy):
        raise TypeError("policy must be GPUExecutionPolicy")
    from quant_evaluator.api.factor_source import _source_request_fingerprint

    metadata = capture_factor_tile_source(source)
    selected = tuple(metrics)
    fingerprint = _source_request_fingerprint(metadata, labels, selected)
    return source_profile_router.capture_source_route_profile_context(
        source=source, metadata=metadata, metrics=selected,
        request_fingerprint=fingerprint,
        requested_tile_size=requested_tile_size, policy=policy)


def _oracle_report(bundle, receipt: Mapping[str, Any],
                   context: SourceRouteProfileContext, *, backend: str,
                   run_index: int, oracle: Callable) -> dict[str, Any]:
    expected = oracle(bundle=bundle, run_receipt=receipt, context=context,
                      backend=backend, run_index=run_index)
    if type(expected) is BatchEvaluationBundle:
        if (expected.factor_ids != bundle.factor_ids
                or expected.metadata.get("source_request_fingerprint")
                != context.request_content_sha256):
            raise IndependentOracleMismatchError(
                "independent oracle bundle differs from the live source request")
        outputs = {}
        for metric in context.metric_ids:
            in_scalar = metric in expected.scalar_metrics
            in_series = metric in expected.series_metrics
            if in_scalar == in_series:
                raise IndependentOracleMismatchError(
                    f"independent oracle bundle is missing or duplicates {metric}")
            outputs[metric] = {
                "values": (expected.series_metrics[metric] if in_series
                           else expected.scalar_metrics[metric]),
                "observation_counts": expected.observation_counts.get(metric),
            }
        expected = outputs
    if not isinstance(expected, Mapping):
        raise IndependentOracleMismatchError(
            "independent oracle must return a metric mapping")
    if set(expected) != set(context.metric_ids):
        raise IndependentOracleMismatchError(
            "independent oracle metric coverage is incomplete")

    coverage = dict(context.metric_coverage)
    tolerances = dict(context.metric_error_tolerances)
    factor_count = context.request_shape[-1]
    report: dict[str, Any] = {
        "pass": True, "backend": backend, "run_index": run_index, "metrics": {}}
    for metric in context.metric_ids:
        item = expected[metric]
        if not isinstance(item, Mapping):
            raise IndependentOracleMismatchError(
                f"independent oracle output is malformed for {metric}")
        expected_values = item.get("values")
        expected_counts = item.get("observation_counts")
        if (not isinstance(expected_values, np.ndarray)
                or expected_values.dtype.kind != "f"):
            raise IndependentOracleMismatchError(
                f"independent oracle values must be floating arrays for {metric}")
        if (not isinstance(expected_counts, np.ndarray)
                or expected_counts.shape != (factor_count,)
                or expected_counts.dtype.kind not in "iu"
                or (expected_counts.dtype.kind == "i" and np.any(expected_counts < 0))
                or (expected_counts.dtype.kind == "u"
                    and np.any(expected_counts > np.iinfo(np.int64).max))):
            raise IndependentOracleMismatchError(
                f"independent oracle counts are malformed for {metric}")

        actual_values, _ = _metric_array(bundle, metric, context)
        expected_shape = ((context.request_shape[0], factor_count)
                          if metric in {"rank_ic_series", "pearson_ic_series"}
                          else (factor_count,))
        shape_equal = (actual_values.shape == expected_shape
                       and expected_values.shape == expected_shape
                       and actual_values.shape == expected_values.shape)
        expected_coverage = coverage[metric]
        observed_coverage = int(actual_values.size)
        coverage_equal = shape_equal and observed_coverage == expected_coverage

        if shape_equal:
            actual_finite = np.isfinite(actual_values)
            expected_finite = np.isfinite(expected_values)
            finite_equal = bool(np.array_equal(actual_finite, expected_finite))
            nan_equal = bool(np.array_equal(np.isnan(actual_values),
                                            np.isnan(expected_values)))
            posinf_equal = bool(np.array_equal(np.isposinf(actual_values),
                                               np.isposinf(expected_values)))
            neginf_equal = bool(np.array_equal(np.isneginf(actual_values),
                                               np.isneginf(expected_values)))
            actual_counts = _counts_array(bundle, metric, factor_count)
            expected_counts = np.asarray(expected_counts, dtype="<i8", order="C")
            counts_equal = bool(np.array_equal(actual_counts, expected_counts))
            finite = actual_finite & expected_finite
            with np.errstate(over="ignore", invalid="ignore"):
                max_error = (float(np.max(np.abs(
                    actual_values[finite].astype(np.float64, copy=False)
                    - expected_values[finite].astype(np.float64, copy=False))))
                    if finite.any() else 0.0)
        else:
            finite_equal = nan_equal = posinf_equal = neginf_equal = counts_equal = False
            max_error = float("inf")

        tolerance = tolerances[metric]
        passed = (coverage_equal and finite_equal and nan_equal and posinf_equal
                  and neginf_equal and counts_equal and max_error <= tolerance)
        report["metrics"][metric] = {
            "pass": bool(passed), "expected_coverage": expected_coverage,
            "observed_coverage": observed_coverage,
            "shape_equal": bool(shape_equal),
            "finite_mask_equal": finite_equal, "nan_mask_equal": nan_equal,
            "positive_infinity_mask_equal": posinf_equal,
            "negative_infinity_mask_equal": neginf_equal,
            "observation_counts_equal": counts_equal,
            "max_abs_error": max_error, "tolerance": tolerance,
        }
        report["pass"] = report["pass"] and bool(passed)
    if not report["pass"]:
        failed = [name for name, item in report["metrics"].items() if not item["pass"]]
        raise IndependentOracleMismatchError(
            f"independent oracle mismatch for {', '.join(failed)} on run {run_index}",
            report=report)
    return report


def produce_source_route_profile_abba(
    *, oracle: Callable | None, run_kwargs: Mapping[str, Any],
    run_backend_fn: Callable | None = None,
    context_observer: Callable = live_source_profile_context_observer,
    comparison_fn: Callable | None = None,
) -> SourceRouteProfileABBAResult:
    """Run CPU/CUDA/CUDA/CPU and independently validate every backend output.

    The oracle must return, per requested metric, independent floating ``values``
    and integer ``observation_counts`` arrays. Each run is checked for exact
    coverage, finite/NaN/+Inf/-Inf masks, exact counts, and the live tolerance.
    It may return either that metric mapping or a ``BatchEvaluationBundle``
    bound to the identical factor axis and semantic request fingerprint.
    The callback is caller-trusted, not cryptographically attested.

    The runner and observer seams support bounded unit tests. Production callers
    should use the real benchmark runner and the default live-context observer.
    """
    if not callable(oracle):
        raise ValueError("an independent oracle callback is required")
    if not isinstance(run_kwargs, Mapping) or "context_observer" in run_kwargs:
        raise TypeError("run_kwargs must be a mapping without context_observer")
    if not callable(context_observer):
        raise TypeError("context_observer must be callable")
    if run_backend_fn is None:
        from quant_evaluator.scripts.benchmark_real_cos_source_batch import run_backend
        run_backend_fn = run_backend
    if not callable(run_backend_fn):
        raise TypeError("run_backend_fn must be callable")
    if comparison_fn is None:
        from quant_evaluator.scripts.benchmark_real_cos_source_batch import compare
        comparison_fn = compare
    if not callable(comparison_fn):
        raise TypeError("comparison_fn must be callable")

    runs = []
    contexts = []
    oracle_reports = []
    for run_index, requested_backend in enumerate(RUN_ORDER):
        bundle, receipt = run_backend_fn(
            requested_backend, context_observer=context_observer, **dict(run_kwargs))
        if not isinstance(receipt, Mapping):
            raise ValueError("benchmark runner must return a run receipt mapping")
        before, after = receipt.get("context_before"), receipt.get("context_after")
        if (type(before) is not SourceRouteProfileContext
                or type(after) is not SourceRouteProfileContext or before != after):
            raise ValueError(f"live source-route context changed or missing on run {run_index}")
        if contexts and before != contexts[0]:
            raise ValueError("ABBA runs do not share one identical live context")
        contexts.append(before)

        expected_backend = "cuda" if requested_backend == "cuda_strict" else "cpu"
        if receipt.get("backend_used") != expected_backend:
            raise ValueError("benchmark runner used a different backend than requested")
        oracle_reports.append(_oracle_report(
            bundle, receipt, before, backend=expected_backend,
            run_index=run_index, oracle=oracle))
        runs.append((bundle, dict(receipt)))

    selected_metrics = contexts[0].metric_ids
    expected_days = contexts[0].request_shape[0]
    records = []
    for first_index, second_index in ((0, 1), (2, 3)):
        first_bundle, first_receipt = runs[first_index]
        second_bundle, second_receipt = runs[second_index]
        pair = ((first_bundle, first_receipt), (second_bundle, second_receipt))
        cpu_bundle, cpu_receipt = next(item for item in pair
                                       if item[1].get("backend_used") == "cpu")
        cuda_bundle, cuda_receipt = next(item for item in pair
                                         if item[1].get("backend_used") == "cuda")
        parity = comparison_fn(cpu_bundle, cuda_bundle, selected_metrics,
                               expected_days=expected_days)
        if not isinstance(parity, Mapping) or parity.get("pass") is not True:
            raise ValueError("CPU/CUDA pairwise parity failed")
        execution_order = tuple(
            "cuda" if receipt.get("backend_used") == "cuda" else "cpu"
            for _, receipt in pair)
        cpu_index = first_index if first_receipt.get("backend_used") == "cpu" else second_index
        cuda_index = first_index if first_receipt.get("backend_used") == "cuda" else second_index
        records.append(build_counterbalanced_profile_record(
            cpu_bundle=cpu_bundle, cuda_bundle=cuda_bundle,
            cpu_run_receipt=cpu_receipt, cuda_run_receipt=cuda_receipt,
            context=contexts[0], comparison_report=parity,
            execution_order=execution_order,
            cpu_correctness_validated=oracle_reports[cpu_index]["pass"] is True,
            cuda_correctness_validated=oracle_reports[cuda_index]["pass"] is True,
        ))

    validate_source_route_profile_qualification(tuple(records), expected_context=contexts[0])
    return SourceRouteProfileABBAResult(
        records=(records[0], records[1]), run_order=RUN_ORDER,
        oracle_reports=tuple(oracle_reports))


__all__ = (
    "IndependentOracleMismatchError", "RUN_ORDER", "SourceRouteProfileABBAResult",
    "live_source_profile_context_observer", "produce_source_route_profile_abba",
)
