"""Run bounded randomized, paired source-route trials without qualifying auto routing."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Callable

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_route_profiles import SourceRouteProfileContext
from quant_evaluator.scripts.source_profile_abba import (
    _oracle_report,
    live_source_profile_context_observer,
)
from quant_evaluator.scripts.source_profile_measurement import (
    build_backend_profile_measurement,
)
from quant_evaluator.scripts.source_profile_paired_trials import (
    aggregate_paired_trials, build_randomized_paired_schedule,
)


def _progress_event(*, pair_index: int, run_index: int, requested_backend: str,
                    backend_used: str, seconds: float) -> dict[str, Any]:
    """Expose only bounded scalar metadata; never forward raw run objects."""
    return {
        "phase": "run_validated",
        "pair_index": pair_index,
        "run_index": run_index,
        "requested_backend": requested_backend,
        "backend_used": backend_used,
        "api_wall_seconds": seconds,
        "oracle_pass": True,
        "measurement_pass": True,
    }


def produce_source_route_profile_randomized_trials(
    *, oracle: Callable,
    run_kwargs: Mapping[str, Any],
    seed: int,
    pairs: int = 3,
    run_backend_fn: Callable | None = None,
    context_observer: Callable = live_source_profile_context_observer,
    comparison_fn: Callable | None = None,
    progress_observer: Callable | None = None,
) -> dict[str, Any]:
    """Run 3–16 counterbalanced full-API trials and aggregate their receipts.

    Every CPU and CUDA output is checked against the independent oracle and the
    real backend measurement contract before pair parity is considered. Results
    contain only lightweight receipts and deliberately do not qualify auto.
    """
    if not callable(oracle):
        raise ValueError("an independent oracle callback is required")
    if not isinstance(run_kwargs, Mapping) or "context_observer" in run_kwargs:
        raise TypeError("run_kwargs must be a mapping without context_observer")
    if not callable(context_observer):
        raise TypeError("context_observer must be callable")
    if progress_observer is not None and not callable(progress_observer):
        raise TypeError("progress_observer must be callable or None")
    schedule = build_randomized_paired_schedule(seed=seed, pairs=pairs)
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

    shared_context: SourceRouteProfileContext | None = None
    context_fingerprint: str | None = None
    trials: list[dict[str, Any]] = []
    run_index = 0
    for pair_index, requested_order in enumerate(schedule):
        pair_bundles: dict[str, Any] = {}
        pair_seconds: dict[str, float] = {}
        oracle_pass: dict[str, bool] = {}
        for requested_backend in requested_order:
            bundle, receipt = run_backend_fn(
                requested_backend, context_observer=context_observer,
                **dict(run_kwargs))
            if not isinstance(receipt, Mapping):
                raise ValueError("benchmark runner must return a run receipt mapping")
            before, after = receipt.get("context_before"), receipt.get("context_after")
            if (type(before) is not SourceRouteProfileContext
                    or type(after) is not SourceRouteProfileContext or before != after):
                raise ValueError(f"live source-route context changed or missing on run {run_index}")
            if shared_context is None:
                shared_context = before
                context_fingerprint = stable_content_hex(
                    tag="SourceRandomizedTrialContext.v1",
                    fields=asdict(before),
                )
            elif before != shared_context:
                raise ValueError("randomized trials do not share one identical live context")

            expected_backend = "cuda" if requested_backend == "cuda_strict" else "cpu"
            if receipt.get("backend_used") != expected_backend:
                raise ValueError("benchmark runner used a different backend than requested")
            oracle_report = _oracle_report(
                bundle, receipt, before, backend=expected_backend,
                run_index=run_index, oracle=oracle)
            measurement = build_backend_profile_measurement(
                bundle, receipt, before, correctness_validated=True)
            if (measurement.backend != expected_backend
                    or measurement.correctness_validated is not True):
                raise ValueError("backend measurement differs from validated run")

            pair_bundles[expected_backend] = bundle
            pair_seconds[expected_backend] = measurement.seconds
            oracle_pass[expected_backend] = oracle_report.get("pass") is True
            if progress_observer is not None:
                progress_observer(_progress_event(
                    pair_index=pair_index, run_index=run_index,
                    requested_backend=requested_backend,
                    backend_used=expected_backend, seconds=measurement.seconds,
                ))
            del measurement, oracle_report, bundle, receipt
            run_index += 1

        parity = comparison_fn(
            pair_bundles["cpu"], pair_bundles["cuda"],
            shared_context.metric_ids,
            expected_days=shared_context.request_shape[0],
        )
        if not isinstance(parity, Mapping) or parity.get("pass") is not True:
            raise ValueError("CPU/CUDA randomized pairwise parity failed")
        trials.append({
            "execution_order": tuple(requested_order),
            "context_fingerprint": context_fingerprint,
            "timing_scope": shared_context.timing_scope,
            "cpu_seconds": pair_seconds["cpu"],
            "cuda_seconds": pair_seconds["cuda"],
            "cpu_oracle_pass": oracle_pass["cpu"],
            "cuda_oracle_pass": oracle_pass["cuda"],
            "parity_pass": True,
        })
        del parity, pair_bundles, pair_seconds, oracle_pass

    aggregate = aggregate_paired_trials(trials, minimum_pairs=3)
    return {
        "seed": seed,
        "pairs": len(schedule),
        "schedule": schedule,
        "trials": trials,
        "aggregate": aggregate,
    }


__all__ = ("produce_source_route_profile_randomized_trials",)
