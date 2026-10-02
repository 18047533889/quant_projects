"""One-shot benchmark-only F48 auto-route injection.

This module patches only the selector reference held by factor_source, and
only while the standalone F48 reference benchmark process is evaluating its
exact pending candidate. It never changes the production evidence registry.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
from unittest.mock import patch

from quant_evaluator.api import factor_source
from quant_evaluator.runtime import source_auto_evidence

F48_SHAPE = (2586, 5461, 48)
F48_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
F48_EVIDENCE_ID = "real_cos_f48_mixed_three_tile2"
F48_BENCHMARK_CANDIDATE_STATUS = "benchmark_only_pending_auto"
_SCOPE_LOCK = threading.Lock()


def _pending_f48_evidence():
    matches = [item for item in source_auto_evidence.SOURCE_AUTO_EVIDENCE
               if item.evidence_id == F48_EVIDENCE_ID]
    if len(matches) != 1:
        raise RuntimeError("the pending F48 source evidence envelope is unavailable")
    item = matches[0]
    if (item.evidence_status != "measured_source_ab_pending_auto"
            or item.shape != F48_SHAPE or item.metrics != frozenset(F48_METRICS)
            or item.exact_requested_tile != 2 or item.certified_tile_widths != (2,)):
        raise RuntimeError("the pending F48 source evidence envelope changed")
    return item


def _candidate_route(*, shape, metrics, source_dtype, label_dtype,
                     requested_tile_width):
    shape = tuple(shape)
    metrics = tuple(metrics)
    ordinary = source_auto_evidence.select_source_auto_route(
        shape=shape, metrics=metrics, source_dtype=source_dtype,
        label_dtype=label_dtype, requested_tile_width=requested_tile_width,
    )
    if ordinary is not None:
        return ordinary
    item = _pending_f48_evidence()
    if (source_dtype != "float64" or label_dtype != "float64"
            or shape != F48_SHAPE or metrics != F48_METRICS
            or type(requested_tile_width) is not int or requested_tile_width != 2
            or not item.matches(shape, metrics, requested_tile_width)):
        return None
    return source_auto_evidence.SourceAutoRoute(
        evidence_id=item.evidence_id,
        legacy_reason=item.legacy_reason,
        effective_tile_width=2,
        minimum_effective_vram_bytes=item.minimum_effective_vram_bytes,
        evidence_artifacts=item.evidence_artifacts,
        evidence_status=F48_BENCHMARK_CANDIDATE_STATUS,
    )


@contextmanager
def f48_benchmark_candidate_scope(*, shape, metrics, source_dtype, label_dtype,
                                  requested_tile_width):
    """Inject the candidate in a non-reentrant standalone CLI scope.

    This patch is process-global and not thread-safe. Use only from the
    standalone benchmark CLI's exact F48 reference mode, never from an
    embedded or long-lived service. All request predicates are checked before
    patching; the scope restores the API selector even when evaluation raises.
    """
    shape = tuple(shape)
    metrics = tuple(metrics)
    item = _pending_f48_evidence()
    if (shape != F48_SHAPE or metrics != F48_METRICS
            or source_dtype != "float64" or label_dtype != "float64"
            or type(requested_tile_width) is not int or requested_tile_width != 2
            or not item.matches(shape, metrics, requested_tile_width)):
        raise ValueError("benchmark candidate scope requires the exact F48 tile-2 profile")
    provenance = {
        "provenance": "benchmark_only_pending_auto_candidate",
        "evidence_id": item.evidence_id,
        "evidence_status": F48_BENCHMARK_CANDIDATE_STATUS,
        "source_auto_evidence_version": source_auto_evidence.SOURCE_AUTO_EVIDENCE_VERSION,
        "source_ab_artifacts": list(item.evidence_artifacts),
        "shape": list(F48_SHAPE),
        "metric_ids": list(F48_METRICS),
        "source_dtype": "float64",
        "label_dtype": "float64",
        "requested_tile_width": 2,
    }
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("F48 benchmark candidate scope is restricted to the CLI main thread")
    if not _SCOPE_LOCK.acquire(blocking=False):
        raise RuntimeError("F48 benchmark candidate scope is non-reentrant")
    try:
        with patch.object(factor_source, "select_source_auto_route", side_effect=_candidate_route):
            yield provenance
    finally:
        _SCOPE_LOCK.release()


__all__ = (
    "F48_BENCHMARK_CANDIDATE_STATUS", "F48_EVIDENCE_ID", "F48_METRICS",
    "F48_SHAPE", "f48_benchmark_candidate_scope",
)
