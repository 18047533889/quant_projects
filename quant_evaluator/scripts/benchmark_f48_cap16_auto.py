"""Benchmark-only F48 default-width auto-route candidate against strict CUDA."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import threading
from unittest.mock import patch
from pathlib import Path

from quant_evaluator.api import factor_source
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime import source_auto_evidence
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts.f48_auto_references import (
    F48_METRICS, F48_SHAPE, HASH_FIELDS, validate_f48_auto_references,
)
from quant_evaluator.scripts.source_tree_provenance import (
    SOURCE_FILES, capture_source_tree, finalize_source_tree,
)

CANDIDATE_STATUS = "benchmark_only_pending_auto"
REGISTERED_EVIDENCE_ID = "real_cos_f48_mixed_three_cap16_tile2"
REFERENCE_EVIDENCE_ID = "real_cos_f48_mixed_three_tile2"
EVIDENCE_ID = "real_cos_f48_mixed_three_cap16_tile2_candidate"
CANDIDATE_REASON = "bounded_f48_mixed_three_gpu_cap16_tile2"
LOCK = threading.Lock()


def _evidence():
    matches = [item for item in source_auto_evidence.SOURCE_AUTO_EVIDENCE
               if item.evidence_id == REFERENCE_EVIDENCE_ID]
    if len(matches) != 1:
        raise RuntimeError("F48 source evidence envelope is unavailable")
    item = matches[0]
    if (item.shape != F48_SHAPE or item.metrics != frozenset(F48_METRICS)
            or item.exact_requested_tile != 2 or item.certified_tile_widths != (2,)):
        raise RuntimeError("F48 source evidence envelope changed")
    if item.evidence_status not in ("measured_source_ab", "measured_source_ab_pending_auto"):
        raise RuntimeError("F48 source evidence status is not eligible for benchmark")
    return item


def _candidate_selector(*, shape, metrics, source_dtype, label_dtype,
                        requested_tile_width):
    if (tuple(shape) != F48_SHAPE
            or len(tuple(metrics)) != len(F48_METRICS)
            or frozenset(metrics) != frozenset(F48_METRICS)
            or source_dtype != "float64" or label_dtype != "float64"
            or type(requested_tile_width) is not int or requested_tile_width != 16):
        return source_auto_evidence.select_source_auto_route(
            shape=shape, metrics=metrics, source_dtype=source_dtype,
            label_dtype=label_dtype, requested_tile_width=requested_tile_width)
    item = _evidence()
    # Reuse the exact cap-2 VRAM floor and artifacts from the published entry.
    return source_auto_evidence.SourceAutoRoute(
        evidence_id=EVIDENCE_ID, legacy_reason=CANDIDATE_REASON,
        effective_tile_width=2,
        minimum_effective_vram_bytes=item.minimum_effective_vram_bytes,
        evidence_artifacts=item.evidence_artifacts,
        evidence_status=CANDIDATE_STATUS)


@contextmanager
def cap16_candidate_scope(*, shape, metrics, source_dtype, label_dtype,
                          requested_tile_width):
    """Main-thread-only temporary selector injection for the exact F48 request."""
    item = _evidence()
    if (tuple(shape) != F48_SHAPE or (len(tuple(metrics)) != len(F48_METRICS) or frozenset(metrics) != frozenset(F48_METRICS))
            or source_dtype != "float64" or label_dtype != "float64"
            or type(requested_tile_width) is not int or requested_tile_width != 16):
        raise ValueError("candidate scope requires exact F48 declared-width-16 default profile")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("candidate scope is restricted to the CLI main thread")
    if not LOCK.acquire(blocking=False):
        raise RuntimeError("candidate scope is non-reentrant")
    candidate = {"provenance": "benchmark_only_pending_auto_candidate",
                 "evidence_id": EVIDENCE_ID, "reference_evidence_id": REFERENCE_EVIDENCE_ID,
                 "evidence_status": CANDIDATE_STATUS,
                 "requested_tile_width": 16, "effective_tile_width": 2,
                 "shape": list(F48_SHAPE), "metric_ids": list(F48_METRICS),
                 "source_dtype": "float64", "label_dtype": "float64",
                 "minimum_effective_vram_bytes": item.minimum_effective_vram_bytes,
                 "source_ab_artifacts": list(item.evidence_artifacts)}
    try:
        with patch.object(factor_source, "select_source_auto_route",
                          side_effect=_candidate_selector):
            yield candidate
    finally:
        LOCK.release()


def _preflight(args):
    result = harness.preflight(args.max_object_mib, args.max_total_mib,
                               args.max_source_memory_mib)
    if not result.get("pass"):
        raise RuntimeError("RAM or COS cache disk headroom failed preflight")
    return result


def _memory_profile(shape):
    cells = shape[0] * shape[1]
    memory = harness.source_width_preparation.width_memory_admission(
        shape=shape, widths=tuple(range(1, 17)), source_budget_bytes=4096 * 1024**2,
        max_prefetch_memory_bytes=512 * 1024**2, prefetch_workers=2,
        prefetch_enabled=True,
        extra_assembly_bytes_per_cell=(256 * 1024**2 + cells - 1) // cells)
    admitted = max((int(width) for width, item in memory.items()
                    if item["admitted"]), default=0)
    if admitted < 2:
        raise RuntimeError("source memory budget cannot admit measured GPU width 2")
    return memory, admitted


def run(args):
    ordinary = getattr(args, "ordinary", False)
    if type(ordinary) is not bool:
        raise TypeError("ordinary must be bool")
    selected = F48_METRICS
    reference = validate_f48_auto_references(
        args.references, selected, tiles.MANIFEST_SHA256)
    source_files = tuple(SOURCE_FILES) + ("quant_evaluator/scripts/benchmark_f48_cap16_auto.py",)
    before = capture_source_tree(harness.SOURCE_ROOT, files=source_files)
    report = {"status": "partial", "kind": "real_cos_f48_cap16_default_auto_benchmark.v1",
              "benchmark_only": not ordinary, "production_route_changed": False,
              "registered_route_verification": ordinary,
              "manifest_sha256": tiles.MANIFEST_SHA256, "shape": list(F48_SHAPE),
              "factor_dtype": "float64", "label_dtype": "float64",
              "metric_ids": list(selected), "declared_tile_size": 16,
              "candidate_effective_tile_size": 2, "explicit_cuda_tile_size": 2,
              "source_adapter": "cos", "cos_prefetch": "auto",
              "cos_prefetch_workers": 2, "max_prefetch_memory_mib": 512,
              "max_object_mib": 128, "max_total_mib": 4096,
              "max_source_memory_mib": 4096,
              "reference_reports": [str(p) for p in args.references], "runs": {}}
    gate = _preflight(args)
    report["preflight_before_axis_preparation"] = gate
    records = tiles.select_source_records(
        tiles.read_manifest(tiles.MANIFEST_SHA256), 48,
        args.max_object_mib, args.max_total_mib)
    if args.axis_index and args.axis_index.is_file():
        dates, assets, source_rows = tiles.read_axis_index(
            args.axis_index, tiles.MANIFEST_SHA256, records)
    else:
        dates, assets, source_rows = tiles.intersect_axes(
            tiles.iter_frames(records, tiles.MANIFEST_SHA256, args.max_object_mib,
                              reuse_manifest=True), len(records))
    dates, assets, labels = tiles.load_labels(dates, assets, 2586, 5461)
    shape = (len(dates), len(assets), len(records))
    if shape != F48_SHAPE or str(labels.values.dtype) != "float64":
        raise ValueError("prepared source/label axes do not match exact F48 float64 envelope")
    report["shape"] = list(shape)
    memory, admitted_width = _memory_profile(shape)
    report["source_memory_preflight"] = memory
    report["memory_admitted_tile_size"] = admitted_width
    policy = GPUExecutionPolicy()
    rejection = _auto_batch_cuda_rejection(policy, _evidence().minimum_effective_vram_bytes)
    if rejection:
        raise RuntimeError("CUDA policy rejected: " + rejection)
    params = (records, source_rows, dates, assets, labels, tiles.MANIFEST_SHA256,
              16, args.max_object_mib, policy, selected)
    gate = _preflight(args)
    scope = (nullcontext(None) if ordinary else cap16_candidate_scope(
        shape=shape, metrics=selected, source_dtype="float64",
        label_dtype="float64", requested_tile_width=16))
    with scope as candidate:
        auto, auto_receipt = harness.run_backend(
            "auto", *params, source_adapter="cos", cos_prefetch="auto",
            max_source_memory_mib=4096, cos_prefetch_workers=2,
            max_prefetch_memory_mib=512, use_default_tile_size=True,
            expected_auto_cuda=True)
    auto_receipt["preflight"] = gate
    auto_receipt["benchmark_auto_candidate"] = candidate
    auto_receipt["auto_backend_reason"] = auto.metadata.get("auto_backend_reason")
    auto_receipt["effective_max_tile_size"] = auto.metadata.get("effective_max_tile_size")
    auto_verified = harness.verify_auto_against_reference(auto, selected, reference)
    gate = _preflight(args)
    cuda, cuda_receipt = harness.run_backend(
        "cuda_strict", *params[:6], 2, args.max_object_mib, policy, selected,
        source_adapter="cos", cos_prefetch="auto",
        max_source_memory_mib=4096, cos_prefetch_workers=2,
        max_prefetch_memory_mib=512)
    cuda_receipt["preflight"] = gate
    direct = harness.compare(auto, cuda, selected, expected_days=len(dates))
    expected_ranges = [(start, start + 2) for start in range(0, 48, 2)]
    coverage_pass = all(
        receipt.get("tile_ranges") == expected_ranges
        and receipt.get("factor_tiles_processed") == 24
        and receipt.get("oom_retries") == 0
        for receipt in (auto_receipt, cuda_receipt))
    identity_pass = all(auto_receipt.get(key) is not None
                        and auto_receipt[key] == cuda_receipt.get(key)
                        for key in HASH_FIELDS)
    route_pass = (auto_receipt["api_default_tile_size"] is True
                  and auto_receipt["declared_source_tile_size"] == 16
                  and auto_receipt["admitted_source_tile_size"] == admitted_width
                  and auto_receipt["effective_max_tile_size"] == 2
                  and auto_receipt["auto_backend_reason"] == CANDIDATE_REASON
                  and auto.metadata.get("auto_backend_evidence_status") == (
                      "measured_source_ab" if ordinary else CANDIDATE_STATUS)
                  and auto.metadata.get("auto_backend_evidence_id") == (
                      REGISTERED_EVIDENCE_ID if ordinary else EVIDENCE_ID)
                  and auto.metadata.get("source_request_fingerprint")
                  == reference["verified_source_request_fingerprint"])
    report.update(status="complete" if route_pass and coverage_pass and identity_pass and auto_verified["pass"] and direct["pass"] else "verification_failed",
                  benchmark_auto_candidate=candidate,
                  source_memory_preflight=memory,
                  runs={"auto_default": auto_receipt, "cuda_strict_tile2": cuda_receipt},
                  route_pass=route_pass, coverage_pass=coverage_pass,
                  identity_pass=identity_pass, reference_comparison=auto_verified,
                  direct_comparison=direct,
                  limitations=["Research metrics only; no PIT or production certification.",
                               "Ordinary auto selected the registered route without injection."
                               if ordinary else "The F48 selector candidate exists only in this benchmark process.",
                               "This driver does not modify the production route registry.",
                               "Stored reference receipts do not attest current source-code equivalence."])
    finalize_source_tree(report, before, harness.SOURCE_ROOT, files=source_files)
    harness.emit_report(report, args.output)
    if report["status"] != "complete":
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", nargs=2, required=True, type=Path,
                        metavar=("CPU_FIRST", "CUDA_FIRST"))
    parser.add_argument("--ordinary", action="store_true",
                        help="verify registered default auto without candidate injection")
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-object-mib", type=int, default=128)
    parser.add_argument("--max-total-mib", type=int, default=4096)
    parser.add_argument("--max-source-memory-mib", type=int, default=4096)
    parser.add_argument("--cos-prefetch-workers", type=int, default=2)
    parser.add_argument("--max-prefetch-memory-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if (args.max_object_mib, args.max_total_mib, args.max_source_memory_mib,
            args.cos_prefetch_workers, args.max_prefetch_memory_mib) != (128, 4096, 4096, 2, 512):
        parser.error("F48 cap16 benchmark resource settings are fixed by the evidence envelope")
    run(args)


if __name__ == "__main__":
    main()
