"""Qualify ordinary F48 Pearson-chain API-default auto at requested cap 16.

This driver is intentionally not executed by unit tests. When explicitly run,
it uses the COS-backed source and ordinary registered auto route.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.runtime.source_auto_evidence import SOURCE_AUTO_EVIDENCE_VERSION
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness
from quant_evaluator.scripts.f48_pearson_auto_references import (
    ARTIFACTS, METRICS, SHAPE, validate_references,
)
from quant_evaluator.scripts.source_tree_provenance import (
    SOURCE_FILES, capture_source_tree, finalize_source_tree,
)

ROUTE_ID = "real_cos_f48_pearson_chain_cap16_tile4"
ROUTE_REASON = "bounded_f48_pearson_chain_gpu_cap16_tile4"
MIN_RAM_BYTES = 32 * 1024**3
MIN_DISK_BYTES = 5 * 1024**3


def _valid_api_execution_receipt(
        receipt, *, request_fingerprint, admitted_source_tile_size):
    if type(receipt) is not dict:
        return False
    if not (
        receipt.get("backend_requested") == "auto"
        and receipt.get("backend_used") == "cuda"
        and receipt.get("auto_backend_reason") == ROUTE_REASON
        and receipt.get("auto_backend_evidence_id") == ROUTE_ID
        and isinstance(receipt.get("auto_backend_evidence_version"), str)
        and receipt["auto_backend_evidence_version"] == SOURCE_AUTO_EVIDENCE_VERSION
        and isinstance(receipt.get("auto_backend_evidence_artifacts"), (list, tuple))
        and bool(receipt["auto_backend_evidence_artifacts"])
        and all(isinstance(path, str) and path
                for path in receipt["auto_backend_evidence_artifacts"])
        and receipt.get("auto_backend_evidence_status") == "measured_source_ab"
        and receipt.get("source_request_fingerprint") == request_fingerprint
        and receipt.get("effective_max_tile_size") == 4
        and receipt.get("admitted_source_tile_size") == admitted_source_tile_size
        and receipt.get("metric_backends") == {metric: "cuda" for metric in METRICS}
        and isinstance(receipt.get("receipt_hash"), str)
        and bool(receipt["receipt_hash"])
    ):
        return False
    fields = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    try:
        expected_hash = stable_content_hex(
            tag="FactorSourceBatchExecutionReceipt.v1", fields=fields)
    except (TypeError, ValueError):
        return False
    return receipt["receipt_hash"] == expected_hash


def _preflight_ok(gate):
    fields = ("available_ram_bytes", "minimum_available_ram_bytes",
              "cos_cache_disk_free_bytes", "required_disk_bytes")
    if type(gate) is not dict or gate.get("pass") is not True:
        return False
    if any(type(gate.get(key)) is not int for key in fields):
        return False
    ram, minimum, disk, required = (gate[key] for key in fields)
    return (minimum >= MIN_RAM_BYTES and ram >= minimum
            and required >= MIN_DISK_BYTES and disk >= required)


def _emit_preflight_failure(args, initial_gate, evaluation_gate, stage):
    harness.emit_report({
        "kind": "real_cos_f48_pearson_chain_default_auto_qualification.v1",
        "status": "preflight_rejected", "evaluation_started": False,
        "failure_stage": stage, "requested_tile_cap": 16,
        "preflight_initial": initial_gate,
        "preflight_before_evaluation": evaluation_gate,
        "limitations": ["No COS source read or GPU evaluation was started after failed preflight."],
    }, args.output)
    raise SystemExit(1)


def run(args):
    reference = validate_references(args.references)
    source_files = tuple(SOURCE_FILES) + (
        "quant_evaluator/scripts/f48_pearson_auto_references.py",
        "quant_evaluator/scripts/benchmark_f48_pearson_default_auto.py",
    )
    before = capture_source_tree(harness.SOURCE_ROOT, files=source_files)
    initial_gate = harness.preflight(args.max_object_mib, args.max_total_mib,
                                     args.max_source_memory_mib)
    if not _preflight_ok(initial_gate):
        _emit_preflight_failure(args, initial_gate, None, "initial")
    records = tiles.select_source_records(
        tiles.read_manifest(tiles.MANIFEST_SHA256), 48,
        args.max_object_mib, args.max_total_mib)
    if args.axis_index and args.axis_index.is_file():
        dates, assets, source_rows = tiles.read_axis_index(
            args.axis_index, tiles.MANIFEST_SHA256, records)
    else:
        dates, assets, source_rows = tiles.intersect_axes(
            tiles.iter_frames(records, tiles.MANIFEST_SHA256,
                              args.max_object_mib, reuse_manifest=True),
            len(records))
    dates, assets, labels = tiles.load_labels(dates, assets, SHAPE[0], SHAPE[1])
    shape = (len(dates), len(assets), len(records))
    if shape != SHAPE or str(labels.values.dtype) != "float64":
        raise ValueError("prepared source does not match the exact F48 float64 envelope")
    evaluation_gate = harness.preflight(args.max_object_mib, args.max_total_mib,
                                        args.max_source_memory_mib)
    if not _preflight_ok(evaluation_gate):
        _emit_preflight_failure(args, initial_gate, evaluation_gate, "before_evaluation")
    policy = GPUExecutionPolicy()
    rejection = _auto_batch_cuda_rejection(policy, 14 * 1024**3)
    if rejection:
        raise RuntimeError("CUDA policy rejected: " + rejection)
    params = (records, source_rows, dates, assets, labels, tiles.MANIFEST_SHA256,
              16, args.max_object_mib, policy, METRICS)
    auto, receipt = harness.run_backend(
        "auto", *params, source_adapter="cos", cos_prefetch="auto",
        max_source_memory_mib=args.max_source_memory_mib,
        cos_prefetch_workers=2, max_prefetch_memory_mib=512,
        use_default_tile_size=True, expected_auto_cuda=True)
    api_receipt = auto.metadata.get("execution_receipt")
    receipt["api_execution_receipt"] = api_receipt
    expected_ranges = tuple((start, start + 4) for start in range(0, 48, 4))
    try:
        actual_ranges = tuple(tuple(row) for row in receipt.get("tile_ranges", ()))
    except (TypeError, ValueError):
        actual_ranges = ()
    route_pass = (
        receipt["api_default_tile_size"] is True
        and receipt["declared_source_tile_size"] == 16
        and type(receipt["admitted_source_tile_size"]) is int
        and receipt["admitted_source_tile_size"] >= 4
        and receipt["effective_max_tile_size"] == 4
        and auto.metadata.get("auto_backend_evidence_id") == ROUTE_ID
        and auto.metadata.get("auto_backend_reason") == ROUTE_REASON
        and auto.metadata.get("source_request_fingerprint")
            == reference["verified_source_request_fingerprint"]
        and _valid_api_execution_receipt(
            api_receipt,
            request_fingerprint=reference["verified_source_request_fingerprint"],
            admitted_source_tile_size=receipt["admitted_source_tile_size"]))
    coverage_pass = (actual_ranges == expected_ranges
                     and receipt["factor_tiles_processed"] == 12
                     and receipt["oom_retries"] == 0
                     and receipt["backend_used"] == "cuda")
    parity = harness.verify_auto_against_reference(auto, METRICS, reference)
    report = {
        "kind": "real_cos_f48_pearson_chain_default_auto_qualification.v1",
        "status": "complete" if route_pass and coverage_pass and parity["pass"] else "verification_failed",
        "benchmark_only": False, "registered_route_verification": True,
        "evidence_id": ROUTE_ID, "requested_tile_cap": 16,
        "effective_tile_width": 4, "api_default_tile_size": True,
        "shape": list(shape), "metric_ids": list(METRICS),
        "manifest_sha256": tiles.MANIFEST_SHA256,
        "reference_reports": [str(p) for p in args.references],
        "reference_artifacts": list(ARTIFACTS),
        "historical_reference_provenance_sha256": reference["historical_source_aggregate_sha256"],
        "preflight_initial": initial_gate,
        "preflight_before_evaluation": evaluation_gate,
        "run": receipt,
        "route_pass": route_pass, "coverage_pass": coverage_pass,
        "reference_comparison": parity,
        "limitations": [
            "The historical references certify exact cap-4 source A/B, not a strict cap-16 A/B.",
            "Cap-16 to effective-width-4 is supported by this ordinary default-auto qualification.",
            "Historical source provenance hash is not a claim of current source-tree or transitive-runtime equivalence.",
            "This research route does not certify PIT or production suitability.",
        ],
    }
    finalize_source_tree(report, before, harness.SOURCE_ROOT, files=source_files)
    if report["source_provenance_verification"]["pass"] is not True:
        report["status"] = "source_tree_changed"
    harness.emit_report(report, args.output)
    if report["status"] != "complete":
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", nargs=2, required=True, type=Path)
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-object-mib", type=int, default=128)
    parser.add_argument("--max-total-mib", type=int, default=4096)
    parser.add_argument("--max-source-memory-mib", type=int, default=4096)
    args = parser.parse_args(argv)
    if (args.max_object_mib, args.max_total_mib, args.max_source_memory_mib) != (128, 4096, 4096):
        parser.error("resource caps are fixed by the measured source envelope")
    run(args)


if __name__ == "__main__":
    main()
