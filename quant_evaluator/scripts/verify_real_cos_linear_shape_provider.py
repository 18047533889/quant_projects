"""Verify a fixed F48 profile through a fresh file-backed provider.

The default invocation performs bounded preflight only. Real COS reads and
runtime execution require --run plus caller-supplied axis, profile, and output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.runtime.source_profile_report_schema import (
    LINEAR_SHAPE_METRICS, LINEAR_SHAPE_ORACLE, LINEAR_SHAPE_REPORT_KIND,
    REPORT_SHAPE, REPORT_TILE_CAP,
)
from quant_evaluator.runtime.source_qualification_cache import has_validated_records
from quant_evaluator.runtime.source_qualification_provider import (
    FileSourceQualificationProvider, clear_source_qualification_provider,
    configure_source_qualification_provider, get_source_qualification_provider,
)
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_source_batch as source_batch
from quant_evaluator.scripts import benchmark_real_cos_linear_shape_profile_abba as shape_benchmark
_checked_run_backend = shape_benchmark._checked_run_backend
_runtime_ready = shape_benchmark._runtime_ready
_warm = shape_benchmark._warm
from quant_evaluator.scripts.source_profile_abba import live_source_profile_context_observer
from quant_evaluator.scripts.source_profile_report_reader import load_source_profile_report

METRICS = LINEAR_SHAPE_METRICS
SHAPE = REPORT_SHAPE
TILE_CAP = REPORT_TILE_CAP
REPORT_KIND = "real_cos_f48_linear_shape_fresh_provider.v1"
MIN_EFFECTIVE_VRAM_BYTES = source_batch.MIN_EFFECTIVE_VRAM_BYTES

def preflight():
    """Reuse the established COS source RAM and cache-disk headroom gate."""
    return shape_benchmark.preflight()



def _diagnostic(code):
    print(f"error_code={code}", file=sys.stderr)


def _write_output(payload, path):
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    encoded = (body + "\n").encode("utf-8")
    if len(encoded) > 1024 * 1024:
        raise ValueError("receipt_size_limit")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(body + "\n")


def _run(args):
    if not args.axis_index.is_file():
        raise ValueError("axis_index_unavailable")
    if not args.profile_report.is_file():
        raise ValueError("profile_report_unavailable")
    if args.output.exists():
        raise FileExistsError("output_exists")
    if get_source_qualification_provider() is not None:
        raise RuntimeError("provider_already_configured")

    # The strict reader bounds input and accepts only the fixed F48 linear-shape schema.
    report = load_source_profile_report(args.profile_report)
    if report.manifest_sha256 != tiles.MANIFEST_SHA256:
        raise ValueError("profile_manifest_mismatch")
    if (report.kind != LINEAR_SHAPE_REPORT_KIND or report.status != "complete"
            or len(report.records) != 2 or report.qualification.winning_backend not in ("cpu", "cuda")):
        raise ValueError("profile_report_contract_invalid")
    for record in report.records:
        context = record.context
        if tuple(context.request_shape) != SHAPE or tuple(context.metric_ids) != METRICS:
            raise ValueError("profile_report_contract_invalid")
    winner = report.qualification.winning_backend

    if has_validated_records():
        raise RuntimeError("validated_source_cache_present")

    policy = GPUExecutionPolicy()
    if winner == "cuda":
        rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
        if rejection:
            raise RuntimeError("cuda_resource_gate_rejected")

    manifest_sha = tiles.MANIFEST_SHA256
    records = tiles.select_source_records(tiles.read_manifest(manifest_sha), 48, 128, 4096)
    dates, assets, source_rows = tiles.read_axis_index(args.axis_index, manifest_sha, records)
    dates, assets, labels = tiles.load_labels(dates, assets, 0, 5500)
    if (len(dates), len(assets), len(records)) != SHAPE:
        raise ValueError("axis_index_shape_mismatch")
    if preflight().get("pass") is not True:
        raise RuntimeError("preflight_rejected_before_source")

    oracle_source = source_batch._make_cos_source(
        records, source_rows, dates, assets, labels, manifest_sha, TILE_CAP, 128,
        prefetch="auto", max_source_memory_mib=4096, prefetch_workers=2,
        max_prefetch_memory_mib=512,
    )
    try:
        from quant_evaluator.scripts.source_linear_shape_oracle import reference_source_linear_shape
        oracle_bundle = reference_source_linear_shape(
            oracle_source, labels, max_tile_size=1, max_result_bytes=64 * 1024**2)
    finally:
        oracle_source.close()

    # CPU winners still need the recorded CUDA context to reproduce the profile identity.
    if not _runtime_ready():
        raise RuntimeError("runtime_unavailable")
    _warm(policy, "cpu")
    if winner == "cuda":
        _warm(policy, "cuda_strict")

    common = dict(records=records, source_rows=source_rows, dates=dates,
        assets=assets, labels=labels, manifest_sha=manifest_sha, tile_size=TILE_CAP,
        max_object_mib=128, policy=policy, selected_metrics=METRICS,
        source_adapter="cos", cos_prefetch="auto", max_source_memory_mib=4096,
        cos_prefetch_workers=2, max_prefetch_memory_mib=512)
    fingerprint = report.records[0].context.request_content_sha256
    provider = FileSourceQualificationProvider({fingerprint: args.profile_report})
    configure_source_qualification_provider(provider)
    installed = True
    try:
        # The provider is deliberately not cleared before this first lookup.
        from quant_evaluator.scripts.linear_shape_provider_verification import verify_provider_run
        run_results = []
        for run_index, require_cache_hit in ((0, False), (1, True)):
            kwargs = dict(common, expected_auto_cuda=(winner == "cuda"),
                context_observer=live_source_profile_context_observer)
            bundle, receipt = _checked_run_backend("auto", **kwargs)
            verification = verify_provider_run(bundle, receipt, report, oracle_bundle,
                run_index=run_index, require_cache_hit=require_cache_hit)
            run_results.append({"run_index": run_index, **verification})
        payload = {"kind": REPORT_KIND, "status": "complete", "run_started": True,
            "shape": list(SHAPE), "metric_ids": list(METRICS),
            "manifest_sha256": manifest_sha, "oracle": LINEAR_SHAPE_ORACLE, "runs": run_results}
        _write_output(payload, args.output)
        return 0
    finally:
        if installed and get_source_qualification_provider() is provider:
            clear_source_qualification_provider()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.run and (args.axis_index is None or args.profile_report is None or args.output is None):
        parser.error("--run requires --axis-index, --profile-report, and --output")
    if args.run:
        if not args.axis_index.is_file():
            parser.error("--axis-index must name an existing file")
        if not args.profile_report.is_file():
            parser.error("--profile-report must name an existing file")
        if args.output.exists():
            parser.error("--output must name a new file")
    try:
        gate = preflight()
        if gate.get("pass") is not True:
            raise RuntimeError("preflight_rejected")
        base = {"kind": REPORT_KIND, "status": "preflight_only", "run_started": False,
            "shape": list(SHAPE), "metric_ids": list(METRICS),
            "manifest_sha256": tiles.MANIFEST_SHA256, "requested_tile_cap": TILE_CAP,
            "preflight": gate}
        if not args.run:
            body = json.dumps(base, ensure_ascii=False, allow_nan=False, indent=2)
            if len(body.encode("utf-8")) > 1024 * 1024:
                raise ValueError("receipt_size_limit")
            print(body)
            return 0
        return _run(args)
    except (Exception, SystemExit) as exc:
        # Error details can contain report data, paths, or COS credentials. Emit a type-only code.
        _diagnostic(type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
