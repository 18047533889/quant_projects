"""Explicit fresh-process F61 provider verification; preflight-only by default.

A separate receipt family records physical provider runs 0/1. The reused auto
verifier uses protocol positions 4/5; these are labels, not extra API calls.
Reports remain caller-trusted, unsigned evidence, not source authentication.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.source_profile_report_schema import F61_ALL15_SCHEMA, F61_ALL24_SCHEMA
from quant_evaluator.runtime.source_qualification_provider import (
    FileSourceQualificationProvider, clear_source_qualification_provider_if_current,
    configure_source_qualification_provider_if_absent,
    get_source_qualification_provider,
)
from quant_evaluator.runtime.source_qualification_cache import has_validated_records
from quant_evaluator.runtime.source_route_profiles import validate_source_route_profile_qualification
from quant_evaluator.scripts.source_profile_report_reader import (
    SourceProfileReport, load_source_profile_report, MAX_REPORT_BYTES,
)
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_source_batch as source_batch
from quant_evaluator.scripts.benchmark_real_cos_f61_profile_abba import (
    preflight, _checked_run_backend, _runtime_ready, warm_source_profile,
)
from quant_evaluator.scripts.source_all24_oracle import reference_source_all24
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.scripts.source_f61_auto_verification import verify_source_profile_auto
from quant_evaluator.scripts.source_profile_abba import live_source_profile_context_observer
from quant_evaluator.scripts.source_f61_axis_admission import validate_f61_raw_axes

REPORT_KIND = "real_cos_f61_fresh_provider.v1"
_SCHEMAS = {schema.kind: schema for schema in (F61_ALL15_SCHEMA, F61_ALL24_SCHEMA)}


def _report_contract(report):
    if type(report) is not SourceProfileReport or report.status != "complete":
        raise ValueError("profile_report_contract_invalid")
    schema = _SCHEMAS.get(report.kind)
    if schema is None or type(report.records) is not tuple or len(report.records) != 2:
        raise ValueError("profile_report_contract_invalid")
    context = report.records[0].context
    if (context.request_shape != schema.shape or context.metric_ids != schema.metric_ids
            or context.requested_tile_size != schema.requested_tile_cap
            or context.metric_error_tolerances != tuple((m, 1e-10) for m in schema.metric_ids)):
        raise ValueError("profile_report_domain_invalid")
    qualification = validate_source_route_profile_qualification(
        report.records, expected_context=context)
    if qualification != report.qualification:
        raise ValueError("profile_qualification_mismatch")
    return schema, context, qualification


def verify_provider_run(bundle, receipt, report, oracle_bundle, *, run_index, require_cache_hit):
    """Verify ordinary auto's provider origin, exact winner outputs and oracle."""
    if (type(run_index) is not int or run_index not in (0, 1)
            or type(require_cache_hit) is not bool or require_cache_hit != (run_index == 1)):
        raise ValueError("provider_run_index_invalid")
    schema, context, qualification = _report_contract(report)
    expected = (
        ("process_cache", "not_checked", "cache_hit") if require_cache_hit
        else ("report_candidate", "candidate_validated", "supplied"))
    metadata = bundle.metadata
    fields = ("source_qualification_origin", "source_qualification_provider_status",
              "source_qualification_cache_status")
    if tuple(metadata.get(field) for field in fields) != expected:
        raise ValueError("provider_origin_status_or_cache_mismatch")
    selected = getattr(report.records[0], qualification.winning_backend)
    verified = verify_source_profile_auto(
        bundle, receipt, qualification, context, oracle_bundle,
        selected_profile=selected, run_index=run_index + 4, require_cache_hit=require_cache_hit)
    # Change only the label of this separate family's already checked oracle
    # receipt. No extra evaluation occurs and no original ABBA receipt is changed.
    verified["oracle_report"] = {**verified["oracle_report"], "run_index": run_index}
    verified.update(
        run_index=run_index, auto_protocol_run_index=run_index + 4,
        profile_kind=schema.kind, context=asdict(context), cache_status=expected[2],
        **dict(zip(fields, expected)),
    )
    return verified


def _write_output(payload, path):
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(body.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("receipt_size_limit")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(body)


def _require_fresh_state():
    if get_source_qualification_provider() is not None:
        raise RuntimeError("provider_already_configured")
    if has_validated_records():
        raise RuntimeError("validated_source_cache_present")


def _run(args):
    _require_fresh_state()
    report = load_source_profile_report(args.profile_report)
    schema, context, qualification = _report_contract(report)
    if report.manifest_sha256 != tiles.MANIFEST_SHA256:
        raise ValueError("profile_manifest_mismatch")
    policy = GPUExecutionPolicy()
    winner = qualification.winning_backend
    if winner == "cuda" and _auto_batch_cuda_rejection(policy, source_batch.MIN_EFFECTIVE_VRAM_BYTES):
        raise RuntimeError("cuda_resource_gate_rejected")
    # Even CPU winners require the recorded CUDA runtime identity in this
    # CPU/CUDA qualification domain. Prepare it before any external COS reads.
    if not _runtime_ready():
        raise RuntimeError("runtime_unavailable")
    records = tiles.select_source_records(
        tiles.read_manifest(tiles.MANIFEST_SHA256), 61, 128, 6144)
    dates, assets, rows = tiles.read_axis_index(args.axis_index, tiles.MANIFEST_SHA256, records)
    try:
        validate_f61_raw_axes(dates, assets, records, schema)
    except SystemExit as error:
        # Preserve this CLI's type-only ValueError contract for bad axes.
        raise ValueError("axis_index_shape_mismatch") from error
    # Labels may trim the raw tail; qualification binds the exact final axes.
    dates, assets, labels = tiles.load_labels(dates, assets, 0, 5500)
    if (len(dates), len(assets), len(records)) != schema.shape:
        raise ValueError("label_axis_shape_mismatch")
    if preflight().get("pass") is not True:
        raise RuntimeError("preflight_rejected_before_oracle")
    source = source_batch._make_cos_source(
        records, rows, dates, assets, labels, tiles.MANIFEST_SHA256,
        schema.requested_tile_cap, 128, prefetch="auto", max_source_memory_mib=4096,
        prefetch_workers=2, max_prefetch_memory_mib=512)
    try:
        oracle = reference_source_all24(
            source, labels, metrics=schema.metric_ids, max_tile_size=1,
            max_result_bytes=64 * 1024**2)
    finally:
        source.close()
    warm_source_profile(policy, "cpu", schema.metric_ids)
    if winner == "cuda":
        warm_source_profile(policy, "cuda_strict", schema.metric_ids)
    # Preparation must not overwrite a provider installed by another caller.
    _require_fresh_state()
    provider = FileSourceQualificationProvider(
        {context.request_content_sha256: args.profile_report})
    if not configure_source_qualification_provider_if_absent(provider):
        raise RuntimeError("provider_slot_already_claimed")
    common = dict(
        records=records, source_rows=rows, dates=dates, assets=assets, labels=labels,
        manifest_sha=tiles.MANIFEST_SHA256, tile_size=schema.requested_tile_cap,
        max_object_mib=128, policy=policy, selected_metrics=schema.metric_ids,
        source_adapter="cos", cos_prefetch="auto", max_source_memory_mib=4096,
        cos_prefetch_workers=2, max_prefetch_memory_mib=512,
        source_auto_policy="qualified_only",
        expected_auto_cuda=winner == "cuda",
        context_observer=live_source_profile_context_observer,
    )
    try:
        results = []
        for run_index in (0, 1):
            # No explicit source_qualification: ordinary auto must find the
            # installed file provider on run 0 and process cache on run 1.
            bundle, receipt = _checked_run_backend("auto", **common)
            results.append(verify_provider_run(
                bundle, receipt, report, oracle, run_index=run_index,
                require_cache_hit=run_index == 1))
        _write_output({
            "kind": REPORT_KIND, "status": "complete", "run_started": True,
            "shape": list(schema.shape), "metric_ids": list(schema.metric_ids),
            "profile_kind": schema.kind, "manifest_sha256": report.manifest_sha256,
            "oracle": schema.oracle, "runs": results,
        }, args.output)
    finally:
        clear_source_qualification_provider_if_current(provider)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.run:
        if args.axis_index is None or args.profile_report is None or args.output is None:
            parser.error("--run requires --axis-index, --profile-report and --output")
        if not args.axis_index.is_file() or not args.profile_report.is_file():
            parser.error("--run inputs must be existing files")
        if args.output.exists():
            parser.error("--output must be a new file")
    try:
        gate = preflight()
        if gate.get("pass") is not True:
            raise RuntimeError("preflight_rejected")
        if not args.run:
            print(json.dumps({
                "kind": REPORT_KIND, "status": "preflight_only", "run_started": False,
                "shape": list(F61_ALL24_SCHEMA.shape),
                "allowed_profile_kinds": list(_SCHEMAS), "preflight": gate,
            }, ensure_ascii=False, allow_nan=False))
            return 0
        return _run(args)
    except (Exception, SystemExit) as error:
        # Values can contain paths, report text or COS credentials.
        print("error_code=" + type(error).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
