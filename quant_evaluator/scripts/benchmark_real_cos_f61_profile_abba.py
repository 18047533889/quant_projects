"""Explicit F61 source-profile producer, preflight-only by default."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.runtime.source_profile_report_schema import F61_ALL15_SCHEMA, F61_ALL24_SCHEMA
from quant_evaluator.runtime.source_route_profiles import validate_source_route_profile_qualification
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_source_batch as source_batch
from quant_evaluator.scripts.benchmark_real_cos_profile_abba import _runtime_ready
from quant_evaluator.scripts.profile_progress_writer import ExclusiveProgressWriter
from quant_evaluator.scripts.research_cos_cli_preflight import preflight_research_cos_cli
from quant_evaluator.scripts.source_all24_oracle import reference_source_all24
from quant_evaluator.scripts.source_f61_auto_verification import verify_source_profile_auto
from quant_evaluator.scripts.source_profile_abba import (
    live_source_profile_context_observer, produce_source_route_profile_abba,
)
from quant_evaluator.scripts.source_profile_report_reader import parse_source_profile_report
from quant_evaluator.scripts.source_profile_warmup import warm_source_profile


def preflight():
    if preflight_research_cos_cli().get("configured_cli_resolvable") is not True:
        raise SystemExit("research COS CLI is unavailable or not executable")
    return source_batch.preflight(128, 6144, 4096)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--metric-set", choices=("all15", "all24"), default="all24")
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.run:
        if args.axis_index is None or args.output is None:
            parser.error("--run requires --axis-index and --output")
        if not args.axis_index.is_file():
            parser.error("axis index must be an existing file")
        progress = args.output.with_suffix(args.output.suffix + ".progress.json")
        if args.output.exists() or progress.exists():
            parser.error("output and progress paths must both be new")
    schema = F61_ALL15_SCHEMA if args.metric_set == "all15" else F61_ALL24_SCHEMA
    gate = preflight()
    if gate.get("pass") is not True:
        raise SystemExit("preflight rejected: insufficient RAM or cache disk headroom")
    base = {"kind": schema.kind, "status": "preflight_only", "run_started": False,
            "shape": list(schema.shape), "metric_ids": list(schema.metric_ids),
            "manifest_sha256": tiles.MANIFEST_SHA256,
            "requested_tile_cap": schema.requested_tile_cap, "preflight": gate}
    if not args.run:
        print(json.dumps(base, ensure_ascii=False, allow_nan=False))
        return 0
    return run_profile(args, schema, base)


def _checked_run_backend(backend, **kwargs):
    """Re-admit full F61 resources before each freshly-created source."""
    if preflight().get("pass") is not True:
        raise SystemExit("preflight rejected before " + backend + " source run")
    if backend == "cuda_strict" or kwargs.get("expected_auto_cuda") is True:
        rejection = _auto_batch_cuda_rejection(kwargs["policy"], source_batch.MIN_EFFECTIVE_VRAM_BYTES)
        if rejection:
            raise SystemExit("CUDA source run rejected: " + rejection)
    return source_batch.run_backend(backend, **kwargs)


def _write_complete_report(report, path):
    body = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    parse_source_profile_report(body)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(body + "\n")
    print(json.dumps({"kind": report["kind"], "status": report["status"],
                      "qualification_winner": report["qualification_winner"]}))


def run_profile(args, schema, base):
    """Run a complete source API profile; never infer correctness from parity."""
    policy = GPUExecutionPolicy()
    rejection = _auto_batch_cuda_rejection(policy, source_batch.MIN_EFFECTIVE_VRAM_BYTES)
    if rejection:
        raise SystemExit("CUDA route rejected by VRAM gate: " + rejection)
    if not _runtime_ready():
        raise SystemExit("CuPy unavailable; strict CUDA runtime cannot be prepared")
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    events, report_written = [], False
    with ExclusiveProgressWriter(progress_path) as writer:
        try:
            writer.write({**base, "phase": "preparing_input"}, events)
            manifest_sha = tiles.MANIFEST_SHA256
            records = tiles.select_source_records(tiles.read_manifest(manifest_sha), 61, 128, 6144)
            dates, assets, rows = tiles.read_axis_index(args.axis_index, manifest_sha, records)
            if (len(dates), len(assets), len(records)) != schema.shape:
                raise SystemExit("axis index does not describe the fixed F61 full-history request")
            dates, assets, labels = tiles.load_labels(dates, assets, 0, 5500)
            if (len(dates), len(assets), len(records)) != schema.shape:
                raise SystemExit("loaded labels changed the fixed F61 full-history axes")
            if preflight().get("pass") is not True:
                raise SystemExit("preflight rejected before independent source oracle")
            writer.write({**base, "phase": "independent_oracle"}, events)
            source = source_batch._make_cos_source(
                records, rows, dates, assets, labels, manifest_sha,
                schema.requested_tile_cap, 128, prefetch="auto", max_source_memory_mib=4096)
            try:
                oracle_bundle = reference_source_all24(
                    source, labels, metrics=schema.metric_ids,
                    max_tile_size=1, max_result_bytes=64 * 1024**2)
            finally:
                source.close()
            warm_source_profile(policy, "cpu", schema.metric_ids)
            warm_source_profile(policy, "cuda_strict", schema.metric_ids)
            common = dict(
                records=records, source_rows=rows, dates=dates, assets=assets, labels=labels,
                manifest_sha=manifest_sha, tile_size=schema.requested_tile_cap,
                max_object_mib=128, policy=policy, selected_metrics=schema.metric_ids,
                source_adapter="cos", cos_prefetch="auto", max_source_memory_mib=4096,
                cos_prefetch_workers=2, max_prefetch_memory_mib=512,
                source_auto_policy="qualified_only")
            def progress(event):
                if len(events) >= 4:
                    raise ValueError("ABBA progress exceeds four runs")
                events.append(event)
                writer.write(base, events)
            result = produce_source_route_profile_abba(
                oracle=lambda **_: oracle_bundle, run_kwargs=common,
                context_observer=live_source_profile_context_observer,
                run_backend_fn=_checked_run_backend, progress_observer=progress)
            context = result.records[0].context
            qualification = validate_source_route_profile_qualification(
                result.records, expected_context=context)
            winner = qualification.winning_backend
            selected_profile = getattr(result.records[0], winner)
            verifications = []
            for index, supplied in enumerate((result.records, None), start=4):
                kwargs = dict(common, expected_auto_cuda=winner == "cuda",
                              context_observer=live_source_profile_context_observer)
                if supplied is not None:
                    kwargs["source_qualification"] = supplied
                bundle, receipt = _checked_run_backend("auto", **kwargs)
                verifications.append(verify_source_profile_auto(
                    bundle, receipt, qualification, context, oracle_bundle,
                    selected_profile=selected_profile, run_index=index,
                    require_cache_hit=index == 5))
            report = {**base, "status": "complete", "run_started": True,
                      "request_shape": list(schema.shape), "oracle": schema.oracle,
                      "profile_records": [asdict(item) for item in result.records],
                      "run_order": list(result.run_order), "oracle_reports": list(result.oracle_reports),
                      "qualification_winner": winner, "auto_verification": verifications[0],
                      "default_auto_verification": verifications[1]}
            _write_complete_report(report, args.output)
            report_written = True
            writer.write(base, events, status="complete")
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            if not report_written:
                writer.write(base, events, status="failed", error_type=type(error).__name__)
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
