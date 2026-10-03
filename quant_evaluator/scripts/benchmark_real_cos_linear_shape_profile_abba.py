"""Explicit F48 real-COS linear-shape ABBA profiling; dry-run by default.

Measures the complete source API, not isolated shape kernels. Independent
reference arithmetic and current-context qualification are both mandatory.
No source paths, credentials or partial qualification records enter receipts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.runtime.source_profile_report_schema import (
    LINEAR_SHAPE_METRICS, LINEAR_SHAPE_ORACLE, LINEAR_SHAPE_REPORT_KIND,
    REPORT_SHAPE, REPORT_TILE_CAP,
)
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_source_batch as source_batch
from quant_evaluator.scripts.benchmark_real_cos_profile_abba import (
    _checked_run_backend, _runtime_ready,
)
from quant_evaluator.scripts.profile_progress_writer import ExclusiveProgressWriter
from quant_evaluator.scripts.research_cos_cli_preflight import preflight_research_cos_cli
from quant_evaluator.scripts.source_profile_abba import (
    _oracle_report, live_source_profile_context_observer,
    produce_source_route_profile_abba,
)

METRICS = LINEAR_SHAPE_METRICS
SHAPE = REPORT_SHAPE
TILE_CAP = REPORT_TILE_CAP
REPORT_KIND = LINEAR_SHAPE_REPORT_KIND
MIN_EFFECTIVE_VRAM_BYTES = source_batch.MIN_EFFECTIVE_VRAM_BYTES


def preflight():
    cli = preflight_research_cos_cli()
    if cli.get("configured_cli_resolvable") is not True:
        raise SystemExit("research COS CLI is unavailable or not executable")
    return source_batch.preflight(128, 4096, 4096)


def _write_report(report, path=None):
    body = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if len(body.encode("utf-8")) > 1024**2:
        raise ValueError("profile report exceeds 1 MiB")
    if path is None:
        print(body)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite another worker's evidence, including a late-created file.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(body + "\n")
    print(json.dumps({"kind": REPORT_KIND, "status": report["status"],
                      "qualification_winner": report.get("qualification_winner")}))


def _warm(policy, backend):
    """Tiny six-metric plan warms JIT/device code outside measured calls."""
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.factor_tile_source import FactorTile
    from quant_evaluator.contracts.label_bundle import LabelBundle
    times = np.arange(24, dtype=np.int64)
    assets = AxisRef("asset", "int64", 72, np.arange(72, dtype=np.int64))
    time_axis = AxisRef("time", "int64", 24, times)
    rng = np.random.default_rng(20261004)
    values = rng.normal(size=(24, 72, 2))
    labels = LabelBundle("shape-warmup", rng.normal(size=(24, 72)), 1,
        decision_time=tuple(times), label_start_time=tuple(times),
        label_end_time=tuple(times + 1), asset_axis=assets)

    class WarmSource:
        factor_ids = ("shape-warm-a", "shape-warm-b")
        dtype, snapshot_id, max_tile_size = "float64", "shape-warmup-only", 2
        def __init__(self):
            self.time_axis, self.asset_axis = time_axis, assets
        def read_tile(self, start, end):
            batch = FactorBatch(self.factor_ids[start:end], time_axis, assets,
                                values[:, :, start:end])
            return FactorTile(start, end, batch, self.snapshot_id)
        def close(self):
            pass

    source = WarmSource()
    try:
        source_batch.evaluate_factor_source_batch(source, labels, metrics=METRICS,
            backend=backend, max_tile_size=2, gpu_policy=policy)
    finally:
        source.close()


def _verify_auto(bundle, receipt, qualification, context, oracle_bundle,
                 *, run_index, require_cache_hit):
    """Validate values AND exact measured route, context and tile width."""
    if receipt.get("context_before") != context or receipt.get("context_after") != context:
        raise ValueError("auto verification context differs from measured profile")
    winner = qualification.winning_backend
    width = dict(qualification.actual_tile_sizes)[winner]
    metadata = bundle.metadata
    if (metadata.get("source_qualification_applied") is not True
            or metadata.get("source_qualification_status") != "qualified_current_source"
            or metadata.get("source_qualification_winner") != winner
            or receipt.get("backend_used") != winner
            or receipt.get("effective_max_tile_size") != width):
        raise ValueError("auto did not apply the exact qualified source profile")
    if require_cache_hit and metadata.get("source_qualification_cache_status") != "cache_hit":
        raise ValueError("default auto did not reuse the exact qualified profile")
    report = _oracle_report(bundle, receipt, context, backend=winner,
        run_index=run_index, oracle=lambda **_: oracle_bundle)
    verification = {
        "backend_used": winner,
        "qualification_status": metadata["source_qualification_status"],
        "qualification_applied": True,
        "qualification_winner": winner,
        "output_matches_independent_oracle": True,
        "oracle_report": report,
        "values_sha256": {name: hashlib.sha256(
            np.ascontiguousarray(bundle.scalar_metrics[name]).tobytes()).hexdigest()
            for name in METRICS},
    }
    if require_cache_hit:
        verification["cache_status"] = metadata["source_qualification_cache_status"]
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.run and (args.axis_index is None or args.output is None):
        parser.error("--run requires --axis-index and --output")
    progress_path = None
    if args.run:
        if not args.axis_index.is_file():
            parser.error("axis index must be an existing file")
        progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
        if args.output.exists() or progress_path.exists():
            parser.error("output and progress paths must both be new")
    gate = preflight()
    if gate.get("pass") is not True:
        raise SystemExit("preflight rejected: insufficient RAM or cache disk headroom")
    base = {"kind": REPORT_KIND, "status": "preflight_only", "run_started": False,
        "shape": list(SHAPE), "metric_ids": list(METRICS),
        "manifest_sha256": tiles.MANIFEST_SHA256, "requested_tile_cap": TILE_CAP,
        "preflight": gate}
    if not args.run:
        _write_report(base)
        return 0
    policy = GPUExecutionPolicy()
    rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
    if rejection:
        raise SystemExit("CUDA route rejected by existing VRAM gate: " + rejection)
    manifest_sha = tiles.MANIFEST_SHA256
    records = tiles.select_source_records(tiles.read_manifest(manifest_sha), 48, 128, 4096)
    dates, assets, rows = tiles.read_axis_index(args.axis_index, manifest_sha, records)
    dates, assets, labels = tiles.load_labels(dates, assets, 0, 5500)
    if (len(dates), len(assets), len(records)) != SHAPE:
        raise SystemExit("axis index does not describe the fixed F48 full-history request")
    if preflight().get("pass") is not True:
        raise SystemExit("preflight rejected before source/oracle execution")
    from quant_evaluator.scripts.source_linear_shape_oracle import reference_source_linear_shape
    source = source_batch._make_cos_source(records, rows, dates, assets, labels,
        manifest_sha, TILE_CAP, 128, prefetch="auto", max_source_memory_mib=4096)
    try:
        oracle_bundle = reference_source_linear_shape(source, labels,
            max_tile_size=1, max_result_bytes=64 * 1024**2)
    finally:
        source.close()
    if not _runtime_ready():
        raise SystemExit("CuPy unavailable; strict CUDA warmup cannot be performed")
    _warm(policy, "cpu")
    _warm(policy, "cuda_strict")
    common = dict(records=records, source_rows=rows, dates=dates, assets=assets,
        labels=labels, manifest_sha=manifest_sha, tile_size=TILE_CAP,
        max_object_mib=128, policy=policy, selected_metrics=METRICS,
        source_adapter="cos", cos_prefetch="auto", max_source_memory_mib=4096,
        cos_prefetch_workers=2, max_prefetch_memory_mib=512)
    if preflight().get("pass") is not True:
        raise SystemExit("preflight rejected before ABBA timing")
    events = []
    def progress(event):
        if len(events) >= 4:
            raise ValueError("ABBA progress exceeds four runs")
        events.append(event)
        progress_writer.write(base, events)
    # Acquire immediately before ABBA and outside the error-saving handler.
    progress_writer = ExclusiveProgressWriter(progress_path)
    progress_writer.__enter__()
    report_written = False
    try:
        result = produce_source_route_profile_abba(
            oracle=lambda **_: oracle_bundle, run_kwargs=common,
            context_observer=live_source_profile_context_observer,
            run_backend_fn=_checked_run_backend, progress_observer=progress)
        from quant_evaluator.runtime.source_route_profiles import (
            validate_source_route_profile_qualification,
        )
        context = result.records[0].context
        qualification = validate_source_route_profile_qualification(
            result.records, expected_context=context)
        verifications = []
        for index, supplied in enumerate((result.records, None), start=4):
            kwargs = dict(common, expected_auto_cuda=qualification.winning_backend == "cuda",
                          context_observer=live_source_profile_context_observer)
            if supplied is not None:
                kwargs["source_qualification"] = supplied
            bundle, receipt = _checked_run_backend("auto", **kwargs)
            verifications.append(_verify_auto(bundle, receipt, qualification,
                context, oracle_bundle, run_index=index, require_cache_hit=index == 5))
        body = {**base, "status": "complete", "run_started": True,
            "request_shape": list(SHAPE),
            "oracle": LINEAR_SHAPE_ORACLE,
            "profile_records": [asdict(row) for row in result.records],
            "run_order": list(result.run_order), "oracle_reports": list(result.oracle_reports),
            "qualification_winner": qualification.winning_backend,
            "auto_verification": verifications[0],
            "default_auto_verification": verifications[1]}
        _write_report(body, args.output)
        report_written = True
        progress_writer.write(base, events, status="complete")
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        if not report_written:
            progress_writer.write(base, events, status="failed", error_type=type(exc).__name__)
        raise
    finally:
        progress_writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
