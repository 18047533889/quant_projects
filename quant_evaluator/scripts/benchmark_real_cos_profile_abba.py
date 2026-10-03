"""Explicit, fail-closed F48 real-COS Pearson-chain ABBA profiler.

Dry-run is the default. A live run requires both --run and a user-supplied
axis-index and report path; receipts contain no source paths or credentials.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_source_batch as source_batch
from quant_evaluator.scripts.source_profile_abba import (
    live_source_profile_context_observer, produce_source_route_profile_abba,
    _oracle_report,
)
from quant_evaluator.scripts.source_pearson_oracle import reference_source_pearson_chain

METRICS = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
SHAPE = (2586, 5461, 48)
TILE_CAP = 16
MIN_EFFECTIVE_VRAM_BYTES = source_batch.MIN_EFFECTIVE_VRAM_BYTES


def preflight():
    return source_batch.preflight(128, 4096, 4096)


def _make_cos_source(*args, **kwargs):
    """Keep source construction behind the established COS adapter."""
    return source_batch._make_cos_source(*args, **kwargs)


class _WarmSource:
    """Small in-memory source used only to initialize CPU/CUDA runtimes."""
    def __init__(self):
        self.factor_ids = ("warm-a", "warm-b")
        self.time_axis = AxisRef("time", "datetime64[ns]", 24,
                                 pd.date_range("2000-01-01", periods=24).to_numpy())
        self.asset_axis = AxisRef("asset", "str", 32,
                                  np.asarray([f"A{i:03d}" for i in range(32)]))
        self.dtype, self.snapshot_id, self.max_tile_size = "float64", "warmup-only", 2
        self._values = np.random.default_rng(312).normal(size=(24, 32, 2))
    def read_tile(self, start, end):
        batch = FactorBatch(self.factor_ids[start:end], self.time_axis,
                            self.asset_axis, self._values[:, :, start:end])
        return FactorTile(start, end, batch, self.snapshot_id)
    def close(self):
        pass


def _warm(policy, backend):
    dates = pd.date_range("2000-01-01", periods=24).to_numpy()
    assets = np.asarray([f"A{i:03d}" for i in range(32)])
    values = np.random.default_rng(78).normal(size=(24, 32))
    labels = LabelBundle("warmup", values, 1, decision_time=tuple(dates),
        observation_time=tuple(dates), signal_available_time=tuple(dates),
        execution_time=tuple(dates), label_start_time=tuple(dates + np.timedelta64(1, "D")),
        label_end_time=tuple(dates + np.timedelta64(2, "D")), validity=np.isfinite(values),
        asset_axis=AxisRef("asset", "str", 32, assets))
    source = _WarmSource()
    try:
        source_batch.evaluate_factor_source_batch(source, labels, metrics=METRICS,
            backend=backend, max_tile_size=2, gpu_policy=policy)
    finally:
        source.close()


def _runtime_ready():
    import threadpoolctl  # noqa: F401
    try:
        import numba
        numba.get_num_threads()
    except ImportError:
        pass
    try:
        import cupy as cp
    except ImportError:
        return False
    cp.cuda.Device().use()
    cp.empty((1,), dtype=cp.uint8)
    cp.cuda.runtime.deviceSynchronize()
    return True


def _oracle_mapping(bundle):
    return {name: {"values": bundle.series_metrics[name] if name in bundle.series_metrics
                                  else bundle.scalar_metrics[name],
                   "observation_counts": bundle.observation_counts[name]}
            for name in METRICS}


def _dump(report, path=None):
    body = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if len(body.encode("utf-8")) > 1024**2:
        raise ValueError("profile report exceeds 1 MiB")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    print(body)


def _checked_run_backend(backend, **kwargs):
    """Re-admit resources before every fresh source, outside API timings."""
    gate = preflight()
    if gate.get("pass") is not True:
        raise SystemExit("preflight rejected before " + backend + " source run")
    if backend == "cuda_strict" or kwargs.get("expected_auto_cuda") is True:
        rejection = _auto_batch_cuda_rejection(kwargs["policy"], MIN_EFFECTIVE_VRAM_BYTES)
        if rejection:
            raise SystemExit("CUDA source run rejected: " + rejection)
    return source_batch.run_backend(backend, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly permit real COS reads and CPU/CUDA runs")
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.run and (args.axis_index is None or args.output is None):
        parser.error("--run requires user-provided --axis-index and --output")
    gate = preflight()
    if not gate.get("pass"):
        raise SystemExit("preflight rejected: insufficient RAM or COS cache disk headroom")
    base = {"kind": "real_cos_profile_abba_f48_pearson.v2",
        "status": "preflight_only", "run_started": False,
        "shape": list(SHAPE), "metric_ids": list(METRICS),
        "manifest_sha256": tiles.MANIFEST_SHA256, "requested_tile_cap": TILE_CAP,
        "preflight": gate}
    if not args.run:
        _dump(base)
        return 0

    policy = GPUExecutionPolicy()
    rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
    if rejection:
        raise SystemExit(f"CUDA route rejected by existing VRAM gate: {rejection}")
    manifest_sha = tiles.MANIFEST_SHA256
    records = tiles.select_source_records(tiles.read_manifest(manifest_sha), 48, 128, 4096)
    dates, assets, source_rows = tiles.read_axis_index(args.axis_index, manifest_sha, records)
    dates, assets, labels = tiles.load_labels(dates, assets, 0, 5500)
    if (len(dates), len(assets), len(records)) != SHAPE:
        raise SystemExit("user axis index does not describe the fixed F48 full-history request")
    initial = preflight()
    if not initial.get("pass"):
        raise SystemExit("preflight rejected before source/oracle execution")

    oracle_source = _make_cos_source(records, source_rows, dates, assets,
        labels, manifest_sha, TILE_CAP, 128, prefetch="auto", max_source_memory_mib=4096)
    try:
        oracle_bundle = reference_source_pearson_chain(oracle_source, labels,
            max_tile_size=1, max_result_bytes=64 * 1024**2)
    finally:
        oracle_source.close()
    expected = _oracle_mapping(oracle_bundle)

    if not _runtime_ready():
        raise SystemExit("CuPy unavailable; strict CUDA warmup cannot be performed")
    _warm(policy, "cpu")
    _warm(policy, "cuda_strict")
    common = dict(records=records, source_rows=source_rows, dates=dates,
        assets=assets, labels=labels, manifest_sha=manifest_sha, tile_size=TILE_CAP,
        max_object_mib=128, policy=policy, selected_metrics=METRICS,
        source_adapter="cos", cos_prefetch="auto", max_source_memory_mib=4096,
        cos_prefetch_workers=2, max_prefetch_memory_mib=512)
    gate2 = preflight()
    if not gate2.get("pass"):
        raise SystemExit("preflight rejected before ABBA timing")
    result = produce_source_route_profile_abba(oracle=lambda **_: oracle_bundle,
        run_kwargs=common, context_observer=live_source_profile_context_observer,
        run_backend_fn=_checked_run_backend)
    qualification_records = result.records
    from quant_evaluator.runtime.source_route_profiles import validate_source_route_profile_qualification
    qualification = validate_source_route_profile_qualification(
        qualification_records, expected_context=qualification_records[0].context)

    auto_gate = preflight()
    if not auto_gate.get("pass"):
        raise SystemExit("preflight rejected before independent auto verification")
    if qualification.winning_backend == "cuda":
        rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
        if rejection:
            raise SystemExit(f"CUDA auto winner rejected by existing VRAM gate: {rejection}")
    auto, receipt = _checked_run_backend("auto", **common,
        expected_auto_cuda=qualification.winning_backend == "cuda",
        source_qualification=qualification_records,
        context_observer=live_source_profile_context_observer)
    context = qualification_records[0].context
    if (receipt.get("context_before") != context
            or receipt.get("context_after") != context):
        raise ValueError("auto verification context differs from measured profile")
    auto_oracle_report = _oracle_report(
        auto, receipt, context, backend=receipt.get("backend_used"),
        run_index=4, oracle=lambda **_: oracle_bundle)
    actual = _oracle_mapping(auto)
    expected_width = dict(qualification.actual_tile_sizes)[qualification.winning_backend]
    if (auto.metadata.get("source_qualification_applied") is not True
            or auto.metadata.get("source_qualification_status") != "qualified_current_source"
            or auto.metadata.get("source_qualification_winner") != qualification.winning_backend
            or receipt.get("effective_max_tile_size") != expected_width):
        raise ValueError("auto route did not apply the exact qualified source profile")
    # The supplied-record call admits the profile into the process-local cache.
    # This fresh source must then use default auto without a qualification token.
    default_auto, default_receipt = _checked_run_backend("auto", **common,
        expected_auto_cuda=qualification.winning_backend == "cuda",
        context_observer=live_source_profile_context_observer)
    if (default_receipt.get("context_before") != context
            or default_receipt.get("context_after") != context):
        raise ValueError("default auto verification context differs from measured profile")
    default_oracle_report = _oracle_report(
        default_auto, default_receipt, context,
        backend=default_receipt.get("backend_used"), run_index=5,
        oracle=lambda **_: oracle_bundle)
    if (default_auto.metadata.get("source_qualification_applied") is not True
            or default_auto.metadata.get("source_qualification_status") != "qualified_current_source"
            or default_auto.metadata.get("source_qualification_cache_status") != "cache_hit"
            or default_auto.metadata.get("source_qualification_winner") != qualification.winning_backend
            or default_receipt.get("backend_used") != qualification.winning_backend
            or default_receipt.get("effective_max_tile_size") != expected_width):
        raise ValueError("default auto did not reuse the exact qualified source profile")
    default_actual = _oracle_mapping(default_auto)
    body = {**base, "status": "complete", "run_started": True,
        "request_shape": list(SHAPE), "oracle": "independent scipy/decimal Pearson chain",
        "profile_records": [asdict(row) for row in result.records],
        "run_order": list(result.run_order), "oracle_reports": list(result.oracle_reports),
        "qualification_winner": qualification.winning_backend,
        "default_auto_verification": {
            "backend_used": default_receipt.get("backend_used"),
            "qualification_status": default_auto.metadata.get("source_qualification_status"),
            "qualification_applied": default_auto.metadata.get("source_qualification_applied"),
            "qualification_winner": default_auto.metadata.get("source_qualification_winner"),
            "cache_status": default_auto.metadata.get("source_qualification_cache_status"),
            "output_matches_independent_oracle": True,
            "oracle_report": default_oracle_report,
            "values_sha256": {name: hashlib.sha256(np.ascontiguousarray(item["values"]).tobytes()).hexdigest()
                              for name, item in default_actual.items()}},
        "auto_verification": {"backend_used": receipt.get("backend_used"),
            "qualification_status": auto.metadata.get("source_qualification_status"),
            "qualification_applied": auto.metadata.get("source_qualification_applied"),
            "qualification_winner": auto.metadata.get("source_qualification_winner"),
            "output_matches_independent_oracle": True,
            "oracle_report": auto_oracle_report,
            "values_sha256": {name: hashlib.sha256(np.ascontiguousarray(item["values"]).tobytes()).hexdigest()
                              for name, item in actual.items()}}}
    _dump(body, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
