"""Paired whole-source CPU/CUDA A/B for bounded real-COS factor batches.

Reads factors through the existing verified manifest path in fixed-width tiles.
Only the small aggregate comparison receipt is persisted; factor values stay
inside the bounded tile loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles

METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024**3
MIN_AVAILABLE_RAM_BYTES = 32 * 1024**3


def available_ram_bytes() -> int:
    with open("/proc/meminfo", encoding="ascii") as stream:
        return next(int(line.split()[1]) * 1024 for line in stream
                    if line.startswith("MemAvailable:"))


def preflight(max_object_mib: int, max_total_mib: int) -> dict:
    available = available_ram_bytes()
    cache_path = tiles.cos_cache_root()
    while not cache_path.exists():
        cache_path = cache_path.parent
    disk_free = shutil.disk_usage(cache_path).free
    disk_required = (max_total_mib + 1024) * 1024**2
    return {
        "pass": available >= MIN_AVAILABLE_RAM_BYTES and disk_free >= disk_required,
        "available_ram_bytes": available,
        "minimum_available_ram_bytes": MIN_AVAILABLE_RAM_BYTES,
        "cos_cache_disk_free_bytes": disk_free,
        "required_disk_bytes": disk_required,
    }


class RealCosSource:
    """Immutable source adapter that materializes only one factor tile."""

    def __init__(self, records, source_rows, dates, assets, label, manifest_sha,
                 tile_size, max_object_mib):
        self.records = tuple(records)
        self.source_rows = tuple(tuple(row) for row in source_rows)
        self.factor_ids = tuple(row[0] for row in self.records)
        self.time_axis = AxisRef(
            "time", "datetime64[ns]", len(dates),
            dates.to_numpy(dtype="datetime64[ns]"),
        )
        self.asset_axis = label.asset_axis
        self.dtype = "float64"
        body = json.dumps(
            [manifest_sha, self.source_rows, self.time_axis.values.tolist(),
             self.asset_axis.values.tolist()],
            separators=(",", ":"), default=str,
        ).encode()
        self.snapshot_id = hashlib.sha256(body).hexdigest()
        self.max_tile_size = tile_size
        self.max_object_mib = max_object_mib
        self.next_start = 0
        self.dates, self.assets, self.label = dates, tuple(assets), label
        self.manifest_sha = manifest_sha
        self.reads = []

    def read_tile(self, start: int, end: int) -> FactorTile:
        if not (0 <= start < end <= len(self.factor_ids)
                and end - start <= self.max_tile_size):
            raise InvalidContractError("source tile range is outside its bound")
        if start != self.next_start:
            raise InvalidContractError("source tiles must be read once in factor order")
        expected = self.source_rows[start:end]
        stream = tiles.iter_frames(
            self.records[start:end], self.manifest_sha, self.max_object_mib,
            reuse_manifest=True,
        )
        batch = tiles.make_tile(stream, expected, self.dates, self.assets, self.label)
        self.reads.append((start, end))
        self.next_start = end
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self) -> None:
        return None


def compare(cpu, cuda) -> dict:
    if cpu.factor_ids != cuda.factor_ids:
        raise ValueError("CPU/CUDA whole-request factor axes differ")
    if (cpu.metadata.get("source_request_fingerprint") !=
            cuda.metadata.get("source_request_fingerprint")):
        raise ValueError("CPU/CUDA source request fingerprints differ")
    metrics = {}
    for metric in METRICS:
        cpu_values = np.asarray(cpu.scalar_metrics[metric])
        cuda_values = np.asarray(cuda.scalar_metrics[metric])
        if cpu_values.shape != (len(cpu.factor_ids),) or cuda_values.shape != cpu_values.shape:
            raise ValueError(f"{metric} does not cover the complete factor axis")
        finite = np.isfinite(cpu_values) & np.isfinite(cuda_values)
        same_mask = np.array_equal(np.isfinite(cpu_values), np.isfinite(cuda_values))
        max_abs = (float(np.max(np.abs(cpu_values[finite] - cuda_values[finite])))
                   if finite.any() else 0.0)
        counts_equal = np.array_equal(
            cpu.observation_counts[metric], cuda.observation_counts[metric])
        passed = same_mask and counts_equal and np.allclose(
            cpu_values, cuda_values, rtol=1e-8, atol=1e-10, equal_nan=True)
        metrics[metric] = {
            "pass": bool(passed),
            "factor_count": len(cpu.factor_ids),
            "finite_value_count": int(finite.sum()),
            "max_abs_error": max_abs,
            "observation_counts_equal": bool(counts_equal),
            "cpu_values_sha256": hashlib.sha256(
                np.ascontiguousarray(cpu_values).tobytes()).hexdigest(),
            "cuda_values_sha256": hashlib.sha256(
                np.ascontiguousarray(cuda_values).tobytes()).hexdigest(),
        }
    return {
        "pass": all(item["pass"] for item in metrics.values()),
        "compared_factor_count": len(cpu.factor_ids),
        "compared_metric_count": len(metrics) * len(cpu.factor_ids),
        "metrics": metrics,
    }


def run_backend(backend, records, source_rows, dates, assets, labels,
                manifest_sha, tile_size, max_object_mib, policy):
    source = RealCosSource(records, source_rows, dates, assets, labels,
                           manifest_sha, tile_size, max_object_mib)
    started = time.perf_counter()
    result = evaluate_factor_source_batch(
        source, labels, metrics=METRICS, backend=backend,
        max_tile_size=tile_size, gpu_policy=policy,
    )
    elapsed = time.perf_counter() - started
    expected_reads = [
        (start, min(start + tile_size, len(records)))
        for start in range(0, len(records), tile_size)
    ]
    if source.reads != expected_reads:
        raise ValueError("source API did not read exact ordered tile coverage")
    if result.metadata.get("factor_tiles_processed") != len(expected_reads):
        raise ValueError("source API tile receipt does not cover the whole request")
    if result.metadata.get("backend_used") != (
            "cuda" if backend == "cuda_strict" else "cpu"):
        raise ValueError("source API used a different backend than requested")
    return result, {
        "backend_requested": backend,
        "backend_used": result.metadata["backend_used"],
        "seconds": elapsed,
        "factor_tiles_processed": len(source.reads),
        "tile_ranges": source.reads,
        "peak_process_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, choices=(48, 61), required=True)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--assets", type=int, default=5500)
    parser.add_argument("--max-object-mib", type=int, default=128)
    parser.add_argument("--max-total-mib", type=int, default=4096)
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.tile_size <= 32:
        parser.error("--tile-size must be 1..32")

    initial_preflight = preflight(args.max_object_mib, args.max_total_mib)
    print(json.dumps({"preflight": initial_preflight}), flush=True)
    if not initial_preflight["pass"]:
        raise SystemExit("insufficient RAM or COS cache disk headroom")

    manifest_sha = tiles.MANIFEST_SHA256
    records = tiles.select_records(
        tiles.read_manifest(manifest_sha), args.factors,
        args.max_object_mib, args.max_total_mib,
    )
    if args.axis_index and args.axis_index.exists():
        common_dates, common_assets, source_rows = tiles.read_axis_index(
            args.axis_index, manifest_sha, records)
    else:
        common_dates, common_assets, source_rows = tiles.intersect_axes(
            tiles.iter_frames(records, manifest_sha, args.max_object_mib,
                              reuse_manifest=True),
            len(records),
        )
    dates, assets, labels = tiles.load_labels(
        common_dates, common_assets, args.days, args.assets)
    policy = GPUExecutionPolicy()
    rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
    if rejection:
        raise SystemExit(f"CUDA A/B preflight rejected: {rejection}")

    cpu_preflight = preflight(args.max_object_mib, args.max_total_mib)
    if not cpu_preflight["pass"]:
        raise SystemExit("RAM or COS cache disk headroom fell below preflight before CPU run")
    cpu, cpu_run = run_backend(
        "cpu", records, source_rows, dates, assets, labels, manifest_sha,
        args.tile_size, args.max_object_mib, policy)
    cpu_run["preflight"] = cpu_preflight
    between_preflight = preflight(args.max_object_mib, args.max_total_mib)
    if not between_preflight["pass"]:
        raise SystemExit("RAM or COS cache disk headroom fell below preflight before CUDA run")
    rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
    if rejection:
        raise SystemExit(f"CUDA A/B preflight rejected: {rejection}")
    cuda, cuda_run = run_backend(
        "cuda_strict", records, source_rows, dates, assets, labels,
        manifest_sha, args.tile_size, args.max_object_mib, policy)
    cuda_run["preflight"] = between_preflight
    comparison = compare(cpu, cuda)
    report = {
        "status": "complete" if comparison["pass"] else "parity_failed",
        "kind": "real_cos_whole_source_batch_ab.v1",
        "manifest_sha256": manifest_sha,
        "shape": [len(dates), len(assets), len(records)],
        "factor_dtype": "float64",
        "tile_size": args.tile_size,
        "metric_ids": list(METRICS),
        "preflight_before_cpu": cpu_run["preflight"],
        "preflight_before_cuda": cuda_run["preflight"],
        "runs": [cpu_run, cuda_run],
        "comparison": comparison,
        "source_api": "columnar_factor_source_v1",
        "limitations": [
            "Research-source metrics only; no PIT or production certification.",
            "This A/B does not itself enable automatic backend routing.",
        ],
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if len(payload.encode("utf-8")) > 1024**2:
        raise ValueError("summary report exceeds 1 MiB")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)
    if not comparison["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
