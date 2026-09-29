"""Paired whole-source CPU/CUDA A/B for bounded real-COS factor batches.

Reads factors through the existing verified manifest path in fixed-width tiles.
Only the small aggregate comparison receipt is persisted; factor values stay
inside the bounded tile loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles

DEFAULT_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
PEARSON_SINGLE = ("pearson_ic",)
PEARSON_CHAIN = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")


def is_pearson_chain(metrics):
    return len(metrics) == len(PEARSON_CHAIN) and frozenset(metrics) == frozenset(PEARSON_CHAIN)
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


def _metric_array(bundle, metric):
    """Resolve one scalar or series artifact without assuming both maps exist."""
    scalar = getattr(bundle, "scalar_metrics", {})
    series = getattr(bundle, "series_metrics", {})
    if metric in scalar and metric in series:
        raise ValueError(f"{metric} appears in both scalar and series metrics")
    if metric in scalar:
        return "scalar", np.asarray(scalar[metric])
    if metric in series:
        return "series", np.asarray(series[metric])
    return None, np.asarray([])


def compare(cpu, cuda, selected_metrics, *, expected_days=None) -> dict:
    if cpu.factor_ids != cuda.factor_ids:
        raise ValueError("CPU/CUDA whole-request factor axes differ")
    cpu_fingerprint = cpu.metadata.get("source_request_fingerprint")
    cuda_fingerprint = cuda.metadata.get("source_request_fingerprint")
    if (not isinstance(cpu_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", cpu_fingerprint) is None
            or cpu_fingerprint != cuda_fingerprint):
        raise ValueError("CPU/CUDA source request fingerprints differ")
    metric_results = {}
    for metric in selected_metrics:
        cpu_group, cpu_values = _metric_array(cpu, metric)
        cuda_group, cuda_values = _metric_array(cuda, metric)
        factors = len(cpu.factor_ids)
        if cpu_group == "scalar":
            cpu_shape_valid = cpu_values.shape == (factors,)
        elif cpu_group == "series":
            cpu_shape_valid = (cpu_values.ndim == 2 and cpu_values.shape[1] == factors
                               and (expected_days is None or cpu_values.shape[0] == expected_days))
        else:
            cpu_shape_valid = False
        if cuda_group == "scalar":
            cuda_shape_valid = cuda_values.shape == (factors,)
        elif cuda_group == "series":
            cuda_shape_valid = (cuda_values.ndim == 2 and cuda_values.shape[1] == factors
                                and (expected_days is None or cuda_values.shape[0] == expected_days))
        else:
            cuda_shape_valid = False
        shapes_equal = cpu_group == cuda_group and cpu_values.shape == cuda_values.shape
        valid_shape = cpu_shape_valid and cuda_shape_valid and shapes_equal
        masks_equal = (valid_shape and
                       np.array_equal(np.isfinite(cpu_values), np.isfinite(cuda_values)))
        finite = (np.isfinite(cpu_values) & np.isfinite(cuda_values)
                  if shapes_equal else np.asarray([], dtype=bool))
        max_abs = (float(np.max(np.abs(cpu_values[finite] - cuda_values[finite])))
                   if finite.any() else 0.0)
        cpu_counts = np.asarray(cpu.observation_counts.get(metric, []))
        cuda_counts = np.asarray(cuda.observation_counts.get(metric, []))
        counts_shape_valid = cpu_counts.shape == (factors,) and cuda_counts.shape == (factors,)
        counts_equal = counts_shape_valid and np.array_equal(cpu_counts, cuda_counts)
        values_close = (valid_shape and np.allclose(
            cpu_values, cuda_values, rtol=1e-8, atol=1e-10, equal_nan=True))
        passed = masks_equal and counts_equal and values_close
        metric_results[metric] = {
            "pass": bool(passed),
            "artifact_kind": cpu_group if cpu_group == cuda_group else None,
            "cpu_shape": list(cpu_values.shape),
            "cuda_shape": list(cuda_values.shape),
            "shape_valid": bool(valid_shape),
            "factor_count": factors,
            "compared_value_count": int(cpu_values.size) if valid_shape else 0,
            "finite_value_count": int(finite.sum()),
            "finite_mask_equal": bool(masks_equal),
            "max_abs_error": max_abs,
            "observation_counts_equal": bool(counts_equal),
            "observation_counts_shape_valid": bool(counts_shape_valid),
            "cpu_values_sha256": hashlib.sha256(
                np.ascontiguousarray(cpu_values).tobytes()).hexdigest(),
            "cuda_values_sha256": hashlib.sha256(
                np.ascontiguousarray(cuda_values).tobytes()).hexdigest(),
        }
    return {
        "pass": all(item["pass"] for item in metric_results.values()),
        "compared_factor_count": len(cpu.factor_ids),
        "compared_metric_count": sum(item["compared_value_count"]
                                     for item in metric_results.values()),
        "metrics": metric_results,
    }


def run_backend(backend, records, source_rows, dates, assets, labels,
                manifest_sha, tile_size, max_object_mib, policy, selected_metrics,
                *, expected_auto_cuda=False):
    source = RealCosSource(records, source_rows, dates, assets, labels,
                           manifest_sha, tile_size, max_object_mib)
    started = time.perf_counter()
    result = evaluate_factor_source_batch(
        source, labels, metrics=selected_metrics, backend=backend,
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
            "cuda" if backend == "cuda_strict" or (backend == "auto" and expected_auto_cuda)
            else "cpu"):
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


def emit_report(report, output):
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if len(payload.encode("utf-8")) > 1024**2:
        raise ValueError("summary report exceeds 1 MiB")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


def _bundle_from_gpu_worker(report, selected_metrics):
    return SimpleNamespace(
        factor_ids=tuple(report["factor_ids"]),
        metadata={"source_request_fingerprint": report["source_request_fingerprint"]},
        scalar_metrics={name: np.asarray(report["scalar_metrics"][name], dtype=np.float64)
                        for name in selected_metrics},
        observation_counts={name: np.asarray(report["observation_counts"][name], dtype=np.int64)
                            for name in selected_metrics},
    )


def run_gpu_tile_width_ab(args, selected_metrics):
    """Interleave widths in isolated workers so source memory is reclaimed."""
    width_a, width_b = args.gpu_tile_widths
    order = (width_a, width_b, width_b, width_a)
    report = {"status": "partial", "kind": "real_cos_gpu_tile_width_ab.v2",
              "run_order": list(order), "metric_ids": list(selected_metrics),
              "runs": [], "comparisons_to_first_run": [],
              "limitations": ["One real COS manifest and host; research use only.",
                              "Separate CPU/CUDA reports establish CPU parity.",
                              "Partial runs must not select an auto backend."]}
    with tempfile.TemporaryDirectory(prefix="qe-gpu-width-") as directory:
        first = None
        for index, width in enumerate(order):
            output = Path(directory) / f"run-{index}.json"
            command = [sys.executable, "-m",
                       "quant_evaluator.scripts.benchmark_real_cos_source_batch",
                       "--factors", str(args.factors), "--tile-size", str(width),
                       "--metrics", ",".join(selected_metrics),
                       "--days", str(args.days), "--assets", str(args.assets),
                       "--max-object-mib", str(args.max_object_mib),
                       "--max-total-mib", str(args.max_total_mib),
                       "--gpu-worker", "--output", str(output)]
            if args.axis_index:
                command.extend(("--axis-index", str(args.axis_index)))
            try:
                worker = subprocess.run(command, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, text=True,
                                        timeout=1200, check=False)
            except subprocess.TimeoutExpired:
                report.update(status="interrupted", failure=f"worker_{index}_timeout")
                emit_report(report, args.output)
                raise SystemExit(1)
            if worker.returncode != 0 or not output.exists():
                report.update(status="interrupted",
                              failure=f"worker_{index}_exit_{worker.returncode}")
                emit_report(report, args.output)
                raise SystemExit(1)
            if output.stat().st_size > 1024**2:
                raise ValueError("GPU worker receipt exceeds 1 MiB")
            item = json.loads(output.read_text(encoding="utf-8"))
            if item["kind"] != "real_cos_source_gpu_worker.v1" or item["tile_size"] != width:
                raise ValueError("GPU worker receipt identity mismatch")
            if tuple(item.get("metric_ids", ())) != tuple(selected_metrics):
                raise ValueError("GPU worker metric identity mismatch")
            bundle = _bundle_from_gpu_worker(item, selected_metrics)
            public_item = {key: value for key, value in item.items()
                           if key not in {"factor_ids", "scalar_metrics", "observation_counts"}}
            if first is None:
                first = bundle
                report["manifest_sha256"] = item["manifest_sha256"]
                report["shape"] = item["shape"]
            else:
                if (item["manifest_sha256"] != report["manifest_sha256"]
                        or item["shape"] != report["shape"]):
                    raise ValueError("GPU workers used different source identities")
                parity = compare(first, bundle, selected_metrics)
                report["comparisons_to_first_run"].append(parity)
                if not parity["pass"]:
                    report["runs"].append(public_item)
                    report.update(status="parity_failed", failure=f"worker_{index}_parity")
                    emit_report(report, args.output)
                    raise SystemExit(1)
            report["runs"].append(public_item)
            emit_report(report, args.output)
    report["median_seconds_by_width"] = {
        str(width): float(np.median([item["run"]["seconds"] for item in report["runs"]
                                     if item["tile_size"] == width]))
        for width in (width_a, width_b)}
    report["status"] = "complete"
    emit_report(report, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, choices=(48, 61), required=True)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="default three metrics, pearson_ic alone, or exact Pearson chain")
    parser.add_argument("--gpu-tile-widths", nargs=2, type=int,
                        metavar=("WIDTH_A", "WIDTH_B"))
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-order", choices=("cpu-first", "cuda-first"),
                        default="cpu-first", help="order for whole-source CPU/CUDA A/B")
    parser.add_argument("--verify-auto", action="store_true",
                        help="also verify the exact certified F61 Pearson chain auto route")
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--assets", type=int, default=5500)
    parser.add_argument("--max-object-mib", type=int, default=128)
    parser.add_argument("--max-total-mib", type=int, default=4096)
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = tuple(args.metrics.split(","))
    if selected not in (DEFAULT_METRICS, PEARSON_SINGLE) and not is_pearson_chain(selected):
        parser.error("--metrics must be the exact default three, pearson_ic alone, or Pearson chain")
    if not 1 <= args.tile_size <= 32:
        parser.error("--tile-size must be 1..32")
    if args.gpu_tile_widths and (len(set(args.gpu_tile_widths)) != 2
                                 or any(not 1 <= width <= 32 for width in args.gpu_tile_widths)):
        parser.error("--gpu-tile-widths requires two distinct widths in 1..32")
    if args.gpu_worker and is_pearson_chain(selected):
        parser.error("Pearson chain is supported only by whole-source CPU/CUDA A/B")
    if args.verify_auto and (args.factors != 61 or not is_pearson_chain(selected)
                             or args.tile_size < 16 or args.gpu_worker
                             or args.gpu_tile_widths):
        parser.error("--verify-auto requires F61 Pearson chain, tile >=16, whole-source mode")
    if args.gpu_tile_widths:
        if is_pearson_chain(selected):
            parser.error("Pearson chain is supported only by whole-source CPU/CUDA A/B")
        if args.gpu_worker or args.output is None:
            parser.error("--gpu-tile-widths requires --output and cannot use --gpu-worker")
        run_gpu_tile_width_ab(args, selected)
        return
    if args.gpu_worker and args.output is None:
        parser.error("--gpu-worker requires --output")

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

    if args.gpu_worker:
        gate = preflight(args.max_object_mib, args.max_total_mib)
        if not gate["pass"]:
            raise SystemExit("RAM or COS cache disk headroom fell below preflight")
        result, run = run_backend(
            "cuda_strict", records, source_rows, dates, assets, labels,
            manifest_sha, args.tile_size, args.max_object_mib, policy, selected)
        run["tile_size"] = args.tile_size
        run["preflight"] = gate
        scalar_metrics, observation_counts = {}, {}
        for metric in selected:
            values = np.asarray(result.scalar_metrics[metric], dtype=np.float64)
            counts = np.asarray(result.observation_counts[metric], dtype=np.int64)
            if values.shape != (len(records),) or counts.shape != values.shape:
                raise ValueError("GPU worker did not cover the complete factor axis")
            scalar_metrics[metric] = [float(value) if np.isfinite(value) else None
                                      for value in values]
            observation_counts[metric] = [int(value) for value in counts]
        report = {
            "status": "complete", "kind": "real_cos_source_gpu_worker.v1",
            "manifest_sha256": manifest_sha,
            "shape": [len(dates), len(assets), len(records)],
            "factor_ids": list(result.factor_ids),
            "source_request_fingerprint": result.metadata["source_request_fingerprint"],
            "tile_size": args.tile_size, "metric_ids": list(selected),
            "run": run, "scalar_metrics": scalar_metrics,
            "observation_counts": observation_counts,
        }
        emit_report(report, args.output)
        return

    runs = {}
    order = ("cpu", "cuda_strict") if args.run_order == "cpu-first" else ("cuda_strict", "cpu")
    for backend in order:
        gate = preflight(args.max_object_mib, args.max_total_mib)
        if not gate["pass"]:
            raise SystemExit(f"RAM or COS cache disk headroom fell below preflight before {backend} run")
        if backend == "cuda_strict":
            rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
            if rejection:
                raise SystemExit(f"CUDA A/B preflight rejected: {rejection}")
        bundle, receipt = run_backend(
            backend, records, source_rows, dates, assets, labels, manifest_sha,
            args.tile_size, args.max_object_mib, policy, selected)
        receipt["preflight"] = gate
        runs[backend] = (bundle, receipt)
    cpu, cpu_run = runs["cpu"]
    cuda, cuda_run = runs["cuda_strict"]
    comparison = compare(cpu, cuda, selected, expected_days=len(dates))
    auto_run = None
    auto_comparison = None
    if args.verify_auto and comparison["pass"]:
        gate = preflight(args.max_object_mib, args.max_total_mib)
        if not gate["pass"]:
            raise SystemExit("RAM or COS cache disk headroom fell below preflight before auto run")
        auto, auto_run = run_backend(
            "auto", records, source_rows, dates, assets, labels, manifest_sha,
            args.tile_size, args.max_object_mib, policy, selected,
            expected_auto_cuda=True)
        auto_run["preflight"] = gate
        auto_run["auto_backend_reason"] = auto.metadata.get("auto_backend_reason")
        auto_run["effective_max_tile_size"] = auto.metadata.get("effective_max_tile_size")
        if (auto_run["auto_backend_reason"] != "bounded_f61_pearson_chain_gpu_tile16"
                or auto_run["effective_max_tile_size"] != 16):
            raise ValueError("auto did not use the certified F61 Pearson chain route")
        auto_comparison = compare(cpu, auto, selected, expected_days=len(dates))
    report = {
        "status": "complete" if comparison["pass"] and
                  (auto_comparison is None or auto_comparison["pass"]) else "parity_failed",
        "kind": "real_cos_whole_source_batch_ab.v1",
        "manifest_sha256": manifest_sha,
        "shape": [len(dates), len(assets), len(records)],
        "factor_dtype": "float64",
        "tile_size": args.tile_size,
        "run_order": list(order),
        "metric_ids": list(selected),
        "preflight_before_cpu": cpu_run["preflight"],
        "preflight_before_cuda": cuda_run["preflight"],
        "runs": [cpu_run, cuda_run],
        "comparison": comparison,
        "auto_run": auto_run,
        "auto_comparison": auto_comparison,
        "source_api": "columnar_factor_source_v1",
        "limitations": [
            "Research-source metrics only; no PIT or production certification.",
            ("Auto exercised only for the exact certified F61 Pearson chain profile."
             if args.verify_auto else
             "This A/B alone does not establish automatic backend routing."),
        ],
    }
    emit_report(report, args.output)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
