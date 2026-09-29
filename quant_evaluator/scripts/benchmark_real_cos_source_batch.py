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
import pandas as pd

from quant_evaluator.api.factor_source import _F61_ALL_SOURCE_SHAPES, evaluate_factor_source_batch
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection
from quant_evaluator.adapters.cos_factor_tile_source import (
    CosFactorTileSource, DataAccessReadContext,
)
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles

DEFAULT_METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")
PEARSON_SINGLE = ("pearson_ic",)
PEARSON_CHAIN = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
ALL_SOURCE_METRICS = (
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
)


def _int64_sha256(values) -> str:
    """Hash observation counts in the canonical contiguous int64 encoding."""
    array = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


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


def _make_cos_source(records, source_rows, dates, assets, labels,
                     manifest_sha, tile_size, max_object_mib, *, prefetch):
    """Use the production adapter with the research descriptors used by this harness."""
    from data_access.core.engine import DuckDBEngine
    from data_access.registry.loader import DatasetRegistry
    from data_access.store import DataAccessStore

    manifest_uri = f"{tiles.BASE}/metadata/{manifest_sha}/landing_manifest.json"
    manifest_ds = tiles._ds("source_manifest", manifest_uri.rsplit("/", 1)[0],
                            "landing_manifest.json", "json")

    def manifest_context_factory():
        engine = DuckDBEngine(threads=1)
        try:
            store = DataAccessStore(DatasetRegistry({manifest_ds.name: manifest_ds}), engine)
        except BaseException:
            engine.close()
            raise
        return DataAccessReadContext(store, manifest_ds.name, close=engine.close)

    def factor_context_factory(record):
        factor_ds = tiles._ds(
            "factor_panel", record.uri.rsplit("/", 1)[0],
            record.factor_id + ".parquet", "parquet")
        engine = DuckDBEngine(threads=2)
        try:
            store = DataAccessStore(
                DatasetRegistry({manifest_ds.name: manifest_ds,
                                 factor_ds.name: factor_ds}), engine)
        except BaseException:
            engine.close()
            raise
        return DataAccessReadContext(
            store, manifest_ds.name, factor_ds.name, close=engine.close)

    def make_tile(start, end, bound_records, payloads):
        expected = source_rows[start:end]
        if len(payloads) != len(expected) or len(bound_records) != len(expected):
            raise ValueError("COS source did not return the selected tile width")
        stream = []
        for record, item, row in zip(bound_records, payloads, expected):
            identity = item.identity
            frame = item.payload
            if "timestamp" not in frame.columns:
                raise ValueError("factor timestamp missing after bound COS read")
            frame = frame.set_index("timestamp")
            frame.index = pd.to_datetime(frame.index).normalize()
            if frame.index.has_duplicates or frame.columns.has_duplicates:
                raise ValueError("duplicate factor axis")
            frame = frame.loc[:, [column for column in frame
                                  if column.endswith((".SZ", ".SH"))]].sort_index()
            if ((record.factor_id, record.uri, record.sha256, record.size_bytes)
                    != (row[0], row[1], row[2], row[3])):
                raise ValueError("COS bound identity differs from the axis preflight")
            if (identity.get("source_etag") != row[4]
                    or tiles.axis_hash(frame) != row[5]):
                raise ValueError("COS source ETag or panel axes changed after preflight")
            stream.append((frame, row))
        return tiles.make_tile(stream, expected, dates, assets, labels)

    return CosFactorTileSource.from_data_access(
        factor_ids=tuple(row[0] for row in records), time_axis=AxisRef(
            "time", "datetime64[ns]", len(dates),
            dates.to_numpy(dtype="datetime64[ns]")),
        asset_axis=labels.asset_axis, dtype="float64", make_tile=make_tile,
        manifest_context_factory=manifest_context_factory,
        factor_context_factory=factor_context_factory,
        max_object_mib=max_object_mib, max_tile_size=tile_size,
        prefetch=prefetch)


class RealCosSource:
    """Immutable source adapter that materializes only one factor tile."""

    def __init__(self, records, source_rows, dates, assets, label, manifest_sha,
                 tile_size, max_object_mib, prefetch_objects=False):
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
        self.prefetch_objects = bool(prefetch_objects)
        self.next_start = 0
        self.dates, self.assets, self.label = dates, tuple(assets), label
        self.manifest_sha = manifest_sha
        self.reads = []
        self.tile_read_timings = []

    def read_tile(self, start: int, end: int) -> FactorTile:
        if not (0 <= start < end <= len(self.factor_ids)
                and end - start <= self.max_tile_size):
            raise InvalidContractError("source tile range is outside its bound")
        if start != self.next_start:
            raise InvalidContractError("source tiles must be read once in factor order")
        expected = self.source_rows[start:end]
        load_phases = {"bound_factor_read_s": 0.0, "arrow_to_pandas_axis_s": 0.0,
                       "reindex_write_s": 0.0}
        read_started = time.perf_counter()
        if self.prefetch_objects:
            stream = tiles.iter_frames_prefetched(
                self.records[start:end], self.manifest_sha, self.max_object_mib,
                load_phases=load_phases,
            )
        else:
            stream = tiles.iter_frames(
                self.records[start:end], self.manifest_sha, self.max_object_mib,
                reuse_manifest=True, load_phases=load_phases,
            )
        batch = tiles.make_tile(
            stream, expected, self.dates, self.assets, self.label,
            load_phases=load_phases,
        )
        self.tile_read_timings.append({
            "tile_range": [start, end],
            "wall_seconds": time.perf_counter() - read_started,
            "load_phases_seconds": load_phases,
        })
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
            "cpu_observation_counts_sha256": _int64_sha256(cpu_counts),
            "cuda_observation_counts_sha256": _int64_sha256(cuda_counts),
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
                *, expected_auto_cuda=False, prefetch_objects=False,
                source_adapter="legacy", cos_prefetch="auto"):
    if source_adapter == "legacy":
        if cos_prefetch != "auto":
            raise ValueError("cos_prefetch applies only to source_adapter='cos'")
        source = RealCosSource(records, source_rows, dates, assets, labels,
                               manifest_sha, tile_size, max_object_mib,
                               prefetch_objects=prefetch_objects)
        prefetch_mode = "on" if prefetch_objects else "off"
    elif source_adapter == "cos":
        if prefetch_objects:
            raise ValueError("prefetch_objects is legacy-only; pass cos_prefetch")
        source = _make_cos_source(
            records, source_rows, dates, assets, labels, manifest_sha,
            tile_size, max_object_mib,
            prefetch=cos_prefetch)
        prefetch_mode = source.prefetch_mode
    else:
        raise ValueError("source_adapter must be 'legacy' or 'cos'")
    started = time.perf_counter()
    try:
        result = evaluate_factor_source_batch(
            source, labels, metrics=selected_metrics, backend=backend,
            max_tile_size=tile_size, gpu_policy=policy,
        )
        elapsed = time.perf_counter() - started
        effective_tile_size = (result.metadata.get("effective_max_tile_size", tile_size)
                               if backend == "auto" else tile_size)
        if not isinstance(effective_tile_size, int) or not 1 <= effective_tile_size <= tile_size:
            raise ValueError("source API reported an invalid effective tile width")
        expected_reads = [
            (start, min(start + effective_tile_size, len(records)))
            for start in range(0, len(records), effective_tile_size)
        ]
        if tuple(source.factor_ids) != tuple(row[0] for row in records):
            raise ValueError("source factor identity order differs from the selected request")
        if source.reads != expected_reads:
            raise ValueError("source API did not read exact ordered tile coverage")
        if result.metadata.get("factor_tiles_processed") != len(expected_reads):
            raise ValueError("source API tile receipt does not cover the whole request")
        if result.metadata.get("backend_used") != (
                "cuda" if backend == "cuda_strict" or (backend == "auto" and expected_auto_cuda)
                else "cpu"):
            raise ValueError("source API used a different backend than requested")
    finally:
        source.close()
    identity_payload = json.dumps(
        [list(row) for row in source_rows], separators=(",", ":"), default=str).encode()
    request_sources_payload = json.dumps(
        [list(row) for row in records], separators=(",", ":"), default=str).encode()
    return result, {
        "source_adapter": source_adapter,
        "backend_requested": backend,
        "backend_used": result.metadata["backend_used"],
        "seconds": elapsed,
        "total_wall_seconds": elapsed,
        "factor_tiles_processed": len(source.reads),
        "tile_ranges": source.reads,
        "factor_ids_sha256": hashlib.sha256(
            json.dumps([row[0] for row in records], separators=(",", ":")).encode()).hexdigest(),
        "source_request_identity_sha256": hashlib.sha256(request_sources_payload + identity_payload).hexdigest(),
        "source_snapshot_id": source.snapshot_id,
        "source_manifest_sha256": getattr(source, "manifest_sha256", manifest_sha),
        "prefetch_objects": bool(getattr(source, "prefetch_objects", prefetch_mode != "off")),
        "prefetch_mode": prefetch_mode,
        "cos_prefetch": cos_prefetch if source_adapter == "cos" else None,
        "prefetch_window": getattr(source, "prefetch_window", 2 if prefetch_mode == "on" else 1),
        "estimated_peak_source_bytes": getattr(source, "estimated_peak_source_bytes", None),
        "max_source_memory_bytes": getattr(source, "max_source_memory_bytes", None),
        "tile_read_timings": source.tile_read_timings,
        "source_read_wall_seconds": sum(item["wall_seconds"]
                                        for item in source.tile_read_timings),
        "observation_counts_sha256": {
            name: _int64_sha256(getattr(result, "observation_counts", {}).get(name, ()))
            for name in selected_metrics
        },
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


def _source_prefetch_report_fields(run_receipt):
    """Project effective source-read settings, never CLI request flags."""
    return {
        "prefetch_objects": bool(run_receipt["prefetch_objects"]),
        "prefetch_mode": run_receipt["prefetch_mode"],
        "prefetch_window": int(run_receipt["prefetch_window"]),
    }


def run_gpu_tile_width_ab(args, selected_metrics):
    """Interleave widths in isolated workers so source memory is reclaimed."""
    width_a, width_b = args.gpu_tile_widths
    order = (width_a, width_b, width_b, width_a)
    report = {"status": "partial", "kind": "real_cos_gpu_tile_width_ab.v2",
              "run_order": list(order), "metric_ids": list(selected_metrics),
              "source_adapter": getattr(args, "source_adapter", "legacy"),
              "cos_prefetch": getattr(args, "cos_prefetch", "auto"),
              "prefetch_objects": None, "prefetch_mode": None, "prefetch_window": None,
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
                       "--source-adapter", getattr(args, "source_adapter", "legacy"),
                       "--cos-prefetch", getattr(args, "cos_prefetch", "auto"),
                       "--metrics", ",".join(selected_metrics),
                       "--days", str(args.days), "--assets", str(args.assets),
                       "--max-object-mib", str(args.max_object_mib),
                       "--max-total-mib", str(args.max_total_mib),
                       "--gpu-worker", "--output", str(output)]
            if getattr(args, "prefetch_objects", False):
                command.append("--prefetch-objects")
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
            if not report["runs"]:
                report.update(_source_prefetch_report_fields(item["run"]))
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


def certified_cuda_hashes(paths, selected_metrics, manifest_sha):
    """Bind auto-only verification to two complete, opposite-order real A/B receipts."""
    if len(paths) != 2:
        raise ValueError("auto-only verification requires two reference reports")
    reports = []
    for path in paths:
        if path.stat().st_size > 1024**2:
            raise ValueError("reference report exceeds 1 MiB")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete"
                or report.get("kind") != "real_cos_whole_source_batch_ab.v1"
                or not report.get("comparison", {}).get("pass")
                or report.get("manifest_sha256") != manifest_sha
                or tuple(report.get("metric_ids", ())) != selected_metrics
                or report.get("factor_dtype") != "float64"
                or report.get("tile_size") != 16
                or {run.get("backend_used") for run in report.get("runs", ())}
                != {"cpu", "cuda"}):
            raise ValueError("reference report is not a matching completed CPU/CUDA A/B")
        reports.append(report)
    first, second = reports
    profile_shape = tuple(first.get("shape", ()))
    if ({tuple(first.get("run_order", ())), tuple(second.get("run_order", ()))}
            != {("cpu", "cuda_strict"), ("cuda_strict", "cpu")}
            or first.get("shape") != second.get("shape")
            or profile_shape not in _F61_ALL_SOURCE_SHAPES
            or first["comparison"].get("compared_metric_count")
            != second["comparison"].get("compared_metric_count")):
        raise ValueError("reference reports lack opposite-order matched coverage")
    for metric in selected_metrics:
        left = first["comparison"]["metrics"][metric]
        right = second["comparison"]["metrics"][metric]
        if (not left.get("pass") or not right.get("pass")
                or left.get("cuda_values_sha256") != right.get("cuda_values_sha256")
                or left.get("cpu_values_sha256") != right.get("cpu_values_sha256")
                or left.get("cuda_shape") != right.get("cuda_shape")
                or left.get("finite_value_count") != right.get("finite_value_count")):
            raise ValueError(f"reference reports disagree for {metric}")
    return first


def verify_auto_against_reference(auto, selected_metrics, reference):
    checks = {}
    factors = len(auto.factor_ids)
    for metric in selected_metrics:
        expected = reference["comparison"]["metrics"][metric]
        kind, values = _metric_array(auto, metric)
        counts = np.asarray(auto.observation_counts.get(metric, ()))
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        counts_digest = _int64_sha256(counts)
        expected_counts_digest = expected.get("cuda_observation_counts_sha256")
        passed = (kind == expected["artifact_kind"]
                  and list(values.shape) == expected["cuda_shape"]
                  and int(np.isfinite(values).sum()) == expected["finite_value_count"]
                  and counts.shape == (factors,)
                  and digest == expected["cuda_values_sha256"]
                  and (expected_counts_digest is None
                       or counts_digest == expected_counts_digest))
        checks[metric] = {"pass": bool(passed), "artifact_kind": kind,
                          "shape": list(values.shape), "finite_value_count": int(np.isfinite(values).sum()),
                          "observation_counts_shape_valid": counts.shape == (factors,),
                          "observation_counts_sha256": counts_digest,
                          "values_sha256": digest}
    return {"pass": all(item["pass"] for item in checks.values()),
            "compared_value_count": sum(int(np.prod(item["shape"])) for item in checks.values()),
            "metrics": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, choices=(48, 61), required=True)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="default three, pearson_ic, Pearson chain, or all_source")
    parser.add_argument("--gpu-tile-widths", nargs=2, type=int,
                        metavar=("WIDTH_A", "WIDTH_B"))
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-order", choices=("cpu-first", "cuda-first"),
                        default="cpu-first", help="order for whole-source CPU/CUDA A/B")
    parser.add_argument("--prefetch-objects", action="store_true",
                        help="legacy source only: prefetch at most two COS factor objects")
    parser.add_argument("--source-adapter", choices=("legacy", "cos"), default="legacy",
                        help="source implementation; cos uses DataAccess bound reads and bounded auto prefetch")
    parser.add_argument("--cos-prefetch", choices=("off", "on", "auto"), default="auto",
                        help="COS adapter object prefetch policy (default: bounded auto)")
    parser.add_argument("--verify-auto", action="store_true",
                        help="also verify the exact certified F61 Pearson or all-source auto route")
    parser.add_argument("--auto-references", nargs=2, type=Path,
                        metavar=("CUDA_CPU_REPORT", "CPU_CUDA_REPORT"),
                        help="run only auto against two opposite-order, matching F61 all-source A/B reports")
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--assets", type=int, default=5500)
    parser.add_argument("--max-object-mib", type=int, default=128)
    parser.add_argument("--max-total-mib", type=int, default=4096)
    parser.add_argument("--axis-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.source_adapter == "cos" and args.prefetch_objects:
        parser.error("--prefetch-objects is legacy-only; use --cos-prefetch for the COS adapter")
    if args.source_adapter == "legacy" and args.cos_prefetch != "auto":
        parser.error("--cos-prefetch requires --source-adapter cos")
    selected = ALL_SOURCE_METRICS if args.metrics == "all_source" else tuple(args.metrics.split(","))
    if (selected not in (DEFAULT_METRICS, PEARSON_SINGLE, ALL_SOURCE_METRICS)
            and not is_pearson_chain(selected)):
        parser.error("--metrics must be default three, pearson_ic, Pearson chain, or all_source")
    if not 1 <= args.tile_size <= 32:
        parser.error("--tile-size must be 1..32")
    if args.gpu_tile_widths and (len(set(args.gpu_tile_widths)) != 2
                                 or any(not 1 <= width <= 32 for width in args.gpu_tile_widths)):
        parser.error("--gpu-tile-widths requires two distinct widths in 1..32")
    if args.gpu_worker and (is_pearson_chain(selected) or selected == ALL_SOURCE_METRICS):
        parser.error("series metrics are supported only by whole-source CPU/CUDA A/B")
    if args.verify_auto and (args.factors != 61
                             or not (is_pearson_chain(selected) or selected == ALL_SOURCE_METRICS)
                             or args.tile_size < 16 or args.gpu_worker
                             or args.gpu_tile_widths):
        parser.error("--verify-auto requires F61 Pearson or all-source, tile >=16, whole-source mode")
    if args.auto_references and (args.factors != 61 or selected != ALL_SOURCE_METRICS
                                  or args.tile_size != 16 or args.gpu_worker
                                  or args.gpu_tile_widths or args.verify_auto
                                  or args.output is None):
        parser.error("--auto-references requires F61 all_source, tile 16, --output, and no other mode")
    if args.gpu_tile_widths:
        if is_pearson_chain(selected) or selected == ALL_SOURCE_METRICS:
            parser.error("series metrics are supported only by whole-source CPU/CUDA A/B")
        if args.gpu_worker or args.output is None:
            parser.error("--gpu-tile-widths requires --output and cannot use --gpu-worker")
        run_gpu_tile_width_ab(args, selected)
        return
    if args.gpu_worker and args.output is None:
        parser.error("--gpu-worker requires --output")

    reference = None
    if args.auto_references:
        reference = certified_cuda_hashes(args.auto_references, selected,
                                          tiles.MANIFEST_SHA256)

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

    if reference is not None:
        if reference["shape"] != [len(dates), len(assets), len(records)]:
            raise ValueError("auto request shape differs from certified A/B coverage")
        gate = preflight(args.max_object_mib, args.max_total_mib)
        if not gate["pass"]:
            raise SystemExit("RAM or COS cache disk headroom fell below preflight before auto run")
        auto, receipt = run_backend(
            "auto", records, source_rows, dates, assets, labels, manifest_sha,
            args.tile_size, args.max_object_mib, policy, selected,
            expected_auto_cuda=True, prefetch_objects=args.prefetch_objects,
            source_adapter=args.source_adapter, cos_prefetch=args.cos_prefetch)
        receipt["preflight"] = gate
        receipt["auto_backend_reason"] = auto.metadata.get("auto_backend_reason")
        receipt["effective_max_tile_size"] = auto.metadata.get("effective_max_tile_size")
        reference_comparison = verify_auto_against_reference(auto, selected, reference)
        cuda_gate = preflight(args.max_object_mib, args.max_total_mib)
        if not cuda_gate["pass"]:
            raise SystemExit("RAM or COS cache disk headroom fell below preflight before explicit CUDA run")
        rejection = _auto_batch_cuda_rejection(policy, MIN_EFFECTIVE_VRAM_BYTES)
        if rejection:
            raise SystemExit(f"CUDA A/B preflight rejected before explicit CUDA run: {rejection}")
        cuda, cuda_receipt = run_backend(
            "cuda_strict", records, source_rows, dates, assets, labels, manifest_sha,
            args.tile_size, args.max_object_mib, policy, selected,
            prefetch_objects=args.prefetch_objects,
            source_adapter=args.source_adapter, cos_prefetch=args.cos_prefetch)
        cuda_receipt["preflight"] = cuda_gate
        direct_comparison = compare(auto, cuda, selected, expected_days=len(dates))
        route_pass = (receipt["auto_backend_reason"] ==
                      "bounded_f61_all_source_15_gpu_tile16"
                      and receipt["effective_max_tile_size"] == 16)
        complete = route_pass and reference_comparison["pass"] and direct_comparison["pass"]
        report = {
            "status": "complete" if complete else "verification_failed",
            "kind": "real_cos_whole_source_auto_reference.v1",
            "manifest_sha256": manifest_sha,
            "shape": [len(dates), len(assets), len(records)],
            "factor_dtype": "float64", "tile_size": args.tile_size,
            "source_adapter": args.source_adapter,
            "cos_prefetch": args.cos_prefetch if args.source_adapter == "cos" else None,
            "metric_ids": list(selected),
            **_source_prefetch_report_fields(receipt),
            "run": receipt,
            "explicit_cuda_run": cuda_receipt,
            "route_pass": route_pass,
            "reference_comparison": reference_comparison,
            "direct_comparison": direct_comparison,
            "reference_reports": [str(path) for path in args.auto_references],
            "limitations": ["Research-source metrics only; no PIT or production certification."],
        }
        emit_report(report, args.output)
        if report["status"] != "complete":
            raise SystemExit(1)
        return

    if args.gpu_worker:
        gate = preflight(args.max_object_mib, args.max_total_mib)
        if not gate["pass"]:
            raise SystemExit("RAM or COS cache disk headroom fell below preflight")
        result, run = run_backend(
            "cuda_strict", records, source_rows, dates, assets, labels,
            manifest_sha, args.tile_size, args.max_object_mib, policy, selected,
            prefetch_objects=args.prefetch_objects,
            source_adapter=args.source_adapter, cos_prefetch=args.cos_prefetch)
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
            "source_adapter": args.source_adapter,
            "cos_prefetch": args.cos_prefetch if args.source_adapter == "cos" else None,
            **_source_prefetch_report_fields(run),
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
            args.tile_size, args.max_object_mib, policy, selected,
            prefetch_objects=args.prefetch_objects,
            source_adapter=args.source_adapter, cos_prefetch=args.cos_prefetch)
        receipt["preflight"] = gate
        runs[backend] = (bundle, receipt)
    cpu, cpu_run = runs["cpu"]
    cuda, cuda_run = runs["cuda_strict"]
    if (cpu_run["source_adapter"] != cuda_run["source_adapter"]
            or cpu_run["prefetch_mode"] != cuda_run["prefetch_mode"]):
        raise ValueError("CPU/CUDA A/B runs used different source adapter or prefetch settings")
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
            expected_auto_cuda=True, prefetch_objects=args.prefetch_objects,
            source_adapter=args.source_adapter, cos_prefetch=args.cos_prefetch)
        auto_run["preflight"] = gate
        auto_run["auto_backend_reason"] = auto.metadata.get("auto_backend_reason")
        auto_run["effective_max_tile_size"] = auto.metadata.get("effective_max_tile_size")
        expected_reason = ("bounded_f61_all_source_15_gpu_tile16"
                           if selected == ALL_SOURCE_METRICS else
                           "bounded_f61_pearson_chain_gpu_tile16")
        if (auto_run["auto_backend_reason"] != expected_reason
                or auto_run["effective_max_tile_size"] != 16):
            raise ValueError("auto did not use the certified F61 source route")
        auto_comparison = compare(cpu, auto, selected, expected_days=len(dates))
    report = {
        "status": "complete" if comparison["pass"] and
                  (auto_comparison is None or auto_comparison["pass"]) else "parity_failed",
        "kind": "real_cos_whole_source_batch_ab.v1",
        "manifest_sha256": manifest_sha,
        "shape": [len(dates), len(assets), len(records)],
        "factor_dtype": "float64",
        "tile_size": args.tile_size,
        "source_adapter": args.source_adapter,
        "cos_prefetch": args.cos_prefetch if args.source_adapter == "cos" else None,
        **_source_prefetch_report_fields(cpu_run),
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
