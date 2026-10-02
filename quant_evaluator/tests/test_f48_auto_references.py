"""Offline validation tests for F48 opposite-order evidence receipts."""
import json

import pytest

from quant_evaluator.scripts.f48_auto_references import (
    F48_METRICS, F48_SHAPE, validate_f48_auto_references,
)
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness


MANIFEST = "a" * 64
IDENTITY = {
    "source_request_identity_sha256": "b" * 64,
    "source_snapshot_id": "c" * 64,
    "source_manifest_sha256": MANIFEST,
    "factor_ids_sha256": "d" * 64,
    "source_request_fingerprint": "e" * 64,
}
RANGES = [[start, start + 2] for start in range(0, 48, 2)]


def _report(order):
    runs = {
        "cpu": dict(backend_requested="cpu", backend_used="cpu", seconds=220.3,
                    source_adapter="cos", cos_prefetch="auto", prefetch_objects=True,
                    prefetch_mode="auto", prefetch_window=2,
                    max_source_memory_bytes=4 * 1024**3,
                    factor_tiles_processed=24,
                    tile_ranges=RANGES, **IDENTITY),
        "cuda_strict": dict(backend_requested="cuda_strict", backend_used="cuda",
                            seconds=57.3, source_adapter="cos", cos_prefetch="auto",
                            prefetch_objects=True, prefetch_mode="auto", prefetch_window=2,
                            vram_budget_bytes=30 * 1024**3,
                            max_source_memory_bytes=4 * 1024**3,
                            factor_tiles_processed=24,
                            tile_ranges=RANGES, **IDENTITY),
    }
    metrics = {}
    for index, metric in enumerate(F48_METRICS):
        metrics[metric] = {
            "pass": True, "artifact_kind": "scalar", "cpu_shape": [48],
            "cuda_shape": [48], "compared_value_count": 48,
            "finite_value_count": 48, "finite_mask_equal": True,
            "observation_counts_equal": True, "observation_counts_shape_valid": True,
            "max_abs_error": 1e-15,
            "cpu_values_sha256": str(index + 1) * 64,
            "cuda_values_sha256": str(index + 4) * 64,
            "cpu_observation_counts_sha256": "f" * 64,
            "cuda_observation_counts_sha256": "f" * 64,
        }
    canonical_runs = [runs["cpu"], runs["cuda_strict"]]
    return {
        "status": "complete", "kind": "real_cos_whole_source_batch_ab.v1",
        "manifest_sha256": MANIFEST, "shape": list(F48_SHAPE),
        "factor_dtype": "float64", "tile_size": 2,
        "metric_ids": list(F48_METRICS), "run_order": list(order),
        "source_adapter": "cos", "cos_prefetch": "auto",
        "prefetch_objects": True, "prefetch_mode": "auto", "prefetch_window": 2,
        "preflight_before_cpu": {"pass": True, "available_ram_bytes": 40 * 1024**3,
                                 "minimum_available_ram_bytes": 32 * 1024**3,
                                 "cos_cache_disk_free_bytes": 8 * 1024**3,
                                 "required_disk_bytes": 5 * 1024**3},
        "preflight_before_cuda": {"pass": True, "available_ram_bytes": 40 * 1024**3,
                                  "minimum_available_ram_bytes": 32 * 1024**3,
                                  "cos_cache_disk_free_bytes": 8 * 1024**3,
                                  "required_disk_bytes": 5 * 1024**3}, "runs": canonical_runs,
        "comparison": {"pass": True, "compared_factor_count": 48,
                       "compared_metric_count": 144, "metrics": metrics},
    }


def _write_pair(tmp_path, mutate_second=None):
    paths = []
    for index, order in enumerate((("cpu", "cuda_strict"),
                                   ("cuda_strict", "cpu"))):
        report = _report(order)
        if index == 1 and mutate_second:
            mutate_second(report)
        path = tmp_path / f"f48-{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def test_f48_reference_pair_requires_opposite_order_and_cuda_faster_each_time(tmp_path):
    paths = _write_pair(tmp_path)
    result = validate_f48_auto_references(paths, F48_METRICS, MANIFEST)
    assert result["verified_source_request_fingerprint"] == IDENTITY[
        "source_request_fingerprint"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(shape=[2586, 5461, 47]),
    lambda r: r.update(tile_size=4),
    lambda r: r.update(factor_dtype="float32"),
    lambda r: r.update(source_adapter="legacy"),
    lambda r: r.update(metric_ids=["rank_ic", "quantile_spread"]),
    lambda r: r.update(manifest_sha256="0" * 64),
    lambda r: r["runs"][1].update(source_request_fingerprint="0" * 64),
    lambda r: r["runs"][1].update(tile_ranges=[[0, 4]]),
    lambda r: r["comparison"]["metrics"]["rank_ic"].update(
        cuda_observation_counts_sha256="0" * 64),
])
def test_f48_reference_rejects_stale_or_incomplete_receipt(tmp_path, mutation):
    paths = _write_pair(tmp_path, mutation)
    with pytest.raises(ValueError):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_f48_reference_rejects_invalid_timing(tmp_path, value):
    paths = _write_pair(tmp_path, lambda r: r["runs"][1].update(seconds=value))
    with pytest.raises(ValueError, match="timing"):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)


def test_f48_reference_rejects_cuda_not_faster(tmp_path):
    paths = _write_pair(tmp_path, lambda r: next(
        run for run in r["runs"] if run["backend_requested"] == "cuda_strict"
    ).update(seconds=221.0))
    with pytest.raises(ValueError, match="faster"):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)


def test_f48_reference_rejects_different_output_hashes_between_orders(tmp_path):
    paths = _write_pair(tmp_path, lambda r: r["comparison"]["metrics"][
        "rank_ic"].update(cuda_values_sha256="0" * 64))
    with pytest.raises(ValueError, match="opposite-order"):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)


def test_f48_reference_rejects_same_order(tmp_path):
    paths = _write_pair(tmp_path, lambda r: r.update(run_order=["cpu", "cuda_strict"]))
    with pytest.raises(ValueError, match="opposite"):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)


def test_f48_reference_rejects_duplicate_metrics_and_unbounded_report(tmp_path):
    paths = _write_pair(tmp_path)
    with pytest.raises(ValueError):
        validate_f48_auto_references(paths, (*F48_METRICS, F48_METRICS[0]), MANIFEST)
    oversized = tmp_path / "oversized.json"
    oversized.write_text(" " * (1024**2 + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="1 MiB"):
        validate_f48_auto_references((oversized, paths[1]), F48_METRICS, MANIFEST)


@pytest.mark.parametrize("override", [
    ["--cos-prefetch", "off"], ["--cos-prefetch", "on"],
    ["--cos-prefetch-workers", "1"], ["--cos-prefetch-workers", "4"],
    ["--max-prefetch-memory-mib", "64"],
])
def test_f48_cli_rejects_nonreference_prefetch_settings_before_io(
        monkeypatch, tmp_path, override):
    monkeypatch.setattr("sys.argv", ["benchmark", "--factors", "48",
        "--days", "2586", "--assets", "5461", "--tile-size", "2",
        "--source-adapter", "cos", "--auto-f48-references", "cpu.json", "cuda.json",
        "--output", str(tmp_path / "out.json"), *override])
    monkeypatch.setattr(harness, "preflight", lambda *args: pytest.fail("unexpected preflight"))
    monkeypatch.setattr(harness.tiles, "read_manifest", lambda *args: pytest.fail("unexpected source read"))
    with pytest.raises(SystemExit) as exc:
        harness.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("field,value", [
    ("max_abs_error", 1e-4), ("finite_mask_equal", "true"),
])
def test_f48_reference_rejects_weak_or_malformed_metric_evidence(tmp_path, field, value):
    paths = _write_pair(tmp_path, lambda report: report["comparison"]["metrics"][
        "rank_ic"].update({field: value}))
    with pytest.raises(ValueError):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)


@pytest.mark.parametrize("malform", [
    lambda report: report.update(comparison=[]),
    lambda report: report.update(runs=[None, None]),
    lambda report: report["comparison"].update(metrics=[]),
])
def test_f48_reference_rejects_malformed_object_fields(tmp_path, malform):
    paths = _write_pair(tmp_path, malform)
    with pytest.raises(ValueError):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)


@pytest.mark.parametrize(("key", "value"), [
    ("cos_prefetch_workers", 1), ("cos_prefetch_workers", True),
    ("max_prefetch_memory_mib", 0), ("max_prefetch_memory_mib", 512.0),
    ("max_object_mib", 64), ("max_object_mib", False),
    ("max_total_mib", 1024), ("max_total_mib", 4096.0),
])
def test_f48_reference_rejects_wrong_or_coerced_caller_settings(tmp_path, key, value):
    paths = _write_pair(tmp_path, lambda report: report.update(caller_settings={
        "cos_prefetch_workers": 2, "max_prefetch_memory_mib": 512,
        "max_object_mib": 128, "max_total_mib": 4096, key: value,
    }))
    with pytest.raises(ValueError):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)


@pytest.mark.parametrize(("receipt", "field", "value"), [
    ("preflight_before_cpu", "available_ram_bytes", 0),
    ("preflight_before_cuda", "minimum_available_ram_bytes", 16 * 1024**3),
    ("preflight_before_cpu", "cos_cache_disk_free_bytes", 0),
    ("preflight_before_cuda", "required_disk_bytes", 1024**3),
    ("preflight_before_cpu", "available_ram_bytes", True),
    ("preflight_before_cuda", "required_disk_bytes", 5 * 1024**3 + 0.5),
])
def test_f48_reference_rejects_inconsistent_preflight_details(
        tmp_path, receipt, field, value):
    paths = _write_pair(tmp_path, lambda report: report[receipt].update({field: value}))
    with pytest.raises(ValueError):
        validate_f48_auto_references(paths, F48_METRICS, MANIFEST)

