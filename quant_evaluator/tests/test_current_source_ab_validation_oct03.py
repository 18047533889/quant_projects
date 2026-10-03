"""Behavioral checks for current-source F48 Pearson A/B validation."""
import copy
import json
from pathlib import Path

import pytest

from quant_evaluator.scripts.current_source_ab_validation import (
    validate_current_f48_pearson_ab,
)

SOURCE = "a" * 64
MANIFEST = "b" * 64
METRICS = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
SCOPE = {
    "directories": ["quant_evaluator/api", "quant_evaluator/adapters",
                    "quant_evaluator/contracts", "quant_evaluator/kernels",
                    "quant_evaluator/metrics", "quant_evaluator/runtime"],
    "files": ["quant_evaluator/scripts/benchmark_real_cos_source_batch.py",
              "quant_evaluator/scripts/benchmark_real_cos_factor_tiles.py",
              "quant_evaluator/scripts/source_width_preparation.py",
              "quant_evaluator/scripts/f48_auto_references.py",
              "quant_evaluator/scripts/f48_benchmark_candidate.py",
              "quant_evaluator/scripts/source_tree_provenance.py"],
}


def _h(c):
    return c * 64


def _gate():
    return {"pass": True, "available_ram_bytes": 40 * 1024**3,
            "minimum_available_ram_bytes": 32 * 1024**3,
            "cos_cache_disk_free_bytes": 8 * 1024**3,
            "required_disk_bytes": 5 * 1024**3}


def _run(name, seconds):
    cuda = name == "cuda_strict"
    return {"source_adapter": "cos", "backend_requested": name,
            "backend_used": "cuda" if cuda else "cpu",
            "effective_max_tile_size": 4, "declared_source_tile_size": 4,
            "admitted_source_tile_size": 4, "seconds": seconds,
            "factor_tiles_processed": 12,
            "tile_ranges": [[i, i + 4] for i in range(0, 48, 4)],
            "source_manifest_sha256": MANIFEST,
            "source_request_fingerprint": _h("c"),
            "source_request_identity_sha256": _h("d"),
            "source_snapshot_id": _h("e"), "factor_ids_sha256": _h("f"),
            "max_source_memory_bytes": 4 * 1024**3,
            "prefetch_objects": True, "prefetch_mode": "auto",
            "prefetch_window": 2, "cos_prefetch": "auto",
            "oom_retries": 0 if cuda else None,
            "vram_budget_bytes": 16 * 1024**3 if cuda else None,
            "peak_vram_bytes": 1024 if cuda else None,
            "pool_reserved_peak_bytes": 1024 if cuda else None,
            "h2d_bytes": 2048 if cuda else None,
            "d2h_bytes": 1024 if cuda else None,
            "preflight": _gate()}


def _comparison():
    rows = {}
    for metric in METRICS:
        series = metric == "pearson_ic_series"
        rows[metric] = {
            "pass": True, "artifact_kind": "series" if series else "scalar",
            "cpu_shape": [2586, 48] if series else [48],
            "cuda_shape": [2586, 48] if series else [48],
            "shape_valid": True, "factor_count": 48,
            "compared_value_count": 124128 if series else 48,
            "finite_value_count": 123529 if series else 48,
            "finite_mask_equal": True, "max_abs_error": 1e-15,
            "observation_counts_equal": True, "observation_counts_shape_valid": True,
            "cpu_observation_counts_sha256": _h("1"),
            "cuda_observation_counts_sha256": _h("1"),
            "cpu_values_sha256": _h("2"), "cuda_values_sha256": _h("3")}
    return {"pass": True, "compared_factor_count": 48,
            "compared_metric_count": 124272, "metrics": rows}


def _report(order, cpu, cuda):
    runs = {"cpu": _run("cpu", cpu), "cuda_strict": _run("cuda_strict", cuda)}
    return {
        "status": "complete", "kind": "real_cos_whole_source_batch_ab.v1",
        "manifest_sha256": MANIFEST, "shape": [2586, 5461, 48],
        "factor_dtype": "float64", "tile_size": 4, "source_adapter": "cos",
        "cos_prefetch": "auto", "prefetch_objects": True,
        "prefetch_mode": "auto", "prefetch_window": 2,
        "caller_settings": {"cos_prefetch_workers": 2,
                            "max_prefetch_memory_mib": 512,
                            "max_object_mib": 128, "max_total_mib": 4096},
        "run_order": list(order), "metric_ids": list(METRICS),
        "preflight_before_cpu": _gate(), "preflight_before_cuda": _gate(),
        "runs": [runs["cpu"], runs["cuda_strict"]], "comparison": _comparison(),
        "source_provenance": {"algorithm": "sha256_path_length_prefixed_v1",
                              "aggregate_sha256": SOURCE, "file_count": 166,
                              "scope": copy.deepcopy(SCOPE),
                              "versions": {"python": "3.12.3", "numpy": "2.2.6",
                                           "pandas": "2.3.3", "cupy": None},
                              "limitations": ["current Python source scope only"]},
        "source_provenance_verification": {
            "pass": True, "status": "unchanged",
            "after_aggregate_sha256": SOURCE}}


@pytest.fixture
def reports(tmp_path):
    result = []
    for label, order, cpu, cuda in (
            ("cpu-first", ("cpu", "cuda_strict"), 80.0, 60.0),
            ("cuda-first", ("cuda_strict", "cpu"), 81.0, 61.0)):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(_report(order, cpu, cuda)), encoding="utf-8")
        result.append(path)
    return result


def _validate(paths, source=SOURCE, manifest=MANIFEST):
    return validate_current_f48_pearson_ab(
        paths, expected_source_sha256=source, expected_manifest_sha256=manifest)


def _run_by_backend(report, backend):
    return next(run for run in report["runs"] if run["backend_requested"] == backend)


def test_accepts_opposite_order_pair_bound_to_caller_identities(reports):
    # Matching arbitrary report hashes must not substitute for caller-bound identity.
    result = _validate(reports)
    assert result["pass"] is True
    assert result["source_sha256"] == SOURCE
    assert result["manifest_sha256"] == MANIFEST
    assert result["thread_settings"] == "not_recorded"
    assert any("runtime closure" in item for item in result["limitations"])
    assert all([run["backend_requested"] for run in json.loads(path.read_text())["runs"]]
               == ["cpu", "cuda_strict"] for path in reports)


@pytest.mark.parametrize("source,manifest", [("9" * 64, MANIFEST),
                                               (SOURCE, "8" * 64)])
def test_rejects_stale_caller_identity(reports, source, manifest):
    with pytest.raises(ValueError):
        _validate(reports, source, manifest)


@pytest.mark.parametrize("mutation", [
    "scope", "report_manifest", "bool_tile", "bool_range", "bool_ram",
    "bool_seconds", "huge_seconds", "missing_preflight", "missing_run_prefetch",
    "cross_report_digest", "same_order", "cuda_slower_first",
    "cuda_slower_second", "bool_vram", "peak_over_budget", "runtime_mismatch", "source_drift",
])
def test_rejects_invalid_or_nonqualifying_receipts(reports, mutation):
    # Each mutation represents stale, malformed, or unbalanced evidence.
    a, b = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    if mutation == "scope":
        a["source_provenance"]["scope"]["files"].pop()
    elif mutation == "report_manifest":
        a["manifest_sha256"] = _h("9")
    elif mutation == "bool_tile":
        a["tile_size"] = True
    elif mutation == "bool_range":
        _run_by_backend(a, "cpu")["tile_ranges"][0][0] = False
    elif mutation == "bool_ram":
        a["preflight_before_cpu"]["available_ram_bytes"] = True
    elif mutation == "bool_seconds":
        _run_by_backend(a, "cpu")["seconds"] = True
    elif mutation == "huge_seconds":
        _run_by_backend(a, "cpu")["seconds"] = 10**1000
    elif mutation == "missing_run_prefetch":
        _run_by_backend(a, "cuda_strict")["cos_prefetch"] = None
    elif mutation == "missing_preflight":
        del a["preflight_before_cuda"]
    elif mutation == "cross_report_digest":
        b["comparison"]["metrics"]["pearson_ic"]["cpu_values_sha256"] = _h("4")
    elif mutation == "same_order":
        b["run_order"] = ["cpu", "cuda_strict"]
    elif mutation == "cuda_slower_first":
        _run_by_backend(a, "cuda_strict")["seconds"] = 90.0
    elif mutation == "cuda_slower_second":
        _run_by_backend(b, "cuda_strict")["seconds"] = 90.0
    elif mutation == "bool_vram":
        _run_by_backend(a, "cuda_strict")["vram_budget_bytes"] = True
    elif mutation == "peak_over_budget":
        _run_by_backend(a, "cuda_strict")["peak_vram_bytes"] = 17 * 1024**3
    elif mutation == "runtime_mismatch":
        b["source_provenance"]["versions"]["numpy"] = "different"
    elif mutation == "source_drift":
        a["source_provenance_verification"]["after_aggregate_sha256"] = _h("0")
    for path, value in zip(reports, (a, b)):
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        _validate(reports)


def test_historical_receipts_are_schema_smoke_not_current_source_evidence():
    # Historical hashes only check wire-schema compatibility; they are not
    # presented as current source identity or performance evidence.
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cpu_first_20261003.json",
        root / "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cuda_first_20261003.json",
    ]
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))["manifest_sha256"]
    result = validate_current_f48_pearson_ab(
        paths, expected_source_sha256="b13bb54505d6da85b29726dd4736e4da71369a4f2074774282f9e0388a97046f",
        expected_manifest_sha256=manifest)
    assert result["pass"] is True
    assert result["thread_settings"] == "not_recorded"
