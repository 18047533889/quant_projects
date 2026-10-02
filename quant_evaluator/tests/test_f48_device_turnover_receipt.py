"""Real source verification after the GPU turnover device-return change."""
import json
from pathlib import Path
import pytest


def _receipt():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "quant_evaluator/docs/benchmarks/f48_cap16_device_turnover_ordinary_20261002.json").read_text())


def test_device_turnover_real_receipt_preserves_numeric_and_route_contracts():
    report = _receipt()
    assert report["status"] == "complete"
    assert report["benchmark_only"] is False
    assert report["registered_route_verification"] is True
    assert report["benchmark_auto_candidate"] is None
    assert report["shape"] == [2586, 5461, 48]
    assert report["route_pass"] and report["coverage_pass"] and report["identity_pass"]
    assert report["reference_comparison"]["pass"]
    assert report["direct_comparison"]["pass"]
    assert report["source_provenance_verification"]["pass"]
    assert report["source_provenance"]["aggregate_sha256"] == report["source_provenance_verification"]["after_aggregate_sha256"]
    auto = report["runs"]["auto_default"]
    assert auto["api_default_tile_size"] is True
    assert auto["declared_source_tile_size"] == 16
    assert auto["admitted_source_tile_size"] == 5
    assert auto["auto_backend_reason"] == "bounded_f48_mixed_three_gpu_cap16_tile2"


@pytest.mark.parametrize("name", ["auto_default", "cuda_strict_tile2"])
def test_real_device_turnover_runs_have_bounded_coverage_and_final_outputs(name):
    run = _receipt()["runs"][name]
    assert run["backend_used"] == "cuda"
    assert run["effective_max_tile_size"] == 2
    assert run["factor_tiles_processed"] == 24
    assert run["tile_ranges"] == [[i, i + 2] for i in range(0, 48, 2)]
    assert run["oom_retries"] == 0
    assert run["d2h_bytes"] == 48 * 3 * (8 + 8)
    assert run["peak_vram_bytes"] == 2586991616
    assert run["estimated_peak_source_bytes"] <= run["max_source_memory_bytes"]
    assert run["source_read_wall_seconds"] <= run["seconds"]
    # Timing samples validate this run, not a performance superiority claim.
    assert run["seconds"] > 0
