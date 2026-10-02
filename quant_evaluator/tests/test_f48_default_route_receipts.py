"""Exact envelope and real receipt checks for the registered F48 default route."""
import itertools
import json
from pathlib import Path
import pytest
from quant_evaluator.runtime.source_auto_evidence import select_source_auto_route

SHAPE = (2586, 5461, 48)
METRICS = ("rank_ic", "quantile_spread", "factor_turnover_rate")


def _route(**changes):
    request = dict(shape=SHAPE, metrics=METRICS, source_dtype="float64",
                   label_dtype="float64", requested_tile_width=16)
    request.update(changes)
    return select_source_auto_route(**request)


@pytest.mark.parametrize("metrics", list(itertools.permutations(METRICS)))
def test_default_cap16_route_is_deterministic_under_metric_reordering(metrics):
    route = _route(metrics=metrics)
    assert route.evidence_id == "real_cos_f48_mixed_three_cap16_tile2"
    assert route.effective_tile_width == 2
    assert route.evidence_status == "measured_source_ab"


@pytest.mark.parametrize("changes", [
    {"shape": (2586, 5461, count)} for count in (47, 49, 63, 64, 65)
] + [
    {"requested_tile_width": cap} for cap in (1, 3, 4, 8, 15, 17, 32)
] + [
    {"source_dtype": "float32"}, {"label_dtype": "float32"},
    {"metrics": ("rank_ic", "rank_ic", "factor_turnover_rate")},
    {"metrics": METRICS + ("coverage",)}, {"metrics": METRICS[:-1]},
])
def test_default_route_does_not_extrapolate_unmeasured_envelopes(changes):
    assert _route(**changes) is None


def test_real_ordinary_default_receipt_has_no_injection_and_complete_coverage():
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / "quant_evaluator/docs/benchmarks/f48_cap16_ordinary_default_auto_20261002.json").read_text())
    assert report["status"] == "complete"
    assert report["benchmark_only"] is False
    assert report["registered_route_verification"] is True
    assert report["benchmark_auto_candidate"] is None
    assert report["shape"] == list(SHAPE)
    assert report["route_pass"] and report["coverage_pass"] and report["identity_pass"]
    assert report["reference_comparison"]["pass"]
    assert report["direct_comparison"]["pass"]
    assert report["source_provenance_verification"]["pass"]
    assert report["source_provenance"]["aggregate_sha256"] == report["source_provenance_verification"]["after_aggregate_sha256"]
    auto = report["runs"]["auto_default"]
    assert auto["api_default_tile_size"] is True
    assert auto["declared_source_tile_size"] == 16
    assert auto["admitted_source_tile_size"] == 5
    assert auto["effective_max_tile_size"] == 2
    assert auto["backend_used"] == "cuda"
    assert auto["oom_retries"] == 0
    assert auto["tile_ranges"] == [[i, i + 2] for i in range(0, 48, 2)]
    assert auto["factor_tiles_processed"] == 24
