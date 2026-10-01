"""Pure equivalence tests for the source API's evidence-backed auto routes."""
from pathlib import Path
from dataclasses import replace

import pytest

from quant_evaluator.runtime.source_auto_evidence import (
    SOURCE_AUTO_EVIDENCE, SOURCE_AUTO_EVIDENCE_VERSION, select_source_auto_route,
)

F8, F32, F61, F61_ALT = ((2586, 5461, 8), (2586, 5461, 32),
                         (2586, 5461, 61), (2400, 5000, 61))
RANK_PAIR = ("rank_ic", "rank_ic_series")
MIXED = ("rank_ic", "quantile_spread", "factor_turnover_rate")
PEARSON_CHAIN = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
ALL_METRICS = tuple(sorted({
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
}))


def _legacy_route(shape, metrics, width, factor_dtype="float64", label_dtype="float64"):
    """Independent transcription of the pre-registry route predicates."""
    metrics = tuple(metrics)
    metric_set = frozenset(metrics)
    f8_rank_pair = shape == F8 and width == 2 and len(metrics) == 2 and metric_set == frozenset(RANK_PAIR)
    f61_mixed = shape == F61 and len(metrics) == 3 and metric_set == frozenset(MIXED) and width >= 8
    tile = (16 if width >= 16 else 8) if f61_mixed else width
    f32 = shape == F32 and tile == 2
    f32_mixed = f32 and len(metrics) == 3 and metric_set == frozenset(MIXED)
    f61_single = shape == F61 and metrics == ("pearson_ic",) and width >= 16
    f61_chain = shape == F61 and len(metrics) == len(PEARSON_CHAIN) and metric_set == frozenset(PEARSON_CHAIN) and width >= 16
    f61_all = shape in (F61, F61_ALT) and len(metrics) == len(ALL_METRICS) and metric_set == frozenset(ALL_METRICS) and width >= 16
    # Exact F8 source-API A/B coverage is deliberately tile-2 only.
    rank_pair = f8_rank_pair or (f32 and len(metrics) == 2 and metric_set == frozenset(RANK_PAIR))
    if factor_dtype != "float64" or label_dtype != "float64" or not (
            rank_pair or f32_mixed or f61_mixed or f61_single or f61_chain or f61_all):
        return None
    if f61_all:
        return "bounded_f61_all_source_15_gpu_tile16", 16
    if f61_chain:
        return "bounded_f61_pearson_chain_gpu_tile16", 16
    if f61_single:
        return "bounded_f61_pearson_ic_gpu_tile16", 16
    if f61_mixed:
        return ("bounded_f61_mixed_three_gpu_tile16" if tile == 16
                else "bounded_f61_mixed_three_gpu"), tile
    if f32_mixed:
        return "bounded_f32_mixed_three_gpu", 2
    if f8_rank_pair:
        return "bounded_f8_rank_pair_gpu", 2
    if f32:
        return "bounded_f32_rank_pair_gpu", 2
    return None


@pytest.mark.parametrize("shape", [F8, F32, F61, F61_ALT, (2586, 5461, 7)])
@pytest.mark.parametrize("metrics", [
    RANK_PAIR, tuple(reversed(RANK_PAIR)), MIXED, tuple(reversed(MIXED)),
    ("pearson_ic",), PEARSON_CHAIN, tuple(reversed(PEARSON_CHAIN)), ALL_METRICS,
    ("rank_ic", "rank_ic_series", "coverage"),
])
@pytest.mark.parametrize("width", [1, 2, 7, 8, 15, 16, 32])
@pytest.mark.parametrize("dtypes", [("float64", "float64"), ("float32", "float64"), ("float64", "float32")])
def test_registry_matches_legacy_predicate_grid(shape, metrics, width, dtypes):
    expected = _legacy_route(shape, metrics, width, *dtypes)
    # Fresh opposite-order source A/B adds only this cap8/effective2 envelope.
    if (shape == F8 and width == 8 and len(metrics) == 2
            and frozenset(metrics) == frozenset(RANK_PAIR)
            and dtypes == ("float64", "float64")):
        expected = ("bounded_f8_rank_pair_gpu_cap8_tile2", 2)
    actual = select_source_auto_route(
        shape=shape, metrics=metrics, source_dtype=dtypes[0], label_dtype=dtypes[1],
        requested_tile_width=width,
    )
    actual_tuple = None if actual is None else (actual.legacy_reason, actual.effective_tile_width)
    assert actual_tuple == expected


def test_registry_evidence_ids_are_stable_and_artifact_paths_exist():
    assert SOURCE_AUTO_EVIDENCE_VERSION
    ids = [entry.evidence_id for entry in SOURCE_AUTO_EVIDENCE]
    assert len(ids) == len(set(ids))
    root = Path(__file__).resolve().parents[2]
    for entry in SOURCE_AUTO_EVIDENCE:
        assert entry.evidence_artifacts
        assert all((root / path).is_file() for path in entry.evidence_artifacts)


def test_measured_f8_is_selected_only_for_exact_tile2(monkeypatch):
    import quant_evaluator.runtime.source_auto_evidence as registry

    entry = next(e for e in SOURCE_AUTO_EVIDENCE
                 if e.evidence_id == "real_cos_f8_rank_pair")
    query = dict(shape=F8, metrics=RANK_PAIR, source_dtype="float64",
                 label_dtype="float64")
    route = select_source_auto_route(**query, requested_tile_width=2)
    assert route is not None
    assert route.evidence_id == "real_cos_f8_rank_pair"
    assert route.effective_tile_width == 2
    assert route.evidence_artifacts == entry.evidence_artifacts

    # Removing measured status must revoke this exact envelope even when its
    # shape, metrics, dtype, and requested width still match.
    inactive = replace(entry, evidence_status="legacy_unverified_source_performance")
    monkeypatch.setattr(
        registry, "SOURCE_AUTO_EVIDENCE",
        tuple(inactive if item.evidence_id == entry.evidence_id else item
              for item in SOURCE_AUTO_EVIDENCE),
    )
    assert select_source_auto_route(**query, requested_tile_width=2) is None
    for width in (1, 3, 16, 32):
        assert select_source_auto_route(**query, requested_tile_width=width) is None
    cap8 = select_source_auto_route(**query, requested_tile_width=8)
    assert cap8.evidence_id == "real_cos_f8_rank_pair_cap8_tile2"
    assert cap8.effective_tile_width == 2


def test_f8_rank_pair_source_receipts_match_across_orders_and_cuda_is_faster():
    """Check recorded A/B receipts without timing this machine's runtime."""
    import json

    root = Path(__file__).resolve().parents[2]
    entry = next(e for e in SOURCE_AUTO_EVIDENCE
                 if e.evidence_id == "real_cos_f8_rank_pair")
    assert entry.evidence_status == "measured_source_ab"
    assert entry.exact_requested_tile == 2
    assert entry.certified_tile_widths == (2,)
    assert entry.evidence_artifacts == (
        "quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_cpu_first_20261001.json",
        "quant_evaluator/docs/benchmarks/real_cos_f8_source_rank_pair_cuda_first_20261001.json",
    )
    documents = [json.loads((root / path).read_text())
                 for path in entry.evidence_artifacts]
    expected_orders = (("cpu", "cuda_strict"), ("cuda_strict", "cpu"))
    expected_hashes = {
        "rank_ic": {
            "cpu_values_sha256": "f13219245d9c4cdf5052b5010a3d2bbe39a7ab0ce7179d8a6417ce95e15f28e8",
            "cuda_values_sha256": "5c1d415affd699e52d8335f2a148af37a2920717485fd4aae46a0a55e7dfaf81",
        },
        "rank_ic_series": {
            "cpu_values_sha256": "91f60bf5ee83c96bc8b2e68c1d6d3924a93f2ba804fedc5b5f4183bac7896edf",
            "cuda_values_sha256": "82b620ebf5ee38b32253215b4b496336f4c367925770706362c4c5ab8b65e78d",
        },
    }
    expected_count_hash = "b3a7876bf3e133513cb01e6d15a18fe1461c9b6e34038c5f2d9ffd06b796cb47"
    shared_run_keys = (
        "shape", "factor_dtype", "tile_size", "source_adapter", "cos_prefetch",
        "prefetch_objects", "prefetch_mode", "prefetch_window", "metric_ids",
        "manifest_sha256",
    )
    for doc, expected_order in zip(documents, expected_orders):
        assert doc["status"] == "complete"
        assert doc["kind"] == "real_cos_whole_source_batch_ab.v1"
        assert doc["manifest_sha256"] == "b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864"
        assert tuple(doc["shape"]) == F8
        assert tuple(doc["run_order"]) == expected_order
        assert doc["factor_dtype"] == "float64"
        assert doc["tile_size"] == 2
        assert doc["source_adapter"] == "cos"
        assert doc["cos_prefetch"] == "auto"
        assert doc["prefetch_objects"] is True
        assert doc["prefetch_mode"] == "auto"
        assert doc["prefetch_window"] == 2
        assert frozenset(doc["metric_ids"]) == frozenset(RANK_PAIR)
        assert doc["preflight_before_cpu"]["pass"] is True
        assert doc["preflight_before_cuda"]["pass"] is True
        assert doc["comparison"]["pass"] is True
        assert doc["comparison"]["compared_factor_count"] == 8
        assert doc["comparison"]["compared_metric_count"] == 20_696
        assert set(doc["comparison"]["metrics"]) == set(RANK_PAIR)
        runs = {run["backend_requested"]: run for run in doc["runs"]}
        assert set(runs) == {"cpu", "cuda_strict"}
        assert runs["cpu"]["backend_used"] == "cpu"
        assert runs["cuda_strict"]["backend_used"] == "cuda"
        assert runs["cuda_strict"]["seconds"] < runs["cpu"]["seconds"]
        for key in ("source_adapter", "source_request_identity_sha256",
                    "source_snapshot_id", "source_manifest_sha256",
                    "factor_ids_sha256", "prefetch_objects", "prefetch_mode",
                    "cos_prefetch", "prefetch_window", "tile_ranges",
                    "max_source_memory_bytes", "estimated_peak_source_bytes"):
            assert runs["cpu"][key] == runs["cuda_strict"][key]
        assert runs["cpu"]["source_adapter"] == "cos"
        assert runs["cpu"]["prefetch_objects"] is True
        assert runs["cpu"]["prefetch_mode"] == runs["cpu"]["cos_prefetch"] == "auto"
        assert runs["cpu"]["prefetch_window"] == 2
        for metric in RANK_PAIR:
            result = doc["comparison"]["metrics"][metric]
            assert result["pass"] is True
            assert result["shape_valid"] is True
            assert result["finite_mask_equal"] is True
            assert result["observation_counts_equal"] is True
            assert result["observation_counts_shape_valid"] is True
            assert result["cpu_observation_counts_sha256"] == result["cuda_observation_counts_sha256"]
            assert result["cpu_observation_counts_sha256"] == expected_count_hash
            assert result["compared_value_count"] == (8 if metric == "rank_ic" else 20_688)
            assert result["finite_value_count"] == (8 if metric == "rank_ic" else 20_643)
            assert {key: result[key] for key in expected_hashes[metric]} == expected_hashes[metric]

    assert all(doc["comparison"]["compared_metric_count"] == 20_696
               for doc in documents)
    for key in shared_run_keys:
        assert documents[0][key] == documents[1][key]
    for metric in RANK_PAIR:
        for key in ("cpu_values_sha256", "cuda_values_sha256",
                    "cpu_observation_counts_sha256", "cuda_observation_counts_sha256"):
            assert len({doc["comparison"]["metrics"][metric][key]
                        for doc in documents}) == 1


def test_f61_pearson_consuming_evidence_has_full_parity_and_stable_outputs():
    """Validate recorded evidence, not a timing assertion on the CI machine."""
    import json

    root = Path(__file__).resolve().parents[2]
    entry = next(e for e in SOURCE_AUTO_EVIDENCE
                 if e.evidence_id == "real_cos_f61_pearson_chain_tile16")
    names = ("real_cos_f61_consuming_source_pearson_ab_20260930.json",
             "real_cos_f61_prefetch4_pearson_ab_20260930.json",
             "real_cos_f61_prefetch4_reverse_pearson_ab_20260930.json")
    documents = []
    for name in names:
        path = "quant_evaluator/docs/benchmarks/" + name
        assert path in entry.evidence_artifacts
        doc = json.loads((root / path).read_text())
        assert doc["status"] == "complete"
        assert tuple(doc["shape"]) == F61
        assert frozenset(doc["metric_ids"]) == frozenset(PEARSON_CHAIN)
        for comparison in (doc["comparison"], doc["auto_comparison"]):
            assert comparison["pass"] is True
            assert comparison["compared_factor_count"] == 61
            # The harness counts compared output elements, not metric IDs.
            assert comparison["compared_metric_count"] == 2586 * 61 + 3 * 61
            assert set(comparison["metrics"]) == set(PEARSON_CHAIN)
            for metric in comparison["metrics"].values():
                assert metric["finite_mask_equal"] is True
                assert metric["observation_counts_equal"] is True
        documents.append(doc)
    assert len({d["manifest_sha256"] for d in documents}) == 1
    for metric_id in PEARSON_CHAIN:
        for key in ("cpu_values_sha256", "cuda_values_sha256"):
            assert len({d["comparison"]["metrics"][metric_id][key]
                        for d in documents}) == 1
