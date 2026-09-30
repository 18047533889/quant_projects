"""Pure equivalence tests for the source API's evidence-backed auto routes."""
from pathlib import Path

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
    f61_mixed = shape == F61 and len(metrics) == 3 and metric_set == frozenset(MIXED) and width >= 8
    tile = (16 if width >= 16 else 8) if f61_mixed else width
    f8, f32 = shape == F8, shape == F32 and tile == 2
    f32_mixed = f32 and len(metrics) == 3 and metric_set == frozenset(MIXED)
    f61_single = shape == F61 and metrics == ("pearson_ic",) and width >= 16
    f61_chain = shape == F61 and len(metrics) == len(PEARSON_CHAIN) and metric_set == frozenset(PEARSON_CHAIN) and width >= 16
    f61_all = shape in (F61, F61_ALT) and len(metrics) == len(ALL_METRICS) and metric_set == frozenset(ALL_METRICS) and width >= 16
    rank_pair = (f8 or f32) and len(metrics) == 2 and metric_set == frozenset(RANK_PAIR)
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
    if f32:
        return "bounded_f32_rank_pair_gpu", 2
    return "bounded_f8_rank_pair_gpu", width


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
        if entry.evidence_id == "real_cos_f8_rank_pair":
            # No source-API whole-request F8 A/B was found; public-facade
            # metric evidence must not be attached as if it were one.
            assert entry.evidence_artifacts == ()
            assert entry.evidence_status == "legacy_unverified_source_performance"
            continue
        assert entry.evidence_artifacts
        assert all((root / path).is_file() for path in entry.evidence_artifacts)


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
