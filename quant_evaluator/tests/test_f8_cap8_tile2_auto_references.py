import json

import pytest

from quant_evaluator.scripts.f8_cap8_tile2_auto_references import (
    validate_cap8_tile2_references,
)


def _reference(order):
    cpu_fp = "a" * 64
    runs = {
        "cpu": {
            "backend_requested": "cpu", "backend_used": "cpu",
            "source_adapter": "cos", "cos_prefetch": "auto",
            "prefetch_mode": "auto", "prefetch_objects": True,
            "prefetch_window": 2, "source_manifest_sha256": "b" * 64,
            "tile_ranges": [[0, 2], [2, 4], [4, 6], [6, 8]],
            "source_request_identity_sha256": "c" * 64,
            "source_snapshot_id": "d" * 64,
            "factor_ids_sha256": "e" * 64,
            "source_request_fingerprint": cpu_fp,
            "factor_tiles_processed": 4,
        },
        "cuda_strict": {
            "backend_requested": "cuda_strict", "backend_used": "cuda",
            "source_adapter": "cos", "cos_prefetch": "auto",
            "prefetch_mode": "auto", "prefetch_objects": True,
            "prefetch_window": 2, "source_manifest_sha256": "b" * 64,
            "tile_ranges": [[0, 2], [2, 4], [4, 6], [6, 8]],
            "source_request_identity_sha256": "c" * 64,
            "source_snapshot_id": "d" * 64,
            "factor_ids_sha256": "e" * 64,
            "source_request_fingerprint": cpu_fp,
            "factor_tiles_processed": 4,
        },
    }
    metrics = {
        "rank_ic": {
            "pass": True, "cpu_shape": [8], "cuda_shape": [8],
            "cpu_values_sha256": "1" * 64, "cuda_values_sha256": "2" * 64,
            "cpu_observation_counts_sha256": "3" * 64,
            "cuda_observation_counts_sha256": "3" * 64,
            "artifact_kind": "scalar", "compared_value_count": 8,
            "finite_mask_equal": True, "observation_counts_equal": True,
            "finite_value_count": 8,
        },
        "rank_ic_series": {
            "pass": True, "cpu_shape": [2586, 8], "cuda_shape": [2586, 8],
            "cpu_values_sha256": "5" * 64, "cuda_values_sha256": "6" * 64,
            "cpu_observation_counts_sha256": "7" * 64,
            "cuda_observation_counts_sha256": "7" * 64,
            "artifact_kind": "series", "compared_value_count": 20688,
            "finite_mask_equal": True, "observation_counts_equal": True,
            "finite_value_count": 20643,
        },
    }
    ordered_runs = [runs[name] for name in order]
    return {
        "status": "complete", "kind": "real_cos_whole_source_batch_ab.v1",
        "manifest_sha256": "b" * 64, "shape": [2586, 5461, 8],
        "factor_dtype": "float64", "tile_size": 2,
        "metric_ids": ["rank_ic", "rank_ic_series"],
        "run_order": list(order), "source_adapter": "cos",
        "cos_prefetch": "auto", "prefetch_mode": "auto",
        "prefetch_objects": True, "prefetch_window": 2,
        "runs": ordered_runs,
        "comparison": {"pass": True, "compared_factor_count": 8,
                       "compared_metric_count": 20696, "metrics": metrics},
    }


def _write_pair(tmp_path, second_mutation=None):
    orders = (("cpu", "cuda_strict"), ("cuda_strict", "cpu"))
    paths = []
    for index, order in enumerate(orders):
        report = _reference(order)
        if index == 1 and second_mutation:
            second_mutation(report)
        path = tmp_path / f"reference-{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return paths


def test_cap8_tile2_reference_pair_binds_fresh_source_request(tmp_path):
    reference = validate_cap8_tile2_references(
        _write_pair(tmp_path), ("rank_ic", "rank_ic_series"), "b" * 64)
    assert reference["tile_size"] == 2
    assert reference["verified_source_request_fingerprint"] == "a" * 64


@pytest.mark.parametrize("mutation", [
    lambda report: report.update(tile_size=8),
    lambda report: report["runs"][0].update(source_snapshot_id="f" * 64),
    lambda report: report["runs"][0].pop("source_request_fingerprint"),
    lambda report: report.update(run_order=["cpu", "cuda_strict"]),
    lambda report: report["runs"].append(dict(report["runs"][0])),
    lambda report: report["comparison"]["metrics"]["rank_ic"].update(cpu_values_sha256="bad"),
    lambda report: report["comparison"]["metrics"]["rank_ic"].update(cpu_observation_counts_sha256="bad"),
    lambda report: report["comparison"]["metrics"]["rank_ic"].update(cuda_observation_counts_sha256="9" * 64),
    lambda report: report["runs"][0].update(source_request_fingerprint="f" * 64),
    lambda report: report["runs"][0].update(source_request_identity_sha256="f" * 64),
    lambda report: report.update(metric_ids=["rank_ic_series", "rank_ic"]),
])
def test_cap8_tile2_reference_rejects_wrong_profile_or_identity(tmp_path, mutation):
    with pytest.raises(ValueError):
        validate_cap8_tile2_references(
            _write_pair(tmp_path, mutation), ("rank_ic", "rank_ic_series"), "b" * 64)
