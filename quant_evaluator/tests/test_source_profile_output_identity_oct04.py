from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from quant_evaluator.runtime.source_profile_output_identity import (
    hash_array_bytes, hash_finite_mask, matches_profile_output_identity,
    metric_output_identity, normalize_counts_array,
)


def test_raw_hash_matches_auto_report_bytes_for_layouts_and_nan_payloads():
    bits = np.array([
        0x7FF8000000000001, 0x7FF80000000000AB, 0x7FF0000000000001,
        0x8000000000000000, 0x0000000000000000,
    ], dtype=np.uint64)
    base = bits.view(np.float64)
    arrays = (base, np.asfortranarray(base.reshape(1, -1)), base[::-1],
              np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2])
    for values in arrays:
        expected = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        assert hash_array_bytes(values) == expected


def test_finite_mask_and_counts_preserve_existing_encoding():
    values = np.array([1.0, np.nan, np.inf, -np.inf, -0.0])
    expected_mask = hashlib.sha256(
        np.isfinite(values).astype(np.uint8).tobytes()).hexdigest()
    assert hash_finite_mask(values) == expected_mask
    counts = np.array([0, 1, 2], dtype=np.uint32)
    normalized = normalize_counts_array(counts)
    assert normalized.dtype == np.dtype("<i8")
    assert normalized.flags.c_contiguous


def _fixture():
    context = SimpleNamespace(
        request_shape=(3, 4, 2), metric_ids=("rank_ic", "pearson_ic_series"),
        metric_coverage=(("rank_ic", 2), ("pearson_ic_series", 6)),
    )
    scalar = {"rank_ic": np.array([0.2, np.nan], dtype=np.float64)}
    series = {"pearson_ic_series": np.arange(6, dtype=np.float64).reshape(3, 2)}
    counts = {name: np.array([3, 2], dtype=np.int64) for name in context.metric_ids}
    receipts = tuple(SimpleNamespace(
        metric_id=name,
        **metric_output_identity(scalar[name] if name in scalar else series[name], counts[name]).__dict__,
    ) for name in context.metric_ids)
    profile = SimpleNamespace(outputs=receipts)
    output = SimpleNamespace(
        factor_ids=("a", "b"), scalar_metrics=scalar, series_metrics=series,
        vector_metrics={}, observation_counts=counts,
    )
    return context, profile, output


def test_matches_profile_receipts_for_scalar_and_series_outputs():
    context, profile, output = _fixture()
    assert matches_profile_output_identity(output, profile, context,
                                            metrics=context.metric_ids,
                                            expected_factor_ids=("a", "b"))


@pytest.mark.parametrize("mutation", ["values", "mask", "counts", "missing", "duplicate",
                                        "vector", "shape", "dtype", "series_kind",
                                        "factor_ids"])
def test_profile_identity_rejects_malformed_or_drifted_output(mutation):
    context, profile, output = _fixture()
    if mutation == "values":
        output.scalar_metrics["rank_ic"] = np.array([0.3, np.nan])
    elif mutation == "mask":
        output.scalar_metrics["rank_ic"] = np.array([0.2, 1.0])
    elif mutation == "counts":
        output.observation_counts["rank_ic"] = np.array([3, 9], dtype=np.int64)
    elif mutation == "missing":
        output.observation_counts.pop("rank_ic")
    elif mutation == "duplicate":
        output.series_metrics["rank_ic"] = np.array([[0.2, 0.0]])
    elif mutation == "vector":
        output.vector_metrics["rank_ic"] = output.scalar_metrics.pop("rank_ic")
    elif mutation == "shape":
        output.series_metrics["pearson_ic_series"] = np.zeros((2, 2))
    elif mutation == "dtype":
        output.scalar_metrics["rank_ic"] = np.array([True, False])
    elif mutation == "factor_ids":
        output.factor_ids = output.factor_ids[::-1]
    else:
        output.scalar_metrics["pearson_ic_series"] = output.series_metrics.pop(
            "pearson_ic_series")
    assert not matches_profile_output_identity(output, profile, context,
                                               metrics=context.metric_ids,
                                               expected_factor_ids=("a", "b"))
