"""QE-METRIC-P0-04: typed metric artifact contracts (validation + roundtrip)."""

import numpy as np
import pytest

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.metric_artifacts import (
    DistributionMetricArtifact,
    MatrixMetricArtifact,
    MetricArtifact,
    ScalarMetricArtifact,
    SeriesMetricArtifact,
    VectorMetricArtifact,
)


def _base_kwargs():
    return {
        "metric_id": "mean_ic",
        "domain": "ic",
        "provenance": {"min_periods": 20},
        "created_from": ("factor_batch", "label_bundle"),
    }


# --- validation -----------------------------------------------------------


def test_scalar_rejects_non_string_and_empty_metric_id():
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id=42, domain="ic", artifact_kind="scalar", values=np.ones(3))
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id="   ", domain="ic", artifact_kind="scalar", values=np.ones(3))


def test_scalar_rejects_control_characters_and_bad_domain():
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id="bad\nid", domain="ic", artifact_kind="scalar", values=np.ones(3))
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id="m", domain=None, artifact_kind="scalar", values=np.ones(3))


def test_scalar_rejects_wrong_ndim_and_non_array():
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id="m", domain="ic", artifact_kind="scalar", values=np.ones((2, 3)))
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(metric_id="m", domain="ic", artifact_kind="scalar", values="nope")


def test_series_requires_time_index_matching_t():
    values = np.ones((4, 2))
    with pytest.raises(InvalidContractError):
        SeriesMetricArtifact(
            metric_id="ic_series", domain="ic", artifact_kind="series",
            values=values, time_index=(0, 1),
        )
    ok = SeriesMetricArtifact(
        metric_id="ic_series", domain="ic", artifact_kind="series",
        values=values, time_index=(0, 1, 2, 3),
    )
    assert ok.values.shape == (4, 2)


def test_matrix_requires_square_k_by_k():
    with pytest.raises(InvalidContractError):
        MatrixMetricArtifact(
            metric_id="transitions", domain="quantile", artifact_kind="matrix",
            values=np.ones((2, 3, 4)),
        )
    ok = MatrixMetricArtifact(
        metric_id="transitions", domain="quantile", artifact_kind="matrix",
        values=np.ones((3, 3, 4)),
    )
    assert ok.values.shape == (3, 3, 4)


def test_distribution_requires_string_stat_names_and_2d_samples():
    with pytest.raises(InvalidContractError):
        DistributionMetricArtifact(
            metric_id="boot", domain="ic", artifact_kind="distribution",
            samples=np.ones((5, 2)), stat_names=("ci_lower", 7),
        )
    with pytest.raises(InvalidContractError):
        DistributionMetricArtifact(
            metric_id="boot", domain="ic", artifact_kind="distribution",
            samples=np.ones(5), stat_names=("ci_lower",),
        )


def test_provenance_must_be_mapping_and_isolated():
    with pytest.raises(InvalidContractError):
        ScalarMetricArtifact(
            metric_id="m", domain="ic", artifact_kind="scalar",
            values=np.ones(2), provenance=[("a", 1)],
        )
    source = {"k": 1}
    art = ScalarMetricArtifact(
        metric_id="m", domain="ic", artifact_kind="scalar",
        values=np.ones(2), provenance=source,
    )
    source["k"] = 999
    assert art.provenance["k"] == 1


# --- immutability ----------------------------------------------------------


def test_arrays_are_read_only():
    art = ScalarMetricArtifact(
        metric_id="m", domain="ic", artifact_kind="scalar", values=np.ones(3)
    )
    assert not art.values.flags.writeable
    with pytest.raises(ValueError):
        art.values[0] = 5.0
    with pytest.raises(Exception):
        art.metric_id = "other"  # type: ignore[misc]


# --- equality --------------------------------------------------------------


def test_value_equality_and_type_mismatch():
    a = ScalarMetricArtifact(metric_id="m", domain="ic", artifact_kind="scalar", values=np.array([1.0, 2.0]))
    b = ScalarMetricArtifact(metric_id="m", domain="ic", artifact_kind="scalar", values=np.array([1.0, 2.0]))
    c = ScalarMetricArtifact(metric_id="m", domain="ic", artifact_kind="scalar", values=np.array([1.0, 9.0]))
    d = VectorMetricArtifact(metric_id="m", domain="ic", artifact_kind="vector", values=np.array([[1.0, 2.0]]))
    assert a == b
    assert not a != b
    assert a != c
    assert a != d
    assert hash(a) == hash(b)


# --- roundtrip -------------------------------------------------------------


def test_scalar_roundtrip():
    art = ScalarMetricArtifact(values=np.array([0.1, 0.2, np.nan]), **_base_kwargs(), artifact_kind="scalar")
    restored = MetricArtifact.from_dict(art.to_dict())
    assert isinstance(restored, ScalarMetricArtifact)
    assert restored == art
    assert not restored.values.flags.writeable


def test_series_roundtrip():
    art = SeriesMetricArtifact(
        values=np.arange(6, dtype=float).reshape(3, 2),
        time_index=(10, 11, 12),
        metric_id="rank_ic_series", domain="ic", artifact_kind="series",
        provenance={"method": "spearman"}, created_from=("factor_batch", "label_bundle"),
    )
    restored = MetricArtifact.from_dict(art.to_dict())
    assert isinstance(restored, SeriesMetricArtifact)
    assert restored == art
    assert restored.time_index == (10, 11, 12)


def test_vector_roundtrip():
    art = VectorMetricArtifact(
        values=np.arange(10, dtype=float).reshape(5, 2),
        metric_id="quantile_returns_full", domain="quantile", artifact_kind="vector",
        provenance={"n_quantiles": 5}, created_from=("factor_batch", "label_bundle"),
    )
    restored = MetricArtifact.from_dict(art.to_dict())
    assert isinstance(restored, VectorMetricArtifact)
    assert restored == art
    assert restored.values.shape == (5, 2)


def test_matrix_and_distribution_roundtrip():
    mat = MatrixMetricArtifact(
        values=np.eye(3)[:, :, None] * np.arange(2),
        metric_id="transitions", domain="quantile", artifact_kind="matrix",
        provenance={}, created_from=("factor_batch",),
    )
    dist = DistributionMetricArtifact(
        samples=np.arange(8, dtype=float).reshape(4, 2),
        stat_names=("ci_lower", "ci_upper"),
        metric_id="block_bootstrap_ci", domain="ic", artifact_kind="distribution",
        provenance={"num_bootstrap": 100}, created_from=("ic_series",),
    )
    for art, cls in ((mat, MatrixMetricArtifact), (dist, DistributionMetricArtifact)):
        restored = MetricArtifact.from_dict(art.to_dict())
        assert isinstance(restored, cls)
        assert restored == art


def test_from_dict_rejects_unknown_kind():
    with pytest.raises(InvalidContractError):
        MetricArtifact.from_dict({"artifact_kind": "hologram"})


def test_quantile_returns_full_artifact_is_vector():
    from quant_evaluator import AxisRef, FactorBatch, LabelBundle
    from quant_evaluator.metrics.registry_adapters import compute_quantile_returns_full_artifact

    rng = np.random.default_rng(0)
    T, N, F, Q = 60, 20, 2, 5
    batch = FactorBatch(
        factor_ids=("f", "g"),
        time_axis=AxisRef(name="time", dtype="int64", size=T, values=np.arange(T)),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N, values=np.arange(N)),
        values=rng.normal(size=(T, N, F)),
    )
    labels = LabelBundle(
        target_id="ret",
        values=rng.normal(size=(T, N)),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    artifact = compute_quantile_returns_full_artifact(
        batch, labels, min_periods=20, n_quantiles=Q
    )
    assert isinstance(artifact, VectorMetricArtifact)
    assert artifact.metric_id == "quantile_returns_full"
    assert artifact.values.shape == (Q, F)
    assert artifact.created_from == ("factor_batch", "label_bundle")
    assert not artifact.values.flags.writeable
