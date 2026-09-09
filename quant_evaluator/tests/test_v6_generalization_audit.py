"""Independent V6 audit regressions for generalization confidence evidence."""
import numpy as np
import pytest

from quant_evaluator.metrics.generalization_evidence import build_train_validation_artifact


IDENTITY = dict(
    train_factor_ids=("f",), validation_factor_ids=("f",),
    train_factor_versions=("v1",), validation_factor_versions=("v1",),
    metric_instance="rank_ic:h10:daily", train_split_ref="train:1",
    validation_split_ref="validation:1",
)


def test_paired_ci_uses_one_joint_support_for_means_variances_and_covariance():
    artifact = build_train_validation_artifact(
        np.array([2.5]), np.array([2.5]),
        train_samples=np.array([[1.0], [2.0], [np.nan], [4.0]]),
        validation_samples=np.array([[1.0], [np.nan], [3.0], [4.0]]),
        sample_unit="fold", sample_dependence="paired",
        train_sample_ids=("a", "b", "c", "d"),
        validation_sample_ids=("a", "b", "c", "d"), **IDENTITY,
    )
    # Only a and d are valid paired observations. Marginal n=3 is not the
    # sample underlying the paired covariance or a paired ratio interval.
    assert artifact.n_train_samples == (2,)
    assert artifact.n_validation_samples == (2,)


def test_ci_is_bound_to_the_supplied_summary_point_estimate():
    with pytest.raises(ValueError, match="summary.*sample|sample.*summary"):
        build_train_validation_artifact(
            np.array([0.8]), np.array([0.64]),
            train_samples=np.array([[0.5], [1.5]]),
            validation_samples=np.array([[1.0], [3.0]]),
            sample_unit="fold", sample_dependence="independent",
            train_sample_ids=("t1", "t2"), validation_sample_ids=("v1", "v2"),
            **IDENTITY,
        )


def test_frozen_artifact_metadata_is_deeply_immutable():
    source = {"nested": {"owner": "original"}}
    artifact = build_train_validation_artifact(
        np.array([0.8]), np.array([0.64]), metadata=source, **IDENTITY,
    )
    source["nested"]["owner"] = "changed"
    assert artifact.metadata["nested"]["owner"] == "original"
    with pytest.raises(TypeError):
        artifact.metadata["new"] = "mutation"


def test_time_samples_require_declared_temporal_uncertainty_model():
    with pytest.raises(ValueError, match="temporal|HAC|dependence"):
        build_train_validation_artifact(
            np.array([1.0]), np.array([0.8]),
            train_samples=np.array([[0.9], [1.1], [1.0]]),
            validation_samples=np.array([[0.7], [0.9], [0.8]]),
            sample_unit="time", sample_dependence="independent",
            train_sample_ids=("t1", "t2", "t3"),
            validation_sample_ids=("v1", "v2", "v3"), **IDENTITY,
        )


def test_public_artifact_requires_explicit_factor_identity_and_metric_instance():
    with pytest.raises(ValueError, match="factor.*identit"):
        build_train_validation_artifact(np.array([1.0]), np.array([0.8]))
