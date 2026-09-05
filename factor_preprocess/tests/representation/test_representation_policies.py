"""
R61-FI-044 — model-specific representation policy tests (plan §28).

Covers the four frozen profile semantics, the FeatureRepresentation artifact
never overwriting the canonical factor asset, and the non-inferiority conflict
guard.
"""
import pytest

from factor_preprocess.representation.policy import (
    RepresentationProfileId,
    REPRESENTATION_POLICY_VERSION,
    RepresentationPolicy,
    UnknownRepresentationProfileError,
    get_representation_policy,
    list_representation_policies,
    ArtifactKind,
    CANONICAL_FACTOR_NAMESPACE_PREFIX,
    CanonicalAssetOverwriteError,
    FeatureRepresentationArtifact,
    register_feature_representation,
    NonInferiorityTolerance,
    NON_INFERIORITY_POLICY_VERSION,
    DEFAULT_NON_INFERIORITY_TOLERANCE,
    non_inferior,
    SignalDestructionConflict,
)
from factor_preprocess.errors import InvalidContractError


# ---------------------------------------------------------------------------
# 1. the four frozen profiles
# ---------------------------------------------------------------------------


def test_four_profiles_are_registered_and_frozen():
    profiles = list_representation_policies()
    assert [p.profile_id.value for p in profiles] == [
        "TREE_TABULAR",
        "LINEAR",
        "NEURAL_TABULAR",
        "SEQUENCE_MODEL",
    ]
    for p in profiles:
        assert isinstance(p, RepresentationPolicy)
        assert p.version == REPRESENTATION_POLICY_VERSION
        # frozen dataclass
        with pytest.raises(Exception):
            p.allowed_representations = ()  # type: ignore[misc]


def test_tree_tabular_semantics():
    p = get_representation_policy(RepresentationProfileId.TREE_TABULAR)
    assert p.allows_representation("raw")
    assert p.allows_representation("rank")
    assert p.allows_representation("clipped")
    assert p.zscore_optional is True
    assert p.required_normalizations == ()
    # No normalization is required for trees.
    p.require_normalizations(())
    assert not p.train_fitted_scaling_required


def test_linear_semantics_require_train_fitted_zscore():
    p = get_representation_policy(RepresentationProfileId.LINEAR)
    assert p.robust_outlier_handling_required is True
    assert p.train_fitted_scaling_required is True
    # raw / rank-only input violates the LINEAR profile.
    with pytest.raises(InvalidContractError):
        p.require_normalizations(("raw",))
    with pytest.raises(InvalidContractError):
        p.require_normalizations(("rank",))
    # train-fitted zscore present -> ok.
    p.require_normalizations(("train_fitted_zscore", "train_fitted_robust_zscore"))
    p.require_normalizations(("train_fitted_robust_zscore", "train_fitted_zscore"))


def test_neural_tabular_semantics():
    p = get_representation_policy(RepresentationProfileId.NEURAL_TABULAR)
    assert p.missing_channel_required is True
    assert p.stable_clipping_required is True
    assert p.rank_gaussian_allowed is True
    assert p.allows_representation("robust_zscore")
    assert p.allows_representation("rank_gaussian")
    p.require_normalizations(("robust_zscore_or_rank_gaussian",))
    with pytest.raises(InvalidContractError):
        p.require_normalizations(())


def test_sequence_model_semantics():
    p = get_representation_policy(RepresentationProfileId.SEQUENCE_MODEL)
    assert p.train_fitted_scaling_required is True
    assert p.temporal_causal_normalization_required is True
    p.require_normalizations(("train_fitted_scaling",))
    with pytest.raises(InvalidContractError):
        p.require_normalizations(("rank",))


def test_unknown_profile_fails_closed():
    with pytest.raises(UnknownRepresentationProfileError):
        get_representation_policy("CNN_VISION")
    with pytest.raises(UnknownRepresentationProfileError):
        get_representation_policy(RepresentationProfileId)  # not an id


def test_profile_version_is_stable():
    for p in list_representation_policies():
        assert p.version == REPRESENTATION_POLICY_VERSION
    assert REPRESENTATION_POLICY_VERSION  # non-empty


# ---------------------------------------------------------------------------
# 2. FeatureRepresentationArtifact never overwrites canonical factor asset
# ---------------------------------------------------------------------------


def test_artifact_records_model_specific_representation():
    artifact = register_feature_representation(
        artifact_id="rep_linear_01",
        factor_id="fac_alpha_1",
        canonical_factor_ref="factor:fac_alpha_1",
        model_id="ridge_v3",
        profile_id=RepresentationProfileId.LINEAR,
        transform_chain=[{"name": "robust_zscore"}, {"name": "train_fitted_zscore"}],
        representation_name="robust_zscore_train_fitted",
    )
    assert artifact.artifact_kind is ArtifactKind.MODEL_SPECIFIC_REPRESENTATION
    assert artifact.profile_version == REPRESENTATION_POLICY_VERSION
    assert artifact.content_hash
    d = artifact.to_dict()
    assert d["model_id"] == "ridge_v3"
    assert d["profile_id"] == "LINEAR"


def test_artifact_canonical_ref_must_point_at_factor_asset():
    # The canonical_factor_ref is the REFERENCE to the canonical alpha asset;
    # the artifact itself is separate. A ref that IS a canonical-asset
    # identity must fail closed even at the artifact level.
    with pytest.raises(InvalidContractError):
        FeatureRepresentationArtifact(
            artifact_id="bad",
            factor_id="f",
            canonical_factor_ref="factor_asset:fac_alpha_1",
            model_id="m",
            profile_id=RepresentationProfileId.TREE_TABULAR,
            profile_version=REPRESENTATION_POLICY_VERSION,
        )


def test_register_forbids_overwriting_canonical_factor_asset():
    with pytest.raises(CanonicalAssetOverwriteError):
        register_feature_representation(
            artifact_id="rep_bad",
            factor_id="fac_alpha_1",
            canonical_factor_ref=f"{CANONICAL_FACTOR_NAMESPACE_PREFIX}fac_alpha_1",
            model_id="ridge_v3",
            profile_id=RepresentationProfileId.LINEAR,
            transform_chain=[{"name": "zscore"}],
        )


def test_model_representation_may_differ_from_alpha_winner():
    # The plan explicitly allows a representation that is NOT the alpha winner
    # only when tracked as a model-specific artifact. The artifact carries the
    # canonical factor ref (alpha winner) and its own transform chain.
    artifact = register_feature_representation(
        artifact_id="rep_tree_rank",
        factor_id="fac_alpha_1",
        canonical_factor_ref="factor:fac_alpha_1",  # canonical alpha form
        model_id="lgbm_prod",
        profile_id=RepresentationProfileId.TREE_TABULAR,
        transform_chain=[{"name": "cs_rank", "params": {"pct": True}}],
        representation_name="rank",
    )
    # Not the RAW-vs-treatment winner: it is a rank view recorded separately.
    assert artifact.representation_name == "rank"
    assert artifact.factor_id == "fac_alpha_1"
    assert artifact.canonical_factor_ref == "factor:fac_alpha_1"


def test_artifact_hash_content_derived():
    a1 = register_feature_representation(
        artifact_id="x", factor_id="f", canonical_factor_ref="factor:f",
        model_id="m", profile_id="TREE_TABULAR", transform_chain=[{"name": "rank"}],
    )
    a2 = register_feature_representation(
        artifact_id="x", factor_id="f", canonical_factor_ref="factor:f",
        model_id="m", profile_id="TREE_TABULAR", transform_chain=[{"name": "rank"}],
    )
    a3 = register_feature_representation(
        artifact_id="x", factor_id="f", canonical_factor_ref="factor:f",
        model_id="m", profile_id="TREE_TABULAR", transform_chain=[{"name": "zscore"}],
    )
    assert a1.content_hash == a2.content_hash
    assert a1.content_hash != a3.content_hash


# ---------------------------------------------------------------------------
# 3. non-inferiority guard
# ---------------------------------------------------------------------------


def test_non_inferior_equal_evidence():
    assert non_inferior(0.05, 0.05) is True


def test_non_inferior_small_destroy_within_tolerance():
    # 0.05 -> 0.048 destroys 4% <= 10% tolerance.
    assert non_inferior(0.05, 0.048) is True


def test_material_destroy_raises_conflict():
    # 0.05 -> 0.03 destroys 40% > 10% tolerance -> explicit conflict.
    with pytest.raises(SignalDestructionConflict):
        non_inferior(0.05, 0.03)


def test_sign_flip_detected_as_full_destruction():
    # 0.05 -> -0.01 destroys 120% of the alpha evidence.
    with pytest.raises(SignalDestructionConflict):
        non_inferior(0.05, -0.01)


def test_negative_alpha_normalized_to_absolute():
    # Evidence measured as a negative metric (e.g. signed direction) treats
    # the absolute magnitude as the protected signal.
    with pytest.raises(SignalDestructionConflict):
        non_inferior(-0.05, 0.02)  # abs base .05, moved to +.02: destroy 140%
    assert non_inferior(-0.05, -0.05) is True
    assert non_inferior(-0.05, -0.052) is True  # 4% destroy


def test_zero_alpha_no_signal_to_protect():
    # No positive/negative alpha evidence -> nothing destroyed -> allowed.
    assert non_inferior(0.0, 0.0) is True
    assert non_inferior(0.0, 0.01) is True


def test_custom_versioned_tolerance():
    tol = NonInferiorityTolerance(
        alpha_metric="rank_ic",
        max_relative_destroy=0.20,
        policy_version="2026-09-05.1",
    )
    assert non_inferior(0.05, 0.042, tolerance=tol) is True  # 16% <= 20%
    with pytest.raises(SignalDestructionConflict):
        non_inferior(0.05, 0.030, tolerance=tol)  # 40% > 20%


def test_tolerance_validation_fail_closed():
    with pytest.raises(InvalidContractError):
        NonInferiorityTolerance(alpha_metric="rank_ic", max_relative_destroy=1.5)
    with pytest.raises(InvalidContractError):
        NonInferiorityTolerance(alpha_metric="", max_relative_destroy=0.1)


def test_default_tolerance_versioned():
    assert DEFAULT_NON_INFERIORITY_TOLERANCE.alpha_metric == "rank_ic"
    assert DEFAULT_NON_INFERIORITY_TOLERANCE.policy_version == NON_INFERIORITY_POLICY_VERSION


def test_conflict_exception_carries_explicit_message():
    with pytest.raises(SignalDestructionConflict) as excinfo:
        non_inferior(0.05, 0.02)
    msg = str(excinfo.value)
    assert "silently forced" in msg or "conflict" in msg.lower()
