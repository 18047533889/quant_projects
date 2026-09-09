import numpy as np
import pytest

from data_access.core.missingness import MissingReason, MissingReasonPlane
from factor_assets.adapters.fingerprint import assign_preprocessed_fingerprint, build_similarity_fingerprint
from factor_preprocess.contracts.factor_profile import FactorProfileArtifact
from factor_preprocess.contracts.feature_bundle import AxisRef, ChannelRef, FeatureBundle


class DailyMoments:
    version = "moments-1"

    def embed(self, values, validity_mask):
        sample = values[validity_mask]
        return (float(sample.mean()), float(sample.std()))


class ConstantEmbedding:
    version = "constant-1"

    def embed(self, values, validity_mask):
        return (1.0, 0.0)


def inputs(*, bundle_id="bundle-1", config_hash="config-1", profile_source="price",
           value_offset=0.0, exclude_last=False, time_offset=0):
    signal = np.arange(12, dtype=float).reshape(3, 4, 1) + value_offset
    values = np.concatenate((signal, np.zeros_like(signal)), axis=-1)
    plane = MissingReasonPlane(
        reasons=np.full(signal.shape, MissingReason.OBSERVED.value),
        original_missing=np.zeros(signal.shape, dtype=bool),
        filled=np.zeros(signal.shape, dtype=bool),
        usable=np.asarray([True] * 11 + [not exclude_last], dtype=bool).reshape(signal.shape),
        age=np.zeros(signal.shape),
    )
    bundle = FeatureBundle(
        bundle_id=bundle_id, time_axis=AxisRef("time", np.arange(3) + time_offset, "int64"),
        asset_axis=AxisRef("asset", np.asarray(["a", "b", "c", "d"]), "str"),
        channels={"signal": ChannelRef("signal", "feature", ("F1",)),
                  "missing": ChannelRef("missing", "missing", ("F1_missing",))},
        values=values, layout="TNF", source_factor_ids=("F1",),
        has_missing_channel=True, missing_reason_plane=plane,
        fitted_state_refs=("winsor-state-1",), policy_id="rank-winsor-v1",
        config_hash=config_hash,
    )
    profile = FactorProfileArtifact(
        factor_id="F1", factor_version="v1", semantic_family="PRICE_VOLUME",
        source_type=profile_source, update_frequency="daily", natural_horizon=5,
        snapshot_ref="snapshot-1", universe_ref="ASHARE", split_ref="train-1",
    )
    return bundle, profile


def build(bundle, profile, **changes):
    params = dict(
        embedding_model=DailyMoments(), embedding_spec="daily-moments",
        mask_policy="authoritative-usable-and-finite", direction="signed",
        aggregation_method="daily-cross-section-then-time-mean",
        embedding_model_version="moments-1", window="2026H1",
    )
    params.update(changes)
    return build_similarity_fingerprint(bundle, profile, **params)


def test_real_feature_bundle_and_profile_produce_complete_fingerprint():
    bundle, profile = inputs()
    fingerprint = build(bundle, profile)
    assert fingerprint.production_spec_complete
    assert fingerprint.factor_id == "F1"
    assert fingerprint.profile_ref == profile.content_hash
    assert fingerprint.embedding == pytest.approx((5.5, np.std(np.arange(12))))


def test_parameter_and_source_evidence_change_fingerprint_identity():
    bundle, profile = inputs()
    base = build(bundle, profile)
    assert build(bundle, profile, aggregation_method="flat-correlation").content_hash != base.content_hash
    assert build(bundle, profile, direction="absolute").content_hash != base.content_hash
    changed_bundle, _ = inputs(bundle_id="bundle-2")
    assert build(changed_bundle, profile).content_hash != base.content_hash
    _, changed_profile = inputs(profile_source="fundamental")
    assert build(bundle, changed_profile).content_hash != base.content_hash
    assert build(bundle, profile, direction="absolute").embedding == base.embedding


def test_value_ref_binds_actual_values_mask_and_axes_even_when_embedding_collides():
    base_bundle, profile = inputs()
    params = dict(embedding_model=ConstantEmbedding(), embedding_model_version="constant-1")
    base = build(base_bundle, profile, **params)
    changed_values = build(inputs(value_offset=1.0)[0], profile, **params)
    changed_mask = build(inputs(exclude_last=True)[0], profile, **params)
    changed_axis = build(inputs(time_offset=1)[0], profile, **params)
    assert base.embedding == changed_values.embedding == changed_mask.embedding == changed_axis.embedding
    assert len({base.value_ref, changed_values.value_ref, changed_mask.value_ref,
                changed_axis.value_ref}) == 4
    assert len({base.content_hash, changed_values.content_hash, changed_mask.content_hash,
                changed_axis.content_hash}) == 4


def test_public_assignment_entry_builds_then_feeds_incremental(monkeypatch):
    bundle, profile = inputs()
    captured = {}

    def fake_incremental(fingerprints, cluster_versions, fingerprints_by_id, **options):
        captured.update(fingerprint=fingerprints[0], map=fingerprints_by_id, options=options)
        return "assigned"

    monkeypatch.setattr("factor_assets.clustering.incremental.incremental_assign", fake_incremental)
    result = assign_preprocessed_fingerprint(
        bundle, profile, embedding_model=DailyMoments(), cluster_versions={"c": object()},
        fingerprints_by_id={}, embedding_spec="daily-moments",
        mask_policy="authoritative-usable-and-finite", direction="signed",
        aggregation_method="daily-cross-section-then-time-mean",
        embedding_model_version="moments-1", window="2026H1", batch_id="b1",
    )
    assert result == "assigned"
    assert captured["fingerprint"].production_spec_complete
    assert captured["map"]["F1"] is captured["fingerprint"]
    assert captured["options"] == {"batch_id": "b1"}


def test_model_version_and_authoritative_mask_fail_closed():
    bundle, profile = inputs()
    with pytest.raises(ValueError, match="version"):
        build(bundle, profile, embedding_model_version="wrong")
    class MissingPlaneBundle:
        missing_reason_plane = None

        def __getattr__(self, name):
            return getattr(bundle, name)

    incomplete = MissingPlaneBundle()
    with pytest.raises(ValueError, match="usable mask"):
        build(incomplete, profile)
