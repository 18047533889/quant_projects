from factor_assets.contracts.fingerprint import ANNIndexArtifact, ANNIndexCapability, SimilarityFingerprintArtifact


def _fp(**changes):
    values = dict(
        factor_id="F1", embedding=(1.0, 0.0), embedding_spec="daily-rank",
        snapshot="s", universe="u", window="w", preprocessing_ref="winsor-v1",
        mask_policy="pairwise-valid", direction="signed", aggregation_method="fisher-z-n",
        embedding_model_version="embed-2.0", value_ref="bundle-content-1",
        profile_ref="profile-content-1",
    )
    values.update(changes)
    return SimilarityFingerprintArtifact(**values)


def test_fingerprint_identity_binds_preprocessing_and_aggregation():
    base = _fp()
    assert base.production_spec_complete
    assert _fp(preprocessing_ref="winsor-v2").content_hash != base.content_hash
    assert _fp(aggregation_method="flat-correlation").content_hash != base.content_hash
    assert _fp(value_ref="bundle-content-2").content_hash != base.content_hash
    assert _fp(profile_ref="profile-content-2").content_hash != base.content_hash


def test_ann_certification_requires_near_duplicate_recall_and_membership_identity():
    weak = ANNIndexArtifact(
        index_id="i", backend="annoy", capability=ANNIndexCapability.APPROXIMATE,
        index_params={}, member_set_hash="members", embedding_spec_hash="spec",
        near_duplicate_recall=0.7, audit_query_count=100,
    )
    assert not weak.certified_for_recall()
    strong = ANNIndexArtifact(
        index_id="i2", backend="annoy", capability=ANNIndexCapability.APPROXIMATE,
        index_params={}, member_set_hash="members", embedding_spec_hash="spec",
        near_duplicate_recall=0.99, audit_query_count=100,
    )
    assert strong.certified_for_recall()
