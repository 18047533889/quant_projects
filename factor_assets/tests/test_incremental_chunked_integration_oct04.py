"""Public exact incremental recall avoids full oversized member matrices."""
import numpy as np

import factor_assets.clustering.incremental as incremental
from factor_assets.contracts.cluster_governance import ClusterScale, ClusterVersionArtifact
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact


def _fingerprint(fid, vector):
    return SimilarityFingerprintArtifact(
        factor_id=fid, embedding=tuple(vector), embedding_spec='unit-test-v1',
        snapshot='snapshot', universe='universe', window='window',
        preprocessing_ref='prep', mask_policy='pairwise-valid', direction='signed',
        aggregation_method='daily-then-time', embedding_model_version='model',
        value_ref=f'values:{fid}', profile_ref=f'profile:{fid}',
    )


def test_oversized_public_recall_matches_full_reference_without_full_preparation(monkeypatch):
    ids = tuple(f'm-{i:04d}' for i in range(600))
    members = {fid: _fingerprint(fid, (1., 0.) if i % 2 == 0 else (0., 1.))
               for i, fid in enumerate(ids)}
    queries = [_fingerprint('q-a', (1., 0.)), _fingerprint('q-b', (0., 1.))]
    mapping = {**members, **{q.factor_id: q for q in queries}}
    clusters = {'cluster': ClusterVersionArtifact(
        logical_cluster_id='cluster', cluster_set_version_ref='set',
        member_factor_ids=ids, representative_factor_id=ids[0],
        scale=ClusterScale.MICRO_CLUSTER,
    )}
    reference = incremental.incremental_assign(queries, clusters, mapping)
    observed = []
    original_normalize = incremental._normalize_rows

    def bounded_normalize(matrix):
        observed.append(matrix.shape[0])
        assert matrix.shape[0] <= 256
        return original_normalize(matrix)

    def forbidden_full_preparation(*args, **kwargs):
        raise AssertionError('oversized recall materialized the full member matrix')

    monkeypatch.setattr(incremental, '_EXACT_MEMBER_CACHE_BYTES', 1)
    monkeypatch.setattr(incremental, '_normalize_rows', bounded_normalize)
    monkeypatch.setattr(incremental, '_prepare_unit_members', forbidden_full_preparation)
    actual = incremental.incremental_assign(queries, clusters, mapping)
    assert actual.assignments == reference.assignments
    assert actual.candidates == reference.candidates
    assert len(observed) >= 8  # two query normalizations + three chunks per query
    assert max(observed) == 256
