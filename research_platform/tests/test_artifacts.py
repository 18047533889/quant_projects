from research_platform import artifacts as A
from datetime import datetime, timezone

TZ = timezone.utc


def _snapshot(**kw):
    kw.setdefault("artifact_id", "snap-1")
    kw.setdefault("artifact_type", "data_snapshot")
    kw.setdefault("asof", datetime(2024, 1, 1, tzinfo=TZ))
    return A.DataSnapshotArtifact(**kw)


def test_snapshot_content_hash_stable():
    s1 = _snapshot().with_hash()
    s2 = _snapshot().with_hash()
    assert s1.content_hash == s2.content_hash
    assert len(s1.content_hash) == 64


def test_hash_differs_when_semantics_change():
    s1 = _snapshot(field_list=("a", "b")).with_hash()
    s2 = _snapshot(field_list=("a",)).with_hash()
    assert s1.content_hash != s2.content_hash


def test_roundtrip():
    s = _snapshot(snapshot_ref="snap-ref-1", field_list=("a", "b"),
                  instrument_count=100).with_hash()
    d = s.to_dict()
    s2 = A.DataSnapshotArtifact.from_dict(d)
    assert s2.to_dict() == s.to_dict()
    assert s2.content_hash == s.content_hash
    assert s2.asof == s.asof


def test_hash_excludes_id():
    s1 = _snapshot(artifact_id="a", snapshot_ref="S").with_hash()
    s2 = _snapshot(artifact_id="b", snapshot_ref="S").with_hash()
    assert s1.content_hash == s2.content_hash


def test_naive_datetime_rejected():
    import pytest
    with pytest.raises(ValueError):
        _snapshot(asof=datetime(2024, 1, 1))  # naive


def test_all_registered_types_exist():
    names = [
        "DataSnapshotArtifact", "FactorDefinitionArtifact",
        "FactorValueArtifact", "EvaluationBundle", "SimilarityArtifact",
        "AdmissionDecisionArtifact", "FactorSetArtifact", "FeatureBundle",
        "ModelDatasetArtifact", "ModelTrainingArtifact", "BacktestArtifact",
        "PortfolioArtifact", "RiskSnapshotArtifact", "ExpectedReturnArtifact",
    ]
    for n in names:
        assert n in A.ARTIFACT_TYPES
