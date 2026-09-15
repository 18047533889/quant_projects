"""New feature artifacts must not inherit the ambiguous legacy hash codec."""
from dataclasses import asdict
import pytest
from quant_platform.app.contracts._contenthash import content_hash
from quant_platform.app.contracts.feature_set import FeatureSetArtifact


@pytest.mark.parametrize("left,right", [
    ({"a":"b,c=d"}, {"a":"b","c":"d"}), (["a,b","c"], ["a","b,c"]),
    (1,"1"), (True,"true"), (None,"null"), ([1],(1,)), ({"1":2},{1:2}),
])
def test_recursive_v2_frames_disambiguate_types_and_delimiters(left, right):
    from quant_platform.app.contracts._contenthash import content_hash_v2
    assert content_hash_v2(left) != content_hash_v2(right)


def test_v2_is_stable_ordered_finite_and_bounded():
    from datetime import datetime, timedelta, timezone
    from quant_platform.app.contracts._contenthash import content_hash_v2
    assert content_hash_v2({"b":[1,2],"a":0}) == content_hash_v2({"a":0,"b":[1,2]})
    assert content_hash_v2(-0.0) == content_hash_v2(0.0)
    t = datetime(2026,9,12,tzinfo=timezone.utc)
    assert content_hash_v2(t) == content_hash_v2(t.astimezone(timezone(timedelta(hours=8))))
    cycle = []
    cycle.append(cycle)
    for value in (float("nan"), float("inf"), datetime(2026,9,12), cycle):
        with pytest.raises(ValueError):
            content_hash_v2(value)
    with pytest.raises(TypeError):
        content_hash_v2(object())


def test_new_feature_artifact_separates_delimiter_collision():
    a = FeatureSetArtifact("set", "v1", ordered_feature_manifest=("a,b", "c"))
    b = FeatureSetArtifact("set", "v1", ordered_feature_manifest=("a", "b,c"))
    assert a.content_hash != b.content_hash
    assert a.hash_codec == b.hash_codec == "semantic-v2"


def test_unversioned_legacy_artifact_remains_readable_without_rewriting_hash():
    old = content_hash("set", "v1", ("lib",), ("a,b", "c"))
    restored = FeatureSetArtifact("set", "v1", ("lib",), ("a,b", "c"), content_hash=old)
    assert restored.content_hash == old
    assert restored.hash_codec == "semantic-v1"
    assert FeatureSetArtifact(**asdict(restored)) == restored


def test_new_artifact_roundtrip_and_wrong_codec_cannot_downgrade():
    current = FeatureSetArtifact("set", "v1", ordered_feature_manifest=("a,b", "c"))
    assert FeatureSetArtifact(**asdict(current)) == current
    with pytest.raises(ValueError):
        FeatureSetArtifact(**(asdict(current) | {"hash_codec": "semantic-v1"}))
    with pytest.raises(ValueError):
        FeatureSetArtifact(**(asdict(current) | {"hash_codec": "unknown"}))
    # New digests must carry the codec through manually reconstructed DTOs.
    missing_codec = asdict(current)
    missing_codec.pop("hash_codec")
    with pytest.raises(ValueError):
        FeatureSetArtifact(**missing_codec)


def test_new_artifact_rejects_tampered_bytes_and_legacy_is_not_recertified():
    current = FeatureSetArtifact("set", "v1", ordered_feature_manifest=("a,b", "c"))
    with pytest.raises(ValueError, match="does not match"):
        FeatureSetArtifact(**(asdict(current) | {"ordered_feature_manifest": ("a", "b,c")}))
    # Legacy verification cannot detect its historical collision. Its explicit
    # v1 label must remain visible, never claimed as a v2 verification.
    old = content_hash("set", "v1", (), ("a,b", "c"))
    legacy = FeatureSetArtifact("set", "v1", ordered_feature_manifest=("a", "b,c"), content_hash=old)
    assert legacy.hash_codec == "semantic-v1"


def test_artifact_copies_caller_sequences_before_hashing():
    members, sources = ["first"], ["lib"]
    artifact = FeatureSetArtifact("set", "v1", sources, members)
    before = artifact.content_hash
    members.append("tampered")
    sources.append("tampered")
    assert artifact.ordered_feature_manifest == ("first",)
    assert artifact.source_library_versions == ("lib",)
    assert artifact.recomputed_hash() == before
