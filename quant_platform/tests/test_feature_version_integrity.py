"""Content integrity and immutability before migrating version hash codecs."""
from dataclasses import asdict
import json
import pytest
from quant_platform.app.contracts.feature_set import FeatureMemberRef, FeatureSetVersion


def member(**kwargs):
    return FeatureMemberRef(0, "x", "definition:x", **kwargs)


@pytest.mark.parametrize("field", ["schema_hash", "semantic_hash"])
def test_version_rejects_forged_carried_hash(field):
    with pytest.raises(ValueError, match=field):
        FeatureSetVersion("set", "v1", (member(),), **{field: "0"*64})


def test_version_rejects_changed_members_with_original_hashes():
    old = FeatureSetVersion("set", "v1", (member(),))
    with pytest.raises(ValueError):
        FeatureSetVersion("set", "v1", (FeatureMemberRef(0,"y","definition:y"),),
                          schema_hash=old.schema_hash, semantic_hash=old.semantic_hash)


def test_version_detaches_lists_and_freezes_nested_member_metadata():
    members = [member(metadata={"audit":{"refs":["ref:1"]}})]
    libs = ["lib:1"]
    version = FeatureSetVersion("set","v1",members,source_library_versions=libs)
    members.append(FeatureMemberRef(1,"y","definition:y"))
    libs.append("lib:2")
    assert len(version.ordered_members) == 1
    assert version.source_library_versions == ("lib:1",)
    with pytest.raises(TypeError):
        version.ordered_members[0].metadata["audit"]["refs"] = ()
    with pytest.raises((TypeError, AttributeError)):
        version.ordered_members[0].metadata["audit"]["refs"].append("bad")


def test_existing_version_hashes_roundtrip_with_nested_json_metadata():
    old = FeatureSetVersion("set","v1",(member(metadata={"audit":{"refs":["ref:1"]}}),))
    # Same JSON path as durable generation (the only non-JSON container is
    # the immutable mapping facade).
    from collections.abc import Mapping
    wire = json.loads(json.dumps(asdict(old), default=lambda x: dict(x) if isinstance(x,Mapping) else x))
    wire["ordered_members"] = tuple(FeatureMemberRef(**m) for m in wire["ordered_members"])
    restored = FeatureSetVersion(**wire)
    assert restored == old


@pytest.mark.parametrize("value", [None, False, 0, []])
def test_invalid_falsy_hash_is_not_interpreted_as_missing(value):
    with pytest.raises(ValueError, match="schema_hash"):
        FeatureSetVersion("set","v1",(member(),),schema_hash=value)


def test_opaque_mutable_metadata_cannot_escape_the_container_freezer():
    from dataclasses import dataclass
    @dataclass
    class MutableProvenance:
        revision: int = 1
    with pytest.raises(TypeError, match="immutable scalars"):
        member(metadata={"source": MutableProvenance()})
