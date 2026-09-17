"""Cross-wave reuse requires a nonempty scalar snapshot identity."""
from types import SimpleNamespace
import pytest

from factor_engine.runtime.streaming_batch_service import _source_has_snapshot_proof


@pytest.mark.parametrize("value", [None, "", "  ", False, True, [], {}, (), set(),
                                   object(), lambda: "snapshot", float("nan"),
                                   float("inf"), 1.5])
def test_invalid_snapshot_markers_cannot_authorize_reuse(value):
    assert not _source_has_snapshot_proof(SimpleNamespace(generation_id=value))


@pytest.mark.parametrize("attribute", ["data_snapshot_id", "snapshot_id", "generation_id", "content_hash"])
@pytest.mark.parametrize("value", ["snapshot-1", 0, 42])
def test_explicit_string_or_integer_identity_is_supported(attribute, value):
    assert _source_has_snapshot_proof(SimpleNamespace(**{attribute: value}))


def test_invalid_marker_does_not_hide_another_valid_identity():
    assert _source_has_snapshot_proof(SimpleNamespace(snapshot_id=False, content_hash="abc123"))

@pytest.mark.parametrize("token", [None, "", " ", False, 42, object()])
def test_authoritative_token_cannot_fall_back_to_weaker_data_id(token):
    assert not _source_has_snapshot_proof(
        SimpleNamespace(snapshot_token=token, data_snapshot_id="unchanged-old-data-id")
    )


def test_authoritative_nonempty_token_is_supported():
    assert _source_has_snapshot_proof(SimpleNamespace(snapshot_token="manifest-1"))
