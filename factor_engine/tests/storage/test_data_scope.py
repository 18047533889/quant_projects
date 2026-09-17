# -*- coding: utf-8 -*-
"""compute_data_scope 含 params / snapshot 键。"""

from __future__ import annotations

from factor_engine.storage.data_scope import compute_data_scope


class _FakeSource:
    dataset = "factor_lake"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    params = {"factor_id": "mom_3d"}
    data_snapshot_id = "snap_abc123"


def test_compute_data_scope_includes_params_and_snapshot():
    scope_a = compute_data_scope(_FakeSource())
    src_b = _FakeSource()
    src_b.params = {"factor_id": "other"}
    scope_b = compute_data_scope(src_b)
    assert scope_a != scope_b

    src_c = _FakeSource()
    src_c.data_snapshot_id = "snap_other"
    scope_c = compute_data_scope(src_c)
    assert scope_a != scope_c


def test_compute_data_scope_tracks_manifest_token_when_snapshot_id_is_unchanged():
    source = _FakeSource()
    source.snapshot_token = "manifest:v1:p1"
    first = compute_data_scope(source)

    source.snapshot_token = "manifest:v1:p2"
    assert source.data_snapshot_id == "snap_abc123"
    assert compute_data_scope(source) != first


def test_unknown_snapshot_token_is_not_stable_identity_evidence():
    class UnknownSnapshotSource:
        snapshot_token = None

    first = UnknownSnapshotSource()
    second = UnknownSnapshotSource()
    assert compute_data_scope(first) == f"ephemeral:{id(first)}"
    assert compute_data_scope(second) == f"ephemeral:{id(second)}"

    # Composite snapshot state without verification is not a public token.
    first.snapshot_token = type(
        "Unverified", (), {"version": "manifest:v1:p1", "verified": False}
    )()
    assert compute_data_scope(first) == f"ephemeral:{id(first)}"
