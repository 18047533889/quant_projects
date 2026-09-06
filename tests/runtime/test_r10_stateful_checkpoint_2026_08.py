# -*- coding: utf-8 -*-
"""Review-10 checkpoint identity binding (R10-P0-019).

A checkpoint must be bound to the DATA SNAPSHOT it was computed on — not just
the formula.  ``StateCheckpoint.input_fingerprint`` already fingerprints
``input_identity`` and ``require_for_segment`` re-checks it on resume; the R10
fix adds the window-independent source-snapshot scope into the identity so a
checkpoint written against an older dataset / read-mode / snapshot is rejected
(fail-closed to full replay) instead of resuming stale state.
"""
from __future__ import annotations

import pytest

from factor_engine.stateful_contract import StateCheckpoint, StatefulCheckpointRegistry, StatefulContractError


def test_source_snapshot_scope_requires_trusted_identity():
    from factor_engine.runtime.stateful_incremental import (
        SourceSnapshotIdentityUnavailableError,
        _source_snapshot_scope,
    )

    class S:
        def __init__(self, identity, start, end):
            self.source_identity = identity
            self.start_date = start
            self.end_date = end

    a = S("ashare-snapshot-v1", "2024-01-01", "2024-03-01")
    b = S("ashare-snapshot-v1", "2024-02-01", "2024-04-01")
    assert _source_snapshot_scope(a) == _source_snapshot_scope(b)
    c = S("ashare-snapshot-v2", "2024-01-01", "2024-03-01")
    assert _source_snapshot_scope(a) != _source_snapshot_scope(c)

    class Anonymous:
        start_date = "2024-01-01"
        end_date = "2024-03-01"

    anonymous = Anonymous()
    assert _source_snapshot_scope(anonymous) != _source_snapshot_scope(anonymous)
    with pytest.raises(SourceSnapshotIdentityUnavailableError):
        _source_snapshot_scope(anonymous, mode="production")


def test_checkpoint_rejected_when_source_snapshot_changes():
    spec = StatefulCheckpointRegistry.get("ts_ema")
    if spec is None:
        pytest.skip("ts_ema checkpoint spec not registered")
    identity_a = {
        "factor_id": "f",
        "canonical": "ts_ema",
        "input_columns": ["close"],
        "params": {},
        "source_snapshot_scope": "scope_dataset_a",
    }
    identity_b = {
        "factor_id": "f",
        "canonical": "ts_ema",
        "input_columns": ["close"],
        "params": {},
        "source_snapshot_scope": "scope_dataset_b",
    }
    state = {"ema": {"weighted_avg": 1.0, "old_wt": 1.0, "valid_count": 1}, "last_timestamp": "2024-01-02T00:00:00+00:00"}
    cp = StatefulCheckpointRegistry.create_checkpoint(
        "ts_ema", instrument="A", as_of="2024-01-01T00:00:00+00:00",
        state=state, input_identity=identity_a,
    )
    # same snapshot -> valid
    StatefulCheckpointRegistry.validate(cp, input_identity=identity_a)
    # a different data snapshot -> fingerprint mismatch -> fail closed
    with pytest.raises(StatefulContractError, match="fingerprint"):
        StatefulCheckpointRegistry.validate(cp, input_identity=identity_b)


def test_fingerprint_distinguishes_source_scope_key():
    a = {"source_snapshot_scope": "scope_a"}
    b = {"source_snapshot_scope": "scope_b"}
    assert StatefulCheckpointRegistry.fingerprint(a) != StatefulCheckpointRegistry.fingerprint(b)
