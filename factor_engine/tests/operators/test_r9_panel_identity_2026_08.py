# -*- coding: utf-8 -*-
"""R9-P0-014 regression tests: ``PanelIdentity`` equality must reject a KNOWN
grain/frequency mismatch (daily vs minute sharing an index by accident must
loud-fail) while UNKNOWN metadata enters compatibility resolution — and the
hash must stay consistent with equality and deterministic."""
from __future__ import annotations

from cleaned_operators.common._polars_bridge import PanelIdentity


def test_known_metadata_match_is_equal():
    a = PanelIdentity(time_index_hash=1, instrument_axis_hash=10, grain="1d", frequency="daily")
    b = PanelIdentity(time_index_hash=1, instrument_axis_hash=10, grain="1d", frequency="daily")
    assert a == b


def test_known_grain_mismatch_rejects():
    daily = PanelIdentity(1, 10, "1d", "daily")
    minute = PanelIdentity(1, 10, "min", "minute")
    assert daily != minute  # same index+axis by accident -> NOT the same identity


def test_known_frequency_mismatch_rejects():
    a = PanelIdentity(1, 10, "1d", "daily")
    b = PanelIdentity(1, 10, "1d", "weekly")
    assert a != b


def test_unknown_metadata_is_compatibility_resolved():
    a = PanelIdentity(1, 10, "1d", "daily")
    unk = PanelIdentity(1, 10, "unknown", "unknown")
    assert a == unk  # one side unknown -> axis equality governs


def test_axis_mismatch_always_rejects():
    a = PanelIdentity(1, 10, "1d", "daily")
    b = PanelIdentity(2, 10, "1d", "daily")
    c = PanelIdentity(1, 11, "1d", "daily")
    assert a != b and a != c


def test_hash_consistent_with_equality():
    a = PanelIdentity(1, 10, "1d", "daily")
    b = PanelIdentity(1, 10, "1d", "daily")
    assert hash(a) == hash(b)
    # equal objects must hash equal; unequal objects MAY collide but the hash
    # must be deterministic (same axes -> same hash regardless of metadata).
    assert hash(PanelIdentity(5, 7, "1d", "daily")) == hash(PanelIdentity(5, 7, "min", "minute"))
    assert len({a, b}) == 1
    assert len({a, PanelIdentity(1, 10, "min", "minute")}) == 2
