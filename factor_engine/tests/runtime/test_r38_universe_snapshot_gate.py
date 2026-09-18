from dataclasses import replace
from datetime import datetime

import pandas as pd
from data_access.r30.universe_snapshot import UniverseSnapshot
from factor_engine.planner.dag import FactorExecutionScope
from factor_engine.runtime.engine import _validated_universe_snapshot


MEMBERS = ("000001.SZ", "000002.SZ")
SCOPE = FactorExecutionScope(universe_id="CSI300", market="A")


def _snapshot(**overrides):
    values = dict(
        universe_id="CSI300",
        market="A",
        members=MEMBERS,
        membership_policy_version="membership-v1",
        tradability_policy_version="tradability-v1",
        source_snapshot="audited-source-v1",
        effective_start="2024-01-01",
        effective_end="2024-12-31",
    )
    values.update(overrides)
    return UniverseSnapshot.build(**values)


def _source(snapshot, **overrides):
    values = dict(
        universe_snapshot_identity=snapshot,
        instrument_filter=MEMBERS,
        start_date="2024-02-01",
        end_date="2024-11-30",
    )
    values.update(overrides)
    return type("Source", (), values)()


def test_official_interval_snapshot_covers_request_and_exact_members():
    assert _validated_universe_snapshot(_source(_snapshot()), SCOPE)


def test_snapshot_without_effective_interval_is_rejected():
    legacy = UniverseSnapshot.build(
        "CSI300", "A", MEMBERS, "membership-v1", "tradability-v1", "audited-source-v1"
    )
    assert not _validated_universe_snapshot(_source(legacy), SCOPE)


def test_request_outside_effective_interval_is_rejected():
    source = _source(_snapshot(), start_date="2023-12-31")
    assert not _validated_universe_snapshot(source, SCOPE)


def test_forged_digest_is_rejected():
    forged = replace(_snapshot(), snapshot_id="forged")
    assert not _validated_universe_snapshot(_source(forged), SCOPE)


def test_members_must_match_snapshot_exactly():
    source = _source(_snapshot(), instrument_filter=("000001.SZ",))
    assert not _validated_universe_snapshot(source, SCOPE)


def test_timezone_interval_can_cover_date_request():
    snapshot = _snapshot(
        effective_start="2023-12-31T16:00:00-08:00",
        effective_end="2025-01-01T00:00:00Z",
    )
    assert _validated_universe_snapshot(_source(snapshot), SCOPE)


def test_reversed_request_interval_is_rejected():
    source = _source(_snapshot(), start_date="2024-12-01", end_date="2024-02-01")
    assert not _validated_universe_snapshot(source, SCOPE)


def test_nat_request_bound_is_rejected():
    source = _source(_snapshot(), start_date=pd.NaT)
    assert not _validated_universe_snapshot(source, SCOPE)


def test_naive_request_datetime_is_rejected():
    source = _source(_snapshot(), start_date=datetime(2024, 2, 1))
    assert not _validated_universe_snapshot(source, SCOPE)
