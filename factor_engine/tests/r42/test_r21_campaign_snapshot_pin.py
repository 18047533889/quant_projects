"""R21-CAMPAIGN-SNAPSHOT-PIN regression tests.

MiningCampaignSession must *truly pin* ExecutionContext/DataAccess: the engine's
actual read identity (source snapshot + universe pool) is asserted before every
evaluation, and all compiled/failure maps are keyed by the semantic CandidateID,
not by ``factor.name``.
"""
from __future__ import annotations

import pytest

from factor_engine.api.dsl_parser import parse_factor
from factor_engine.mining.campaign import (
    CampaignSnapshotPinError,
    MiningCampaignSession,
    MiningCampaignSnapshot,
    candidate_semantic_hash,
    source_identity_hex,
    universe_identity_hex,
)


@pytest.fixture(scope="module")
def source_cls():
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    return DataAccessSource


def _build_engine(source_cls):
    class _Engine:
        def __init__(self, data_source):
            self.data_source = data_source
            self.stream_calls = 0

        def analyze_batch(self, factors, *, enable_cse=True):
            if any(factor.name.startswith("bad") for factor in factors):
                raise ValueError("deterministic bad candidate")
            return {
                "dag": "dag:" + ",".join(factor.name for factor in factors),
                "analyses": {factor.name: object() for factor in factors},
            }

        def run_many_iter(self, factors, **kwargs):
            self.stream_calls += 1
            for factor in factors:
                yield factor.name, [factor.name], {"backend": "fake"}

    return _Engine


def _pinned_source(source_cls, *, snapshot: str = "snap-A", instruments=None):
    source = source_cls(
        dataset="daily_bars",
        instrument_filter=instruments,
    )
    source._data_snapshot_id = snapshot
    return source


def _pinned_snapshot(source) -> MiningCampaignSnapshot:
    return MiningCampaignSnapshot(
        source_identity_hex(source),
        universe_identity_hex(source),
        "compiler-gen-v1",
    )


def test_run_many_iter_pins_engine_read_identity_before_streaming(source_cls):
    source = _pinned_source(source_cls, instruments=["000001.SZ"])
    snapshot = _pinned_snapshot(source)
    engine = _build_engine(source_cls)(source)
    session = MiningCampaignSession(engine, snapshot)

    factor = parse_factor("close", name="pinned_factor")
    rows = list(session.run_many_iter([factor]))

    assert rows == [("pinned_factor", ["pinned_factor"], {"backend": "fake"})]
    assert engine.stream_calls == 1


def test_run_many_iter_hard_fails_when_engine_reads_different_snapshot(source_cls):
    pinned = _pinned_source(source_cls, snapshot="snap-A")
    snapshot = _pinned_snapshot(pinned)
    # The engine's active source now reads a *different* data snapshot.
    drifted = _pinned_source(source_cls, snapshot="snap-B")
    engine = _build_engine(source_cls)(drifted)
    session = MiningCampaignSession(engine, snapshot)

    factor = parse_factor("close", name="unpinned_factor")
    with pytest.raises(CampaignSnapshotPinError, match="source_snapshot"):
        list(session.run_many_iter([factor]))

    assert engine.stream_calls == 0


def test_compile_and_failure_maps_are_keyed_by_candidate_id_not_factor_name(source_cls):
    source = _pinned_source(source_cls)
    snapshot = _pinned_snapshot(source)
    session = MiningCampaignSession(_build_engine(source_cls)(source), snapshot)

    good = parse_factor("add(close, volume)", name="first_name")
    renamed = parse_factor("add(close, volume)", name="renamed")
    bad = parse_factor("volume", name="bad_invalid")
    same_name = parse_factor("volume", name="bad_invalid")

    result = session.compile_many([good, renamed, bad, same_name])

    # Same semantic candidate, different names -> one record under its hash.
    assert candidate_semantic_hash(good) == candidate_semantic_hash(renamed)
    assert candidate_semantic_hash(bad) == candidate_semantic_hash(same_name)
    # factor names never appear as map keys.
    for name in ("first_name", "renamed", "bad_invalid"):
        assert name not in result["analyses"]
        assert name not in result["failures"]
        assert name not in session.compiled
    # Distinct semantic candidates -> distinct CandidateID keys.
    assert candidate_semantic_hash(good) in result["analyses"]
    assert candidate_semantic_hash(bad) in result["failures"]
    assert set(result["analyses"]) == {candidate_semantic_hash(good)}
    assert set(result["failures"]) == {candidate_semantic_hash(bad)}
    # session.compiled carries (dag, analysis, factor_name) per CandidateID.
    dag, analysis, name = session.compiled[candidate_semantic_hash(good)]
    assert dag == "dag:first_name,renamed"
    assert name == "renamed"
    assert analysis is not None
