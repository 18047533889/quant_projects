# -*- coding: utf-8 -*-
"""P0-PLAT-004 durable orchestration state — SqliteRunStateStore + Pipeline.

Proves the durable anti-replay backing store keeps a replayed batch from
passing anti-replay across Pipeline instances (a process restart) while a
store-less pipeline keeps today's pure in-memory behavior.

* two Pipeline instances constructed against the SAME ``SqliteRunStateStore``:
  instance A consumes a batch, instance B run()'d on the same batch
  short-circuits with ``num_replayed == 1`` and ZERO new consumed;
* a fresh Pipeline against a fresh store re-processes normally;
* the store is idempotent (recording the same fingerprint twice is fine).
"""

from __future__ import annotations

import hashlib
import os
import sys
import pytest

# The quant_platform workspace is FLAT-LAYOUT: ``quant_platform/`` *is* the
# package root. Importing ``quant_platform`` requires the *parent* of this
# file's tree — ``.../quant_projects`` — on ``sys.path``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from quant_platform.app.candidate.ingest import batch_fingerprint, normalize_candidate
from quant_platform.app.db.durable_store import (
    PostgresRunStateStore,
    SqliteRunStateStore,
)
from quant_platform.app.contracts import (
    AdmissionRequest,
    AdmissionVerdict,
    DECISION_APPROVED,
    DECISION_REJECTED,
)
from quant_platform.app.orchestrator import (
    Pipeline,
    PipelineStatusReason,
    build_with_evidence,
)
from quant_platform.app.worker.jobs import JobRunner
from quant_platform.app.worker.publish import InMemoryOutbox


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class _ApproveAuthority:
    """Test authority: approves every request (domain-owned seam)."""

    def decide(self, request: AdmissionRequest) -> AdmissionVerdict:
        return AdmissionVerdict(
            decision=DECISION_APPROVED,
            reason_codes=("qualifies",),
            policy_ref="policy:test",
            authority="tests._ApproveAuthority",
            content_hash=request.content_hash,
        )


def _raw_candidate(
    *,
    seed: str,
    factor_name: str = "durable_f",
    formula: str = "sma(close, 20)",
    market: str = "ashare",
    frequency: str = "1d",
) -> dict:
    definition_ref = f"factor-definition://{factor_name}/{_h(formula)[:16]}"
    return {
        "semantic_id": _h(f"sem:{factor_name}:{formula}"),
        "content_hash": _h(seed),
        "generator_type": "trailing_sma",
        "generator_version": "1.0",
        "submitted_by": "tester",
        "market": market,
        "frequency": frequency,
        "factor_name": factor_name,
        "formula": formula,
        "factor_spec_uri": f"recipe://{factor_name}/{seed}",
        "factor_definition_ref": definition_ref,
        "semantic_ref": f"factor-semantic://{factor_name}",
        "factor_value_ref": f"factor-value://{factor_name}/{seed}",
        "formula_language": "recipe",
        "published_at": "2026-08-28T00:00:00Z",
    }


def _library_snapshot() -> dict:
    return {
        "library_version_ref": "lib:v7",
        "library_version_id": "lib:v7",
        "member_refs": [],
    }


def _evaluate(candidate, context):
    return build_with_evidence(
        {
            "candidate_ref": str(getattr(candidate, "candidate_id", "")),
            "rank_ic": 0.1,
            "evidence_status": "computed",
            "return_basis": "vwap_to_vwap",
        }
    )


def _make_pipeline(*, run_storage=None) -> Pipeline:
    return Pipeline(
        outbox=InMemoryOutbox(),
        runner=JobRunner(worker_id="durable-w"),
        evaluate_candidate=_evaluate,
        admission_authority=_ApproveAuthority(),
        run_storage=run_storage,
    )


def _batch():
    """Two genuinely NEW candidates (distinct content hashes)."""
    return [
        _raw_candidate(seed="durable-a", factor_name="durable_a", formula="sma(close, 5)"),
        _raw_candidate(seed="durable-b", factor_name="durable_b", formula="ema(close, 5)"),
    ]


def test_missing_definition_ref_never_enters_feature_manifest():
    """PL-05: the platform must not invent a definition identity."""
    raw = _raw_candidate(seed="missing-definition")
    raw.pop("factor_definition_ref")
    with pytest.raises(ValueError, match="factor_definition_ref is required"):
        _make_pipeline().run([raw], library_snapshot=_library_snapshot())


def test_pipeline_replay_short_circuits_across_instances_with_same_store():
    """Instance A consumes a batch; a fresh instance B on the SAME store
    short-circuits: num_replayed == 1, ZERO new consumed."""
    store = SqliteRunStateStore(":memory:")
    batch = _batch()

    # Instance A: fresh store -> the batch is genuinely NEW.
    pipeline_a = _make_pipeline(run_storage=store)
    first = pipeline_a.run(batch, library_snapshot=_library_snapshot())
    assert first.num_consumed == 2
    assert first.num_replayed == 0

    # The fingerprint + both consumed content hashes are durable now.
    assert store.consumed_fingerprints() == {first.batch_fingerprint}
    assert len(store.consumed_hashes()) == 2

    # Instance B: a NEW Pipeline constructed against the SAME store (a process
    # restart in production) — the replay must be caught.
    pipeline_b = _make_pipeline(run_storage=store)
    second = pipeline_b.run(batch, library_snapshot=_library_snapshot())
    assert second.num_replayed == 1
    assert second.num_consumed == 0
    assert second.num_approved == 0
    assert second.num_events_published == 0
    assert all(
        st.reason == PipelineStatusReason.QRP_REPLAY_DETECTED for st in second.states
    )
    # The store is unchanged: no new consumed rows, same fingerprints.
    assert len(store.consumed_hashes()) == 2
    assert store.consumed_fingerprints() == {first.batch_fingerprint}


def test_fresh_pipeline_with_fresh_store_reprocesses_normally():
    """A fresh Pipeline against a fresh (empty) store re-processes the same raw
    batch normally — the durable state is per-store, not global."""
    store = SqliteRunStateStore(":memory:")
    batch = _batch()
    pipeline = _make_pipeline(run_storage=store)
    report = pipeline.run(batch, library_snapshot=_library_snapshot())
    assert report.num_consumed == 2
    assert report.num_replayed == 0
    assert report.num_approved == 2
    assert len(store.consumed_hashes()) == 2


def test_store_is_idempotent_recording_same_fingerprint_twice():
    """Recording the same fingerprint / consumption twice is fine (no error,
    no duplicate rows)."""
    store = SqliteRunStateStore(":memory:")
    store.record_fingerprint("fp-1")
    store.record_fingerprint("fp-1")  # idempotent
    assert store.has_fingerprint("fp-1") is True
    assert store.consumed_fingerprints() == {"fp-1"}

    from datetime import datetime, timezone

    store.record_consumed("c1", "h1", "sem1", datetime.now(timezone.utc))
    store.record_consumed("c1", "h1", "sem1", datetime.now(timezone.utc))  # idempotent
    assert store.consumed_hashes() == {"h1"}


def test_durable_reconcile_seed_prevents_reconsume_after_restart():
    """A candidate consumed by a previous process still reconciles as a
    duplicate after restart (the known-set is seeded from the store), so it
    never re-enters the NEW path even in a different batch fingerprint."""
    store = SqliteRunStateStore(":memory:")
    batch = _batch()

    pipeline_a = _make_pipeline(run_storage=store)
    first = pipeline_a.run(batch, library_snapshot=_library_snapshot())
    assert first.num_consumed == 2

    # A NEW pipeline (restart) on the same store sees a DIFFERENT batch that
    # shares ONE candidate.  The shared candidate's content hash is in the
    # store -> reconciles as DUPLICATE_EXACT (never consumed again); the new
    # candidate is genuinely NEW.
    pipeline_b = _make_pipeline(run_storage=store)
    mixed = [
        _raw_candidate(seed="durable-a", factor_name="durable_a", formula="sma(close, 5)"),
        _raw_candidate(seed="durable-new", factor_name="durable_new", formula="rsi(9)"),
    ]
    report = pipeline_b.run(mixed, library_snapshot=_library_snapshot())
    assert report.num_replayed == 0  # different batch fingerprint
    assert report.num_consumed == 1  # only the genuinely new one
    assert report.num_duplicates == 1  # the previously consumed one
    # The store now has three consumed hashes (2 original + 1 new).
    assert len(store.consumed_hashes()) == 3


def test_postgres_run_state_store_surface_adapts_sqlite_backend():
    """PostgresRunStateStore works over the ``Db`` surface (here a SqliteDb —
    the adapter's surface is what PostgresDb.transaction() yields).  No
    psycopg2 needed for the surface contract."""
    from quant_platform.app.db.sqlite_backend import SqliteDb

    db = SqliteDb(":memory:")
    store = PostgresRunStateStore(db)
    store.record_fingerprint("fp-pg")
    assert store.has_fingerprint("fp-pg") is True
    assert store.has_fingerprint("fp-other") is False
    assert store.consumed_fingerprints() == {"fp-pg"}

    from datetime import datetime, timezone

    store.record_consumed("c1", "h1", "sem1", datetime.now(timezone.utc))
    assert store.consumed_hashes() == {"h1"}


def test_batch_fingerprint_same_store_matches_processed_batches():
    """The fingerprint recorded on a consumed batch is exactly the one reported
    by the in-memory anti-replay gate (processed_batches) for both instances."""
    store = SqliteRunStateStore(":memory:")
    batch = _batch()
    pipeline_a = _make_pipeline(run_storage=store)
    first = pipeline_a.run(batch, library_snapshot=_library_snapshot())
    assert first.batch_fingerprint in pipeline_a.processed_batches
    assert first.batch_fingerprint in store.consumed_fingerprints()

    pipeline_b = _make_pipeline(run_storage=store)
    second = pipeline_b.run(batch, library_snapshot=_library_snapshot())
    assert second.batch_fingerprint == first.batch_fingerprint
    assert first.batch_fingerprint in pipeline_b.processed_batches
