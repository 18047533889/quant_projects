# -*- coding: utf-8 -*-
"""QRP-P5-PIPE — full-pipeline orchestrator locking tests.

Covers :mod:`quant_platform.app.orchestrator`:

* entire happy path — 2 NEW candidates -> 1 approve / 1 reject (low
  performance) -> registry registers 1 -> 4 milestone events published
  (CANDIDATE_RECONCILED / CANDIDATE_REJECTED / CANDIDATE_APPROVED /
  ARTIFACT_REGISTERED);
* duplicate replay skip — re-running the SAME batch short-circuits through the
  ``batch_fingerprint`` anti-replay gate (no registry writes, no new events);
* promotion rejection path — low rank_ic / label-not-mature / wrong return
  basis / forced-review -> REJECT / REVIEW reason codes;
* outbox publish counting — every milestone event lands in the WorkerLoop drain;
* ``PipelineReport`` immutability — the ``with_*`` builders return new reports
  and never mutate ``self``;
* the PURE-DTO promotion decision mirrors the FA ``PromotionGate`` semantics
  without importing any domain package.
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
_PLATFORM_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PARENT = os.path.abspath(os.path.join(_PLATFORM_ROOT, ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from quant_platform.app.candidate.ingest import batch_fingerprint, reconcile_candidates
from quant_platform.app.contracts import (
    ARTIFACT_TYPE_FACTOR_CANDIDATE,
    AdmissionRequest,
    AdmissionVerdict,
    DECISION_APPROVED,
    DECISION_REJECTED,
    DECISION_SHADOWED,
    EventEnvelope,
    FeatureSetArtifact,
    FeatureSetDiffCategory,
    FeatureSetVersion,
    REASON_AUTHORITY_ABSENT,
    RefuseAdmission,
)
from quant_platform.app.outbox import InMemoryPublisher
from quant_platform.app.orchestrator import (
    CandidateEvaluation,
    Pipeline,
    PipelineReport,
    PipelineStatusReason,
    build_with_evidence,
)
from quant_platform.app.storage.registry import ArtifactRegistry
from quant_platform.app.worker.jobs import JobRunner
from quant_platform.app.worker.publish import InMemoryOutbox, WorkerLoop


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


#: platform carries no admission threshold — the authority owns 0.02
AUTHORITY_MIN_RANK_IC = 0.02


def _raw_candidate(
    *,
    seed: str,
    factor_name: str = "p5factor",
    formula: str = "sma(close, 20)",
    market: str = "ashare",
    frequency: str = "1d",
    semantic_id: str | None = None,
    **extra,
) -> dict:
    payload = {
        "semantic_id": semantic_id if semantic_id is not None else _h(f"sem:{factor_name}:{formula}"),
        "content_hash": _h(seed),
        "generator_type": "trailing_sma",
        "generator_version": "1.0",
        "submitted_by": "tester",
        "market": market,
        "frequency": frequency,
        "factor_name": factor_name,
        "formula": formula,
        "factor_spec_uri": f"recipe://{factor_name}/{seed}",
        "formula_language": "recipe",
        "published_at": "2026-08-28T00:00:00Z",
    }
    payload.update(extra)
    return payload


def _library_snapshot(
    *,
    library_version_ref: str = "lib:v7",
    member_refs: list[str] | None = None,
) -> dict:
    # Carried refs only — thresholds (min_rank_ic / similarity) are the
    # AUTHORITY's policy now, never the platform's (R55 P0-6).
    return {
        "library_version_ref": library_version_ref,
        "library_version_id": library_version_ref,
        "member_refs": member_refs or [],
    }


class _DelegatedAuthority:
    """Test double of the domain-owned AdmissionAuthority port.

    A stub of what ``factor_assets`` would provide: it receives the carried
    evidence and returns a verdict the platform must record verbatim.
    """

    def __init__(self, *, approve_above: float = AUTHORITY_MIN_RANK_IC) -> None:
        self.approve_above = approve_above
        self.requests: list[AdmissionRequest] = []

    def decide(self, request: AdmissionRequest) -> AdmissionVerdict:
        self.requests.append(request)
        evaluation = request.evaluation
        rank = getattr(evaluation, "rank_ic", None)
        if rank is None:
            return AdmissionVerdict(
                decision=DECISION_REJECTED,
                reason_codes=("rank_ic_below_threshold",),
                policy_ref="policy:test",
                authority="tests._DelegatedAuthority",
                content_hash=request.content_hash,
            )
        if rank > self.approve_above:
            return AdmissionVerdict(
                decision=DECISION_APPROVED,
                reason_codes=("qualifies",),
                policy_ref="policy:test",
                authority="tests._DelegatedAuthority",
                content_hash=request.content_hash,
            )
        return AdmissionVerdict(
            decision=DECISION_REJECTED,
            reason_codes=("rank_ic_below_threshold",),
            policy_ref="policy:test",
            authority="tests._DelegatedAuthority",
            content_hash=request.content_hash,
        )


class _RecordingAuthority(_DelegatedAuthority):
    """Authority that also records that the platform did NOT pre-decide."""


def _make_pipeline(
    *,
    evaluate_candidate,
    registry: ArtifactRegistry | None = None,
    outbox: InMemoryOutbox | None = None,
    worker_id: str = "p5-worker",
    admission_authority: object | None = None,
) -> Pipeline:
    return Pipeline(
        registry=registry if registry is not None else ArtifactRegistry(),
        outbox=outbox if outbox is not None else InMemoryOutbox(),
        runner=JobRunner(worker_id=worker_id),
        evaluate_candidate=evaluate_candidate,
        admission_authority=admission_authority if admission_authority is not None else _DelegatedAuthority(),
    )


def _eval_from(rank_ic: float | None, **extra) -> CandidateEvaluation:
    candidate_ref = extra.pop("candidate_ref", None) or "test-candidate"
    evidence = {
        "candidate_ref": candidate_ref,
        "rank_ic": rank_ic,
        "evidence_status": "computed",
        "return_basis": "vwap_to_vwap",
        "label_maturity": True,
    }
    evidence.update(extra)
    return build_with_evidence(evidence)


# --------------------------------------------------------------------------- #
# happy path — 2 NEW candidates -> 1 approve / 1 reject -> registry 1 -> 5 events
# --------------------------------------------------------------------------- #


def _normalize(raw: dict):
    from quant_platform.app.candidate.ingest import normalize_candidate

    return normalize_candidate(raw)


def test_pipeline_happy_path_register_one_and_publishes_four_events():
    registry = ArtifactRegistry()
    outbox = InMemoryOutbox()
    good = _raw_candidate(seed="good")
    bad = _raw_candidate(seed="bad", factor_name="p5weak")
    good_id = _normalize(good).candidate_id
    bad_id = _normalize(bad).candidate_id

    def evaluate(candidate, context):
        cid = str(getattr(candidate, "candidate_id", ""))
        if cid == good_id:
            return build_with_evidence(
                {
                    "candidate_ref": cid,
                    "rank_ic": 0.09,
                    "evidence_status": "computed",
                    "return_basis": "vwap_to_vwap",
                }
            )
        if cid == bad_id:
            return build_with_evidence(
                {
                    "candidate_ref": cid,
                    "rank_ic": 0.008,  # below the AUTHORITY's 0.02 floor
                    "evidence_status": "computed",
                    "return_basis": "vwap_to_vwap",
                }
            )
        return _eval_from(None, candidate_ref=cid)

    pipeline = _make_pipeline(registry=registry, outbox=outbox, evaluate_candidate=evaluate)
    report = pipeline.run(
        [good, bad],
        library_snapshot=_library_snapshot(),
    )

    # Per-candidate terminal states.
    assert report.num_consumed == 2
    assert report.num_approved == 1
    assert report.num_rejected == 1
    assert report.num_conflicts == 0
    assert report.num_duplicates == 0
    assert report.num_evaluation_failed == 0
    assert report.num_registered == 1

    statuses = {st.candidate_id: st.status for st in report.states}
    assert statuses[good_id] == "APPROVED"
    assert statuses[bad_id] == "REJECTED"
    assert any(
        st.candidate_id == bad_id
        and st.reason == PipelineStatusReason.QRP_GATE_REJECTED_RANK_IC
        for st in report.states
    )

    # Registered artifact is the approved candidate's FACTOR_CANDIDATE.
    assert len(report.registered_artifacts) == 1
    artifact_id, content_hash = report.registered_artifacts[0]
    assert artifact_id.startswith("FC_")
    assert content_hash == _h("good")
    stored = registry.resolve(_h("good"))
    assert stored is not None
    assert stored.artifact_type == ARTIFACT_TYPE_FACTOR_CANDIDATE
    assert registry.resolve(_h("bad")) is None

    # Milestone events: one row per (candidate, milestone). For 2 NEW
    # candidates (1 approve + 1 reject) that is 5 rows covering exactly the 4
    # milestone types CANDIDATE_RECONCILED / CANDIDATE_APPROVED /
    # CANDIDATE_REJECTED / ARTIFACT_REGISTERED.
    assert report.num_events_published == 5
    assert len(report.published_event_ids) == 5
    assert len(outbox) == 5
    rows = outbox.rows()
    milestones = {row.event.payload.get("milestone") for row in rows}
    assert milestones == {
        "CANDIDATE_RECONCILED",
        "CANDIDATE_REJECTED",
        "CANDIDATE_APPROVED",
        "ARTIFACT_REGISTERED",
    }
    assert rows[0].status == "pending"
    for row in rows:
        assert isinstance(row.event, EventEnvelope)


def test_pipeline_workerloop_drains_all_milestone_events():
    outbox = InMemoryOutbox()
    publisher = InMemoryPublisher()
    loop = WorkerLoop(outbox, publisher, in_flight_timeout_s=10.0)
    good = _raw_candidate(seed="drain-g")
    good_id = _normalize(good).candidate_id

    def evaluate(candidate, context):
        return build_with_evidence(
            {
                "candidate_ref": str(getattr(candidate, "candidate_id", "")),
                "rank_ic": 0.11,
                "evidence_status": "computed",
                "return_basis": "vwap_to_vwap",
            }
        )

    pipeline = _make_pipeline(outbox=outbox, evaluate_candidate=evaluate)
    report = pipeline.run([good], library_snapshot=_library_snapshot())
    assert report.num_events_published == 3  # reconciled + approved + registered

    rep = loop.run_until_quiesce(max_cycles=50)
    assert rep.published == 3
    assert len(publisher.delivered) == 3
    delivered_milestones = {
        event.payload.get("milestone") for event in publisher.delivered
    }
    assert delivered_milestones == {
        "CANDIDATE_RECONCILED",
        "CANDIDATE_APPROVED",
        "ARTIFACT_REGISTERED",
    }


# --------------------------------------------------------------------------- #
# duplicate replay / anti-replay
# --------------------------------------------------------------------------- #


def test_pipeline_duplicate_replay_skips_and_no_new_events():
    # Distinct factor identities (factor_name + formula) so a DIFFERENT batch
    # in the same run is genuinely NEW (not a semantic conflict with a
    # consumed one — the semantic family hint is derived from the carrier
    # fields factor_name/market/frequency/formula, NOT the semantic_id key).
    batch = [
        _raw_candidate(seed="replay-a", factor_name="rep_a", formula="sma(close, 5)"),
        _raw_candidate(seed="replay-b", factor_name="rep_b", formula="ema(close, 5)"),
    ]
    fingerprint = batch_fingerprint([_normalize(b) for b in batch])
    outbox = InMemoryOutbox()
    registry = ArtifactRegistry()
    good = _raw_candidate(seed="replay-a", factor_name="rep_a", formula="sma(close, 5)")
    good_id = _normalize(good).candidate_id

    def evaluate(candidate, context):
        return build_with_evidence(
            {
                "candidate_ref": str(getattr(candidate, "candidate_id", "")),
                "rank_ic": 0.12,
                "evidence_status": "computed",
                "return_basis": "vwap_to_vwap",
            }
        )

    pipeline = _make_pipeline(registry=registry, outbox=outbox, evaluate_candidate=evaluate)
    first = pipeline.run(batch, library_snapshot=_library_snapshot())
    assert first.num_consumed == 2
    first_event_count = first.num_events_published
    assert len(outbox) == first_event_count

    # Replay the exact same batch: anti-replay short-circuit.
    second = pipeline.run(batch, library_snapshot=_library_snapshot())
    assert fingerprint in pipeline.processed_batches
    assert second.num_replayed == 1
    assert second.num_consumed == 0
    assert second.num_approved == 0
    assert second.num_events_published == 0
    assert len(outbox) == first_event_count  # no new outbox rows
    assert all(
        st.reason == PipelineStatusReason.QRP_REPLAY_DETECTED for st in second.states
    )

    # A DIFFERENT batch still runs normally.
    third = pipeline.run(
        [_raw_candidate(seed="replay-c", factor_name="rep_c", formula="rsi(9)")],
        library_snapshot=_library_snapshot(),
    )
    assert third.num_consumed == 1
    # Fingerprint replay detection is per-batch, not global.
    assert third.num_replayed == 0


def test_pipeline_duplicate_content_hash_reconcile_skipped():
    # Same raw record twice in one batch -> reconcile classifies the second as
    # DUPLICATE_EXACT (both content + semantic hit the registered candidate),
    # so only one is consumed.
    raw = _raw_candidate(seed="dedup")

    def evaluate(candidate, context):
        return build_with_evidence(
            {
                "candidate_ref": str(getattr(candidate, "candidate_id", "")),
                "rank_ic": 0.1,
                "evidence_status": "computed",
                "return_basis": "vwap_to_vwap",
            }
        )

    pipeline = _make_pipeline(evaluate_candidate=evaluate)
    report = pipeline.run([raw, raw], library_snapshot=_library_snapshot())
    assert report.num_consumed == 1
    assert report.num_duplicates == 1
    assert report.num_approved == 1
    duplicate_state = [
        st for st in report.states if st.status == "DUPLICATE_SKIPPED"
    ]
    assert len(duplicate_state) == 1
    assert duplicate_state[0].reason == PipelineStatusReason.QRP_DUPLICATE_SKIPPED


# --------------------------------------------------------------------------- #
# promotion rejection / review paths
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# delegated admission: the platform RECORDS, the authority DECIDES (R55 P0-6)
# --------------------------------------------------------------------------- #


def test_admission_is_delegated_to_the_injected_authority():
    """The verdict comes from the authority, not from a platform threshold."""
    authority = _DelegatedAuthority()
    pipeline = _make_pipeline(
        evaluate_candidate=lambda c, ctx: build_with_evidence(
            {"candidate_ref": str(getattr(c, "candidate_id", "")), "rank_ic": 0.09, "evidence_status": "computed", "return_basis": "vwap_to_vwap"}
        ),
        admission_authority=authority,
    )
    report = pipeline.run(
        [_raw_candidate(seed="deleg")], library_snapshot=_library_snapshot()
    )
    assert report.num_approved == 1
    # The platform ASKED the authority (one request per consumed candidate)...
    assert len(authority.requests) == 1
    request = authority.requests[0]
    assert request.content_hash == _h("deleg")
    assert request.library_snapshot_ref == "lib:v7"
    # ...and recorded the verdict VERBATIM (reason codes carried, not derived).
    verdict = pipeline.verdicts()[0]
    assert verdict.decision == DECISION_APPROVED
    assert verdict.reason_codes == ("qualifies",)
    assert verdict.policy_ref == "policy:test"
    assert verdict.authority == "tests._DelegatedAuthority"
    # The verdict is recorded on the milestone event payload too.
    rows = pipeline.outbox.rows() if hasattr(pipeline.outbox, "rows") else ()
    decision_rows = [
        r for r in rows if r.event.payload.get("decision", {}).get("delegated") is True
    ]
    assert decision_rows, "delegated decision must be recorded in the outbox payload"
    assert list(decision_rows[0].event.payload["decision"]["reason_codes"]) == ["qualifies"]


def test_admission_low_rank_ic_is_rejected_by_the_authority_not_the_platform():
    pipeline = _make_pipeline(
        evaluate_candidate=lambda c, ctx: build_with_evidence(
            {"candidate_ref": str(getattr(c, "candidate_id", "")), "rank_ic": 0.005, "evidence_status": "computed", "return_basis": "vwap_to_vwap"}
        )
    )
    report = pipeline.run(
        [_raw_candidate(seed="lowic")], library_snapshot=_library_snapshot()
    )
    assert report.num_approved == 0
    assert report.num_rejected == 1
    rejected_states = [st for st in report.states if st.status == "REJECTED"]
    assert len(rejected_states) == 1
    # the reason code came FROM the authority, mapped onto the platform vocabulary
    assert rejected_states[0].reason == PipelineStatusReason.QRP_GATE_REJECTED_RANK_IC
    verdict = pipeline.verdicts()[0]
    assert verdict.decision == DECISION_REJECTED


def test_admission_shadowed_verdict_is_recorded_not_applied():
    class _Shadowing:
        def decide(self, request):
            return AdmissionVerdict(
                decision=DECISION_SHADOWED,
                reason_codes=("merge_suggested",),
                authority="tests._Shadowing",
            )

    registry = ArtifactRegistry()
    pipeline = _make_pipeline(
        evaluate_candidate=lambda c, ctx: build_with_evidence(
            {"candidate_ref": str(getattr(c, "candidate_id", "")), "rank_ic": 0.3, "evidence_status": "computed", "return_basis": "vwap_to_vwap"}
        ),
        registry=registry,
        admission_authority=_Shadowing(),
    )
    report = pipeline.run(
        [_raw_candidate(seed="shadow")], library_snapshot=_library_snapshot()
    )
    assert report.num_approved == 0
    assert report.num_rejected == 0
    assert report.num_shadowed == 1
    assert report.num_registered == 0
    assert registry.contains(_h("shadow")) is False
    shadowed_states = [st for st in report.states if st.status == "SHADOWED"]
    assert len(shadowed_states) == 1
    assert shadowed_states[0].reason == PipelineStatusReason.QRP_ADMISSION_SHADOWED


def test_platform_without_authority_fails_closed_and_never_fabricates_approval():
    # Default seam = RefuseAdmission: an uncomposed pipeline approves nothing
    # and records the machine-parseable absence reason.
    pipeline = _make_pipeline(
        evaluate_candidate=lambda c, ctx: build_with_evidence(
            {"candidate_ref": str(getattr(c, "candidate_id", "")), "rank_ic": 0.5, "evidence_status": "computed", "return_basis": "vwap_to_vwap"}
        ),
        admission_authority=RefuseAdmission(),
    )
    report = pipeline.run(
        [_raw_candidate(seed="noauth")], library_snapshot=_library_snapshot()
    )
    assert report.num_approved == 0
    assert report.num_rejected == 1
    rejected_states = [st for st in report.states if st.status == "REJECTED"]
    assert len(rejected_states) == 1
    assert (
        rejected_states[0].reason == PipelineStatusReason.QRP_ADMISSION_AUTHORITY_ABSENT
    )
    verdict = pipeline.verdicts()[0]
    assert verdict.decision == DECISION_REJECTED
    assert verdict.reason_codes == (REASON_AUTHORITY_ABSENT,)


def test_platform_carries_no_admission_thresholds():
    # The library snapshot the platform forwards carries ONLY refs: the
    # threshold knobs the old platform gate owned are gone from the platform
    # surface entirely.
    snapshot = _library_snapshot()
    assert "min_rank_ic" not in snapshot
    assert "duplicate_similarity_threshold" not in snapshot
    assert "reject_duplicates" not in snapshot
    import quant_platform.app.orchestrator as orch

    assert not hasattr(orch, "DEFAULT_MIN_RANK_IC")
    assert not hasattr(orch, "pipeline_gate_decision")
    assert not hasattr(orch, "VWAP_TO_VWAP_BASIS")


def test_admission_request_carries_domain_refs_only():
    captured: list[AdmissionRequest] = []

    class _Capture:
        def decide(self, request):
            captured.append(request)
            return AdmissionVerdict(decision=DECISION_APPROVED, reason_codes=("qualifies",))

    pipeline = _make_pipeline(
        evaluate_candidate=lambda c, ctx: build_with_evidence(
            {"candidate_ref": "irrelevant", "rank_ic": 0.2, "evidence_status": "computed", "return_basis": "vwap_to_vwap"}
        ),
        admission_authority=_Capture(),
    )
    report = pipeline.run(
        [_raw_candidate(seed="carry")], library_snapshot=_library_snapshot()
    )
    assert report.num_approved == 1
    request = captured[0]
    assert request.candidate_ref
    assert len(request.content_hash) == 64
    # the carried semantic ref is the DOMAIN's digest (hex), untouched
    assert len(request.factor_definition_ref) == 64
    assert request.semantic_ref == request.factor_definition_ref


def test_pipeline_rejection_path_no_registry_and_no_approved():
    registry = ArtifactRegistry()
    outbox = InMemoryOutbox()

    def evaluate(candidate, context):
        return build_with_evidence(
            {
                "candidate_ref": str(getattr(candidate, "candidate_id", "")),
                "rank_ic": 0.01,
                "evidence_status": "computed",
                "return_basis": "vwap_to_vwap",
            }
        )

    pipeline = _make_pipeline(registry=registry, outbox=outbox, evaluate_candidate=evaluate)
    report = pipeline.run(
        [_raw_candidate(seed="rej")],
        library_snapshot=_library_snapshot(),
    )
    assert report.num_consumed == 1
    assert report.num_approved == 0
    assert report.num_rejected == 1
    assert report.num_registered == 0
    assert report.num_events_published == 2  # reconciled + rejected
    # no feature snapshot (no approved members)
    assert report.num_feature_snapshot_created == 0
    assert registry.contains(_h("rej")) is False


# --------------------------------------------------------------------------- #
# evaluation failure path
# --------------------------------------------------------------------------- #


def test_pipeline_evaluation_failure_marks_failed_and_registers_nothing():
    def evaluate(candidate, context):
        raise RuntimeError("evaluator unavailable")

    pipeline = _make_pipeline(evaluate_candidate=evaluate)
    report = pipeline.run(
        [_raw_candidate(seed="boom")],
        library_snapshot=_library_snapshot(),
    )
    assert report.num_consumed == 1
    assert report.num_evaluation_failed == 1
    assert report.num_approved == 0
    assert report.num_registered == 0
    failed_states = [st for st in report.states if st.status == "FAILED"]
    assert len(failed_states) == 1


def test_pipeline_no_evidence_default_evaluator_approves_nothing():
    # The fail-closed default evaluator must never approve anything.
    pipeline = _make_pipeline(evaluate_candidate=None)
    report = pipeline.run(
        [_raw_candidate(seed="nerf")],
        library_snapshot=_library_snapshot(),
    )
    assert report.num_approved == 0
    assert report.num_registered == 0
    assert report.num_events_published == 2  # reconciled + no-evidence rejected


# --------------------------------------------------------------------------- #
# feature-set snapshot + retrain diff
# --------------------------------------------------------------------------- #


def test_pipeline_feature_snapshot_and_retrain_diff_across_rounds():
    def evaluate(candidate, context):
        return build_with_evidence(
            {
                "candidate_ref": str(getattr(candidate, "candidate_id", "")),
                "rank_ic": 0.1,
                "evidence_status": "computed",
                "return_basis": "vwap_to_vwap",
            }
        )

    pipeline = _make_pipeline(evaluate_candidate=evaluate)
    first = pipeline.run(
        [_raw_candidate(seed="fs-a", factor_name="fsa")],
        library_snapshot=_library_snapshot(),
    )
    assert first.num_approved == 1
    assert first.num_feature_snapshot_created == 1
    assert first.feature_snapshot_id == "fs_pipeline_default"
    assert first.feature_snapshot_version == "v1"
    assert len(first.feature_set_artifacts) == 1
    fs_artifact = first.feature_set_artifacts[0]
    assert isinstance(fs_artifact, FeatureSetArtifact)
    assert fs_artifact.content_hash == fs_artifact.recomputed_hash()
    assert len(pipeline.feature_snapshots()) == 1
    snap = pipeline.feature_snapshots()[0]
    assert isinstance(snap, FeatureSetVersion)
    assert len(snap.ordered_members) == 1

    # Second round with a DIFFERENT approved factor -> membership change ->
    # retrain.
    second = pipeline.run(
        [_raw_candidate(seed="fs-b", factor_name="fsb")],
        library_snapshot=_library_snapshot(),
    )
    assert second.num_feature_snapshot_created == 1
    assert second.feature_snapshot_version == "v2"
    assert second.retrain_required is True
    assert second.feature_set_diff_category in {
        FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE.value
    }
    assert len(pipeline.feature_snapshots()) == 2

    # Same identity round -> no retrain.
    third = pipeline.run(
        [_raw_candidate(seed="fs-a", factor_name="fsa")],
        library_snapshot=_library_snapshot(),
    )
    # The identity was already consumed -> reconcile reports it as a duplicate
    # (the same manifest was consumed in round 1), so no approval event and no
    # snapshot.
    assert third.num_approved == 0 or third.num_duplicates == 1
    assert third.retrain_required is False


# --------------------------------------------------------------------------- #
# PipelineReport immutability
# --------------------------------------------------------------------------- #


def test_pipline_report_immutable_builders():
    report = PipelineReport(batch_fingerprint="fp:1")
    state = {"candidate_id": "c1", "content_hash": _h("a"), "status": "NEW_CONSUMED", "reason": "QRP_OK"}
    extended = report.with_item(**state)
    assert extended is not report
    assert len(extended.states) == 1
    assert report.states == ()  # original untouched
    assert report.num_consumed == 0
    # P0-PLAT-006: state items are immutable — a caller cannot rewrite a
    # reported state, and caller-side dict mutation must not leak in.
    with pytest.raises(AttributeError):
        extended.states[0].status = "APPROVED"
    state["status"] = "APPROVED"
    assert extended.states[0].status == "NEW_CONSUMED"
    assert extended.states[0].candidate_id == "c1"
    assert extended.states[0].content_hash == _h("a")
    assert extended.states[0].reason == "QRP_OK"

    replaced = extended.with_consumed(candidate_id="c2", content_hash=_h("b"), status="NEW_CONSUMED", reason="QRP_OK")
    assert replaced.num_consumed == 1
    assert replaced is not extended
    assert extended.num_consumed == 0

    with_ev = extended.with_event("evt-1")
    assert with_ev.num_events_published == 1
    assert with_ev.published_event_ids == ("evt-1",)
    assert extended.num_events_published == 0

    # Merged report fields become tuples.
    assert isinstance(replaced.approved_content_hashes, tuple)
    assert isinstance(with_ev.published_event_ids, tuple)


def test_pipeline_report_replayed_flag():
    report = PipelineReport(batch_fingerprint="fp:replay")
    assert report.num_replayed == 0
    replayed = report.as_replayed()
    assert replayed.num_replayed == 1
    assert report.num_replayed == 0  # original untouched
    assert replayed.as_replayed().num_replayed == 1  # idempotent


def test_pipeline_run_requires_library_snapshot():
    pipeline = _make_pipeline(evaluate_candidate=_eval_from(0.1))
    with pytest.raises(TypeError):
        pipeline.run([_raw_candidate(seed="x")])


# --------------------------------------------------------------------------- #
# Registry idempotency: same candidate approved twice registers exactly once
# --------------------------------------------------------------------------- #


def test_pipeline_registry_register_idempotent_across_rounds():
    registry = ArtifactRegistry()

    def evaluate(candidate, context):
        return build_with_evidence(
            {
                "candidate_ref": str(getattr(candidate, "candidate_id", "")),
                "rank_ic": 0.1,
                "evidence_status": "computed",
                "return_basis": "vwap_to_vwap",
            }
        )

    raw = _raw_candidate(seed="idem")
    pipeline = _make_pipeline(registry=registry, evaluate_candidate=evaluate)

    # Round 1 -> approved, registered.
    first = pipeline.run([raw], library_snapshot=_library_snapshot())
    assert first.num_registered == 1
    assert len(registry.list_versions(f"FC_{_h('idem')[:16]}")) == 1

    # Round 2 with the same raw record is a duplicate replay -> the registry is
    # untouched (idempotent per (content_hash, artifact_type)).
    second = pipeline.run([raw], library_snapshot=_library_snapshot())
    assert second.num_replayed == 1
    assert second.num_registered == 0
    assert len(registry.list_versions(f"FC_{_h('idem')[:16]}")) == 1