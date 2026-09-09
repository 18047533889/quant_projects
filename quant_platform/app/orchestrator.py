# -*- coding: utf-8 -*-
"""QRP-P5-PIPE — full-pipeline platform orchestrator (reconcile -> job -> registry -> library).

Glues the per-stage pure-logic layers already built in P2/P3/P4/P8 into one
runnable pipeline orchestration, WITHOUT touching any domain implementation:

    raw candidate records + registered library snapshot
      -> replay-safe normalization (``candidate.ingest.normalize_candidate``)
      -> dual-key reconciliation (``reconcile_candidates``) — only NEW enters
      -> ``batch_fingerprint`` anti-replay (a replayed batch short-circuits)
      -> JobRunner-scheduled per-candidate handler (FP treatment -> QE evaluation
         through the platform's own ``contracts/jobs`` DTOs; the actual domain
         evaluation behind the injectable ``evaluate_candidate`` seam, so no
         domain import leaks in)
      -> ADMISSION DELEGATED to the injected
         :class:`~quant_platform.app.contracts.admission.AdmissionAuthority`
         (the owning domain package — ``factor_assets`` for factors/clusters —
         implements the Protocol; the platform only RECORDS the returned
         :class:`~quant_platform.app.contracts.admission.AdmissionVerdict`,
         R55 P0-6)
      -> approved candidates registered into ``ArtifactRegistry``
         (artifact_type=FACTOR_CANDIDATE)
      -> per-round ``FeatureSetVersion`` snapshot + ``FeatureSetArtifact`` and
         ``retrain_required_for_diff`` (P8, quant_platform-native)
      -> milestone events published through the in-memory outbox
         (``InMemoryOutbox``): CANDIDATE_RECONCILED / CANDIDATE_APPROVED /
         CANDIDATE_REJECTED / ARTIFACT_REGISTERED

Purity rules (mirroring ``candidate/ingest.py``):
  * stdlib + quant_platform internal modules only (PURE-DTO);
  * no import of domain internals — ``quant_evaluator`` / ``factor_optimizer`` /
    ``factor_assets`` are referenced as literal ``PURE-DTO dict`` shapes only;
  * all in-memory; the outbox is the ``worker.publish.InMemoryOutbox``.

The eight-step domain chain (factor -> preprocess -> evaluate -> optimize) is
WIRED, not owned, here: every step that touches a domain authority lives behind
an injectable seam (``evaluate_candidate`` for evaluation, ``admission_authority``
for the admit/reject decision). The default admission seam fails closed — it
refuses everything with ``admission_authority_absent`` and approves nothing
without a delegated verdict (never a fabricated pass, never a platform-side
threshold call on rank_ic / label maturity / return basis / similarity).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace, asdict
from datetime import datetime, timezone
from typing import Any, Callable
import hashlib
import json
import uuid

from quant_platform.app.candidate.ingest import (
    CandidateNormalizationError,
    ReconcileReason,
    batch_fingerprint,
    normalize_candidate,
    reconcile_candidates,
)
from quant_platform.app.contracts import (
    ARTIFACT_TYPE_FACTOR_CANDIDATE,
    ARTIFACT_TYPE_FEATURE_SET,
    AdmissionAuthority,
    AdmissionRequest,
    AdmissionVerdict,
    ArtifactRef,
    DECISION_APPROVED,
    DECISION_SHADOWED,
    EventEnvelope,
    FeatureMemberRef,
    FeatureSetArtifact,
    FeatureSetVersion,
    JobResult,
    JobSpec,
    RefuseAdmission,
    RetrainPolicy,
    retrain_required_for_diff,
)
from quant_platform.app.contracts._contenthash import content_hash
from quant_platform.app.storage.registry import ArtifactRegistry
from quant_platform.app.worker.jobs import ErrorClass, JobError, JobRunner
from quant_platform.app.worker.publish import InMemoryOutbox

__all__ = [
    "PipelineConfig",
    "PipelineReport",
    "PipelineStateItem",
    "PipelineStatusReason",
    "CandidateEvaluation",
    "Pipeline",
    "AdmissionAuthority",
    "AdmissionRequest",
    "AdmissionVerdict",
    "build_with_evidence",
]

# Platform ``EventEnvelope`` validates the event_type against a fixed enum; the
# pipeline milestone discriminator travels in the payload under ``milestone``.
_PIPELINE_EVENT_TYPE = "FactorCandidateValidated"
_EVENT_SCHEMA_VERSION = "1.0"
_PRODUCER = "quant_platform.orchestrator"
_PRODUCER_VERSION = "1.0.0"

#: Markers recorded on a JobResult summary when the candidate evaluation step
#: produced real computed evidence (vs a fail-closed no-evidence JobResult).
_GOOD_EVIDENCE_MARKER = "evaluation:ok"
_BAD_EVIDENCE_MARKER = "evaluation:no-evidence"


# --------------------------------------------------------------------------- #
# Status and report
# --------------------------------------------------------------------------- #


class PipelineStatusReason:
    """Machine-parseable per-candidate status reason strings.

    ``QRP_GATE_*`` values are now RECORDED verdict attributions from the
    delegated authority (R55 P0-6) — the platform maps the authority's reason
    codes onto them but computes none of them.
    """

    QRP_OK = "QRP_OK"
    QRP_DUPLICATE_SKIPPED = "QRP_DUPLICATE_SKIPPED"
    QRP_CONFLICT_SKIPPED = "QRP_CONFLICT_SKIPPED"
    QRP_GATE_QUALIFIES = "QRP_GATE_QUALIFIES"
    QRP_GATE_REJECTED_RANK_IC = "QRP_GATE_REJECTED_RANK_IC"
    QRP_GATE_REJECTED_LABEL = "QRP_GATE_REJECTED_LABEL"
    QRP_GATE_REJECTED_BASIS = "QRP_GATE_REJECTED_BASIS"
    QRP_GATE_REVIEW_UNKNOWN = "QRP_GATE_REVIEW_UNKNOWN"
    QRP_ADMISSION_REJECTED = "QRP_ADMISSION_REJECTED"
    QRP_ADMISSION_SHADOWED = "QRP_ADMISSION_SHADOWED"
    QRP_ADMISSION_AUTHORITY_ABSENT = "QRP_ADMISSION_AUTHORITY_ABSENT"
    QRP_EVALUATION_FAILED = "QRP_EVALUATION_FAILED"
    QRP_NO_EVIDENCE = "QRP_NO_EVIDENCE"
    QRP_REPLAY_DETECTED = "QRP_REPLAY_DETECTED"
    QRP_NORMALIZATION_FAILED = "QRP_NORMALIZATION_FAILED"

    @classmethod
    def catalog(cls) -> frozenset[str]:
        return frozenset(v for k, v in vars(cls).items() if k.startswith("QRP_"))


class PipelineStateItem:
    """Immutable per-candidate terminal state (P0-PLAT-006).

    ``PipelineReport.states`` used to be ``tuple[dict[str, str], ...]`` — the
    dicts were caller-owned and mutable, so a caller could mutate a reported
    state after the fact.  A frozen dataclass makes each state item immutable.
    """

    __slots__ = ("candidate_id", "ingestion_record_id", "content_hash", "status", "reason")

    candidate_id: str
    ingestion_record_id: str
    content_hash: str | None
    status: str
    reason: str

    def __init__(self, candidate_id: str, content_hash: str | None, status: str, reason: str, ingestion_record_id: str | None = None) -> None:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if content_hash is not None and (not isinstance(content_hash, str) or not content_hash):
            raise ValueError("content_hash must be a non-empty string or None")
        if not isinstance(status, str) or not status:
            raise ValueError("status must be a non-empty string")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "ingestion_record_id", ingestion_record_id or content_hash or f"ingestion:{candidate_id}")
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"PipelineStateItem is a frozen value object; cannot assign {name!r}")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "candidate_id": self.candidate_id,
            "ingestion_record_id": self.ingestion_record_id,
            "content_hash": self.content_hash,
            "status": self.status,
            "reason": self.reason,
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PipelineStateItem(candidate_id={self.candidate_id!r}, "
            f"status={self.status!r}, reason={self.reason!r})"
        )


@dataclass(frozen=True)
class PipelineReport:
    """Immutable result of one pipeline run.

    Every field is immutable (tuples / ints / str). ``with_item`` /
    ``with_event`` / ``with_registered`` / ``as_replayed`` return a NEW report
    and never mutate ``self``.
    """

    batch_fingerprint: str
    num_consumed: int = 0
    num_duplicates: int = 0
    num_conflicts: int = 0
    num_replayed: int = 0
    num_approved: int = 0
    num_rejected: int = 0
    num_shadowed: int = 0
    num_evaluation_failed: int = 0
    num_normalization_failed: int = 0
    num_registered: int = 0
    num_events_published: int = 0
    num_feature_snapshot_created: int = 0
    retrain_required: bool = False
    feature_set_diff_category: str | None = None
    feature_snapshot_id: str | None = None
    feature_snapshot_version: str | None = None
    feature_set_artifacts: tuple[FeatureSetArtifact, ...] = ()

    #: Per-candidate terminal states: immutable ``PipelineStateItem`` records
    #: (candidate_id / content_hash / status / reason).  P0-PLAT-006: never a
    #: mutable dict — a caller must not be able to rewrite a reported state.
    states: tuple[PipelineStateItem, ...] = ()
    #: ids of candidates consumed as NEW (reconciled + scheduled), in order.
    consumed_ids: tuple[str, ...] = ()
    #: EventEnvelope event_ids appended to the outbox.
    published_event_ids: tuple[str, ...] = ()
    #: Registered ``(artifact_id, content_hash)`` pairs for approved candidates.
    registered_artifacts: tuple[tuple[str, str], ...] = ()
    #: Approved candidate content hashes (ordered identity of the feature set).
    approved_content_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.batch_fingerprint:
            raise ValueError("batch_fingerprint is required")
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "consumed_ids", tuple(self.consumed_ids))
        object.__setattr__(self, "published_event_ids", tuple(self.published_event_ids))
        object.__setattr__(
            self, "registered_artifacts", tuple(self.registered_artifacts)
        )
        object.__setattr__(
            self, "approved_content_hashes", tuple(self.approved_content_hashes)
        )
        object.__setattr__(
            self, "feature_set_artifacts", tuple(self.feature_set_artifacts)
        )

    # ---- immutable builders ------------------------------------------------- #
    def with_item(
        self,
        *,
        candidate_id: str,
        status: str,
        reason: str,
        content_hash: str,
        ingestion_record_id: str | None = None,
    ) -> "PipelineReport":
        """New report with one per-candidate state appended (never mutates)."""
        item = PipelineStateItem(
            candidate_id=candidate_id,
            content_hash=content_hash,
            status=status,
            reason=reason,
            ingestion_record_id=ingestion_record_id,
        )
        return replace(self, states=self.states + (item,))

    def with_consumed(
        self,
        *,
        candidate_id: str,
        content_hash: str,
        status: str,
        reason: str,
    ) -> "PipelineReport":
        return replace(
            self,
            num_consumed=self.num_consumed + 1,
            consumed_ids=self.consumed_ids + (candidate_id,),
        ).with_item(
            candidate_id=candidate_id,
            status=status,
            reason=reason,
            content_hash=content_hash,
        )

    def with_duplicate(
        self, *, candidate_id: str, content_hash: str, reason: str
    ) -> "PipelineReport":
        return replace(self, num_duplicates=self.num_duplicates + 1).with_item(
            candidate_id=candidate_id,
            status="DUPLICATE_SKIPPED",
            reason=reason,
            content_hash=content_hash,
        )

    def with_conflict(
        self, *, candidate_id: str, content_hash: str, reason: str
    ) -> "PipelineReport":
        return replace(self, num_conflicts=self.num_conflicts + 1).with_item(
            candidate_id=candidate_id,
            status="REJECTED",
            reason=reason,
            content_hash=content_hash,
        )

    def with_approved(
        self,
        *,
        candidate_id: str,
        content_hash: str,
        artifact_id: str,
        registry_content_hash: str,
    ) -> "PipelineReport":
        """New report with one approved + registered candidate appended."""
        return (
            replace(
                self,
                num_approved=self.num_approved + 1,
                num_registered=self.num_registered + 1,
                registered_artifacts=self.registered_artifacts
                + ((artifact_id, registry_content_hash),),
                approved_content_hashes=self.approved_content_hashes
                + (registry_content_hash,),
            )
        ).with_item(
            candidate_id=candidate_id,
            status="APPROVED",
            reason=PipelineStatusReason.QRP_GATE_QUALIFIES,
            content_hash=content_hash,
        )

    def with_rejected(
        self, *, candidate_id: str, content_hash: str, reason: str
    ) -> "PipelineReport":
        return replace(self, num_rejected=self.num_rejected + 1).with_item(
            candidate_id=candidate_id,
            status="REJECTED",
            reason=reason,
            content_hash=content_hash,
        )

    def with_shadowed(
        self, *, candidate_id: str, content_hash: str, reason: str
    ) -> "PipelineReport":
        """Record a delegated SHADOWED verdict (kept, never applied)."""
        return replace(self, num_shadowed=self.num_shadowed + 1).with_item(
            candidate_id=candidate_id,
            status="SHADOWED",
            reason=reason,
            content_hash=content_hash,
        )

    def with_failed(
        self, *, candidate_id: str, content_hash: str, reason: str
    ) -> "PipelineReport":
        return replace(self, num_evaluation_failed=self.num_evaluation_failed + 1).with_item(
            candidate_id=candidate_id,
            status="FAILED",
            reason=reason,
            content_hash=content_hash,
        )

    def with_normalization_failed(
        self, *, candidate_id: str, ingestion_record_id: str, reason: str
    ) -> "PipelineReport":
        return replace(
            self, num_normalization_failed=self.num_normalization_failed + 1
        ).with_item(
            candidate_id=candidate_id,
            ingestion_record_id=ingestion_record_id,
            content_hash=None,
            status="FAILED",
            reason=reason,
        )

    def with_event(self, event_id: str) -> "PipelineReport":
        return replace(
            self,
            num_events_published=self.num_events_published + 1,
            published_event_ids=self.published_event_ids + (event_id,),
        )

    def with_feature_snapshot(
        self,
        *,
        artifact: FeatureSetArtifact,
        revision: int,
        diff_category: str | None,
        retrain: bool,
    ) -> "PipelineReport":
        return replace(
            self,
            num_feature_snapshot_created=self.num_feature_snapshot_created + 1,
            feature_snapshot_id=artifact.feature_set_id,
            feature_snapshot_version=artifact.feature_set_version,
            feature_set_artifacts=self.feature_set_artifacts + (artifact,),
            retrain_required=retrain,
            feature_set_diff_category=diff_category,
        )

    def as_replayed(self) -> "PipelineReport":
        if self.num_replayed:
            return self
        return replace(self, num_replayed=self.num_replayed + 1)


# --------------------------------------------------------------------------- #
# Evaluation surface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateEvaluation:
    """Carried evaluation DTO (opaque evidence the AUTHORITY will judge).

    The platform never interprets these fields: ``rank_ic`` magnitude,
    ``label_maturity``, ``evidence_status`` and ``return_basis`` are read by the
    injected :class:`AdmissionAuthority` (the owning domain package) — NOT by
    this orchestrator. R55 P0-6 removed the platform-side promotion gate that
    thresholded them.
    """

    candidate_ref: str
    rank_ic: float | None = None
    label_maturity: bool = True
    evidence_status: str | None = None
    return_basis: str = "vwap_to_vwap"
    evidence_ref: str | None = None
    evaluation_ref: str | None = None
    treatment_optimization_ref: str | None = None
    recipe_ref: str | None = None
    preprocess_state_ref: str | None = None
    factor_value_ref: str | None = None
    metric_policy_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_ref:
            raise ValueError("candidate_ref is required")
        if not isinstance(self.label_maturity, bool):
            raise TypeError("label_maturity must be a bool")
        if self.rank_ic is not None:
            if isinstance(self.rank_ic, bool) or not isinstance(self.rank_ic, (int, float)):
                raise TypeError("rank_ic must be a finite non-boolean number or None")
            numeric = float(self.rank_ic)
            if numeric != numeric or numeric in (float("inf"), float("-inf")):
                raise ValueError("rank_ic must be finite")
        if self.evidence_status is not None and not isinstance(self.evidence_status, str):
            raise TypeError("evidence_status must be a string or None")
        if not isinstance(self.return_basis, str) or not self.return_basis:
            raise ValueError("return_basis must be a non-empty string")


def build_with_evidence(evidence: Mapping[str, Any]) -> CandidateEvaluation:
    """Public helper: carry a computed-evidence dict into the DTO.

    ``evidence`` keys: ``candidate_ref``, ``rank_ic``, ``label_maturity``
    (default True), ``evidence_status`` (default ``"computed"``),
    ``return_basis``, ``evidence_ref`` / ``evaluation_ref`` /
    ``treatment_optimization_ref``.

    This is TRANSPORT ONLY — the values are passed to the admission authority
    untouched; the platform draws no conclusion from them (no threshold, no
    maturity or 口径 judgment). Label not mature raises — a not-computed label
    must never be dressed up as computed evidence by the carrier either.
    """
    candidate_ref = str(evidence.get("candidate_ref") or "")
    if not candidate_ref:
        raise ValueError("build_with_evidence: candidate_ref is required")
    # P0-PLAT-007: no optimistic defaults.  A label that has not matured must
    # never be dressed up as computed evidence; the carrier either has real
    # evidence or says so explicitly.
    maturity = evidence.get("label_maturity")
    if maturity is None:
        maturity = True
    maturity = bool(maturity)
    if not maturity:
        raise ValueError(
            "build_with_evidence: label_maturity must be True for computed evidence"
        )
    evidence_status = evidence.get("evidence_status")
    if not evidence_status:
        raise ValueError(
            "build_with_evidence: evidence_status is required (no 'computed' "
            "default — a not-computed carrier must say 'not_computed')"
        )
    return_basis = evidence.get("return_basis")
    if not return_basis:
        raise ValueError(
            "build_with_evidence: return_basis is required (no 'vwap_to_vwap' "
            "default — the basis must be carried explicitly)"
        )
    return CandidateEvaluation(
        candidate_ref=candidate_ref,
        rank_ic=evidence.get("rank_ic"),
        label_maturity=True,
        evidence_status=str(evidence_status),
        return_basis=str(return_basis),
        evidence_ref=evidence.get("evidence_ref"),
        evaluation_ref=evidence.get("evaluation_ref"),
        treatment_optimization_ref=evidence.get("treatment_optimization_ref"),
        recipe_ref=evidence.get("recipe_ref"),
        preprocess_state_ref=evidence.get("preprocess_state_ref"),
        factor_value_ref=evidence.get("factor_value_ref"),
        metric_policy_ref=evidence.get("metric_policy_ref"),
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration.

    ``min_rank_ic`` is retained ONLY as a carried pass-through for the injected
    :class:`AdmissionAuthority` — the platform itself never reads it. All
    admission thresholds live with the owning domain package (R55 P0-6).
    """

    scope: str = "pipeline:test"
    min_rank_ic: float = 0.02
    retrain_policy: RetrainPolicy | None = None
    feature_set_id: str = "fs_pipeline_default"
    require_complete_execution_trace: bool = False
    feature_set_update_mode: str = 'MERGE'

    def __post_init__(self):
        if self.feature_set_update_mode not in {'MERGE', 'REPLACE'}:
            raise ValueError('feature_set_update_mode must be MERGE or REPLACE')


def _default_evaluate(candidate: Any, context: Mapping[str, Any]) -> CandidateEvaluation:
    """Fail-closed default evaluator: never fabricates computed evidence.

    When no per-candidate evaluator is injected the pipeline cannot carry real
    vwap-derived rank_ic — the default produces a no-evidence DTO that the
    admission authority will see as not-computed. Silent fabrication of a pass
    is exactly the failure mode admission delegation is built to reject.
    """
    return CandidateEvaluation(
        candidate_ref=str(getattr(candidate, "candidate_id", "")),
        rank_ic=None,
        label_maturity=False,
        evidence_status="not_computed",
        return_basis="vwap_to_vwap",
    )


class Pipeline:
    """In-memory orchestrator over the per-stage pure-logic layers.

    Parameters
    ----------
    registry : ``ArtifactRegistry`` approved FACTOR_CANDIDATE artifacts register
        into (idempotent per (content_hash, artifact_type)).
    outbox : ``worker.publish.InMemoryOutbox`` milestone events append to.
    runner : ``worker.jobs.JobRunner`` per-candidate handlers schedule on.
    evaluate_candidate : injectable domain seam ``(manifest, context) ->
        CandidateEvaluation`` — produces the evidence CARRIED to the authority.
    admission_authority : the :class:`AdmissionAuthority` port implemented by
        the owning domain package (e.g. a ``factor_assets`` promotion/selection
        adapter). Every approve/reject verdict is its output; the platform only
        records it. Default: :class:`RefuseAdmission` (fail closed — an
        uncomposed pipeline approves nothing).
    config : sizing / policy switches.
    run_storage : OPTIONAL durable anti-replay backing store (P0-PLAT-004). A
        :class:`~quant_platform.app.db.durable_store.RunStateStore` (SQLite or
        Postgres) persists processed batch fingerprints + consumed candidates
        so a PROCESS RESTART does not let a replayed batch through anti-replay.
        When ``None`` (default) the pipeline keeps its anti-replay state in
        memory only — EXACTLY today's behavior.

    ``run`` is deterministic and idempotent in the sense that replaying the same
    batch short-circuits through ``batch_fingerprint`` anti-replay.
    """

    def __init__(
        self,
        *,
        registry: ArtifactRegistry | None = None,
        outbox: InMemoryOutbox | None = None,
        runner: JobRunner | None = None,
        evaluate_candidate: Callable[[Any, Mapping[str, Any]], CandidateEvaluation] | None = None,
        admission_authority: AdmissionAuthority | None = None,
        config: PipelineConfig | None = None,
        worker_id: str = "qrpp5-worker",
        run_storage: Any | None = None,
        artifact_publisher: Any | None = None,
        generation_coordinator: Any | None = None,
        failure_injector: Callable[[str, ArtifactRef], None] | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ArtifactRegistry()
        self.outbox = outbox if outbox is not None else InMemoryOutbox()
        self.runner = runner if runner is not None else JobRunner(worker_id=worker_id)
        self.config = config if config is not None else PipelineConfig()
        self.evaluate_candidate = evaluate_candidate or _default_evaluate
        #: the delegated decision-maker (domain-owned). Never None: without a
        #: composed authority the pipeline refuses admission (fail closed).
        self.admission_authority = admission_authority if admission_authority is not None else RefuseAdmission()
        #: OPTIONAL durable backing store for anti-replay state (P0-PLAT-004).
        #: ``None`` = pure in-memory (today's behavior, unchanged).
        self.run_storage = run_storage
        # Production composition injects the durable two-phase publisher.  A
        # missing publisher deliberately retains the historical research-only
        # in-memory path; it must not be mistaken for a durable publication.
        self.artifact_publisher = artifact_publisher
        self.generation_coordinator = generation_coordinator
        self.failure_injector = failure_injector

        self._processed_fingerprints: set[str] = set()
        self._consumed_manifests: list[Any] = []
        self._feature_snapshots: list[FeatureSetVersion] = []
        self._run_counter = 0
        self._evaluation_by_job: dict[str, CandidateEvaluation] = {}
        self._verdict_by_job: dict[str, AdmissionVerdict] = {}
        self._definition_by_artifact_hash: dict[str, str] = {}
        self._feature_provenance_by_artifact_hash: dict[str, dict[str, Any]] = {}
        self._modeling_feature_manifests: list[dict[str, Any]] = []
        self._research_payloads: dict[str, bytes] = {}
        self._execution_trace_by_candidate: dict[str, dict[str, str]] = {}
        self._reservation_context = ContextVar(
            f"pipeline_reservation_{id(self)}", default=(None, ())
        )

        # P0-PLAT-004: seed anti-replay state from the durable store so a
        # process restart preserves it. Fingerprints recorded by past runs make
        # a replayed batch short-circuit; consumed content hashes make an
        # already-consumed candidate reconcile as a duplicate (never re-enters
        # the NEW path).
        if run_storage is not None:
            for fp in run_storage.consumed_fingerprints():
                self._processed_fingerprints.add(fp)
            self._consumed_hash_seed = set(run_storage.consumed_hashes())
        else:
            self._consumed_hash_seed = set()

    # ---- queries -------------------------------------------------------------
    @property
    def processed_batches(self) -> tuple[str, ...]:
        return tuple(sorted(self._processed_fingerprints))

    def feature_snapshots(self) -> tuple[FeatureSetVersion, ...]:
        return tuple(self._feature_snapshots)

    def job_records(self) -> tuple[Any, ...]:
        return self.runner.records()

    def verdicts(self) -> tuple[AdmissionVerdict, ...]:
        """All delegated admission verdicts recorded this pipeline's lifetime."""
        return tuple(self._verdict_by_job.values())

    def modeling_feature_manifests(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._modeling_feature_manifests)

    # ---- run -----------------------------------------------------------------
    def run(
        self, raw_records: Iterable[Mapping[str, Any]], *,
        library_snapshot: Mapping[str, Any], now: datetime | None = None,
    ) -> PipelineReport:
        """Reserve discovered candidates before entering any external reader."""
        if self.run_storage is None:
            return self._run(raw_records, library_snapshot=library_snapshot, now=now)
        records = list(raw_records)
        candidates = {}
        for raw in records:
            try:
                manifest = normalize_candidate(raw)
            except CandidateNormalizationError:
                continue
            candidates[manifest.factor_spec_sha256] = json.dumps(dict(raw),sort_keys=True)
        token = str(uuid.uuid4())
        reserved_hashes = self.run_storage.reserve_candidates(candidates, token)
        context_token = self._reservation_context.set((token, tuple(reserved_hashes)))
        try:
            return self._run(records, library_snapshot=library_snapshot, now=now)
        finally:
            self.run_storage.release_reservations(token)
            self._reservation_context.reset(context_token)

    def _check_reservations(self):
        if self.run_storage is not None:
            token, hashes = self._reservation_context.get()
            self.run_storage.assert_reservations(hashes, token)

    def _publication_fence(self):
        if self.run_storage is None:
            return nullcontext()
        token, hashes = self._reservation_context.get()
        return self.run_storage.fence_reservations(hashes, token)

    def _run(
        self,
        raw_records: Iterable[Mapping[str, Any]],
        *,
        library_snapshot: Mapping[str, Any],
        now: datetime | None = None,
    ) -> PipelineReport:
        """Execute one full pipeline pass and return an immutable report.

        Steps: normalize -> fingerprint anti-replay -> reconcile (NEW only) ->
        JobRunner-scheduled treatment+evaluation per candidate -> DELEGATED
        admission verdict (recorded, never computed here) -> register approved
        -> FeatureSet snapshot + diff. Milestone events are appended to the
        outbox as candidates move between phases.
        """
        if not isinstance(library_snapshot, Mapping):
            raise TypeError("library_snapshot is required (must be a Mapping)")
        now = now if now is not None else datetime.now(timezone.utc)

        # 1. Normalize raw discovery records (fail-closed per record).
        manifests: list[Any] = []
        normalization_failures: list[dict[str, str]] = []
        for raw in raw_records:
            try:
                manifests.append(normalize_candidate(raw))
            except CandidateNormalizationError as exc:
                candidate_id = str(raw.get("candidate_id") or "?")
                normalization_failures.append(
                    {
                        "candidate_id": candidate_id,
                        "content_hash": None,
                        "ingestion_record_id": f"ingestion:{len(normalization_failures)}:{candidate_id}",
                        "status": "FAILED",
                        "reason": f"{PipelineStatusReason.QRP_NORMALIZATION_FAILED}:{exc.reason}",
                    }
                )
        fingerprint = batch_fingerprint(manifests)

        # 2. Anti-replay: a batch whose fingerprint we already consumed is a
        #    replay — short-circuit (no new events, all candidates skipped).
        #    P0-PLAT-004: with a durable store the check is also durable — a
        #    fingerprint recorded by a PREVIOUS process restarts the pipeline
        #    with the same in-memory short-circuit (the store is consulted for
        #    any fingerprint the in-memory seed may have missed).
        if fingerprint in self._processed_fingerprints or (
            self.run_storage is not None and self.run_storage.has_fingerprint(fingerprint)
        ):
            if self.run_storage is not None:
                self._processed_fingerprints.add(fingerprint)
            report = PipelineReport(batch_fingerprint=fingerprint).as_replayed()
            for manifest in manifests:
                report = report.with_duplicate(
                    candidate_id=manifest.candidate_id,
                    content_hash=manifest.factor_spec_sha256,
                    reason=PipelineStatusReason.QRP_REPLAY_DETECTED,
                )
            if hasattr(report, "num_replayed"):
                report = report.as_replayed()  # no-op re-entrant
            for item in normalization_failures:
                report = report.with_normalization_failed(
                    candidate_id=item["candidate_id"],
                    ingestion_record_id=item["ingestion_record_id"],
                    reason=item["reason"],
                )
            return report

        self._run_counter += 1
        run_index = self._run_counter
        report = PipelineReport(batch_fingerprint=fingerprint)
        for item in normalization_failures:
            report = report.with_normalization_failed(
                candidate_id=item["candidate_id"],
                ingestion_record_id=item["ingestion_record_id"],
                reason=item["reason"],
            )

        # Known registry for reconciliation: every manifest this Pipeline has
        # consumed across past runs (a duplicate content/semantic identity is a
        # replay of an earlier consumed candidate). P0-PLAT-004: with a durable
        # store the consumed-manifest known-set is ALSO seeded from the store
        # (all consumed content hashes), so a candidate consumed by a PREVIOUS
        # process still reconciles as a duplicate.
        known = list(self._consumed_manifests)
        if self.run_storage is not None:
            known_hash_entries: list[Any] = []
            for manifest in manifests:
                if (manifest.factor_spec_sha256 in self._consumed_hash_seed
                        or self.run_storage.is_consumed_hash(manifest.factor_spec_sha256)):
                    self._consumed_hash_seed.add(manifest.factor_spec_sha256)
                    known_hash_entries.append(manifest)
            known = known + known_hash_entries
        reconcile = reconcile_candidates(manifests, known)

        # Per-candidate classification via the dual-key reconcile logic.
        # reconcile_candidates reconciles within the batch itself as well as
        # against the known registry, so a same-raw-record-twice batch surfaces
        # DUPLICATE_EXACT on the second occurrence. Recompute the dual-key
        # classes here (content + semantic exact-match = duplicate) so only
        # genuinely NEW candidates enter the job + admission path.
        reconcile_conflict_by_hash: dict[str, str] = {}
        for conflict in reconcile.conflicts:
            reconcile_conflict_by_hash[conflict.content_hash] = conflict.reason

        seen_dual_key: set[tuple[str, str]] = set()
        retryable_failure = False
        approved_pending_consumption = []
        for manifest in manifests:
            ch = manifest.factor_spec_sha256
            semantic = manifest.semantic_family_hint or ""
            dual_key = (ch, semantic)
            reason = reconcile_conflict_by_hash.get(ch)

            # Explicit intra-batch duplicate: same content+semantic identity
            # already consumed this run.
            if reason is None or reason != ReconcileReason.DUPLICATE_EXACT.value:
                if dual_key in seen_dual_key:
                    reason = ReconcileReason.DUPLICATE_EXACT.value
            seen_dual_key.add(dual_key)

            if reason == ReconcileReason.DUPLICATE_EXACT.value:
                report = report.with_duplicate(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=PipelineStatusReason.QRP_DUPLICATE_SKIPPED,
                )
                continue
            if reason == ReconcileReason.CONFLICT_SEMANTIC_TO_HASH.value:
                report = report.with_conflict(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=PipelineStatusReason.QRP_CONFLICT_SKIPPED,
                )
                continue
            if reason == ReconcileReason.CONFLICT_HASH_TO_SEMANTIC.value:
                report = report.with_conflict(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=PipelineStatusReason.QRP_CONFLICT_SKIPPED,
                )
                continue

            # NEW candidate -> RECONCILED milestone + JobRunner schedule.
            event_id = self._publish_event(
                milestone="CANDIDATE_RECONCILED",
                candidate=manifest,
                reason=PipelineStatusReason.QRP_OK,
                status="NEW_CONSUMED",
                library_snapshot=library_snapshot,
                now=now,
                run_index=run_index,
            )
            report = report.with_event(event_id).with_consumed(
                candidate_id=manifest.candidate_id,
                content_hash=ch,
                status="NEW_CONSUMED",
                reason=PipelineStatusReason.QRP_OK,
            )
            # Schedule the candidate on the JobRunner (treatment + evaluation).
            job_key = (
                f"candidate:{manifest.candidate_id}:{ch}:scenario:{run_index}"
            )
            spec = JobSpec(
                job_type="pipeline.candidate.evaluate",
                idempotency_key=job_key,
                inputs=(ch,),
                activity_kind="candidate_evaluation",
                resource_class="light",
            )
            record = self.runner.execute(
                spec,
                handler=self._make_candidate_handler(manifest, library_snapshot, job_key),
            )
            if record.status.name != "SUCCEEDED":
                if record.status.name == "FAILED_RETRYABLE":
                    retryable_failure = True
                else:
                    self._record_terminal_consumption(manifest, now)
                report = report.with_failed(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=PipelineStatusReason.QRP_EVALUATION_FAILED,
                )
                continue

            evaluation = self._evaluation_by_job.get(job_key)
            if evaluation is None:
                event_id = self._publish_event(
                    milestone="CANDIDATE_REJECTED",
                    candidate=manifest,
                    reason=PipelineStatusReason.QRP_NO_EVIDENCE,
                    status="REJECTED",
                    library_snapshot=library_snapshot,
                    now=now,
                    run_index=run_index,
                )
                report = report.with_event(event_id).with_rejected(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=PipelineStatusReason.QRP_NO_EVIDENCE,
                )
                continue

            # ADMISSION IS DELEGATED (R55 P0-6): the platform asks the injected
            # AdmissionAuthority (implemented by the owning domain package) and
            # RECORDS the returned verdict. It never thresholds rank_ic, never
            # judges label maturity / return basis / duplicate similarity.
            request = AdmissionRequest(
                candidate_ref=str(manifest.candidate_id),
                content_hash=ch,
                factor_definition_ref=str(getattr(manifest, "factor_definition_ref", "") or ""),
                semantic_ref=str(getattr(manifest, "semantic_ref", "") or ""),
                evaluation=evaluation,
                library_snapshot_ref=self._library_ref(library_snapshot),
                context={"job_key": job_key},
            )
            verdict = self.admission_authority.decide(request)
            self._verdict_by_job[job_key] = verdict
            if self.config.require_complete_execution_trace:
                self._execution_trace_by_candidate[manifest.candidate_id] = (
                    self._build_execution_trace(manifest, evaluation, verdict, library_snapshot)
                )
            approved = verdict.decision == DECISION_APPROVED
            decision_payload = {
                "decision": verdict.decision,
                "reason_codes": list(verdict.reason_codes),
                "library_version_ref": self._library_ref(library_snapshot),
                "authority": verdict.authority,
                "policy_ref": verdict.policy_ref,
                "delegated": True,
            }
            # Approval is a visibility claim.  For approved candidates it is
            # emitted only after bytes have been durably published and the
            # registry contains the verified reference.  Rejection/shadowing
            # can be emitted immediately because they expose no artifact.
            event_id = None
            if not approved:
                event_id = self._publish_event(
                    milestone=(
                        "CANDIDATE_SHADOWED"
                        if verdict.decision == DECISION_SHADOWED
                        else "CANDIDATE_REJECTED"
                    ),
                    candidate=manifest,
                    reason=_verdict_reason(verdict),
                    status=verdict.decision,
                    library_snapshot=library_snapshot,
                    now=now,
                    run_index=run_index,
                    decision=decision_payload,
                )
                report = report.with_event(event_id)

            if approved:
                with self._publication_fence():
                    artifact = self._build_candidate_artifact(manifest, now, library_snapshot)
                    payload = self._research_payloads[artifact.content_hash]
                    if self.generation_coordinator is not None:
                        self.generation_coordinator.stage(artifact, payload)
                        self.generation_coordinator.outbox.publish_pending()
                        active = self.generation_coordinator.resolve_active(artifact.artifact_id)
                        if active is None:
                            raise RuntimeError(
                                "artifact generation is not COMPLETE; approval remains invisible"
                            )
                    elif self.artifact_publisher is not None:
                        artifact = self.artifact_publisher.publish(artifact, payload)
                        self._inject_failure("blob_written_before_registry", artifact)
                    stored = self.registry.register(artifact)
                self._inject_failure("registry_written_before_visibility", stored)
                self._definition_by_artifact_hash[stored.content_hash] = (
                    manifest.factor_definition_ref or ""
                )
                detail = dict(verdict.detail or {})
                self._feature_provenance_by_artifact_hash[stored.content_hash] = {
                    "raw_value_ref": manifest.factor_value_ref,
                    "treatment_selection_ref": detail.get("treatment_selection_ref"),
                    "preprocess_state_ref": detail.get("preprocess_state_ref"),
                    "cluster_version_ref": detail.get("cluster_version_ref"),
                    "treated_feature_ref": detail.get("treated_feature_ref"),
                    "dtype": detail.get("dtype"),
                    "evaluation_ref": evaluation.evaluation_ref,
                    "metric_policy_ref": evaluation.metric_policy_ref,
                    "health_policy_ref": verdict.policy_ref,
                    "health_verdict_ref": verdict.content_hash,
                    "library_version_ref": self._library_ref(library_snapshot),
                }
                report = report.with_approved(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    artifact_id=stored.artifact_id,
                    registry_content_hash=stored.content_hash,
                )
                event_id = self._publish_event(
                    milestone="CANDIDATE_APPROVED",
                    candidate=manifest,
                    reason=_verdict_reason(verdict),
                    status="APPROVED",
                    library_snapshot=library_snapshot,
                    now=now,
                    run_index=run_index,
                    decision=decision_payload,
                    artifact_id=stored.artifact_id,
                )
                report = report.with_event(event_id)
                event_id = self._publish_event(
                    milestone="ARTIFACT_REGISTERED",
                    candidate=manifest,
                    reason=PipelineStatusReason.QRP_GATE_QUALIFIES,
                    status="APPROVED",
                    library_snapshot=library_snapshot,
                    now=now,
                    run_index=run_index,
                    artifact_id=stored.artifact_id,
                )
                report = report.with_event(event_id)
            elif verdict.decision == DECISION_SHADOWED:
                # A SHADOWED verdict is recorded, never applied to the registry.
                report = report.with_shadowed(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=_verdict_reason(verdict),
                )
            else:
                report = report.with_rejected(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=_verdict_reason(verdict),
                )
            if approved:
                # Approval bytes alone do not mean the feature-set update is
                # durable. A crash before its publication must remain retryable.
                approved_pending_consumption.append(manifest)
            else:
                self._record_terminal_consumption(manifest, now)

        # 5. FeatureSet snapshot for the approved group + diff vs previous round.
        if report.approved_content_hashes:
            report = self._snapshot_feature_set(report, library_snapshot)
            if self.config.require_complete_execution_trace:
                manifest = self._modeling_feature_manifests[-1]
                manifest_ref = str(manifest["feature_set_version_ref"])
                for candidate in manifests:
                    trace = self._execution_trace_by_candidate.get(candidate.candidate_id)
                    if trace is None:
                        continue
                    trace["feature_manifest_ref"] = manifest_ref
                    event_id = self._publish_event(
                        milestone="FEATURE_SET_CREATED", candidate=candidate,
                        reason=PipelineStatusReason.QRP_GATE_QUALIFIES,
                        status="APPROVED", library_snapshot=library_snapshot,
                        now=now, run_index=run_index, trace=trace,
                    )
                    report = report.with_event(event_id)

        for completed in approved_pending_consumption:
            self._record_terminal_consumption(completed, now)

        if not retryable_failure:
            self._processed_fingerprints.add(fingerprint)
        # P0-PLAT-004: durably record the batch fingerprint after the batch so
        # a process restart still anti-replays THIS batch. The store is
        # idempotent — a crash mid-run that partially consumed candidates and a
        # replay of the same batch records the fingerprint again harmlessly.
        if self.run_storage is not None and not retryable_failure:
            self.run_storage.record_fingerprint(fingerprint)
        # Anti-replay must survive run() returning: a replay of THIS batch is
        # caught at the top of the next run() via _processed_fingerprints.
        if hasattr(report, "num_replayed") and report.num_replayed:
            report = report.as_replayed()
        return report

    # ---- internals ------------------------------------------------------------
    def _inject_failure(self, stage: str, artifact: ArtifactRef) -> None:
        """Test-only crash seam at publication durability boundaries."""
        if self.failure_injector is not None:
            self.failure_injector(stage, artifact)

    def _build_execution_trace(self, candidate, evaluation, verdict, library_snapshot):
        if evaluation.candidate_ref != candidate.candidate_id:
            raise ValueError("complete execution trace mismatched refs: ['candidate_ref']")
        detail = dict(verdict.detail or {})
        refs = {
            "request_ref": getattr(candidate, "campaign_id", None),
            "parent_trial_ref": getattr(candidate, "attempt_id", None),
            "definition_ref": getattr(candidate, "factor_definition_ref", None),
            "recipe_ref": evaluation.recipe_ref,
            "state_ref": evaluation.preprocess_state_ref,
            "value_ref": evaluation.factor_value_ref,
            "evaluation_ref": evaluation.evaluation_ref,
            "metric_policy_ref": evaluation.metric_policy_ref,
            "health_policy_ref": verdict.policy_ref,
            "health_verdict_ref": verdict.content_hash,
            "library_version_ref": self._library_ref(library_snapshot),
        }
        missing = sorted(name for name, value in refs.items()
                         if not isinstance(value, str) or not value)
        if missing:
            raise ValueError(f"complete execution trace missing refs: {missing}")
        expected = {
            "factor_value_ref": getattr(candidate, "factor_value_ref", None),
            "recipe_ref": detail.get("treatment_selection_ref"),
            "state_ref": detail.get("preprocess_state_ref"),
            "evaluation_ref": detail.get("evaluation_ref"),
        }
        actual = {
            "factor_value_ref": refs["value_ref"], "recipe_ref": refs["recipe_ref"],
            "state_ref": refs["state_ref"], "evaluation_ref": refs["evaluation_ref"],
        }
        mismatched = sorted(name for name in expected
                            if expected[name] != actual[name])
        if mismatched:
            raise ValueError(f"complete execution trace mismatched refs: {mismatched}")
        return {name: str(value) for name, value in refs.items()}

    def _record_terminal_consumption(self, manifest: Any, now: datetime) -> None:
        if any(
            existing.factor_spec_sha256 == manifest.factor_spec_sha256
            for existing in self._consumed_manifests
        ):
            return
        if self.run_storage is not None:
            token, _ = self._reservation_context.get()
            self.run_storage.consume_reserved(
                candidate_id=str(manifest.candidate_id),
                content_hash=manifest.factor_spec_sha256,
                semantic=str(manifest.semantic_family_hint or ""),
                consumed_at=now,
                owner_token=token,
            )
            _, hashes = self._reservation_context.get()
            self._reservation_context.set((
                token, tuple(ch for ch in hashes if ch != manifest.factor_spec_sha256)
            ))
        self._consumed_manifests.append(manifest)
        self._consumed_hash_seed.add(manifest.factor_spec_sha256)

    def _make_candidate_handler(
        self,
        candidate: Any,
        library_snapshot: Mapping[str, Any],
        job_key: str,
    ) -> Callable[[JobSpec], JobResult]:
        def handler(spec: JobSpec) -> JobResult:
            try:
                evaluation = self.evaluate_candidate(
                    candidate,
                    {"library_snapshot": library_snapshot, "job_key": job_key},
                )
            except JobError:
                raise
            except (TimeoutError, ConnectionError, OSError) as exc:
                raise JobError(ErrorClass.RETRYABLE_INFRASTRUCTURE, str(exc)) from exc
            except MemoryError as exc:
                raise JobError(ErrorClass.RESOURCE_EXCEEDED, str(exc)) from exc
            except (TypeError, ValueError) as exc:
                raise JobError(ErrorClass.INVALID_INPUT, str(exc)) from exc
            except Exception as exc:
                raise JobError(ErrorClass.SEMANTIC_CONTRACT, str(exc)) from exc
            if evaluation is None:
                return JobResult(summary=_BAD_EVIDENCE_MARKER)
            self._evaluation_by_job[job_key] = evaluation
            output_refs = tuple(
                r for r in (evaluation.evidence_ref, evaluation.evaluation_ref) if r
            )
            return JobResult(
                summary=_GOOD_EVIDENCE_MARKER, output_artifact_refs=output_refs
            )

        return handler

    def _snapshot_feature_set(
        self,
        report: PipelineReport,
        library_snapshot: Mapping[str, Any],
    ) -> PipelineReport:
        """Build FeatureSetVersion + FeatureSetArtifact and diff vs previous round.

        Returns a new report carrying the snapshot + retrain decision. The
        FeatureSetArtifact is generated (per the 8-step chain) and surfaced on
        the report; only the FACTOR_CANDIDATE artifacts are registered into the
        artifact registry (num_registered stays == approved count).
        """
        parent_generation = None
        self._check_reservations()
        if self.generation_coordinator is not None:
            self.generation_coordinator.outbox.publish_pending()
            active = self.generation_coordinator.resolve_active('feature-set:' + self.config.feature_set_id)
            if active is not None:
                parent_generation = active['generation_id']
                raw = json.loads(bytes.fromhex(active['payload_hex']))['version']
                raw['ordered_members'] = tuple(FeatureMemberRef(**m) for m in raw['ordered_members'])
                raw['source_library_versions'] = tuple(raw['source_library_versions'])
                if raw.get('created_at'):
                    raw['created_at'] = datetime.fromisoformat(raw['created_at'])
                restored = FeatureSetVersion(**raw)
                if not self._feature_snapshots or self._feature_snapshots[-1].version != restored.version:
                    self._feature_snapshots.append(restored)
        revision = int(self._feature_snapshots[-1].version.removeprefix('v')) + 1 if self._feature_snapshots else 1
        existing_members = (
            tuple(self._feature_snapshots[-1].ordered_members)
            if self._feature_snapshots and self.config.feature_set_update_mode == 'MERGE' else ()
        )
        additions = [
            (ch, self._definition_by_artifact_hash.get(ch, ""))
            for ch in report.approved_content_hashes
        ]
        incoming_members = tuple(
            FeatureMemberRef(
                position=offset,
                feature_name=f"factor_{ch[:10]}",
                factor_definition_ref=definition_ref,
                raw_value_ref=self._feature_provenance_by_artifact_hash.get(ch, {}).get("raw_value_ref"),
                treatment_selection_ref=self._feature_provenance_by_artifact_hash.get(ch, {}).get("treatment_selection_ref"),
                treated_feature_ref=self._feature_provenance_by_artifact_hash.get(ch, {}).get("treated_feature_ref"),
                dtype=self._feature_provenance_by_artifact_hash.get(ch, {}).get("dtype"),
                source_artifact_id=f"FC_{ch[:16]}",
                metadata={
                    key: value for key, value in {
                        "preprocess_state_ref": self._feature_provenance_by_artifact_hash.get(ch, {}).get("preprocess_state_ref"),
                        "cluster_version_ref": self._feature_provenance_by_artifact_hash.get(ch, {}).get("cluster_version_ref"),
                        "evaluation_ref": self._feature_provenance_by_artifact_hash.get(ch, {}).get("evaluation_ref"),
                        "metric_policy_ref": self._feature_provenance_by_artifact_hash.get(ch, {}).get("metric_policy_ref"),
                        "health_policy_ref": self._feature_provenance_by_artifact_hash.get(ch, {}).get("health_policy_ref"),
                        "health_verdict_ref": self._feature_provenance_by_artifact_hash.get(ch, {}).get("health_verdict_ref"),
                        "library_version_ref": self._feature_provenance_by_artifact_hash.get(ch, {}).get("library_version_ref"),
                    }.items() if value
                },
            )
            for offset, (ch, definition_ref) in enumerate(additions)
        )
        # A definition ref is the stable logical slot, but provenance is part
        # of its executable identity. MERGE replaces that slot when recipe,
        # state, value, cluster, evaluation, or source artifact changes.
        incoming_by_definition = {
            member.factor_definition_ref: member for member in incoming_members
        }
        merged = []
        for member in existing_members:
            merged.append(incoming_by_definition.pop(member.factor_definition_ref, member))
        merged.extend(incoming_by_definition.values())
        combined_members = tuple(
            replace(member, position=position)
            for position, member in enumerate(merged if existing_members else incoming_members)
        )
        unchanged = bool(self._feature_snapshots and combined_members == self._feature_snapshots[-1].ordered_members)
        if unchanged:
            revision = int(self._feature_snapshots[-1].version.removeprefix('v'))
        version = self._feature_snapshots[-1] if unchanged else FeatureSetVersion(
            feature_set_id=self.config.feature_set_id,
            version=f"v{revision}",
            ordered_members=combined_members,
            consumer_profile=self.config.scope,
            source_library_versions=(
                str(library_snapshot.get("library_version_ref") or "lib:unknown"),
            ),
        )
        if self.generation_coordinator is not None and not unchanged:
            payload = json.dumps({'schema':'feature-set-generation.v1',
                'update_mode':self.config.feature_set_update_mode,
                'parent_generation':parent_generation, 'version':asdict(version)},
                sort_keys=True, separators=(',', ':'), default=lambda obj: obj.isoformat()
                    if isinstance(obj,datetime) else obj.value).encode()
            digest = hashlib.sha256(payload).hexdigest()
            ref = ArtifactRef(artifact_id='feature-set:' + self.config.feature_set_id,
                artifact_type=ARTIFACT_TYPE_FEATURE_SET, schema_version='1.0',
                content_hash=digest, storage_uri=f'feature-set://{self.config.feature_set_id}/{digest}',
                size_bytes=len(payload), created_at=datetime.now(timezone.utc),
                producer_type=_PRODUCER, producer_version='2.0.0')
            with self._publication_fence():
                generation = self.generation_coordinator.stage(
                    ref, payload, expected_parent_generation=parent_generation
                )
                self.generation_coordinator.outbox.publish_pending()
                published = self.generation_coordinator.resolve_active(ref.artifact_id)
                if published is None or published['generation_id'] != generation:
                    raise RuntimeError('feature-set generation not COMPLETE; candidate remains retryable')
                self._inject_failure('feature_set_complete_before_consumption', ref)
        if not unchanged:
            self._feature_snapshots.append(version)
        from quant_platform.app.modeling_manifest_bridge import build_modeling_feature_manifest
        self._modeling_feature_manifests.append(build_modeling_feature_manifest(version))
        artifact = FeatureSetArtifact(
            feature_set_id=version.feature_set_id,
            feature_set_version=version.version,
            source_library_versions=version.source_library_versions,
            ordered_feature_manifest=tuple(
                m.feature_name for m in version.ordered_members
            ),
            created_at=datetime.now(timezone.utc),
        )

        diff_category: str | None = None
        retrain = False
        if len(self._feature_snapshots) >= 2 and not unchanged:
            prev = self._feature_snapshots[-2]
            event = retrain_required_for_diff(prev, version)
            retrain = event is not None
            if event is not None:
                diff_category = event.diff_category.value
            else:
                from quant_platform.app.contracts import classify_feature_set_diff  # noqa: PLC0415

                diff_category = classify_feature_set_diff(prev, version).category.value
                retrain = prev.schema_hash != version.schema_hash
        return report.with_feature_snapshot(
            artifact=artifact,
            revision=revision,
            diff_category=diff_category,
            retrain=retrain,
        )

    def _batch_id_from(self) -> str:
        """Deterministic batch-scoped correlation id for the current run."""
        if self._processed_fingerprints:
            return content_hash(*sorted(self._processed_fingerprints))[:16]
        return content_hash("fresh")[:16]

    @staticmethod
    def _library_ref(library_snapshot: Mapping[str, Any]) -> str:
        """The carried library version ref (platform reads it, never mints it)."""
        return str(
            library_snapshot.get("library_version_ref")
            or library_snapshot.get("library_version_id")
            or ""
        )

    @staticmethod
    def _library_ref(library_snapshot: Mapping[str, Any]) -> str:
        """The carried library version ref (platform reads it, never mints it)."""
        return str(
            library_snapshot.get("library_version_ref")
            or library_snapshot.get("library_version_id")
            or ""
        )

    def _build_candidate_artifact(
        self,
        candidate: Any,
        now: datetime,
        library_snapshot: Mapping[str, Any],
    ) -> ArtifactRef:
        """Build the FACTOR_CANDIDATE ArtifactRef registered on approval."""
        source_spec_hash = candidate.factor_spec_sha256
        payload = json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "factor_definition_ref": candidate.factor_definition_ref,
                "semantic_ref": candidate.semantic_ref,
                "factor_value_ref": candidate.factor_value_ref,
                "source_spec_hash": source_spec_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        ch = hashlib.sha256(payload).hexdigest()
        artifact_id = f"FC_{ch[:16]}"
        self._research_payloads[ch] = payload
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE,
            schema_version="1.0",
            content_hash=ch,
            storage_uri=f"mem://candidates/{artifact_id}.json",
            size_bytes=len(payload),
            created_at=now,
            producer_type=_PRODUCER,
            producer_version=_PRODUCER_VERSION,
            semantic_hash=candidate.semantic_ref or "",
            media_type="application/json",
            producer_source_ref=candidate.factor_spec_uri,
            snapshot_ref=str(
                library_snapshot.get("library_version_ref")
                or library_snapshot.get("library_version_id")
                or ""
            ),
            universe_ref=None,
            security_classification="INTERNAL_RESEARCH",
        )

    def _publish_event(
        self,
        *,
        milestone: str,
        candidate: Any,
        reason: str,
        status: str,
        library_snapshot: Mapping[str, Any],
        now: datetime,
        run_index: int,
        decision: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        trace: Mapping[str, str] | None = None,
    ) -> str:
        """Append one milestone EventEnvelope to the outbox; returns the id."""
        content_ch = getattr(candidate, "factor_spec_sha256", "") or str(
            getattr(candidate, "candidate_id", "")
        )
        candidate_id = str(getattr(candidate, "candidate_id", "") or content_ch[:16])
        key = (
            f"{milestone}:{content_ch}:run:{run_index}:{reason}:"
            f"{'ok' if status == 'APPROVED' else status}"
        )
        if artifact_id:
            key = f"{key}:artifact:{artifact_id}"
        event_id = content_hash(milestone, content_ch, run_index, reason, status)
        if artifact_id:
            event_id = content_hash(event_id, artifact_id)
        payload: dict[str, Any] = {
            "milestone": milestone,
            "status": status,
            "candidate_id": candidate_id,
            "content_hash": content_ch,
            "reason": reason,
            "library_version_ref": str(
                library_snapshot.get("library_version_ref")
                or library_snapshot.get("library_version_id")
                or ""
            ),
        }
        if decision is not None:
            payload["decision"] = dict(decision)
            codes = decision.get("reason_codes")
            if isinstance(codes, (list, tuple)):
                payload["reason_codes"] = tuple(str(c) for c in codes)
        if artifact_id:
            payload["artifact_id"] = artifact_id
        if trace is not None:
            payload["execution_trace"] = dict(trace)

        envelope = EventEnvelope(
            event_id=event_id,
            event_type=_PIPELINE_EVENT_TYPE,
            schema_version=_EVENT_SCHEMA_VERSION,
            occurred_at=now,
            producer=_PRODUCER,
            aggregate_type="factor_candidate",
            aggregate_id=f"scenario:{run_index}",
            correlation_id=f"batch:{self._batch_id_from()}",
            causation_id=None,
            payload=payload,
            idempotency_key=key,
            trace_id=(content_hash(*[trace[k] for k in sorted(trace)]) if trace else None),
        )
        # Deterministic correlation id: batch-level fingerprint of the consumed
        # fingerprint is not available here; use the per-run consumed set.
        object.__setattr__(
            envelope,
            "correlation_id",
            f"batch:{content_hash(*(sorted(self._processed_fingerprints) or ('__fresh__',))):.16s}",
        )
        self.outbox.append(envelope, idempotency_key=key, event_id=event_id)
        return event_id


def _reject_reason(reason_codes: Iterable[str]) -> str:
    """Map a delegated authority's REJECT reason codes to a QRP reason string.

    Pure attribution mapping (P0-6): the codes were produced by the domain
    authority; this only renames them into the platform status vocabulary.
    """
    codes = [str(c) for c in reason_codes]
    if "rank_ic_below_threshold" in codes:
        return PipelineStatusReason.QRP_GATE_REJECTED_RANK_IC
    if "label_not_mature" in codes:
        return PipelineStatusReason.QRP_GATE_REJECTED_LABEL
    if "return_basis_wrong" in codes:
        return PipelineStatusReason.QRP_GATE_REJECTED_BASIS
    if "duplicate_of_existing_member" in codes:
        return PipelineStatusReason.QRP_DUPLICATE_SKIPPED
    if "admission_authority_absent" in codes:
        return PipelineStatusReason.QRP_ADMISSION_AUTHORITY_ABSENT
    return PipelineStatusReason.QRP_ADMISSION_REJECTED


def _verdict_reason(verdict: AdmissionVerdict) -> str:
    """Map a DELEGATED admission verdict to a QRP pipeline reason string."""
    if verdict.decision == DECISION_APPROVED:
        return PipelineStatusReason.QRP_GATE_QUALIFIES
    if verdict.decision == DECISION_SHADOWED:
        return PipelineStatusReason.QRP_ADMISSION_SHADOWED
    return _reject_reason(verdict.reason_codes)
