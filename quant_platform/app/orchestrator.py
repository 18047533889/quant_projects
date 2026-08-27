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
      -> PromotionGate decision over a PURE-DTO dict surface (stdlib only —
         mirrors ``factor_assets.library.promotion_gate`` semantics with NO
         cross-package import: label maturity / vwap->vwap basis / |rank_ic|
         floor / duplicate-similarity)
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
the injectable ``evaluate_candidate`` seam. The default seam fails closed —
approving nothing without real computed evidence (never a fabricated pass).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable

from quant_platform.app.candidate.ingest import (
    CandidateNormalizationError,
    ReconcileReason,
    batch_fingerprint,
    normalize_candidate,
    reconcile_candidates,
)
from quant_platform.app.contracts import (
    ARTIFACT_TYPE_FACTOR_CANDIDATE,
    ArtifactRef,
    EventEnvelope,
    FeatureMemberRef,
    FeatureSetArtifact,
    FeatureSetVersion,
    JobResult,
    JobSpec,
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
    "PipelineStatusReason",
    "CandidateEvaluation",
    "Pipeline",
    "pipeline_gate_decision",
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

#: Default minimum abs rank_ic for promotion. Matches the FA promotion-gate
#: default (0.02) — see Memory: vwap-to-vwap 收益口径.
DEFAULT_MIN_RANK_IC = 0.02
VWAP_TO_VWAP_BASIS = "vwap_to_vwap"

#: QE ``EvidenceStatus`` string values meaning "label not mature / evidence not
#: computed" (reference-only; this module imports no domain package).
_EVIDENCE_NOT_COMPUTED: tuple[str, ...] = (
    "not_computed",
    "label_not_mature",
    "insufficient_data",
    "unavailable",
    "unsupported",
    "invalid_evidence",
    "failed",
)


# --------------------------------------------------------------------------- #
# Status and report
# --------------------------------------------------------------------------- #


class PipelineStatusReason:
    """Machine-parseable per-candidate status reason strings."""

    QRP_OK = "QRP_OK"
    QRP_DUPLICATE_SKIPPED = "QRP_DUPLICATE_SKIPPED"
    QRP_CONFLICT_SKIPPED = "QRP_CONFLICT_SKIPPED"
    QRP_GATE_QUALIFIES = "QRP_GATE_QUALIFIES"
    QRP_GATE_REJECTED_RANK_IC = "QRP_GATE_REJECTED_RANK_IC"
    QRP_GATE_REJECTED_LABEL = "QRP_GATE_REJECTED_LABEL"
    QRP_GATE_REJECTED_BASIS = "QRP_GATE_REJECTED_BASIS"
    QRP_GATE_REVIEW_UNKNOWN = "QRP_GATE_REVIEW_UNKNOWN"
    QRP_EVALUATION_FAILED = "QRP_EVALUATION_FAILED"
    QRP_NO_EVIDENCE = "QRP_NO_EVIDENCE"
    QRP_REPLAY_DETECTED = "QRP_REPLAY_DETECTED"
    QRP_NORMALIZATION_FAILED = "QRP_NORMALIZATION_FAILED"

    @classmethod
    def catalog(cls) -> frozenset[str]:
        return frozenset(v for k, v in vars(cls).items() if k.startswith("QRP_"))


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

    #: Per-candidate terminal states: dicts with ``candidate_id`` /
    #: ``content_hash`` / ``status`` (APPROVED / REJECTED / DUPLICATE_SKIPPED /
    #: FAILED / NEW_CONSUMED) and a machine-parseable ``reason``.
    states: tuple[dict[str, str], ...] = ()
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
    ) -> "PipelineReport":
        """New report with one per-candidate state appended (never mutates)."""
        item = {
            "candidate_id": candidate_id,
            "content_hash": content_hash,
            "status": status,
            "reason": reason,
        }
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

    def with_failed(
        self, *, candidate_id: str, content_hash: str, reason: str
    ) -> "PipelineReport":
        return replace(self, num_evaluation_failed=self.num_evaluation_failed + 1).with_item(
            candidate_id=candidate_id,
            status="FAILED",
            reason=reason,
            content_hash=content_hash,
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
    """PURE-DTO evaluation surface for the promotion decision.

    Mirrors the ``CandidateEvaluationRef`` shape of
    ``factor_assets.library.promotion_gate`` as a plain frozen dataclass — this
    orchestrator never imports factor_assets; the gate decision is a local pure
    decision over exactly these fields.
    """

    candidate_ref: str
    rank_ic: float | None
    label_maturity: bool = True
    evidence_status: str | None = None
    return_basis: str = VWAP_TO_VWAP_BASIS
    evidence_ref: str | None = None
    evaluation_ref: str | None = None
    treatment_optimization_ref: str | None = None

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
    """Public helper: build a PURE-DTO evaluation from a computed-evidence dict.

    ``evidence`` keys: ``candidate_ref``, ``rank_ic``, ``label_maturity``
    (default True), ``evidence_status`` (default ``"computed"``),
    ``return_basis`` (default vwap->vwap), ``evidence_ref`` / ``evaluation_ref``
    / ``treatment_optimization_ref``. Label not mature raises — a not-computed
    label must never be dressed up as computed evidence.
    """
    candidate_ref = str(evidence.get("candidate_ref") or "")
    if not candidate_ref:
        raise ValueError("build_with_evidence: candidate_ref is required")
    maturity = bool(evidence.get("label_maturity", True))
    if not maturity:
        raise ValueError(
            "build_with_evidence: label_maturity must be True for computed evidence"
        )
    return CandidateEvaluation(
        candidate_ref=candidate_ref,
        rank_ic=evidence.get("rank_ic"),
        label_maturity=True,
        evidence_status=str(
            evidence.get("evidence_status") or "computed"
        ),
        return_basis=str(
            evidence.get("return_basis") or VWAP_TO_VWAP_BASIS
        ),
        evidence_ref=evidence.get("evidence_ref"),
        evaluation_ref=evidence.get("evaluation_ref"),
        treatment_optimization_ref=evidence.get("treatment_optimization_ref"),
    )


# --------------------------------------------------------------------------- #
# Promotion decision (PURE-DTO, stdlib only)
# --------------------------------------------------------------------------- #


def pipeline_gate_decision(
    evaluation: CandidateEvaluation,
    library_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure promotion decision mirroring the FA ``PromotionGate`` semantics.

    Operates on the PURE-DTO dict surface only (no cross-package import) over
    how ``factor_assets.library.promotion_gate.PromotionGate.evaluate`` decides:

    * label not mature / evidence-status in the not-computed set -> REJECT
      ``label_not_mature`` (fail closed);
    * return basis != ``vwap_to_vwap`` -> REJECT ``return_basis_wrong``
      (defensive — see Memory: vwap-to-vwap 收益口径);
    * ``rank_ic`` absent or ``|rank_ic|`` below the floor -> REJECT
      ``rank_ic_below_threshold``;
    * candidate similar (above threshold) to an existing library member ->
      REJECT ``duplicate_of_existing_member`` when ``reject_duplicates=True``
      (default), else REVIEW ``merge_suggested``;
    * library has members but uniqueness cannot be measured (no similarity_fn
      or a ``None`` measurement) -> forced REVIEW ``merge_suggested`` (never a
      silent APPROVE);
    * otherwise APPROVE ``qualifies``.

    ``library_snapshot`` keys: ``library_version_ref`` (required),
    ``min_rank_ic``, ``reject_duplicates``, ``duplicate_similarity_threshold``,
    ``member_refs`` (existing library member refs), ``similarity_fn``.

    Returns a plain ``{"decision", "reason_codes", "library_version_ref"}``.
    """
    library_version_ref = str(
        library_snapshot.get("library_version_ref")
        or library_snapshot.get("library_version_id")
        or ""
    )
    if not library_version_ref:
        raise ValueError("pipeline_gate_decision requires library_version_ref")
    min_rank_ic = float(library_snapshot.get("min_rank_ic", DEFAULT_MIN_RANK_IC))
    if min_rank_ic < 0:
        raise ValueError("min_rank_ic must be non-negative")
    reject_duplicates = bool(library_snapshot.get("reject_duplicates", True))
    similarity_threshold = float(
        library_snapshot.get("duplicate_similarity_threshold", 0.7)
    )
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("duplicate_similarity_threshold must be in [0, 1]")
    member_refs = tuple(str(m) for m in (library_snapshot.get("member_refs") or ()))
    similarity_fn = library_snapshot.get("similarity_fn")

    reject_codes: list[str] = []
    if not evaluation.label_maturity or (
        evaluation.evidence_status is not None
        and evaluation.evidence_status in _EVIDENCE_NOT_COMPUTED
    ):
        reject_codes.append("label_not_mature")
    if evaluation.return_basis != VWAP_TO_VWAP_BASIS:
        reject_codes.append("return_basis_wrong")
    if evaluation.rank_ic is None or abs(evaluation.rank_ic) < min_rank_ic:
        reject_codes.append("rank_ic_below_threshold")

    measured: dict[str, float] = {}
    unmeasured = False
    no_measurement = False
    is_duplicate = False
    if member_refs:
        if similarity_fn is None:
            no_measurement = True
        else:
            for member_ref in member_refs:
                value = similarity_fn(evaluation.candidate_ref, member_ref)
                if value is None:
                    unmeasured = True
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"similarity_fn returned a non-number for "
                        f"({evaluation.candidate_ref!r}, {member_ref!r}): {value!r}"
                    )
                number = float(value)
                if number != number or number in (float("inf"), float("-inf")):
                    raise ValueError(
                        f"similarity_fn returned a non-finite value — fail closed"
                    )
                measured[member_ref] = number
            if measured and any(v > similarity_threshold for v in measured.values()):
                is_duplicate = True

    if is_duplicate:
        if not reject_duplicates:
            return {
                "decision": "REVIEW",
                "reason_codes": ["merge_suggested"],
                "library_version_ref": library_version_ref,
            }
        reject_codes.append("duplicate_of_existing_member")

    if reject_codes:
        return {
            "decision": "REJECT",
            "reason_codes": reject_codes,
            "library_version_ref": library_version_ref,
        }
    if unmeasured or no_measurement:
        return {
            "decision": "REVIEW",
            "reason_codes": ["merge_suggested"],
            "library_version_ref": library_version_ref,
        }
    return {
        "decision": "APPROVE",
        "reason_codes": ["qualifies"],
        "library_version_ref": library_version_ref,
    }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""

    scope: str = "pipeline:test"
    min_rank_ic: float = DEFAULT_MIN_RANK_IC
    retrain_policy: RetrainPolicy | None = None
    feature_set_id: str = "fs_pipeline_default"


def _default_evaluate(candidate: Any, context: Mapping[str, Any]) -> CandidateEvaluation:
    """Fail-closed default evaluator: never fabricates computed evidence.

    When no per-candidate evaluator is injected the pipeline cannot claim real
    vwap-derived rank_ic — the default refuses to approve anything (label not
    mature / no evidence). Silent fabrication of a pass is exactly the failure
    mode the promotion gate is built to reject.
    """
    return CandidateEvaluation(
        candidate_ref=str(getattr(candidate, "candidate_id", "")),
        rank_ic=None,
        label_maturity=False,
        evidence_status="not_computed",
        return_basis=VWAP_TO_VWAP_BASIS,
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
        CandidateEvaluation``. Default: fail-closed (nothing approved).
    config : sizing / policy switches.

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
        config: PipelineConfig | None = None,
        worker_id: str = "qrpp5-worker",
    ) -> None:
        self.registry = registry if registry is not None else ArtifactRegistry()
        self.outbox = outbox if outbox is not None else InMemoryOutbox()
        self.runner = runner if runner is not None else JobRunner(worker_id=worker_id)
        self.config = config if config is not None else PipelineConfig()
        self.evaluate_candidate = evaluate_candidate or _default_evaluate

        self._processed_fingerprints: set[str] = set()
        self._consumed_manifests: list[Any] = []
        self._feature_snapshots: list[FeatureSetVersion] = []
        self._run_counter = 0
        self._evaluation_by_job: dict[str, CandidateEvaluation] = {}

    # ---- queries -------------------------------------------------------------
    @property
    def processed_batches(self) -> tuple[str, ...]:
        return tuple(sorted(self._processed_fingerprints))

    def feature_snapshots(self) -> tuple[FeatureSetVersion, ...]:
        return tuple(self._feature_snapshots)

    def job_records(self) -> tuple[Any, ...]:
        return self.runner.records()

    # ---- run -----------------------------------------------------------------
    def run(
        self,
        raw_records: Iterable[Mapping[str, Any]],
        *,
        library_snapshot: Mapping[str, Any],
        now: datetime | None = None,
    ) -> PipelineReport:
        """Execute one full pipeline pass and return an immutable report.

        Steps: normalize -> fingerprint anti-replay -> reconcile (NEW only) ->
        JobRunner-scheduled treatment+evaluation per candidate -> promotion gate
        decision -> register approved -> FeatureSet snapshot + diff. Milestone
        events are appended to the outbox as candidates move between phases.
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
                        "content_hash": "",
                        "status": "FAILED",
                        "reason": f"{PipelineStatusReason.QRP_NORMALIZATION_FAILED}:{exc.reason}",
                    }
                )
        fingerprint = batch_fingerprint(manifests)

        # 2. Anti-replay: a batch whose fingerprint we already consumed is a
        #    replay — short-circuit (no new events, all candidates skipped).
        if fingerprint in self._processed_fingerprints:
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
                report = report.with_item(**item)
            return report

        self._run_counter += 1
        run_index = self._run_counter
        report = PipelineReport(batch_fingerprint=fingerprint)
        for item in normalization_failures:
            report = report.with_item(**item)

        # Known registry for reconciliation: every manifest this Pipeline has
        # consumed across past runs (a duplicate content/semantic identity is a
        # replay of an earlier consumed candidate).
        known = list(self._consumed_manifests)
        reconcile = reconcile_candidates(manifests, known)

        # Per-candidate classification via the dual-key reconcile logic.
        # reconcile_candidates reconciles within the batch itself as well as
        # against the known registry, so a same-raw-record-twice batch surfaces
        # DUPLICATE_EXACT on the second occurrence. Recompute the dual-key
        # classes here (content + semantic exact-match = duplicate) so only
        # genuinely NEW candidates enter the job + gate path.
        reconcile_conflict_by_hash: dict[str, str] = {}
        for conflict in reconcile.conflicts:
            reconcile_conflict_by_hash[conflict.content_hash] = conflict.reason

        seen_dual_key: set[tuple[str, str]] = set()
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
            self._consumed_manifests.append(manifest)

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
                report = report.with_failed(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=PipelineStatusReason.QRP_EVALUATION_FAILED,
                )
                continue

            evaluation = self._evaluation_by_job.get(job_key)
            if evaluation is None or evaluation.rank_ic is None:
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

            # Promotion gate decision over the PURE-DTO surface.
            decision = pipeline_gate_decision(evaluation, library_snapshot).copy()
            approved = decision["decision"] == "APPROVE"
            event_id = self._publish_event(
                milestone="CANDIDATE_APPROVED" if approved else "CANDIDATE_REJECTED",
                candidate=manifest,
                reason=(
                    PipelineStatusReason.QRP_GATE_QUALIFIES
                    if approved
                    else _decision_reason(decision)
                ),
                status="APPROVED" if approved else "REJECTED",
                library_snapshot=library_snapshot,
                now=now,
                run_index=run_index,
                decision=decision,
            )
            report = report.with_event(event_id)

            if approved:
                artifact = self._build_candidate_artifact(manifest, now, library_snapshot)
                stored = self.registry.register(artifact)
                report = report.with_approved(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    artifact_id=stored.artifact_id,
                    registry_content_hash=stored.content_hash,
                )
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
            else:
                reason = _decision_reason(decision)
                report = report.with_rejected(
                    candidate_id=manifest.candidate_id,
                    content_hash=ch,
                    reason=reason,
                )

        # 5. FeatureSet snapshot for the approved group + diff vs previous round.
        if report.approved_content_hashes:
            report = self._snapshot_feature_set(report, library_snapshot)

        self._processed_fingerprints.add(fingerprint)
        # Anti-replay must survive run() returning: a replay of THIS batch is
        # caught at the top of the next run() via _processed_fingerprints.
        if hasattr(report, "num_replayed") and report.num_replayed:
            report = report.as_replayed()
        return report

    # ---- internals ------------------------------------------------------------
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
            except Exception as exc:
                raise JobError(ErrorClass.CAPABILITY, str(exc)) from exc
            if evaluation is None:
                return JobResult(summary=_BAD_EVIDENCE_MARKER)
            self._evaluation_by_job[job_key] = evaluation
            if evaluation.rank_ic is None:
                return JobResult(summary=_BAD_EVIDENCE_MARKER)
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
        revision = len(self._feature_snapshots) + 1
        version = FeatureSetVersion(
            feature_set_id=self.config.feature_set_id,
            version=f"v{revision}",
            ordered_members=tuple(
                FeatureMemberRef(
                    position=idx,
                    feature_name=f"factor_{ch[:10]}",
                    factor_definition_ref=ch,
                    source_artifact_id="",
                )
                for idx, ch in enumerate(report.approved_content_hashes)
            ),
            consumer_profile=self.config.scope,
            source_library_versions=(
                str(library_snapshot.get("library_version_ref") or "lib:unknown"),
            ),
        )
        self._feature_snapshots.append(version)
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
        if len(self._feature_snapshots) >= 2:
            prev = self._feature_snapshots[-2]
            event = retrain_required_for_diff(prev, version)
            retrain = event is not None
            if event is not None:
                diff_category = event.diff_category.value
            else:
                from quant_platform.app.contracts import classify_feature_set_diff  # noqa: PLC0415

                diff_category = classify_feature_set_diff(prev, version).category.value
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

    def _build_candidate_artifact(
        self,
        candidate: Any,
        now: datetime,
        library_snapshot: Mapping[str, Any],
    ) -> ArtifactRef:
        """Build the FACTOR_CANDIDATE ArtifactRef registered on approval."""
        ch = candidate.factor_spec_sha256
        artifact_id = f"FC_{ch[:16]}"
        payload = f"{candidate.candidate_id}:{ch}"
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE,
            schema_version="1.0",
            content_hash=ch,
            storage_uri=f"mem://candidates/{artifact_id}.json",
            size_bytes=len(payload.encode("utf-8")),
            created_at=now,
            producer_type=_PRODUCER,
            producer_version=_PRODUCER_VERSION,
            semantic_hash=candidate.semantic_family_hint or "",
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
    """Map promote-gate REJECT reason codes to a QRP pipeline reason string."""
    codes = [str(c) for c in reason_codes]
    if "rank_ic_below_threshold" in codes:
        return PipelineStatusReason.QRP_GATE_REJECTED_RANK_IC
    if "label_not_mature" in codes:
        return PipelineStatusReason.QRP_GATE_REJECTED_LABEL
    if "return_basis_wrong" in codes:
        return PipelineStatusReason.QRP_GATE_REJECTED_BASIS
    if "duplicate_of_existing_member" in codes:
        return PipelineStatusReason.QRP_DUPLICATE_SKIPPED
    return "QRP_GATE_REJECTED"


def _decision_reason(decision: Mapping[str, Any]) -> str:
    """Map a promote-gate decision dict to a QRP pipeline reason string."""
    decision_value = str(decision.get("decision", ""))
    reason_codes = decision.get("reason_codes") or ()
    if decision_value == "APPROVE":
        return PipelineStatusReason.QRP_GATE_QUALIFIES
    if decision_value == "REVIEW":
        return PipelineStatusReason.QRP_GATE_REVIEW_UNKNOWN
    return _reject_reason(reason_codes)