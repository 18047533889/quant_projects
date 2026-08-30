"""AdmissionAuthority — the delegation seam for admission decisions (R55 P0-6).

THE RULE (task #95 / R55 audit P0-6): whether a factor / cluster / model is
admitted to production is a **domain decision**. The authority is the owning
domain package — for factors and clusters that is ``factor_assets``
(``factor_assets.contracts.admission.FactorAdmissionArtifact`` /
``factor_assets.library.promotion_gate.PromotionGate``); the platform layer
must NOT import it (platform contract, see ``quant_platform/__init__.py``).

So the platform defines only a **port**: the :class:`AdmissionAuthority`
``typing.Protocol`` plus the two value objects it exchanges:

* :class:`AdmissionRequest` — the *carried* evidence refs for one candidate
  (identity refs + the domain's own evaluation DTO), all opaque to the platform;
* :class:`AdmissionVerdict` — the domain's decision **verbatim**: decision label,
  reason codes, the evidence/decision content hash the domain computed and the
  domain policy that produced it. The platform records it (outbox / registry /
  report) but never computes, re-derives, threshold-checks or second-guesses it.

Fail-closed default: with no authority injected, :class:`RefuseAdmission`
returns a ``REJECTED`` verdict with reason ``admission_authority_absent`` — the
platform never approves anything it did not delegate, and never fabricates a
pass. The adapter implementing the Protocol lives in the domain package or the
app-level composition root (dependency inversion: platform <- domain).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .identities import require_non_empty, sha256_hex

__all__ = [
    "DECISION_APPROVED",
    "DECISION_REJECTED",
    "DECISION_SHADOWED",
    "ADMISSION_DECISIONS",
    "REASON_AUTHORITY_ABSENT",
    "AdmissionRequest",
    "AdmissionVerdict",
    "AdmissionAuthority",
    "RefuseAdmission",
]

#: Verdict labels — the vocabulary of the owning domain package
#: (``factor_assets.contracts.admission.AdmissionDecision``), carried as plain
#: strings so the platform needs no domain import.
DECISION_APPROVED = "APPROVED"
DECISION_REJECTED = "REJECTED"
DECISION_SHADOWED = "SHADOWED"
ADMISSION_DECISIONS: frozenset[str] = frozenset(
    {DECISION_APPROVED, DECISION_REJECTED, DECISION_SHADOWED}
)

#: Reason recorded when no authority was injected — the fail-closed default.
REASON_AUTHORITY_ABSENT = "admission_authority_absent"


@dataclass(frozen=True)
class AdmissionRequest:
    """The carried evidence for one admission decision (all domain-owned).

    The platform assembles this from refs it already carries — it never
    computes any of them:

    ``candidate_ref``
        The platform's own pointer to the candidate row (platform-owned id).
    ``content_hash``
        The candidate's *reported* spec sha256 (publisher-carried, format-checked).
    ``factor_definition_ref``
        The domain's factor-definition identity digest (opaque).
    ``semantic_ref``
        The domain's semantic identity digest (opaque; may equal
        ``factor_definition_ref`` for producers that mint one digest).
    ``evaluation``
        The domain's own evaluation DTO verbatim (opaque to the platform — the
        platform does not read ``rank_ic`` / return-basis / maturity out of it;
        judging them is the authority's job).
    ``library_snapshot_ref``
        The library version the decision is asked against (carried ref).
    """

    candidate_ref: str
    content_hash: str
    factor_definition_ref: str = ""
    semantic_ref: str = ""
    evaluation: Any = None
    library_snapshot_ref: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.candidate_ref, "candidate_ref")
        sha256_hex(self.content_hash, "content_hash")


@dataclass(frozen=True)
class AdmissionVerdict:
    """The delegated decision, recorded by the platform verbatim.

    ``decision`` ∈ :data:`ADMISSION_DECISIONS`; ``reason_codes`` and
    ``content_hash`` are the authority's own outputs (carried, never recomputed
    by the platform). ``content_hash`` is the authority's digest over its own
    decision content (e.g. ``FactorAdmissionArtifact.content_hash``) — the
    platform stores it so two runs of the same authority on the same evidence
    are detectably identical.
    """

    decision: str
    reason_codes: tuple[str, ...] = ()
    content_hash: str = ""
    policy_ref: str = ""
    authority: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in ADMISSION_DECISIONS:
            raise ValueError(
                f"unknown admission decision {self.decision!r} — the authority's "
                f"verdict vocabulary is {sorted(ADMISSION_DECISIONS)}"
            )
        object.__setattr__(
            self, "reason_codes", tuple(str(c) for c in (self.reason_codes or ()))
        )
        object.__setattr__(self, "detail", dict(self.detail or {}))
        if self.content_hash:
            # carried from the authority; format-checked only
            sha256_hex(self.content_hash, "content_hash")

    @property
    def approved(self) -> bool:
        return self.decision == DECISION_APPROVED

    @property
    def rejected(self) -> bool:
        return self.decision == DECISION_REJECTED

    @property
    def review_required(self) -> bool:
        return self.decision == DECISION_SHADOWED

    def reason_codes_list(self) -> list[str]:
        """JSON-friendly ordered reason-code list."""
        return list(self.reason_codes)


@runtime_checkable
class AdmissionAuthority(Protocol):
    """Port the platform calls; the domain adapter implements it.

    Implementations live in the owning domain package (or the app composition
    root) — e.g. a thin adapter over ``factor_assets.library.promotion_gate`` /
    ``factor_assets`` selection policy. The platform only types the seam; it
    never imports the implementation, so the dependency arrow stays
    ``domain -> platform`` (adapter side) and never ``platform -> domain``.
    """

    def decide(self, request: AdmissionRequest) -> AdmissionVerdict:
        """Return the domain's admission verdict for one candidate."""
        ...


class RefuseAdmission:
    """Fail-closed authority: refuses everything with a machine-parseable reason.

    This is the default when the composition root injects nothing. It guarantees
    the platform can never approve a candidate on its own judgement — the exact
    P0-6 failure mode — while keeping the pipeline runnable end-to-end.
    """

    def decide(self, request: AdmissionRequest) -> AdmissionVerdict:
        return AdmissionVerdict(
            decision=DECISION_REJECTED,
            reason_codes=(REASON_AUTHORITY_ABSENT,),
            policy_ref="",
            authority="quant_platform.contracts.admission.RefuseAdmission",
            detail={"candidate_ref": request.candidate_ref},
        )