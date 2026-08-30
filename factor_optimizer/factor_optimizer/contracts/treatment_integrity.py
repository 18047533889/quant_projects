"""TreatmentIntegrityEvidence: real integrity evidence for a treatment run (R55 P0-9).

The historical "integrity gates" in this package were FAKE:

- ``factor_optimizer.search.treatment_decision.TreatmentDecisionPolicy`` ran
  "STEP 1 Integrity Hard Gates" that re-derived their verdicts from the very
  scalar metrics the pipeline was about to score.  Nothing recorded WHAT had
  been checked, WHICH treatment had actually been applied to the data, or what
  the treated data looked like before/after — so a treatment that was never
  applied (or applied with different parameters than claimed) sailed through
  the gates as long as its headline metrics looked plausible.
- ``factor_optimizer.policy.decisions.AdmissionPolicy`` declared
  ``require_parent_evidence`` without ever reading it: the criterion was
  dead code and the admission verdict passed unconditionally.

This module supplies the evidence artifact those gates now REQUIRE, fail
closed:

- :class:`IntegrityCheckResult` — one named check with its measured value and
  pass/fail verdict.
- :class:`TreatmentIntegrityStatus` — PASSED / FAILED / NOT_RUN (derived from
  the checks, never self-reported).
- :class:`TreatmentIntegrityEvidence` — deep-immutable, content-hashed record
  of what was actually applied and verified: treatment id/kind, the frozen
  applied parameters, the ordered checks, the before/after content digests of
  the treated data, and a derived sha256 over all of it.
- :func:`digest_value` / :func:`build_integrity_evidence` — the honest way to
  PRODUCE evidence: the digests are computed here from the actual data, never
  accepted from the caller.
- :func:`require_integrity_evidence` — the fail-closed gate.  Missing, stale,
  mis-bound, tampered, NOT_RUN or FAILED evidence raises
  :class:`~factor_optimizer.errors.TreatmentIntegrityError`.
- :func:`describe_integrity_problem` — the reporting variant used by the
  admission policy, which records a REJECTED decision instead of raising.

Hash style follows the FO contracts (``MultiplicityArtifact``,
``TreatmentOptimizationResultArtifact``): a canonical ``json.dumps(...,
sort_keys=True, separators=(",", ":"))`` payload over the semantic fields,
sha256 hex digest, and a ``verify()`` that fails closed on post-construction
tampering.  ``content_hash`` is *derived-only*: it is a property, so a caller
cannot self-report an arbitrary hash, and ``from_dict`` rejects any supplied
hash that disagrees with the recomputed value.

``created_at`` / ``produced_by`` are provenance and are deliberately excluded
from the hash: two evidences of the same verified run share a hash regardless
of when they were written.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from factor_optimizer.errors import TreatmentIntegrityError

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "RAW_TREATMENT_KIND",
    "IntegrityCheckResult",
    "TreatmentIntegrityStatus",
    "TreatmentIntegrityEvidence",
    "digest_value",
    "build_integrity_evidence",
    "require_integrity_evidence",
    "describe_integrity_problem",
]


#: Schema version of the evidence artifact (bumped on any field change).
EVIDENCE_SCHEMA_VERSION = "1.0"

#: The treatment kind of the un-treated RAW baseline.  RAW is the only kind
#: whose before/after digests are EXPECTED to be identical: applying "no
#: treatment" must not change the data.  Every other kind must change it —
#: a treatment whose output is byte-identical to its input did not run.
RAW_TREATMENT_KIND = "raw"


def _freeze(value: Any) -> Any:
    """Recursively freeze a nested structure into immutable equivalents.

    dict/Mapping -> ``MappingProxyType``, list/tuple -> tuple,
    set/frozenset -> frozenset, scalars pass through.  Any other type is
    rejected so the frozen field is provably immutable (fail closed rather
    than silently hashing an unstable object).
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    raise TypeError(
        "cannot deep-freeze value of unsupported type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _jsonable(value: Any) -> Any:
    """Canonical JSON-safe projection of a frozen/plain nested structure.

    Fail closed: unsupported types and non-finite floats raise instead of
    being coerced into an unstable ``repr``.  Mappings are key-sorted and
    sets are order-normalized so equal content always encodes identically.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "non-finite float is not evidence content (NaN/Inf rejected)"
            )
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(
                ((str(k), v) for k, v in value.items()), key=lambda kv: kv[0]
            )
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(
        "cannot canonicalize value of unsupported type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _json_digest(payload: Any) -> str:
    """sha256 hex digest over a canonical JSON payload."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .encode("utf-8")
    ).hexdigest()


def _data_envelope(value: Any) -> bytes:
    """Deterministic byte envelope of the treated data, for content digesting.

    Supports bytes, str, numpy arrays (``dtype``/``shape``/``tobytes``),
    pandas objects (``to_numpy``), mappings and sequences.  Everything else is
    rejected — a digest that silently falls back to ``repr`` would embed
    memory addresses and could not be reproduced across processes.
    """
    if value is None:
        return b"none:"
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    # numpy ndarray (duck-typed: no numpy import needed in this module).
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        header = (
            f"ndarray:{value.dtype.str}:{tuple(value.shape)}:".encode("ascii")
        )
        return header + value.tobytes()
    # pandas Series / DataFrame (duck-typed).
    if hasattr(value, "to_numpy"):
        columns = getattr(value, "columns", None)
        header = b"pandas:"
        if columns is not None:
            header += (
                f"columns={tuple(str(c) for c in columns)}:".encode("utf-8")
            )
        return header + value.to_numpy().tobytes()
    if isinstance(value, Mapping):
        return b"map:" + json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    if isinstance(value, (list, tuple, set, frozenset)):
        return b"seq:" + json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    raise TypeError(
        "cannot digest data of unsupported type "
        f"{type(value).__module__}.{type(value).__qualname__}; pass bytes, "
        "str, an ndarray/Series/DataFrame, or a JSON-safe mapping/sequence"
    )


def digest_value(value: Any) -> str:
    """Content digest of the treated data (sha256 over :func:`_data_envelope`).

    This is the ONLY way a before/after digest enters a
    :class:`TreatmentIntegrityEvidence`: the digest is computed from the actual
    data here, so the evidence can never carry a self-reported digest that does
    not match the data it claims to describe.
    """
    return hashlib.sha256(_data_envelope(value)).hexdigest()


@dataclass(frozen=True)
class IntegrityCheckResult:
    """One named integrity check with its measured value (R55 P0-9).

    Attributes:
        name: Check name (e.g. ``"treatment_changed_data"``, ``"pit_no_leakage"``).
        passed: Verdict.  ``False`` fails the whole evidence fail-closed.
        expected: What the check asserts (human-readable contract).
        measured_value: The value actually measured, when numeric.  ``None``
            for boolean/structural checks.  NaN/Inf are rejected: a
            non-finite measurement is a failed measurement, not evidence.
        detail: Free-form human-readable detail (e.g. truncated digests).
    """

    name: str
    passed: bool
    expected: str = ""
    measured_value: Optional[float] = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("check name must be a non-empty string")
        if not isinstance(self.passed, bool):
            raise TypeError("check passed must be a boolean")
        if self.measured_value is not None:
            if isinstance(self.measured_value, bool) or not isinstance(
                self.measured_value, (int, float)
            ):
                raise TypeError(
                    "measured_value must be a non-boolean number or None"
                )
            measured = float(self.measured_value)
            if not math.isfinite(measured):
                raise ValueError(
                    "measured_value must be finite (NaN/Inf is a failed "
                    "measurement, not evidence)"
                )
            object.__setattr__(self, "measured_value", measured)
        for field_name in ("expected", "detail"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "measured_value": self.measured_value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrityCheckResult":
        if not isinstance(data, Mapping):
            raise TypeError("IntegrityCheckResult.from_dict requires a mapping")
        known = {
            "name", "passed", "expected", "measured_value", "detail",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown IntegrityCheckResult fields: {sorted(unknown)}"
            )
        return cls(
            name=data["name"],
            passed=data["passed"],
            expected=data.get("expected", ""),
            measured_value=data.get("measured_value"),
            detail=data.get("detail", ""),
        )


@dataclass(frozen=True)
class TreatmentIntegrityEvidence:
    """Deep-immutable, content-hashed record of a treatment's integrity (R55 P0-9).

    Produced by the code that actually applied the treatment, and REQUIRED by
    the FO integrity gates before a treatment candidate may be scored or
    admitted.  Missing evidence is a rejection, never a pass.

    Attributes:
        treatment_id: The treatment/trial this evidence is bound to.  The gate
            rejects evidence bound to a different id (stale / shuffled
            evidence is indistinguishable from no evidence).
        treatment_kind: What was applied (e.g. ``"raw"``, ``"ewma"``,
            ``"winsor_zscore_neutralize"``).
        applied_parameters: The parameters actually applied, deep-frozen.
        integrity_checks: Ordered :class:`IntegrityCheckResult` records —
            what was checked, what was measured, and whether it passed.
        before_digest: Content digest of the pre-treatment data.
        after_digest: Content digest of the post-treatment data.
        data_digest_algorithm: Digest algorithm identifier (``"sha256"``).
        produced_by: Identity of the producer that ran the treatment.
        created_at: Creation timestamp (excluded from the content hash).
    """

    treatment_id: str
    treatment_kind: str
    applied_parameters: Mapping[str, Any]
    integrity_checks: Tuple[IntegrityCheckResult, ...]
    before_digest: str
    after_digest: str
    data_digest_algorithm: str = "sha256"
    produced_by: str = "factor_optimizer"
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not isinstance(self.treatment_id, str) or not self.treatment_id.strip():
            raise ValueError("treatment_id must be a non-empty string")
        if not isinstance(self.treatment_kind, str) or not self.treatment_kind.strip():
            raise ValueError("treatment_kind must be a non-empty string")
        if not isinstance(self.applied_parameters, Mapping):
            raise TypeError("applied_parameters must be a mapping")
        if isinstance(self.integrity_checks, (list, tuple)):
            checks = tuple(self.integrity_checks)
        else:
            raise TypeError(
                "integrity_checks must be a sequence of IntegrityCheckResult"
            )
        for check in checks:
            if not isinstance(check, IntegrityCheckResult):
                raise TypeError(
                    "integrity_checks entries must be IntegrityCheckResult, got "
                    f"{type(check).__name__}"
                )
        names = [check.name for check in checks]
        if len(names) != len(set(names)):
            raise ValueError(
                "integrity_checks names must be unique; a duplicated check "
                "name would let one check's verdict stand in for another's"
            )
        for name in ("before_digest", "after_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.data_digest_algorithm != "sha256":
            raise ValueError(
                "data_digest_algorithm must be 'sha256'; any other algorithm "
                "cannot be verified by the gate"
            )
        if not isinstance(self.produced_by, str) or not self.produced_by.strip():
            raise ValueError("produced_by must be a non-empty string")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")

        object.__setattr__(self, "integrity_checks", checks)
        object.__setattr__(
            self, "applied_parameters", _freeze(dict(self.applied_parameters))
        )
        # Snapshot the canonical payload so verify() can detect any
        # post-construction tampering (fail closed).
        object.__setattr__(self, "_canonical", self._canonical_payload())

    # -- derived status (never self-reported) -------------------------------

    @property
    def overall_status(self) -> "TreatmentIntegrityStatus":
        """PASSED / FAILED / NOT_RUN, derived from the checks.

        Derived, not stored: a self-reported status is precisely the fake-gate
        bug this artifact exists to close.  NOT_RUN means no check was ever
        recorded (an evidence object with zero checks proves nothing and every
        gate rejects it).
        """
        if not self.integrity_checks:
            return TreatmentIntegrityStatus.NOT_RUN
        if all(check.passed for check in self.integrity_checks):
            return TreatmentIntegrityStatus.PASSED
        return TreatmentIntegrityStatus.FAILED

    @property
    def failed_checks(self) -> Tuple[str, ...]:
        """Names of the checks that did not pass (empty when all passed)."""
        return tuple(
            check.name for check in self.integrity_checks if not check.passed
        )

    @property
    def all_checks_passed(self) -> bool:
        return bool(self.integrity_checks) and all(
            check.passed for check in self.integrity_checks
        )

    # -- content identity ----------------------------------------------------

    def _canonical_payload(self) -> str:
        return json.dumps(
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "treatment_id": self.treatment_id,
                "treatment_kind": self.treatment_kind,
                "applied_parameters": _jsonable(self.applied_parameters),
                "integrity_checks": [check.to_dict() for check in self.integrity_checks],
                "before_digest": self.before_digest,
                "after_digest": self.after_digest,
                "data_digest_algorithm": self.data_digest_algorithm,
                "produced_by": self.produced_by,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def content_hash(self) -> str:
        """sha256 over every semantic field of the evidence.

        Any change to any check's name/verdict/measured value/detail, to the
        applied parameters, to the kind, or to either data digest changes the
        hash — so evidence cannot be forged by editing a single field.
        """
        return _json_digest(self._canonical_payload())

    def verify(self) -> None:
        """Fail closed if the evidence was altered after construction."""
        if self._canonical_payload() != self._canonical:
            raise TreatmentIntegrityError(
                "treatment integrity evidence content was tampered after "
                f"construction (treatment_id={self.treatment_id!r})"
            )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "treatment_id": self.treatment_id,
            "treatment_kind": self.treatment_kind,
            "applied_parameters": dict(self.applied_parameters),
            "integrity_checks": [check.to_dict() for check in self.integrity_checks],
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "data_digest_algorithm": self.data_digest_algorithm,
            "produced_by": self.produced_by,
            "created_at": self.created_at.isoformat(),
            "overall_status": self.overall_status.value,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TreatmentIntegrityEvidence":
        if not isinstance(data, Mapping):
            raise TypeError("TreatmentIntegrityEvidence.from_dict requires a mapping")
        created = data["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        checks = tuple(
            IntegrityCheckResult.from_dict(item)
            for item in data.get("integrity_checks", ())
        )
        obj = cls(
            treatment_id=data["treatment_id"],
            treatment_kind=data["treatment_kind"],
            applied_parameters=dict(data.get("applied_parameters", {})),
            integrity_checks=checks,
            before_digest=data["before_digest"],
            after_digest=data["after_digest"],
            data_digest_algorithm=data.get("data_digest_algorithm", "sha256"),
            produced_by=data.get("produced_by", "factor_optimizer"),
            created_at=created,
        )
        supplied = data.get("content_hash")
        if supplied not in (None, obj.content_hash):
            raise TreatmentIntegrityError(
                "treatment integrity evidence content_hash does not match its "
                "payload; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )
        return obj


class TreatmentIntegrityStatus(str, Enum):
    """Overall status of a treatment's integrity evidence.

    Defined after :class:`TreatmentIntegrityEvidence` only for readability of
    this module; it is a ``str`` enum so serialized forms compare equal to
    their values.
    """

    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_RUN = "NOT_RUN"

    @classmethod
    def from_value(cls, value: Any) -> "TreatmentIntegrityStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown TreatmentIntegrityStatus value {value!r}; expected "
                    f"one of {[member.value for member in cls]}"
                ) from None
        raise TypeError(
            "TreatmentIntegrityStatus.from_value expects str or "
            f"TreatmentIntegrityStatus, got {type(value).__name__}"
        )


def build_integrity_evidence(
    treatment_id: str,
    treatment_kind: str,
    applied_parameters: Mapping[str, Any],
    before: Any,
    after: Any,
    *,
    expected_parameters: Optional[Mapping[str, Any]] = None,
    extra_checks: Sequence[IntegrityCheckResult] = (),
    produced_by: str = "factor_optimizer",
    created_at: Optional[datetime] = None,
) -> TreatmentIntegrityEvidence:
    """Build evidence by MEASURING the actual before/after data (R55 P0-9).

    The before/after digests are computed here from the data the caller
    passes — never accepted as strings — so the evidence cannot claim a
    digest the data does not have.

    Automatic checks:
      - ``parameters_recorded``: the applied parameters were recorded
        (non-empty), unless this is the RAW baseline which legitimately has
        none.
      - ``treatment_changed_data`` / ``raw_is_identity``: for every non-RAW
        kind the output must differ from the input — a treatment whose output
        is byte-identical to its input did not run.  For the RAW baseline the
        check is inverted: applying no treatment must NOT change the data.
      - ``parameters_match_recipe`` (when ``expected_parameters`` is given):
        the applied parameters must equal the recipe's parameters.

    ``extra_checks`` carries the producer's own measurements (e.g. PIT /
    no-leakage, NaN fraction, coverage) so the gate sees the whole picture.
    """
    if not isinstance(treatment_kind, str) or not treatment_kind.strip():
        raise ValueError("treatment_kind must be a non-empty string")
    if applied_parameters is None or not isinstance(applied_parameters, Mapping):
        raise TypeError("applied_parameters must be a mapping")
    if expected_parameters is not None and not isinstance(
        expected_parameters, Mapping
    ):
        raise TypeError("expected_parameters must be a mapping or None")

    before_digest = digest_value(before)
    after_digest = digest_value(after)
    is_raw = treatment_kind == RAW_TREATMENT_KIND
    changed = before_digest != after_digest

    checks: List[IntegrityCheckResult] = [
        IntegrityCheckResult(
            name="parameters_recorded",
            passed=bool(applied_parameters) or is_raw,
            expected=(
                "no parameters for the RAW baseline"
                if is_raw
                else "applied_parameters must be recorded (non-empty)"
            ),
            measured_value=float(len(applied_parameters)),
            detail=f"{len(applied_parameters)} parameter(s) recorded",
        ),
        IntegrityCheckResult(
            name="raw_is_identity" if is_raw else "treatment_changed_data",
            passed=(not changed) if is_raw else changed,
            expected=(
                "applying the RAW baseline must not change the data"
                if is_raw
                else "the treated output must differ from the untreated input"
            ),
            measured_value=0.0 if changed else 1.0,
            detail=(
                f"before={before_digest[:12]} after={after_digest[:12]} "
                f"changed={changed}"
            ),
        ),
    ]
    if expected_parameters is not None:
        same = _jsonable(applied_parameters) == _jsonable(expected_parameters)
        checks.append(
            IntegrityCheckResult(
                name="parameters_match_recipe",
                passed=same,
                expected="applied parameters must equal the recipe parameters",
                detail=(
                    "applied parameters match the recipe"
                    if same
                    else "applied parameters differ from the recipe parameters"
                ),
            )
        )
    for check in extra_checks:
        if not isinstance(check, IntegrityCheckResult):
            raise TypeError(
                "extra_checks entries must be IntegrityCheckResult, got "
                f"{type(check).__name__}"
            )
        checks.append(check)

    return TreatmentIntegrityEvidence(
        treatment_id=treatment_id,
        treatment_kind=treatment_kind,
        applied_parameters=applied_parameters,
        integrity_checks=tuple(checks),
        before_digest=before_digest,
        after_digest=after_digest,
        produced_by=produced_by,
        created_at=created_at or datetime.now(),
    )


def describe_integrity_problem(
    treatment_id: str,
    evidence: Optional[TreatmentIntegrityEvidence],
    *,
    treatment_kind: Optional[str] = None,
) -> Optional[str]:
    """Return a human-readable reason the evidence is unacceptable, or None.

    This is the reporting twin of :func:`require_integrity_evidence`: the
    decision pipeline raises on any problem, while the admission policy
    records the reason on a REJECTED decision.
    """
    if evidence is None:
        return (
            f"missing TreatmentIntegrityEvidence for treatment {treatment_id!r} "
            "— a treatment may not be admitted without measured integrity "
            "evidence (fail closed)"
        )
    if not isinstance(evidence, TreatmentIntegrityEvidence):
        return (
            f"integrity evidence for treatment {treatment_id!r} has unexpected "
            f"type {type(evidence).__name__} — fail closed"
        )
    try:
        evidence.verify()
    except TreatmentIntegrityError as exc:
        return f"integrity evidence for treatment {treatment_id!r} is tampered: {exc}"
    if evidence.treatment_id != treatment_id:
        return (
            f"integrity evidence is bound to treatment "
            f"{evidence.treatment_id!r}, not {treatment_id!r} — stale or "
            "mis-bound evidence fails closed"
        )
    if treatment_kind is not None and evidence.treatment_kind != treatment_kind:
        return (
            f"integrity evidence for treatment {treatment_id!r} claims kind "
            f"{evidence.treatment_kind!r}, expected {treatment_kind!r}"
        )
    if not evidence.integrity_checks:
        return (
            f"integrity evidence for treatment {treatment_id!r} recorded no "
            "checks (NOT_RUN) — evidence with no measured checks proves "
            "nothing and fails closed"
        )
    failed = tuple(check.name for check in evidence.integrity_checks if not check.passed)
    if failed:
        return (
            f"integrity evidence for treatment {treatment_id!r} failed checks: "
            + ", ".join(failed)
        )
    return None


def require_integrity_evidence(
    treatment_id: str,
    evidence: Optional[TreatmentIntegrityEvidence],
    *,
    treatment_kind: Optional[str] = None,
) -> TreatmentIntegrityEvidence:
    """Fail-closed integrity gate (R55 P0-9).

    Returns the evidence when it is present, self-consistent, bound to
    ``treatment_id``, and every check passed.  Otherwise raises
    :class:`~factor_optimizer.errors.TreatmentIntegrityError` — missing
    evidence is a rejection, never a pass.
    """
    problem = describe_integrity_problem(
        treatment_id, evidence, treatment_kind=treatment_kind
    )
    if problem is not None:
        raise TreatmentIntegrityError(problem)
    assert evidence is not None  # narrowed by describe_integrity_problem
    return evidence