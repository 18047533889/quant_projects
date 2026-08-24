# -*- coding: utf-8 -*-
"""Per-(canonical, backend, PhysicalImplementationID, parameter_domain) certification schema for DuckDB backend.

Certification key:
  canonical x backend x PhysicalImplementationID x parameter_domain x
  ddof/min_periods x NULL/NaN/Inf x tie_policy x window_frame x
  oracle_hash x source_hash x SHA
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from factor_engine.backend.contracts import (
    PhysicalImplementationID,
    PhysicalImplementationSpec,
)


BackendLiteral = Literal["duckdb_sql"]


class NullPolicy(str, Enum):
    """NULL/NaN/Inf handling policy for a certified operator window."""
    NULL_EXPLICIT = "null_explicit"
    NULL_COALESCE = "null_coalesce"
    NAN_PROPAGATE = "nan_propagate"
    NAN_DROP = "nan_drop"
    INF_CLAMP = "inf_clamp"
    INF_PROPAGATE = "inf_propagate"


class TiePolicy(str, Enum):
    """Tie-breaking policy for rank/order-dependent windows."""
    FIRST = "first"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    DENSE = "dense"


class WindowFrame(str, Enum):
    """Window frame semantics certified by the row."""
    ROWS = "rows"
    RANGE = "range"
    GROUP = "group"
    EXPANDING = "expanding"


@dataclass(frozen=True)
class ParameterDomainDigest:
    """Canonicalised parameter-domain fingerprint for a certified window.

    ``raw`` is the untrusted input; ``canonical`` is the canonicalised form
    suitable for hashing.
    """
    raw: str
    canonical: str

    def to_dict(self) -> dict[str, str]:
        return {"raw": self.raw, "canonical": self.canonical}


@dataclass(frozen=True)
class PerRowCertification:
    """One row of the per-(canonical, backend, PhysicalImplementationID, parameter_domain) certification matrix."""

    canonical: str
    backend: BackendLiteral
    physical_implementation_id: PhysicalImplementationID
    parameter_domain: ParameterDomainDigest
    ddof: int | None = None
    min_periods: int | None = None
    null_policy: NullPolicy = NullPolicy.NULL_EXPLICIT
    tie_policy: TiePolicy = TiePolicy.FIRST
    window_frame: WindowFrame = WindowFrame.ROWS
    oracle_hash: str = ""
    source_hash: str = ""
    sha: str = ""

    def _canonical_payload(self) -> dict[str, Any]:
        """Deterministic canonical payload (keys sorted, separators compact)."""
        payload: dict[str, Any] = {
            "canonical": self.canonical.strip(),
            "backend": self.backend,
            "physical_implementation_id": str(self.physical_implementation_id),
            "parameter_domain": self.parameter_domain.canonical,
            "null_policy": self.null_policy.value,
            "tie_policy": self.tie_policy.value,
            "window_frame": self.window_frame.value,
        }
        # Include ddof/min_periods only when meaningful.
        if self.ddof is not None:
            payload["ddof"] = int(self.ddof)
        if self.min_periods is not None:
            payload["min_periods"] = int(self.min_periods)
        if self.oracle_hash:
            payload["oracle_hash"] = self.oracle_hash.strip()
        if self.source_hash:
            payload["source_hash"] = self.source_hash.strip()
        return payload

    def compute_sha(self) -> str:
        """Compute deterministic SHA-256 hex digest of the canonical payload."""
        blob = json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def ensure_sha(self) -> PerRowCertification:
        """Return a copy with ``sha`` populated (idempotent)."""
        if self.sha:
            return self
        return PerRowCertification(
            canonical=self.canonical,
            backend=self.backend,
            physical_implementation_id=self.physical_implementation_id,
            parameter_domain=self.parameter_domain,
            ddof=self.ddof,
            min_periods=self.min_periods,
            null_policy=self.null_policy,
            tie_policy=self.tie_policy,
            window_frame=self.window_frame,
            oracle_hash=self.oracle_hash,
            source_hash=self.source_hash,
            sha=self.compute_sha(),
        )

    def to_dict(self) -> dict[str, Any]:
        certified = self.ensure_sha()
        return {
            "canonical": certified.canonical,
            "backend": certified.backend,
            "physical_implementation_id": str(certified.physical_implementation_id),
            "parameter_domain": certified.parameter_domain.to_dict(),
            "ddof": certified.ddof,
            "min_periods": certified.min_periods,
            "null_policy": certified.null_policy.value,
            "tie_policy": certified.tie_policy.value,
            "window_frame": certified.window_frame.value,
            "oracle_hash": certified.oracle_hash,
            "source_hash": certified.source_hash,
            "sha": certified.sha,
        }


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _derive_physical_implementation_id(spec: PhysicalImplementationSpec) -> PhysicalImplementationID:
    """Derive PhysicalImplementationID from a validated spec.

    Raises ``ValueError`` when the spec is incomplete.
    """
    pid = spec.physical_implementation_id
    if pid is None:
        raise ValueError(
            "PhysicalImplementationSpec is not complete enough to derive "
            f"PhysicalImplementationID (errors={spec.validation_errors()})"
        )
    return pid


def _canonicalise_parameter_domain(raw: str | ParameterDomainDigest) -> ParameterDomainDigest:
    """Canonicalise a parameter domain string."""
    if isinstance(raw, ParameterDomainDigest):
        return raw
    raw_str = (raw or "").strip()
    # Deterministic canonical form: normalise whitespace and lowercase.
    canonical = " ".join(raw_str.split()).lower()
    return ParameterDomainDigest(raw=raw_str, canonical=canonical)


def build_duckdb_certification_row(
    canonical: str,
    *,
    spec: PhysicalImplementationSpec,
    parameter_domain: str | ParameterDomainDigest,
    ddof: int | None = None,
    min_periods: int | None = None,
    null_policy: NullPolicy = NullPolicy.NULL_EXPLICIT,
    tie_policy: TiePolicy = TiePolicy.FIRST,
    window_frame: WindowFrame = WindowFrame.ROWS,
    oracle_hash: str = "",
    source_hash: str = "",
) -> PerRowCertification:
    """Build and hash a single DuckDB certification row.

    Rules:
    * ``backend`` is always ``"duckdb_sql"``.
    * ``spec.backend`` must equal ``"duckdb_sql"``.
    * ``PhysicalImplementationID`` is derived from the spec (not caller-supplied).
    """
    if spec.backend != "duckdb_sql":
        raise ValueError(f"expected spec.backend='duckdb_sql', got {spec.backend!r}")
    pid = _derive_physical_implementation_id(spec)
    row = PerRowCertification(
        canonical=canonical.strip(),
        backend="duckdb_sql",
        physical_implementation_id=pid,
        parameter_domain=_canonicalise_parameter_domain(parameter_domain),
        ddof=ddof,
        min_periods=min_periods,
        null_policy=null_policy,
        tie_policy=tie_policy,
        window_frame=window_frame,
        oracle_hash=oracle_hash,
        source_hash=source_hash,
    )
    return row.ensure_sha()


def verify_certification_row(row: PerRowCertification) -> tuple[bool, str]:
    """Verify that ``row.sha`` matches the recomputed canonical digest."""
    expected = row.compute_sha()
    if row.sha == expected:
        return True, "ok"
    return False, f"sha mismatch: expected={expected}, got={row.sha}"


def certification_key(row: PerRowCertification) -> str:
    """Return a deterministic lookup key for the certification matrix."""
    certified = row.ensure_sha()
    return "|".join([
        certified.canonical,
        certified.backend,
        str(certified.physical_implementation_id),
        certified.parameter_domain.canonical,
        certified.sha,
    ])
