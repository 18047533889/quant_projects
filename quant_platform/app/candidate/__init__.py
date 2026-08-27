"""QRP-P3 — candidate ingestion & reconciliation (pure logic layer).

This package hosts the *pure-logic* ingestion stage for factor candidates:
heterogeneous raw records (factor_recipes / discovery manifests / …) are
normalized into canonical candidate manifests, reconciled against the known
registry by dual key (``content_hash`` × ``semantic_hash``), and batch-level
Merkle fingerprints are computed for later idempotent ingestion.

PURE-DTO discipline: the *contracts* directory is stdlib-only and must not
import this layer. This layer MAY import domain contracts
(``quant_platform.app.contracts``) but must NOT import domain internals
(``factor_engine.*``, ``factor_assets.*`` adapters, DB/API/storage layers).

Authoritative helpers are NOT re-implemented here — identity comes from
``contracts/identities.py`` (``canonicalize`` / ``FactorValueIdentity``),
content hashing from ``contracts/_contenthash.py`` (``content_hash`` /
``canonical_str``).

Import-resolution note (ref: ``contracts/__init__.py`` docstring): a family of
mutually re-importing pure-logic modules ("bundles") may freely reference each
other. Resolving which ``app/`` tree a bare ``import quant_platform`` returns
is an environment concern — the tests for this layer run from the checkout so
the source tree wins on ``sys.path``.
"""

from __future__ import annotations

from .candidate import (
    ConflictRecord,
    NORMALIZATION_ERROR_CODES,
    NormalizationError,
    NormalizedCandidate,
    RECONCILE_REASON_NEW,
    RECONCILE_REASON_DUPLICATE,
    RECONCILE_REASON_CONFLICT_SEMANTIC,
    RECONCILE_REASON_CONFLICT_CONTENT,
    RECONCILE_REASON_INVALID_DUPLICATE,
    RECONCILE_REASON_INVALID_CONFLICT,
    RECONCILE_REASONS,
    ReconcileReport,
    batch_fingerprint,
    normalize_candidate,
    reconcile_candidates,
)

__all__ = [
    "ConflictRecord",
    "NORMALIZATION_ERROR_CODES",
    "NormalizationError",
    "NormalizedCandidate",
    "RECONCILE_REASON_NEW",
    "RECONCILE_REASON_DUPLICATE",
    "RECONCILE_REASON_CONFLICT_SEMANTIC",
    "RECONCILE_REASON_CONFLICT_CONTENT",
    "RECONCILE_REASON_INVALID_DUPLICATE",
    "RECONCILE_REASON_INVALID_CONFLICT",
    "RECONCILE_REASONS",
    "ReconcileReport",
    "batch_fingerprint",
    "normalize_candidate",
    "reconcile_candidates",
]
