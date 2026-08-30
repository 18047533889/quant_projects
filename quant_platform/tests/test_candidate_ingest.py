# -*- coding: utf-8 -*-
"""QRP-P3 candidate ingestion / reconciliation — locking tests (R55 P0-5).

Covers :mod:`quant_platform.app.candidate.ingest`:

* ``normalize_candidate`` — success + fail-closed failures (missing content
  hash / bad hash format / non-finite parameter domain / MISSING OR NON-HEX
  DOMAIN-MINTED SEMANTIC ID / missing generator type). The semantic id is the
  OWNING DOMAIN PACKAGE's digest: the platform validates its hex FORMAT and
  carries it verbatim — it never hashes factor semantics itself.
* ``reconcile_candidates`` — NEW / DUPLICATE_EXACT / CONFLICT_SEMANTIC_TO_HASH
  / CONFLICT_HASH_TO_SEMANTIC classification; ``ReconcileReport.counts`` keyed
  by category (enum-indexable), ``conflicts`` detail tuple, ``total`` count,
  ``batch_fingerprint``;
* ``batch_fingerprint`` — order-independent, content-sensitive, idempotent,
  ``merkle-v1:`` prefix (an anti-replay token over CARRIED hashes, not a minted
  identity).
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

import pytest

# The quant_platform workspace is FLAT-LAYOUT: ``quant_platform/`` *is* the
# package root, and ``app/`` lives under ``quant_platform.app``. Importing
# ``quant_platform`` therefore requires the *parent* of this file's tree —
# ``<platform>/..`` == the directory that physically contains the package dir
# ``quant_platform/`` — on ``sys.path``. (The umbrella repo used to provide
# this via sitecustomize; that hook is absent in this sandbox.) Register the
# parent *before* any ``import quant_platform...`` so the source tree wins
# over any installed copy.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../quant_platform/tests
_PLATFORM_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # .../quant_platform
_PARENT = os.path.abspath(os.path.join(_PLATFORM_ROOT, ".."))    # .../quant_projects
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from quant_platform.app.candidate.ingest import (
    CandidateNormalizationError,
    ReconcileReason,
    ReconciliationConflict,
    ReconcileReport,
    batch_fingerprint,
    normalize_candidate,
    reconcile_candidates,
)
from quant_platform.app.contracts import FactorCandidateManifest


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _raw(*, semantic_id: str | None = None, content_hash: str | None = None,
         generator_type: str = "trailing_sma", param_domain: dict | None = None,
         factor_name: str = "myfactor", formula: str = "sma(close, 20)",
         market: str = "ashare", frequency: str = "1d", **extra) -> dict:
    payload = {
        # the DOMAIN package's own semantic digest — carried, never derived here
        "semantic_id": semantic_id if semantic_id is not None else _h(f"sem:{factor_name}:{formula}"),
        "content_hash": content_hash if content_hash is not None else _h("seed-a"),
        "generator_type": generator_type,
        "generator_version": "1.0",
        "submitted_by": "tester",
        "market": market,
        "frequency": frequency,
        "factor_name": factor_name,
        "formula": formula,
        "factor_spec_uri": f"recipe://{factor_name}",
        "formula_language": "recipe",
        "source": "test",
        "published_at": "2026-08-28T00:00:00Z",
    }
    payload.update(extra)
    if param_domain is not None:
        payload["param_domain"] = param_domain
    return payload


def _manifest(*, semantic_id: str | None = None, seed: str = "a",
              factor_name: str = "myfactor", formula: str = "sma(close, 20)",
              **extra) -> FactorCandidateManifest:
    return normalize_candidate(
        _raw(semantic_id=semantic_id, content_hash=_h(seed), factor_name=factor_name,
             formula=formula, **extra)
    )


# --- normalize -------------------------------------------------------------


def test_normalize_success():
    manifest = _manifest()
    assert isinstance(manifest, FactorCandidateManifest)
    assert manifest.factor_spec_sha256 == _h("a")  # content hash mirrored
    # semantic id is CARRIED verbatim (the domain's digest), not re-derived
    assert manifest.semantic_family_hint == _h("sem:myfactor:sma(close, 20)")
    assert manifest.market == "ashare"
    assert manifest.frequency == "1d"
    assert manifest.candidate_id  # auto-derived from the content hash


def test_normalize_carries_domain_semantic_id_verbatim():
    # A different producer semantic id for the same spec is carried as-is —
    # the platform cannot know better than the domain package.
    domain_digest = _h("whatever-the-domain-minted")
    manifest = _manifest(semantic_id=domain_digest, formula="totally-different")
    assert manifest.semantic_family_hint == domain_digest


def test_normalize_missing_content_hash_fails():
    # content_hash=None: `_raw` falls back to a default sha — so explicitly pop
    # it AND every *_hash carrier to exercise the true fail-closed path.
    raw = _raw(content_hash=None)
    for k in ("content_hash", "factor_spec_sha256", "sha256", "hash"):
        raw.pop(k, None)
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(raw)
    assert ei.value.reason == "missing_content_hash"


def test_normalize_bad_content_hash_format_fails():
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(_raw(content_hash="abc-not-a-sha256"))
    assert ei.value.reason == "bad_content_hash_format"


def test_normalize_missing_semantic_id_fails():
    # No domain-minted semantic digest -> fail closed. Carrier fields
    # (factor_name/market/frequency/formula) must NOT be hashed into a
    # platform-made identity any more (R55 P0-5).
    raw = _raw(semantic_id="")
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(raw)
    assert ei.value.reason == "missing_semantic_id"


def test_normalize_missing_semantic_id_fails_even_with_carrier_fields():
    raw = _raw(semantic_id="", factor_name="named", formula="ema(close,5)")
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(raw)
    assert ei.value.reason == "missing_semantic_id"


def test_normalize_non_hex_semantic_id_fails_format_check():
    # A discovery-side slug is NOT a domain digest: format-only validation
    # rejects it instead of hashing it into something else.
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(_raw(semantic_id="SMOOTH:trailing_sma"))
    assert ei.value.reason == "bad_semantic_hash_format"


def test_normalize_carries_semantic_hash_alias_key():
    raw = _raw()
    raw.pop("semantic_id")
    raw["semantic_hash"] = _h("alias-digest")
    manifest = normalize_candidate(raw)
    assert manifest.semantic_family_hint == _h("alias-digest")


def test_normalize_nonfinite_param_domain_fails():
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(_raw(param_domain={"halflife": (0.0, float("inf"))}))
    assert ei.value.reason == "bad_parameter_domain"


def test_normalize_missing_generator_type_fails():
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(_raw(generator_type=""))
    assert ei.value.reason == "missing_string_field"


def test_normalize_inverted_param_domain_fails():
    with pytest.raises(CandidateNormalizationError):
        normalize_candidate(_raw(param_domain={"halflife": (5.0, 1.0)}))


# --- reconcile -------------------------------------------------------------


def test_reconcile_new_classified():
    known = [_manifest(seed="b", factor_name="knownname", formula="ema(close, 30)")]
    report = reconcile_candidates([_manifest(seed="a")], known)
    assert report[ReconcileReason.NEW] == 1
    assert report.new == 1
    assert report.conflicts == ()


def test_reconcile_duplicate_exact():
    known = [_manifest(seed="a")]
    report = reconcile_candidates([_manifest(seed="a")], known)
    assert report[ReconcileReason.DUPLICATE_EXACT] == 1
    assert report.duplicates == 1
    assert len(report.conflicts) == 1  # duplicate detail recorded


def test_reconcile_conflict_semantic_to_hash():
    # same domain semantic digest, different content hash
    known = [_manifest(seed="a")]
    report = reconcile_candidates([_manifest(seed="b")], known)
    assert report[ReconcileReason.CONFLICT_SEMANTIC_TO_HASH] == 1
    assert report.conflicts[0].reason == ReconcileReason.CONFLICT_SEMANTIC_TO_HASH.value


def test_reconcile_conflict_hash_to_semantic():
    # same content hash, different domain semantic digest
    known = [_manifest(seed="a", factor_name="knownname", formula="ema(close, 30)")]
    report = reconcile_candidates([_manifest(seed="a")], known)
    assert report[ReconcileReason.CONFLICT_HASH_TO_SEMANTIC] == 1
    assert report.conflicts[0].reason == ReconcileReason.CONFLICT_HASH_TO_SEMANTIC.value


def test_reconcile_mixed_batch_counts():
    known = [
        _manifest(seed="dup"),
        _manifest(seed="sem", factor_name="knownname", formula="ema(close, 30)"),
    ]
    report = reconcile_candidates(
        [
            _manifest(seed="new", factor_name="newname", formula="rsi(14)"),          # NEW
            _manifest(seed="dup"),                                                    # DUPLICATE_EXACT
            _manifest(factor_name="knownname", formula="ema(close, 30)", seed="mut"),  # CONFLICT_SEMANTIC_TO_HASH
        ],
        known,
    )
    assert report[ReconcileReason.NEW] == 1
    assert report[ReconcileReason.DUPLICATE_EXACT] == 1
    assert report[ReconcileReason.CONFLICT_SEMANTIC_TO_HASH] == 1
    assert report.conflicts_count == 1
    assert report.total == 3


def test_reconcile_empty_registry_all_new():
    report = reconcile_candidates([_manifest(seed="h1"), _manifest(seed="h2")])
    assert report[ReconcileReason.NEW] == 2
    assert report.batch_fingerprint


def test_reconcile_report_conflict_details_shape():
    known = [_manifest(seed="existing")]
    report = reconcile_candidates([_manifest(seed="other")], known)
    assert isinstance(report, ReconcileReport)
    assert isinstance(report.conflicts, tuple)
    conflict = report.conflicts[0]
    assert isinstance(conflict, ReconciliationConflict)
    assert conflict.candidate_id
    assert conflict.content_hash != conflict.semantic_hash
    assert conflict.matched_registry


# --- batch fingerprint -----------------------------------------------------


def test_fingerprint_order_independent():
    a = _manifest(seed="a")
    b = _manifest(seed="b")
    assert batch_fingerprint([a, b]) == batch_fingerprint([b, a])


def test_fingerprint_content_sensitive():
    assert batch_fingerprint([_manifest(seed="a")]) != batch_fingerprint([_manifest(seed="a2")])


def test_fingerprint_idempotent():
    batch = [_manifest(seed="a"), _manifest(seed="b")]
    assert batch_fingerprint(batch) == batch_fingerprint(batch)
    assert batch_fingerprint(batch).startswith("merkle-v1:")


# --- R55 P0-5: the platform mints NO domain identity ------------------------


def test_platform_does_not_hash_factor_semantics_into_identity():
    # Changing factor-relevant RAW fields (formula / factor_name) must NOT
    # change the carried semantic identity — only the domain's own digest field
    # can do that. The platform hashes nothing semantic.
    a = _raw(formula="sma(close, 20)")
    b = _raw(formula="ema(close, 99)", factor_name="renamed")
    digest = _h("domain-minted-semantic-digest")
    a["semantic_id"] = digest
    b["semantic_id"] = digest
    ma = normalize_candidate(a)
    mb = normalize_candidate(b)
    assert ma.semantic_family_hint == digest
    assert mb.semantic_family_hint == digest


def test_platform_never_derives_semantic_identity_from_carrier_fields():
    raw = _raw()
    raw.pop("semantic_id")
    raw.pop("semantic_hash", None)
    # every carrier field present — still refused: deriving would be minting
    with pytest.raises(CandidateNormalizationError) as ei:
        normalize_candidate(raw)
    assert ei.value.reason == "missing_semantic_id"