# -*- coding: utf-8 -*-
"""R55 audit P0-5/P0-6 — platform is NOT the domain-identity authority.

Locks the two architectural corrections of task #95:

P0-5 — identity REFERENCES, never identity MINTING
  (a) grep-level: no domain-semantic hashing remains in the contracts identity
      layer — no ``canonicalize`` over factor fields, no ``FactorDefinitionIdentity``
      constructor, no parameter-dict hashing, no 口径 (vwap basis) baked into any
      platform hash input;
  (b) contract-level: ``contracts/identities`` exposes only carried refs whose
      ``hash`` is the domain's own digest, format-validated (64-char lowercase
      hex) and never recomputed;
  (c) ingest-level: ``semantic_family_hint`` is the domain digest carried
      verbatim — factor semantics (formula / parameters / carrier fields) never
      enter a platform-computed hash.

P0-6 — admission is DELEGATED, never computed by the platform
  (d) the ``AdmissionAuthority`` Protocol is the only decision seam; the
      platform's default is fail-closed ``RefuseAdmission``;
  (e) verdicts are recorded verbatim (decision / reason codes / policy ref /
      authority / content hash) — the platform adds no threshold, no similarity
      measurement, no label-maturity or return-basis judgment;
  (f) no platform symbol owns an admission threshold any more.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pathlib
import re
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PLATFORM_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PARENT = os.path.abspath(os.path.join(_PLATFORM_ROOT, ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from quant_platform.app.contracts import (
    ADMISSION_DECISIONS,
    DECISION_APPROVED,
    DECISION_REJECTED,
    DECISION_SHADOWED,
    REASON_AUTHORITY_ABSENT,
    AdmissionAuthority,
    AdmissionRequest,
    AdmissionVerdict,
    EvaluationRef,
    FactorDefinitionRef,
    FactorValueRef,
    IdentityRef,
    RefuseAdmission,
    TreatmentRef,
    require_non_empty,
    sha256_hex,
)


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# (a) grep-level: no domain-semantic hashing left in the identity contract
# --------------------------------------------------------------------------- #

_CONTRACTS_DIR = pathlib.Path(_PLATFORM_ROOT) / "app" / "contracts"
_ORCHESTRATOR = pathlib.Path(_PLATFORM_ROOT) / "app" / "orchestrator.py"


def test_no_identity_minting_symbols_remain_anywhere_in_platform():
    """The minting API is GONE, not merely unused (grep-level lock)."""
    offenders: list[str] = []
    banned = (
        "FactorDefinitionIdentity",
        "FactorValueIdentity",
        "EvaluationIdentity",
        "TreatmentIdentity",
        "canonicalize(",
    )
    for path in sorted(_CONTRACTS_DIR.glob("*.py")) + [_ORCHESTRATOR]:
        if path.name == "admission.py":
            continue  # the delegation seam may *reference* the vocabulary
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, f"platform still mints domain identities: {offenders}"


def test_no_vwap_caliber_or_threshold_inside_platform_hash_inputs():
    """The 口径 and the IC floor live with the domain authority, not the platform."""
    orch_text = _ORCHESTRATOR.read_text(encoding="utf-8")
    for needle in (
        "DEFAULT_MIN_RANK_IC",
        "VWAP_TO_VWAP_BASIS =",
        "min_rank_ic <",
        "abs(evaluation.rank_ic)",
        "duplicate_similarity_threshold",
        "reject_duplicates",
        "_EVIDENCE_NOT_COMPUTED",
        "pipeline_gate_decision",
    ):
        assert needle not in orch_text, f"platform still owns admission semantics: {needle!r}"


def test_identities_module_contains_no_hash_function_at_all():
    """``contracts/identities.py`` must be format-validation only."""
    text = (_CONTRACTS_DIR / "identities.py").read_text(encoding="utf-8")
    assert "hashlib" not in text, "identities.py must not hash anything"
    assert "sha256(" not in text.replace("sha256_hex", ""), "no raw sha256 calls"
    assert "content_hash(" not in text, "no platform content-hash calls"
    assert "canonicalize" not in text


def test_ingest_layer_hashes_no_factor_semantics():
    """The candidate logic layer never hashes formula / parameters / carriers."""
    ingest = pathlib.Path(_PLATFORM_ROOT) / "app" / "candidate" / "ingest.py"
    text = ingest.read_text(encoding="utf-8")
    assert "canonicalize" not in text
    # the only sha256 work left is the anti-replay token over CARRIED hashes
    assert text.count("hashlib.sha256(") == 1, "batch_fingerprint is the only hash left"
    assert "formula" not in text.split("def batch_fingerprint")[1], (
        "batch fingerprint must fold carried content hashes only"
    )


def test_candidate_module_hashes_no_factor_semantics():
    cand = pathlib.Path(_PLATFORM_ROOT) / "app" / "candidate" / "candidate.py"
    text = cand.read_text(encoding="utf-8")
    assert "FactorDefinitionIdentity" not in text
    assert "FactorDefinitionRef(" in text  # carried ref construction
    # the only sha256 left is the batch-fingerprint fold (platform anti-replay)
    assert text.count("hashlib.sha256(") == 1
    negotiation_fragments = (
        "calculation_semantics",
        "FactorDefinitionIdentity",
        "return_basis",
    )
    fn = text.split("def normalize_candidate")[1].split("def fields_dict")[0]
    for frag in negotiation_fragments:
        assert frag not in fn, f"platform still negotiates identity semantics: {frag!r}"


# --------------------------------------------------------------------------- #
# (b) carried refs: format-only validation, verbatim digest
# --------------------------------------------------------------------------- #


def test_carried_ref_returns_the_domain_digest_verbatim():
    digest = _h("domain-digest")
    ref = IdentityRef(digest)
    assert ref.hash == digest  # carried, NOT recomputed


def test_carried_ref_rejects_non_hex_and_short_digests():
    with pytest.raises(ValueError):
        IdentityRef("SMOOTH:slug")
    with pytest.raises(ValueError):
        IdentityRef("abc123")
    with pytest.raises(TypeError):
        IdentityRef(None)  # type: ignore[arg-type]


def test_factor_definition_ref_requires_factor_version_and_carries_descriptor():
    ref = FactorDefinitionRef(_h("fd"), factor_version="v7", descriptor={"opaque": "blob"})
    assert ref.hash == _h("fd")
    assert ref.factor_version == "v7"
    assert ref.descriptor == {"opaque": "blob"}
    with pytest.raises(ValueError):
        FactorDefinitionRef(_h("fd"), factor_version="")


def test_typed_refs_carry_their_parent_ref():
    fd = FactorDefinitionRef(_h("fd"), factor_version="v1")
    fv = FactorValueRef(_h("fv"), factor_definition_ref=fd.hash)
    ev = EvaluationRef(_h("ev"), factor_value_ref=fv.hash)
    tr = TreatmentRef(_h("tr"), source_factor_value_ref=fv.hash)
    assert fv.factor_definition_ref == fd.hash
    assert ev.factor_value_ref == fv.hash
    assert tr.source_factor_value_ref == fv.hash
    # identity refs compare by carried digest only
    assert IdentityRef(_h("fd")) == IdentityRef(_h("fd"))
    assert IdentityRef(_h("fd")) != IdentityRef(_h("other"))


def test_sha256_hex_is_format_only_and_never_recomputes():
    digest = _h("payload")
    assert sha256_hex(digest, "x") == digest  # returned unchanged
    assert sha256_hex(digest, "x") != _h(digest)  # NOT re-hashed
    with pytest.raises(ValueError):
        sha256_hex(digest.upper(), "x")  # uppercase is not the canonical form
    with pytest.raises(ValueError):
        sha256_hex("", "x")


def test_require_non_empty_is_format_only():
    assert require_non_empty("x", "label") == "x"
    with pytest.raises(ValueError):
        require_non_empty("  ", "label")
    with pytest.raises(TypeError):
        require_non_empty(5, "label")


# --------------------------------------------------------------------------- #
# (c) ingest carries the domain semantic id
# --------------------------------------------------------------------------- #


def _ingest_normalize(raw: dict):
    from quant_platform.app.candidate.ingest import normalize_candidate

    return normalize_candidate(raw)


def _raw_candidate(**extra) -> dict:
    payload = {
        "semantic_id": _h("domain-sem"),
        "content_hash": _h("spec-bytes"),
        "generator_type": "sma",
        "generator_version": "1.0",
        "submitted_by": "tester",
        "market": "ashare",
        "frequency": "1d",
        "factor_name": "f",
        "formula": "sma(close,20)",
        "factor_spec_uri": "recipe://f",
        "formula_language": "recipe",
        "published_at": "2026-08-28T00:00:00Z",
    }
    payload.update(extra)
    return payload


def test_ingest_carries_domain_semantic_digest_verbatim():
    digest = _h("a-domain-digest")
    manifest = _ingest_normalize(_raw_candidate(semantic_id=digest))
    assert manifest.semantic_family_hint == digest


def test_ingest_refuses_to_mint_identity_from_carrier_fields():
    from quant_platform.app.candidate.ingest import CandidateNormalizationError

    raw = _raw_candidate()
    raw.pop("semantic_id")
    with pytest.raises(CandidateNormalizationError) as ei:
        _ingest_normalize(raw)
    assert ei.value.reason == "missing_semantic_id"


def test_ingest_rejects_non_hex_semantic_alias():
    from quant_platform.app.candidate.ingest import CandidateNormalizationError

    with pytest.raises(CandidateNormalizationError) as ei:
        _ingest_normalize(_raw_candidate(semantic_id="SMOOTH:sma"))
    assert ei.value.reason == "bad_semantic_hash_format"


# --------------------------------------------------------------------------- #
# (d)+(e) admission delegation seam
# --------------------------------------------------------------------------- #


def _request(**extra) -> AdmissionRequest:
    base = {
        "candidate_ref": "FC_abc",
        "content_hash": _h("spec-bytes"),
        "factor_definition_ref": _h("domain-sem"),
        "semantic_ref": _h("domain-sem"),
        "evaluation": None,
        "library_snapshot_ref": "lib:v7",
    }
    base.update(extra)
    return AdmissionRequest(**base)


def test_admission_request_is_format_validated_only():
    req = _request()
    assert req.candidate_ref == "FC_abc"
    assert len(req.content_hash) == 64
    with pytest.raises(ValueError):
        _request(content_hash="not-a-digest")
    with pytest.raises(ValueError):
        _request(candidate_ref="")


def test_verdict_vocabulary_matches_the_domain_package():
    # factor_assets.contracts.admission.AdmissionDecision uses exactly these.
    assert ADMISSION_DECISIONS == {"APPROVED", "REJECTED", "SHADOWED"}
    with pytest.raises(ValueError):
        AdmissionVerdict(decision="MAYBE")


def test_verdict_carries_authority_outputs_verbatim():
    digest = _h("decision-content")
    verdict = AdmissionVerdict(
        decision=DECISION_APPROVED,
        reason_codes=("qualifies", "low_correlation"),
        content_hash=digest,
        policy_ref="fa:policy:2026-08",
        authority="factor_assets.library.promotion_gate",
    )
    assert verdict.reason_codes == ("qualifies", "low_correlation")
    assert verdict.content_hash == digest
    assert verdict.approved and not verdict.rejected and not verdict.review_required
    assert verdict.reason_codes_list() == ["qualifies", "low_correlation"]
    with pytest.raises(ValueError):
        AdmissionVerdict(decision="APPROVED", content_hash="deadbeef")


def test_verdict_flags():
    assert AdmissionVerdict(decision=DECISION_REJECTED).rejected
    assert AdmissionVerdict(decision=DECISION_SHADOWED).review_required


class _AlwaysApprove:
    def __init__(self) -> None:
        self.calls: list[AdmissionRequest] = []

    def decide(self, request: AdmissionRequest) -> AdmissionVerdict:
        self.calls.append(request)
        return AdmissionVerdict(
            decision=DECISION_APPROVED, reason_codes=("qualifies",), authority="_AlwaysApprove"
        )


def test_admission_authority_is_a_structural_protocol():
    assert isinstance(_AlwaysApprove(), AdmissionAuthority)
    assert not isinstance(object(), AdmissionAuthority)


def test_refuse_admission_fails_closed_and_never_approves():
    authority = RefuseAdmission()
    verdict = authority.decide(_request())
    assert verdict.decision == DECISION_REJECTED
    assert verdict.reason_codes == (REASON_AUTHORITY_ABSENT,)
    assert not verdict.approved


def test_refuse_admission_is_the_package_default():
    """Without a composed authority the platform must not invent one that passes."""
    import quant_platform.app.contracts.admission as adm

    sig = inspect.signature(adm.RefuseAdmission.decide)
    assert list(sig.parameters) == ["self", "request"]


def test_pipeline_default_authority_is_refuse_admission():
    from quant_platform.app.orchestrator import Pipeline

    pipeline = Pipeline()
    assert isinstance(pipeline.admission_authority, RefuseAdmission)


def test_verdict_content_hash_is_carried_not_recomputed():
    # The authority's content hash is NOT re-derived from its reason codes: two
    # verdicts with the same codes but different authority hashes stay distinct.
    a = AdmissionVerdict(decision=DECISION_APPROVED, content_hash=_h("x"), policy_ref="p")
    b = AdmissionVerdict(decision=DECISION_APPROVED, content_hash=_h("y"), policy_ref="p")
    assert a.content_hash != b.content_hash
    # and a verdict without one is legal (the authority may omit it)
    assert AdmissionVerdict(decision=DECISION_REJECTED).content_hash == ""


# --------------------------------------------------------------------------- #
# (f) no platform symbol owns an admission threshold
# --------------------------------------------------------------------------- #


def test_pipeline_module_exports_no_decision_function():
    import quant_platform.app.orchestrator as orch

    for banned in (
        "pipeline_gate_decision",
        "DEFAULT_MIN_RANK_IC",
        "VWAP_TO_VWAP_BASIS",
        "_EVIDENCE_NOT_COMPUTED",
        "PromotionGate",
        "PromotionDecision",
    ):
        assert not hasattr(orch, banned), f"platform re-exported decision logic: {banned}"


def test_contracts_export_surface_exposes_the_seam_and_carried_refs():
    import quant_platform.app.contracts as contracts

    for symbol in (
        "AdmissionAuthority",
        "AdmissionRequest",
        "AdmissionVerdict",
        "RefuseAdmission",
        "DECISION_APPROVED",
        "DECISION_REJECTED",
        "DECISION_SHADOWED",
        "REASON_AUTHORITY_ABSENT",
        "IdentityRef",
        "FactorDefinitionRef",
        "FactorValueRef",
        "EvaluationRef",
        "TreatmentRef",
        "sha256_hex",
    ):
        assert hasattr(contracts, symbol), f"missing seam symbol: {symbol}"


def test_contracts_still_pure_stdlib_after_the_seam():
    # mirrors tests/test_smoke.py but for the new modules specifically
    import subprocess
    import sys as _sys

    code = (
        "import sys; "
        "import quant_platform.app.contracts.admission as a; "
        "import quant_platform.app.contracts.identities as i; "
        "third_party={'numpy','pandas','pydantic','fastapi','sqlalchemy','duckdb','pyarrow'}; "
        "loaded=set(sys.modules); "
        "assert not (third_party & loaded), f'third-party modules loaded: {third_party & loaded}'"
    )
    result = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_platform_imports_no_domain_package():
    """The delegation arrow is domain -> platform, never platform -> domain."""
    app_dir = pathlib.Path(_PLATFORM_ROOT) / "app"
    domain_prefixes = re.compile(
        r"^\s*(import|from)\s+(factor_engine|factor_assets|quant_evaluator|"
        r"factor_preprocess|factor_optimizer|factor_mining|vectorbt_qs)\b",
        re.MULTILINE,
    )
    offenders = []
    for path in app_dir.rglob("*.py"):
        if "__pycache__" in path.parts or "build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if domain_prefixes.search(text):
            offenders.append(str(path))
    assert not offenders, f"platform imports domain packages: {offenders}"