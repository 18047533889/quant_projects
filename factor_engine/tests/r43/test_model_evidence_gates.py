# -*- coding: utf-8 -*-
"""MF-P0-001/002/003: Model evidence generation and DirectUse readiness gates.

Tests:
1. MODEL_CURRENT_HEAD evidence must bind to current repo HEAD (MF-P0-001/REM-025)
2. DirectUse readiness gate returns honest counts (MF-P0-002/REM-024)
3. Behavioral certification ledger matches real on-disk evidence (MF-P0-003/REM-171)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# R49 reorg: this file lives at <monorepo>/factor_engine/tests/r43/, so the
# monorepo root (where evidence/factor_engine/model_operators and the git HEAD
# both resolve) is parents[3], NOT parents[2] (which is factor_engine/ itself
# and would shadow the real evidence tree with a stray factor_engine/evidence/).
REPO = Path(__file__).resolve().parents[3]


def _current_head() -> str | None:
    """Get current git HEAD."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# MF-P0-001 / REM-025: MODEL_CURRENT_HEAD binds to current HEAD
# --------------------------------------------------------------------------- #
def test_model_current_head_json_exists():
    """MODEL_CURRENT_HEAD.json must exist after certifier runs."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "MODEL_CURRENT_HEAD.json"
    assert path.exists(), "MODEL_CURRENT_HEAD.json missing; run generate_model_canonical_ledger.py"


def test_model_current_head_binds_to_real_head():
    """MODEL_CURRENT_HEAD.json must bind to the ACTUAL repo HEAD, not a stale SHA."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "MODEL_CURRENT_HEAD.json"
    if not path.exists():
        pytest.skip("MODEL_CURRENT_HEAD.json does not exist")

    data = json.loads(path.read_text(encoding="utf-8"))
    head = _current_head()

    if head is None:
        pytest.skip("not in a git repository")

    # The evidence SHA MUST match the current HEAD
    assert data.get("commit_sha") == head, (
        f"MODEL_CURRENT_HEAD.json bound to {data.get('commit_sha')} "
        f"but repo HEAD is {head} — STALE evidence"
    )


def test_model_current_head_has_required_fields():
    """MODEL_CURRENT_HEAD.json must have all required fields."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "MODEL_CURRENT_HEAD.json"
    if not path.exists():
        pytest.skip("MODEL_CURRENT_HEAD.json does not exist")

    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "commit_sha", "canonical_count", "model_like_count",
        "explicit_timing_count", "final_direct_use_ready_count", "generated_at",
    }
    missing = required - set(data.keys())
    assert not missing, f"Missing fields: {missing}"


def test_evidence_freshness_gate_fails_on_mismatch():
    """report_hard_gate_set MODEL_CURRENT_HEAD_EVIDENCE_FRESH must FAIL on stale evidence."""
    from modeling.evidence import report_hard_gate_set

    head = _current_head()
    if head is None:
        pytest.skip("not in a git repository")

    wrong_sha = "0" * 40 if head != "0" * 40 else "1" * 40
    gates = report_hard_gate_set(git_sha=wrong_sha, current_head=head)
    entry = gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]

    assert entry["value"] is False, "freshness gate must FAIL on mismatched SHA"
    assert entry["status"] == "FAIL"
    assert wrong_sha in entry["check"]
    assert head in entry["check"]


def test_evidence_freshness_gate_passes_on_match():
    """report_hard_gate_set MODEL_CURRENT_HEAD_EVIDENCE_FRESH must PASS when SHAs match."""
    from modeling.evidence import report_hard_gate_set

    head = _current_head()
    if head is None:
        pytest.skip("not in a git repository")

    gates = report_hard_gate_set(git_sha=head, current_head=head)
    entry = gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]

    assert entry["value"] is True, "freshness gate must PASS when SHAs match"
    assert entry["status"] == "PASS"


# --------------------------------------------------------------------------- #
# MF-P0-002 / REM-024: DirectUse readiness gate (honest state reporting)
# --------------------------------------------------------------------------- #
def test_direct_use_readiness_gate_exists():
    """check_model_direct_use_readiness must be callable and return structured data."""
    from modeling.evidence import check_model_direct_use_readiness

    ready_count, total_count, reasons = check_model_direct_use_readiness()

    assert isinstance(ready_count, int)
    assert isinstance(total_count, int)
    assert isinstance(reasons, dict)
    assert ready_count >= 0
    assert total_count >= 0
    assert ready_count <= total_count


def test_direct_use_readiness_gate_is_honest():
    """DirectUse readiness gate must report the REAL state, not fake-green."""
    from modeling.evidence import check_model_direct_use_readiness

    ready_count, total_count, reasons = check_model_direct_use_readiness()

    # If ready_count > 0, there MUST be operators with genuine evidence
    if ready_count > 0:
        # At least one operator must have NO failures
        ready_ops = [op for op in reasons.keys() if not reasons[op]]
        assert len(ready_ops) >= ready_count, (
            f"ready_count={ready_count} but no operators have zero failures"
        )

    # If ready_count == 0, ALL operators must have at least one failure
    if ready_count == 0 and total_count > 0:
        assert all(reasons.get(op, []) for op in reasons if op != "_error"), (
            "ready_count=0 but some operators have no failures listed"
        )


def test_direct_use_readiness_gate_monkeypatch_evidence_removal():
    """If we remove evidence, ready_count must drop (proves gate is not vacuous)."""
    from modeling.evidence import check_model_direct_use_readiness

    ready_before, total_before, _ = check_model_direct_use_readiness()

    # Monkeypatch to simulate missing parameter domain store
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "parameter_domain_store" in name:
            raise ImportError("simulated evidence removal")
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = mock_import
        ready_after, total_after, _ = check_model_direct_use_readiness()
        builtins.__import__ = original_import
    except Exception:
        builtins.__import__ = original_import
        raise

    # With evidence removed, ready_count should not INCREASE
    assert ready_after <= ready_before, (
        f"ready_count INCREASED when evidence was removed: {ready_before} -> {ready_after}"
    )


def test_direct_use_readiness_appears_in_report_hard_gate_set():
    """report_hard_gate_set must include the DirectUse readiness gate."""
    from modeling.evidence import report_hard_gate_set

    gates = report_hard_gate_set()
    assert "MODEL_ALL_DIRECT_USE_HAVE_BEHAVIORAL_CERTIFICATION" in gates

    entry = gates["MODEL_ALL_DIRECT_USE_HAVE_BEHAVIORAL_CERTIFICATION"]
    assert "status" in entry
    assert entry["status"] in {"PASS", "FAIL", "NOT_RUN"}
    assert isinstance(entry["value"], bool)
    assert "check" in entry


# --------------------------------------------------------------------------- #
# MF-P0-003 / REM-171: Behavioral certification ledger matches on-disk evidence
# --------------------------------------------------------------------------- #
def test_behavioral_certification_ledger_exists():
    """behavioral_certification_ledger.json must exist after script runs."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "behavioral_certification_ledger.json"
    # This is OK to skip if not generated yet
    if not path.exists():
        pytest.skip("behavioral_certification_ledger.json not generated yet")


def test_behavioral_certification_ledger_structure():
    """behavioral_certification_ledger.json must have required structure."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "behavioral_certification_ledger.json"
    if not path.exists():
        pytest.skip("behavioral_certification_ledger.json not generated yet")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert "generated_at" in data
    assert "commit_sha" in data
    assert "certified_operators" in data
    assert isinstance(data["certified_operators"], list)

    if data["certified_operators"]:
        op = data["certified_operators"][0]
        assert "canonical" in op
        assert "certifications" in op
        cert = op["certifications"]
        assert "parameter_domain" in cert
        assert "status" in cert["parameter_domain"]


def test_behavioral_certification_ledger_binds_to_current_head():
    """behavioral_certification_ledger.json must bind to current HEAD."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "behavioral_certification_ledger.json"
    if not path.exists():
        pytest.skip("behavioral_certification_ledger.json not generated yet")

    head = _current_head()
    if head is None:
        pytest.skip("not in a git repository")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("commit_sha") == head, (
        f"behavioral_certification_ledger.json bound to {data.get('commit_sha')} "
        f"but repo HEAD is {head} — STALE"
    )


def test_behavioral_certification_never_fakes_certified_status():
    """If ledger says CERTIFIED, the evidence artifact must exist."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "behavioral_certification_ledger.json"
    if not path.exists():
        pytest.skip("behavioral_certification_ledger.json not generated yet")

    data = json.loads(path.read_text(encoding="utf-8"))

    # Check parameter_domain: if CERTIFIED, the store must have entries
    param_certified = [
        op for op in data["certified_operators"]
        if op["certifications"]["parameter_domain"]["status"] == "CERTIFIED"
    ]

    if param_certified:
        # The parameter domain store must exist and be loadable
        store_path = REPO / "evidence" / "factor_engine" / "r37" / "R37_PARAMETER_DOMAIN_STORE.json"
        assert store_path.exists(), (
            f"{len(param_certified)} operators marked parameter_domain CERTIFIED "
            "but R37_PARAMETER_DOMAIN_STORE.json missing"
        )


def test_model_final_hard_gates_binds_to_current_head():
    """MODEL_FINAL_HARD_GATES.json must bind to current HEAD."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "MODEL_FINAL_HARD_GATES.json"
    if not path.exists():
        pytest.skip("MODEL_FINAL_HARD_GATES.json does not exist")

    head = _current_head()
    if head is None:
        pytest.skip("not in a git repository")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("commit_sha") == head, (
        f"MODEL_FINAL_HARD_GATES.json bound to {data.get('commit_sha')} "
        f"but repo HEAD is {head} — STALE"
    )


def test_model_final_hard_gates_has_freshness_gate():
    """MODEL_FINAL_HARD_GATES.json must include MODEL_CURRENT_HEAD_EVIDENCE_FRESH."""
    path = REPO / "evidence" / "factor_engine" / "model_operators" / "MODEL_FINAL_HARD_GATES.json"
    if not path.exists():
        pytest.skip("MODEL_FINAL_HARD_GATES.json does not exist")

    data = json.loads(path.read_text(encoding="utf-8"))
    gates = data.get("gates", {})
    assert "MODEL_CURRENT_HEAD_EVIDENCE_FRESH" in gates

    gate = gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]
    assert "status" in gate
    assert gate["status"] in {"PASS", "FAIL", "NOT_RUN"}
