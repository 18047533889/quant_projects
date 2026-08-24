# -*- coding: utf-8 -*-
"""VER-P0-02 — exact-identity gate aggregation tests.

A gate is moved to PASS ONLY when the set of machine-verified evidence
``test`` paths EXACTLY equals the expected test-file set from the gate spec
(no missing, no extra, no duplicate).  A partial run (only one of a
two-test gate) must leave the gate NOT_RUN — never a fabricated PASS.

The tests call ``scripts.gen_verification_manifest.apply_verified_evidence``
directly with a constructed gates list and monkeypatched evidence so they do
not depend on the real ``evidence/verified_gate_evidence.json`` or on
``scripts.gate_runner`` being importable mid-refactor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_verification_manifest as g  # noqa: E402

# A gate with two expected test files.  gate_runner's SERIALIZATION gate is a
# real 2-command gate, but we build our own expectation map inside the test so
# the assertions are independent of the sibling agent's in-flight schema edit.
EXPECTED_A = "pkg_a/tests/test_a.py"
EXPECTED_B = "pkg_b/tests/test_b.py"


def _valid_evidence_item(test: str, **overrides) -> dict:
    item = {
        "test": test,
        "command_hash": "a" * 64,
        "exit_code": 0,
        "passed": 5,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "tests": 5,
        "duration_sec": 0.1,
        "executed_at": "2026-08-23T00:00:00+00:00",
        "run_at": "2026-08-23T00:00:00+00:00",
    }
    item.update(overrides)
    return item


def _make_gates(*names: str) -> list[dict]:
    """A minimal gate list containing the requested gate names."""
    gates = [{
        "name": name,
        "description": "",
        "status": "NOT_RUN",
        "evidence": [],
    } for name in (names or ("EXACT_IDENTITY_GATE", "UNKNOWN_GATE"))]
    return gates


def _expectations_two_test_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the expectation map at a 2-test gate named EXACT_IDENTITY_GATE."""
    class _FakeSpec:
        gate_id = "EXACT_IDENTITY_GATE"
        tests = (EXPECTED_A, EXPECTED_B)
        skip_policy = "allowed"
        allowed_skip_inventory = None

    monkeypatch.setattr(
        g,
        "_build_gate_expectations",
        lambda: (
            {"EXACT_IDENTITY_GATE": {EXPECTED_A, EXPECTED_B}},
            {"EXACT_IDENTITY_GATE": "allowed"},
            {},
        ),
    )


def _monkeypatch_evidence(monkeypatch: pytest.MonkeyPatch, entries: list[dict]) -> None:
    monkeypatch.setattr(g, "_load_verified_evidence", lambda: entries)


def test_partial_evidence_leaves_gate_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ONE of the two expected tests has valid evidence -> NOT_RUN."""
    _expectations_two_test_gate(monkeypatch)
    _monkeypatch_evidence(monkeypatch, [
        {"name": "EXACT_IDENTITY_GATE", "status": "PASS", "evidence": [
            _valid_evidence_item(EXPECTED_A),
        ]},
    ])
    gates = g.apply_verified_evidence(_make_gates(), verbose=False)
    status = next(x["status"] for x in gates if x["name"] == "EXACT_IDENTITY_GATE")
    assert status == "NOT_RUN", (
        "a gate with only one of two expected test files verified must stay "
        f"NOT_RUN (exact identity), got {status!r}"
    )


def test_exact_evidence_passes_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both expected tests verified (exact set) -> PASS."""
    _expectations_two_test_gate(monkeypatch)
    _monkeypatch_evidence(monkeypatch, [
        {"name": "EXACT_IDENTITY_GATE", "status": "PASS", "evidence": [
            _valid_evidence_item(EXPECTED_A),
            _valid_evidence_item(EXPECTED_B),
        ]},
    ])
    gates = g.apply_verified_evidence(_make_gates(), verbose=False)
    gate = next(x for x in gates if x["name"] == "EXACT_IDENTITY_GATE")
    assert gate["status"] == "PASS"
    assert {e["test"] for e in gate["evidence"]} == {EXPECTED_A, EXPECTED_B}


def test_extra_evidence_keeps_gate_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both expected tests verified PLUS an extra test file -> NOT_RUN."""
    _expectations_two_test_gate(monkeypatch)
    _monkeypatch_evidence(monkeypatch, [
        {"name": "EXACT_IDENTITY_GATE", "status": "PASS", "evidence": [
            _valid_evidence_item(EXPECTED_A),
            _valid_evidence_item(EXPECTED_B),
            _valid_evidence_item("pkg_c/tests/test_extra.py"),
        ]},
    ])
    gates = g.apply_verified_evidence(_make_gates(), verbose=False)
    status = next(x["status"] for x in gates if x["name"] == "EXACT_IDENTITY_GATE")
    assert status == "NOT_RUN", (
        "a gate with extra (out-of-set) evidence must stay NOT_RUN, got "
        f"{status!r}"
    )


def test_duplicate_evidence_keeps_gate_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same test verified twice (duplicate) -> NOT_RUN."""
    _expectations_two_test_gate(monkeypatch)
    _monkeypatch_evidence(monkeypatch, [
        {"name": "EXACT_IDENTITY_GATE", "status": "PASS", "evidence": [
            _valid_evidence_item(EXPECTED_A),
            _valid_evidence_item(EXPECTED_A),
            _valid_evidence_item(EXPECTED_B),
        ]},
    ])
    gates = g.apply_verified_evidence(_make_gates(), verbose=False)
    status = next(x["status"] for x in gates if x["name"] == "EXACT_IDENTITY_GATE")
    assert status == "NOT_RUN", (
        "a gate with duplicate evidence for one test file must stay NOT_RUN, "
        f"got {status!r}"
    )


def test_unknown_gate_stays_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate with no expectation-map entry can never be promoted to PASS."""
    _expectations_two_test_gate(monkeypatch)
    _monkeypatch_evidence(monkeypatch, [
        {"name": "UNKNOWN_GATE", "status": "PASS", "evidence": [
            _valid_evidence_item("some/test.py"),
        ]},
    ])
    gates = g.apply_verified_evidence(_make_gates(), verbose=False)
    status = next(x["status"] for x in gates if x["name"] == "UNKNOWN_GATE")
    assert status == "NOT_RUN", (
        "a gate with no expected-test-set entry cannot be proven by exact "
        f"identity; got {status!r}"
    )


def test_fail_on_skip_policy_blocks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """fail_on_skip gate with skipped > 0 and no allowed inventory -> NOT_RUN."""
    class _FakeSpec:
        gate_id = "SKIP_GATE"
        tests = (EXPECTED_A,)
        skip_policy = "fail_on_skip"
        allowed_skip_inventory = None

    monkeypatch.setattr(
        g,
        "_build_gate_expectations",
        lambda: (
            {"SKIP_GATE": {EXPECTED_A}},
            {"SKIP_GATE": "fail_on_skip"},
            {},
        ),
    )
    _monkeypatch_evidence(monkeypatch, [
        {"name": "SKIP_GATE", "status": "PASS", "evidence": [
            _valid_evidence_item(EXPECTED_A, skipped=2),
        ]},
    ])
    gates = g.apply_verified_evidence(_make_gates("SKIP_GATE"), verbose=False)
    status = next(x["status"] for x in gates if x["name"] == "SKIP_GATE")
    assert status == "NOT_RUN", (
        "fail_on_skip gate with skipped tests must not PASS, got {status!r}"
    )


def test_allowed_skip_inventory_permits_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """skipped > 0 covered by allowed_skip_inventory may still PASS."""
    class _FakeSpec:
        gate_id = "SKIP_GATE"
        tests = (EXPECTED_A,)
        skip_policy = "fail_on_skip"
        allowed_skip_inventory = (EXPECTED_A,)

    monkeypatch.setattr(
        g,
        "_build_gate_expectations",
        lambda: (
            {"SKIP_GATE": {EXPECTED_A}},
            {"SKIP_GATE": "fail_on_skip"},
            {"SKIP_GATE": {EXPECTED_A}},
        ),
    )
    _monkeypatch_evidence(monkeypatch, [
        {"name": "SKIP_GATE", "status": "PASS", "evidence": [
            _valid_evidence_item(EXPECTED_A, skipped=2),
        ]},
    ])
    gates = g.apply_verified_evidence(_make_gates("SKIP_GATE"), verbose=False)
    status = next(x["status"] for x in gates if x["name"] == "SKIP_GATE")
    assert status == "PASS", (
        "an allowed skip inventory may permit a pass, got {status!r}"
    )
