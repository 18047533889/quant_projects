# -*- coding: utf-8 -*-
"""VER-P0-01 — regression tests proving the gate runner can never fake a PASS.

A gate PASS must be produced ONLY by actually executing pytest and reading
back a structured (JUnit XML) result.  These tests prove the three failure
modes:

  (a) a test file that EXISTS but FAILS      -> status == FAIL
  (b) a test file that does NOT exist        -> status != PASS (BLOCKED/FAIL)
  (c) a test file that PASSES                -> status == PASS

The test fixtures below are real pytest files (not mocks), so a hypothetical
"pretend to run" implementation would be caught.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

from gate_runner import GateRunner, GateSpec, Status  # noqa: E402

HERE = Path(__file__).resolve().parent

# A passing test file (real pytest file, real pytest execution).
PASSING_TEST = HERE / "_gr_fixture_pass_test.py"
# A test file that exists but always FAILS.
FAILING_TEST = HERE / "_gr_fixture_fail_test.py"
# A path that does not exist.
MISSING_TEST = HERE / "_gr_fixture_does_not_exist_test.py"

PASSING_TEST.write_text(
    "def test_pass_true():\n    assert True\n",
    encoding="utf-8",
)
FAILING_TEST.write_text(
    "def test_fail_always():\n    assert False\n",
    encoding="utf-8",
)


def _spec(test_file: Path, gate_id: str = "GR_HONESTY") -> GateSpec:
    rel = str(test_file.relative_to(Path(__file__).resolve().parents[2]))
    return GateSpec(
        gate_id=gate_id,
        commands=((rel, "-q", "--junitxml={junitxml}", "-p", "no:cacheprovider"),),
        tests=(rel,),
        timeout_sec=300,
    )


def test_failing_test_file_cannot_produce_pass() -> None:
    """(a) file EXISTS but the test FAILS -> status must be FAIL, never PASS."""
    assert FAILING_TEST.is_file(), "fixture file must exist for this scenario"
    spec = _spec(FAILING_TEST, "GR_HONESTY_FAIL")
    result = GateRunner().run_gate(spec)
    assert result["status"] == Status.FAIL, (
        f"expected FAIL for a failing test, got {result['status']!r}: "
        f"{result['evidence']}"
    )
    ev = result["evidence"][0]
    assert ev["exit_code"] != 0
    assert ev["failed"] is not None and ev["failed"] >= 1
    assert ev["tests"] is not None and ev["tests"] >= 1


def test_missing_test_file_cannot_produce_pass() -> None:
    """(b) test file does NOT exist -> status != PASS (BLOCKED/FAIL)."""
    assert not MISSING_TEST.exists()
    spec = _spec(MISSING_TEST, "GR_HONESTY_MISSING")
    result = GateRunner().run_gate(spec)
    assert result["status"] != Status.PASS, (
        "a gate pointing at a non-existent test file must never be PASS"
    )


def test_passing_test_file_produces_pass() -> None:
    """(c) file EXISTS and the test PASSES -> status == PASS."""
    assert PASSING_TEST.is_file()
    spec = _spec(PASSING_TEST, "GR_HONESTY_PASS")
    result = GateRunner().run_gate(spec)
    assert result["status"] == Status.PASS, (
        f"expected PASS for a passing test, got {result['status']!r}: "
        f"{result['evidence']}"
    )
    ev = result["evidence"][0]
    assert ev["exit_code"] == 0
    assert ev["passed"] == 1
    assert ev["failed"] == 0 and ev["errors"] == 0
    assert ev["command_hash"]
    assert ev["executed_at"]


def test_artifact_shape_has_machine_fields() -> None:
    """Every evidence entry must carry the machine fields (no hard-coded counts)."""
    spec = _spec(PASSING_TEST, "GR_HONESTY_ARTIFACT")
    artifact = GateRunner().build_artifact([spec])
    gate = artifact["gates"][0]
    ev = gate["evidence"][0]
    for field in (
        "test", "argv", "command_hash", "exit_code", "passed", "failed",
        "errors", "skipped", "duration_sec", "executed_at",
    ):
        assert field in ev, f"missing machine field {field!r} in evidence entry"
    assert ev["command_hash"], "command_hash must be non-empty"
    assert ev["executed_at"], "executed_at must be non-empty"


def test_fabricated_evidence_cannot_produce_pass() -> None:
    """Manifest consumer must reject evidence without real machine fields.

    This mirrors the VER-P0-01 contract in
    scripts/gen_verification_manifest.apply_verified_evidence(): an entry that
    claims ``result: "passed"`` with only a bare ``count`` (the old hard-coded
    shape) but no command_hash / exit_code / failed / errors must be treated as
    NOT_RUN — never PASS.
    """
    import json
    import sys as _sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gen_verification_manifest as gvm

    # Build the same gate list load_gates() produces.
    cfg_path = gvm.REPO_ROOT / "config" / "gates.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    gates = [
        {
            "name": g["name"],
            "description": g.get("description", ""),
            "status": g.get("default_status", "NOT_RUN"),
            "evidence": [],
        }
        for g in cfg.get("gates", [])
    ]
    # A fabricated entry: result="passed" + count, but NO command_hash / exit_code.
    fabricated = {
        "name": "NUMERICAL_ORACLE",
        "status": "PASS",
        "evidence": [
            {
                "test": "quant_evaluator/tests/test_numerical_oracle.py",
                "result": "passed",
                "count": 13,
                "run_at": "2026-08-21T00:00:00+00:00",
            }
        ],
    }
    tmp = Path(__file__).resolve().parent / "_gr_fabricated_evidence.json"
    tmp.write_text(json.dumps({"gates": [fabricated]}), encoding="utf-8")
    try:
        orig = gvm.VERIFIED_EVIDENCE_PATH
        gvm.VERIFIED_EVIDENCE_PATH = tmp
        try:
            res = gvm.apply_verified_evidence(gates, verbose=False)
        finally:
            gvm.VERIFIED_EVIDENCE_PATH = orig
    finally:
        tmp.unlink(missing_ok=True)
    gate = next(g for g in res if g["name"] == "NUMERICAL_ORACLE")
    assert gate["status"] == Status.NOT_RUN, (
        f"fabricated evidence (no command_hash/exit_code) must NOT produce PASS, "
        f"got {gate['status']!r}"
    )
