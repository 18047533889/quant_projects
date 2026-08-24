#!/usr/bin/env python3
"""Gate Matrix — honest gate status from verified evidence.

Reads verified_gate_evidence.json and gates.json, computes the current git SHA,
and outputs a Gate Matrix (PASS/FAIL/STALE/NOT_RUN/BLOCKED) + per-package
classification (PRODUCTION_CANDIDATE/SAFE_BY_GATING/RESEARCH_ONLY/OFFLINE_ONLY/BROKEN).

VER-P0-05: "ALL GREEN" (exit 0) requires EVERY gate in REQUIRED_GATES == PASS
AND the evidence git SHA == current HEAD AND a clean working tree
(`git status --porcelain` empty).  SUBMODULE_REACHABILITY is NOT_RUN (never
PASS) when the repo declares no git submodules (no mode-160000 index entries).
Exit code 1 otherwise.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_PATH = os.path.join(REPO_ROOT, "evidence", "verified_gate_evidence.json")
GATES_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "gates.json")

# R46 P0-X: single authoritative verdict system.  GateMatrix only RENDERS
# verdicts from it — it never re-derives PASS.
try:
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from gate_evidence_verifier import (
        GateEvidenceVerifier,
        NOT_APPLICABLE,
        NOT_RUN,
    )
except Exception as _gve_import_err:  # pragma: no cover - import hardening
    print(f"ERROR: cannot import GateEvidenceVerifier: {_gve_import_err}", file=sys.stderr)
    sys.exit(2)


def get_current_git_sha() -> str:
    """Return the current HEAD SHA of the repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"ERROR: could not get git SHA: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(2)
    return result.stdout.strip()


def load_evidence(path: str) -> dict[str, Any]:
    """Load the verified gate evidence JSON."""
    with open(path, "r") as f:
        return json.load(f)


def load_gates_config(path: str) -> list[dict[str, Any]]:
    """Load the gates.json config."""
    with open(path, "r") as f:
        return json.load(f)["gates"]


def git_tree_dirty() -> bool:
    """True when `git status --porcelain` is non-empty (uncommitted changes)."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return True  # fail-closed: cannot prove clean
    return bool(result.stdout.strip())


def gitlink_count() -> int:
    """Number of real git submodule pins (git index entries with mode 160000)."""
    result = subprocess.run(
        ["git", "ls-files", "--stage"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return 0
    n = 0
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "160000":
            n += 1
    return n


# ── Package-to-gate mapping ──────────────────────────────────────────────────
# Determined from test file paths in the evidence.
# Cross-package / integration gates apply to every package.
CROSS_PACKAGE_GATES = {
    "CROSS_PACKAGE",
    "ASHARE_SEMANTIC_CONTRACT_GOLDEN",
    "ASHARE_REAL_DATA_SHADOW",
}

PACKAGE_GATES: dict[str, set[str]] = {
    "factor_engine": {
        "UNIT", "LEAKAGE", "PIT", "DETERMINISM",
        "CHECKPOINT_RESUME", "FRESH_WHEEL",
    },
    "quant_evaluator": {
        "NUMERICAL_ORACLE", "PROPERTY", "SERIALIZATION",
        "1K_SCALE", "10K_SCALE", "100K_SCALE",
    },
    "factor_optimizer": set(),
    "factor_assets": {"SERIALIZATION"},
    "factor_preprocess": {"LEAKAGE"},
    "dataaccess": set(),
}

# Gates that are critical for BROKEN classification
CRITICAL_GATES = {"UNIT", "PROPERTY", "NUMERICAL_ORACLE"}

# VER-P0-05: gates that must ALL be PASS for exit 0 / "all green".
# (the 15 existing gates + the 5 hard/structural gates)
REQUIRED_GATES = {
    "UNIT", "NUMERICAL_ORACLE", "PROPERTY", "CROSS_PACKAGE",
    "LEAKAGE", "PIT", "DETERMINISM", "SERIALIZATION",
    "CHECKPOINT_RESUME", "FRESH_WHEEL", "1K_SCALE", "10K_SCALE",
    "100K_SCALE", "ASHARE_SEMANTIC_CONTRACT_GOLDEN",
    "ASHARE_REAL_DATA_SHADOW",
    "SOURCE_AUTHORITY", "SUPPLY_CHAIN", "SUBMODULE_REACHABILITY",
    "FRESH_WHEEL_MATRIX", "EVIDENCE_CURRENT",
}

# Gates not yet wired to a specific runner; they are read from evidence when
# present (a runner populates them later) and stay NOT_RUN otherwise.
EVIDENCE_BACKED_GATES = {
    "SOURCE_AUTHORITY", "SUPPLY_CHAIN", "FRESH_WHEEL_MATRIX", "EVIDENCE_CURRENT",
}

# R46 P0-Z: the platform structural gates that gate EVERY production candidate.
# All must be PASS (or legitimate NOT_APPLICABLE) for ANY package to be
# PRODUCTION_CANDIDATE; a single FAIL/BLOCKED/STALE floors the platform to
# PRE_PRODUCTION / SAFE_BY_GATING — never PRODUCTION_CANDIDATE.
PLATFORM_STRUCTURAL_GATES = {
    "SOURCE_AUTHORITY", "SUPPLY_CHAIN", "FRESH_WHEEL_MATRIX", "EVIDENCE_CURRENT",
    "SUBMODULE_REACHABILITY",
}

# R46 P0-Z: every production package must at least carry this core set of gates
# (cross-package + the structural package-specific responsibilities).  A
# package with an empty package-specific gate set can no longer be
# PRODUCTION_CANDIDATE via CROSS_PACKAGE alone.
PLATFORM_CORE_GATES = {
    "SOURCE_AUTHORITY", "PACKAGE_UNIT", "PACKAGE_CONTRACT",
    "SERIALIZATION", "DETERMINISM", "FRESH_WHEEL", "SUPPLY_CHAIN",
    "CROSS_PACKAGE",
}

ASPIRATIONAL_GATES: set[str] = set()


def _first_entry_note(gate_obj: dict) -> str:
    """Pull the first evidence entry's note (verifier reason) if present."""
    ev = (gate_obj or {}).get("evidence", []) or []
    for e in ev:
        if e.get("note"):
            return e["note"]
    return ""


def _find_spec(gate_name: str):
    """Look up the live GateSpec for a gate (None when unavailable)."""
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        from gate_runner import find_spec as _fs  # noqa: PLC0415
        return _fs(gate_name)
    except Exception:
        return None


def _spec_expected_hashes(gate_name: str) -> set[str] | None:
    """ExpectedCommandIdentitySet for the live spec (None for unknown/verifier gates)."""
    spec = _find_spec(gate_name)
    if spec is None:
        return None
    return GateEvidenceVerifier().expected_command_identity_set(spec)


def _verdict_expected_commands(gate_name: str, spec) -> set[str] | None:
    """ExpectedCommandIdentitySet to bind the verdict; empty for verifier gates.

    Returns ``None`` for verifier-backed gates so the verifier trusts its
    authoritative per-entry status instead of command identity.
    """
    if spec is not None and getattr(spec, "verifier", None):
        return None
    if spec is not None:
        return GateEvidenceVerifier().expected_command_identity_set(spec)
    try:
        return _spec_expected_hashes(gate_name)
    except Exception:
        return None


def compute_gate_status(
    gate_name: str,
    evidence_gates: dict[str, list[dict[str, Any]]],
    evidence_sha: str | None,
    current_sha: str,
    verifier: GateEvidenceVerifier,
) -> tuple[str, int, str, str]:
    """Compute a SINGLE authoritative status via GateEvidenceVerifier.

    R46 P0-X: the verifier is the only source of gate truth; GateMatrix only
    renders.  Returns (status, total_tests, timestamp, reason).
    """
    entries = evidence_gates.get(gate_name)
    spec = _find_spec(gate_name)
    live_hashes = _verdict_expected_commands(gate_name, spec)
    verdict = verifier.verify_gate(
        gate_name=gate_name,
        spec=spec,
        entries=entries or [],
        live_command_hashes=live_hashes,
    )
    return (verdict.status, verdict.tests, verdict.timestamp, verdict.reason_text())


def classify_package(
    package: str,
    gate_statuses: dict[str, str],
) -> str:
    """Classify a package based on its applicable gate statuses.

    PRODUCTION_CANDIDATE: all applicable gates PASS
    SAFE_BY_GATING: all applicable gates are at least NOT_RUN (no FAIL/BLOCKED)
    RESEARCH_ONLY: some gates FAIL or are BLOCKED (but not critical gates)
    OFFLINE_ONLY: no gate evidence at all for this package
    BROKEN: critical gates (UNIT, PROPERTY, NUMERICAL_ORACLE) are FAIL/BLOCKED
    """
    applicable = PACKAGE_GATES.get(package, set()) | CROSS_PACKAGE_GATES

    # OFFLINE_ONLY: no gates apply to this package at all
    if not applicable:
        return "OFFLINE_ONLY"

    # Check if there's any evidence at all for this package's gates
    has_any_evidence = any(
        gate_statuses.get(g) not in ("NOT_RUN", None)
        for g in applicable
    )
    if not has_any_evidence:
        return "OFFLINE_ONLY"

    # Check critical gates
    for g in applicable:
        if g in CRITICAL_GATES:
            s = gate_statuses.get(g)
            if s in ("FAIL", "BLOCKED"):
                return "BROKEN"

    # Check for any FAIL or BLOCKED
    has_fail_or_blocked = any(
        gate_statuses.get(g) in ("FAIL", "BLOCKED")
        for g in applicable
    )
    if has_fail_or_blocked:
        return "RESEARCH_ONLY"

    # Check if all applicable gates PASS
    all_pass = all(
        gate_statuses.get(g) == "PASS"
        for g in applicable
    )
    if all_pass:
        return "PRODUCTION_CANDIDATE"

    # Otherwise: all gates at least NOT_RUN (no FAIL/BLOCKED)
    return "SAFE_BY_GATING"


def print_matrix(
    gate_order: list[str],
    gate_statuses: dict[str, str],
    gate_tests: dict[str, int],
    gate_timestamps: dict[str, str],
) -> None:
    """Print the Gate Matrix as a markdown table."""
    print("# Gate Matrix\n")
    print("| Gate | Status | Tests | Evidence Timestamp |")
    print("|------|--------|-------|--------------------|")
    for gate in gate_order:
        status = gate_statuses.get(gate, "NOT_RUN")
        tests = gate_tests.get(gate, 0)
        ts = gate_timestamps.get(gate, "—")
        print(f"| {gate} | {status} | {tests} | {ts} |")
    print()


def print_package_classification(
    packages: list[str],
    gate_statuses: dict[str, str],
) -> None:
    """Print the Package Classification as a markdown table."""
    print("# Package Classification\n")
    print("| Package | Classification |")
    print("|---------|---------------|")
    for pkg in packages:
        classification = classify_package(pkg, gate_statuses)
        print(f"| {pkg} | {classification} |")
    print()


def main() -> int:
    # Load evidence
    if not os.path.exists(EVIDENCE_PATH):
        print(f"ERROR: evidence not found at {EVIDENCE_PATH}", file=sys.stderr)
        sys.exit(2)
    evidence = load_evidence(EVIDENCE_PATH)
    evidence_sha = evidence.get("git_sha")
    evidence_timestamp = evidence.get("timestamp", "—")

    # Load gates config
    if not os.path.exists(GATES_CONFIG_PATH):
        print(f"ERROR: gates config not found at {GATES_CONFIG_PATH}", file=sys.stderr)
        sys.exit(2)
    gates_config = load_gates_config(GATES_CONFIG_PATH)

    # Get current git SHA
    current_sha = get_current_git_sha()
    tree_dirty = git_tree_dirty()
    n_gitlinks = gitlink_count()

    # Build evidence lookup: gate_name -> list of entries, and gate_name -> gate obj
    evidence_gates: dict[str, list[dict[str, Any]]] = {}
    evidence_gate_objs: dict[str, dict[str, Any]] = {}
    for gate_entry in evidence.get("gates", []):
        name = gate_entry["name"]
        evidence_gates[name] = gate_entry.get("evidence", [])
        evidence_gate_objs[name] = gate_entry

    # VER-P0-05: SUBMODULE_REACHABILITY — with no git submodules declared there
    # is nothing pinned to prove, so the gate is NOT_RUN (never PASS).
    submodule_declared = n_gitlinks > 0

    # Determine gate order: from gates.json, then any extras from evidence
    configured_gate_names = [g["name"] for g in gates_config]
    all_gate_names = list(configured_gate_names)
    for g in evidence_gates:
        if g not in all_gate_names:
            all_gate_names.append(g)

    # Compute status for each gate
    gate_statuses: dict[str, str] = {}
    gate_tests: dict[str, int] = {}
    gate_timestamps: dict[str, str] = {}
    gate_notes: dict[str, str] = {}

    for gate_name in all_gate_names:
        if gate_name == "SUBMODULE_REACHABILITY":
            if submodule_declared:
                # Submodules ARE declared: normal evidence evaluation.
                status, tests, ts = compute_gate_status(
                    gate_name, evidence_gates, evidence_sha, current_sha,
                )
            else:
                status, tests, ts = ("NOT_RUN", 0, "—")
                gate_notes[gate_name] = "no submodules declared; nothing pinned to prove"
        elif gate_name in EVIDENCE_BACKED_GATES:
            # SOURCE_AUTHORITY / SUPPLY_CHAIN / FRESH_WHEEL_MATRIX /
            # EVIDENCE_CURRENT: the runner now emits an authoritative per-gate
            # ``status`` (PASS/FAIL/BLOCKED/NOT_RUN) derived from REAL verifier
            # output.  Prefer that over recomputing from synthetic entries.
            gate_status = evidence_gate_objs.get(gate_name, {}).get("status")
            if gate_status is not None:
                gate_notes[gate_name] = _first_entry_note(evidence_gate_objs.get(gate_name, {}))
                status, tests, ts = gate_status, 0, evidence_timestamp
            elif evidence_gates.get(gate_name):
                status, tests, ts = compute_gate_status(
                    gate_name, evidence_gates, evidence_sha, current_sha,
                )
            else:
                status, tests, ts = ("NOT_RUN", 0, "—")
                gate_notes[gate_name] = "no evidence recorded (runner not yet wired)"
        else:
            status, tests, ts = compute_gate_status(
                gate_name, evidence_gates, evidence_sha, current_sha,
            )
        gate_statuses[gate_name] = status
        gate_tests[gate_name] = tests
        gate_timestamps[gate_name] = ts

    # Print header info
    print(f"# Gate Matrix Report")
    print(f"# Evidence SHA: {evidence_sha or 'N/A'}")
    print(f"# Current HEAD: {current_sha}")
    print(f"# Evidence timestamp: {evidence_timestamp}")
    print(f"# Evidence fresh: {evidence_sha == current_sha}")
    print(f"# Tree dirty: {tree_dirty}")
    print()

    # Print Gate Matrix
    print_matrix(all_gate_names, gate_statuses, gate_tests, gate_timestamps)

    # Print Package Classification
    packages = ["factor_engine", "quant_evaluator", "factor_optimizer",
                "factor_assets", "factor_preprocess", "dataaccess"]
    print_package_classification(packages, gate_statuses)

    # Summary
    pass_count = sum(1 for s in gate_statuses.values() if s == "PASS")
    stale_count = sum(1 for s in gate_statuses.values() if s == "STALE")
    fail_count = sum(1 for s in gate_statuses.values() if s == "FAIL")
    blocked_count = sum(1 for s in gate_statuses.values() if s == "BLOCKED")
    not_run_count = sum(1 for s in gate_statuses.values() if s == "NOT_RUN")

    print("## Summary")
    print(f"- PASS: {pass_count}")
    print(f"- STALE: {stale_count}")
    print(f"- FAIL: {fail_count}")
    print(f"- BLOCKED: {blocked_count}")
    print(f"- NOT_RUN: {not_run_count}")
    print(f"- Total gates: {len(gate_statuses)}")
    print()

    # VER-P0-05: ALL GREEN requires every REQUIRED_GATES == PASS AND the
    # evidence git SHA == current HEAD AND a clean working tree.
    reasons: list[str] = []
    for req in sorted(REQUIRED_GATES):
        status = gate_statuses.get(req)
        if status != "PASS":
            note = gate_notes.get(req, "")
            suffix = f" ({note})" if note else ""
            reasons.append(f"{req}={status}{suffix}")
    if evidence_sha is not None and evidence_sha != current_sha:
        reasons.append(f"evidence SHA {evidence_sha} != HEAD {current_sha}")
    if tree_dirty:
        reasons.append("working tree is dirty (git status --porcelain non-empty)")

    if not reasons:
        print("Result: ALL GREEN — every required gate PASS, evidence fresh, "
              "tree clean — exit code 0")
        return 0
    print("Result: NOT ALL GREEN — exit code 1")
    for r in reasons:
        print(f"  - {r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())