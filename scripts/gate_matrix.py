#!/usr/bin/env python3
"""Gate Matrix — honest gate status from verified evidence.

Reads verified_gate_evidence.json and gates.json, computes the current git SHA,
and outputs a Gate Matrix (PASS/FAIL/STALE/NOT_RUN/BLOCKED) + per-package
classification (PRODUCTION_CANDIDATE/SAFE_BY_GATING/RESEARCH_ONLY/OFFLINE_ONLY/BROKEN).

Exit code: 0 if all gates PASS, 1 if any gate is FAIL/BLOCKED/STALE.
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

# Gates specified by the user that may not be in evidence/gates.json
# (SOURCE_AUTHORITY, SUPPLY_CHAIN are aspirational / future gates)
ASPIRATIONAL_GATES = {"SOURCE_AUTHORITY", "SUPPLY_CHAIN"}


def compute_gate_status(
    gate_name: str,
    evidence_gates: dict[str, list[dict[str, Any]]],
    evidence_sha: str | None,
    current_sha: str,
) -> tuple[str, int, str]:
    """Compute status for a single gate.

    Returns (status, total_test_count, timestamp).
    Status: NOT_RUN -> STALE -> BLOCKED -> FAIL -> PASS (priority order).
    """
    entries = evidence_gates.get(gate_name)

    # NOT_RUN: no evidence entries
    if not entries:
        return ("NOT_RUN", 0, "—")

    # STALE: evidence git SHA doesn't match current HEAD
    if evidence_sha is not None and evidence_sha != current_sha:
        total_tests = sum(e.get("tests", 0) or 0 for e in entries)
        timestamps = [e.get("executed_at", "") or "" for e in entries if e.get("executed_at")]
        ts = timestamps[0] if timestamps else "—"
        return ("STALE", total_tests, ts)

    total_tests = 0
    has_blocked = False
    has_fail = False
    timestamps = []

    for entry in entries:
        tests = entry.get("tests", 0) or 0
        total_tests += tests
        ts = entry.get("executed_at") or ""
        if ts:
            timestamps.append(ts)

        if tests == 0:
            has_blocked = True
        elif entry.get("failed", 0) or entry.get("errors", 0):
            has_fail = True

    timestamp = timestamps[0] if timestamps else "—"

    if has_blocked:
        return ("BLOCKED", total_tests, timestamp)
    if has_fail:
        return ("FAIL", total_tests, timestamp)

    # PASS: all entries have exit_code==0, failures==0, errors==0, tests>0
    return ("PASS", total_tests, timestamp)


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

    # Build evidence lookup: gate_name -> list of entries
    evidence_gates: dict[str, list[dict[str, Any]]] = {}
    for gate_entry in evidence.get("gates", []):
        name = gate_entry["name"]
        evidence_gates[name] = gate_entry.get("evidence", [])

    # Determine gate order: from gates.json, then aspirational, then any extras
    configured_gate_names = [g["name"] for g in gates_config]
    all_gate_names = list(configured_gate_names)
    for ag in sorted(ASPIRATIONAL_GATES):
        if ag not in all_gate_names:
            all_gate_names.append(ag)
    # Add any extra gates from evidence not in the list
    for g in evidence_gates:
        if g not in all_gate_names:
            all_gate_names.append(g)

    # Compute status for each gate
    gate_statuses: dict[str, str] = {}
    gate_tests: dict[str, int] = {}
    gate_timestamps: dict[str, str] = {}

    for gate_name in all_gate_names:
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

    # Determine exit code: 0 if all PASS, 1 otherwise
    any_non_pass = any(
        s in ("FAIL", "BLOCKED", "STALE")
        for s in gate_statuses.values()
    )
    if any_non_pass:
        print("Result: SOME GATES NOT PASSING — exit code 1")
        return 1
    else:
        print("Result: ALL GATES PASSING — exit code 0")
        return 0


if __name__ == "__main__":
    sys.exit(main())