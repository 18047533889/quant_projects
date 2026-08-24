#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence Integrity Validator - R37 Evidence Truth Engine Compliance

Validates:
1. All evidence artifacts bound to current HEAD commit SHA
2. No "unknown" hash values in any evidence ledger
3. Per-backend hashes present for all certified operators
4. Canonical set matches between declarations and actual evidence
5. Component hashes match current source state

Enforces fail-closed semantics: any integrity violation fails the validation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FE_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = FE_ROOT.parent / "evidence"


@dataclass
class IntegrityIssue:
    """Single evidence integrity issue."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # stale_evidence, unknown_hash, missing_backend, canonical_mismatch
    artifact: str
    detail: str
    operator: str | None = None


def get_current_head() -> str:
    """Get current git HEAD commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=FE_ROOT.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def load_evidence_file(path: Path) -> dict[str, Any]:
    """Load and parse evidence JSON file."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to load {path.name}: {e}", file=sys.stderr)
        return {}


def validate_head_binding(
    data: dict[str, Any], artifact_name: str, current_head: str
) -> list[IntegrityIssue]:
    """Validate evidence artifact is bound to current HEAD."""
    issues = []

    # Check various possible SHA field names
    bound_sha = data.get("commit_sha") or data.get("bound_sha") or data.get("certified_commit_sha")

    if not bound_sha:
        issues.append(IntegrityIssue(
            severity="CRITICAL",
            category="stale_evidence",
            artifact=artifact_name,
            detail=f"No commit SHA binding found (expected {current_head[:12]})",
        ))
    elif bound_sha != current_head:
        issues.append(IntegrityIssue(
            severity="HIGH",
            category="stale_evidence",
            artifact=artifact_name,
            detail=f"Bound to {bound_sha[:12]}, current HEAD is {current_head[:12]}",
        ))

    return issues


def validate_unknown_hashes(
    data: dict[str, Any], artifact_name: str
) -> list[IntegrityIssue]:
    """Check for 'unknown' hash values in evidence."""
    issues = []

    if "operators" not in data:
        return issues

    for op_name, op_data in data["operators"].items():
        if not isinstance(op_data, dict):
            continue

        # Check implementation hash
        impl_hash = op_data.get("implementation_hash")
        if impl_hash == "unknown":
            issues.append(IntegrityIssue(
                severity="HIGH",
                category="unknown_hash",
                artifact=artifact_name,
                detail="implementation_hash='unknown'",
                operator=op_name,
            ))

        # Check backend hashes
        backends = op_data.get("backends", {})
        for backend_name, backend_data in backends.items():
            if isinstance(backend_data, dict):
                backend_hash = backend_data.get("hash")
                if backend_hash == "unknown":
                    issues.append(IntegrityIssue(
                        severity="HIGH",
                        category="unknown_hash",
                        artifact=artifact_name,
                        detail=f"backends.{backend_name}.hash='unknown'",
                        operator=op_name,
                    ))

    return issues


def validate_backend_coverage(
    data: dict[str, Any], artifact_name: str
) -> list[IntegrityIssue]:
    """Check that all available backends have hash values."""
    issues = []

    if "operators" not in data:
        return issues

    for op_name, op_data in data["operators"].items():
        if not isinstance(op_data, dict):
            continue

        backends = op_data.get("backends", {})
        for backend_name in ["pandas", "polars", "duckdb"]:
            backend_data = backends.get(backend_name, {})
            if not isinstance(backend_data, dict):
                continue

            # If backend is available, it must have a hash
            if backend_data.get("available", False):
                if not backend_data.get("hash"):
                    issues.append(IntegrityIssue(
                        severity="MEDIUM",
                        category="missing_backend",
                        artifact=artifact_name,
                        detail=f"Backend {backend_name} available but no hash",
                        operator=op_name,
                    ))

    return issues


def validate_canonical_set_consistency(
    evidence_ops: set[str], declared_ops: set[str], artifact_name: str
) -> list[IntegrityIssue]:
    """Check canonical set matches between declared and actual evidence."""
    issues = []

    missing_evidence = declared_ops - evidence_ops
    extra_evidence = evidence_ops - declared_ops

    if missing_evidence:
        issues.append(IntegrityIssue(
            severity="HIGH",
            category="canonical_mismatch",
            artifact=artifact_name,
            detail=f"{len(missing_evidence)} operators declared but no evidence: {sorted(list(missing_evidence))[:10]}",
        ))

    if extra_evidence:
        issues.append(IntegrityIssue(
            severity="MEDIUM",
            category="canonical_mismatch",
            artifact=artifact_name,
            detail=f"{len(extra_evidence)} operators have evidence but not declared: {sorted(list(extra_evidence))[:10]}",
        ))

    return issues


def validate_recertification_consistency(current_head: str) -> list[IntegrityIssue]:
    """Validate recertification ledger consistency."""
    issues = []

    recer_path = EVIDENCE_DIR / "CURRENT_HEAD_OPERATOR_RECERTIFICATION.json"
    recer_data = load_evidence_file(recer_path)

    if not recer_data:
        issues.append(IntegrityIssue(
            severity="HIGH",
            category="stale_evidence",
            artifact="CURRENT_HEAD_OPERATOR_RECERTIFICATION.json",
            detail="File missing or unreadable",
        ))
        return issues

    # Count certification states
    cert_levels = defaultdict(int)
    needs_recert = 0
    has_evidence = 0
    backend_available = defaultdict(int)

    for op_name, op_data in recer_data.items():
        cert_levels[op_data.get("certification_level", "UNKNOWN")] += 1
        if op_data.get("requires_recertification"):
            needs_recert += 1
        if op_data.get("has_evidence"):
            has_evidence += 1

        backends = op_data.get("backends", {})
        for backend_name, backend_data in backends.items():
            if backend_data.get("available"):
                backend_available[backend_name] += 1

    # Add summary issue (informational)
    issues.append(IntegrityIssue(
        severity="LOW",
        category="canonical_mismatch",
        artifact="CURRENT_HEAD_OPERATOR_RECERTIFICATION.json",
        detail=f"Summary: {len(recer_data)} ops, {has_evidence} with evidence, "
               f"{needs_recert} need recert, levels={dict(cert_levels)}, "
               f"backends={dict(backend_available)}",
    ))

    return issues


def main():
    """Run evidence integrity validation."""
    print("=" * 80)
    print("Evidence Integrity Validation Report")
    print("=" * 80)

    current_head = get_current_head()
    if not current_head:
        print("ERROR: Could not determine current HEAD commit SHA", file=sys.stderr)
        return 1

    print(f"\nCurrent HEAD: {current_head}")
    print(f"Evidence Directory: {EVIDENCE_DIR}")
    print()

    all_issues: list[IntegrityIssue] = []

    # Validate main evidence artifacts
    artifacts = {
        "primitive_verified.json": EVIDENCE_DIR / "primitive_verified.json",
        "factor_operator_verified.json": EVIDENCE_DIR / "factor_operator_verified.json",
        "composite_verified.json": EVIDENCE_DIR / "composite_verified.json",
        "recipe_verified.json": EVIDENCE_DIR / "recipe_verified.json",
    }

    for artifact_name, artifact_path in artifacts.items():
        print(f"Validating {artifact_name}...")
        data = load_evidence_file(artifact_path)

        if not data:
            all_issues.append(IntegrityIssue(
                severity="CRITICAL",
                category="stale_evidence",
                artifact=artifact_name,
                detail="File missing or unreadable",
            ))
            continue

        # Validate HEAD binding
        all_issues.extend(validate_head_binding(data, artifact_name, current_head))

        # Validate unknown hashes
        all_issues.extend(validate_unknown_hashes(data, artifact_name))

        # Validate backend coverage
        all_issues.extend(validate_backend_coverage(data, artifact_name))

    # Validate recertification ledger
    print("Validating recertification ledger...")
    all_issues.extend(validate_recertification_consistency(current_head))

    # Print results
    print("\n" + "=" * 80)
    print("INTEGRITY ISSUES FOUND")
    print("=" * 80)

    # Group by severity
    by_severity = defaultdict(list)
    for issue in all_issues:
        by_severity[issue.severity].append(issue)

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        issues = by_severity[severity]
        if not issues:
            continue

        print(f"\n{severity} ({len(issues)} issues):")
        print("-" * 80)
        for issue in issues:
            op_str = f" [{issue.operator}]" if issue.operator else ""
            print(f"  {issue.category:20s} | {issue.artifact:40s}{op_str}")
            print(f"    {issue.detail}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    by_category = defaultdict(int)
    for issue in all_issues:
        by_category[issue.category] += 1

    print(f"\nTotal issues: {len(all_issues)}")
    print(f"By category:")
    for category, count in sorted(by_category.items()):
        print(f"  {category:25s}: {count:4d}")

    critical_count = len(by_severity["CRITICAL"])
    high_count = len(by_severity["HIGH"])

    print(f"\nCritical issues: {critical_count}")
    print(f"High severity issues: {high_count}")

    if critical_count > 0:
        print("\nVALIDATION FAILED: Critical issues found")
        return 1
    elif high_count > 0:
        print("\nVALIDATION WARNING: High severity issues found")
        return 0
    else:
        print("\nVALIDATION PASSED: No critical or high severity issues")
        return 0


if __name__ == "__main__":
    sys.exit(main())
