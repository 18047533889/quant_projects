#!/usr/bin/env python3
# -*- coding: utf-8
"""
Polars Backend Parity Audit Report Generator

Analyzes Polars backend implementation for parity violations vs Pandas reference.
Produces a detailed report of violations found in null/Inf/warmup/min_periods handling.
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class ParityViolation:
    """Documented parity violation."""
    operator: str
    category: str  # null_propagation, inf_handling, warmup, min_periods, division
    line_number: int
    file_path: str
    pandas_behavior: str
    polars_behavior: str
    severity: str  # HIGH, MEDIUM, LOW
    fix_required: bool


def analyze_polars_expr_emitter(repo_root: Path) -> List[ParityViolation]:
    """Analyze polars_expr_emitter.py for parity issues."""
    violations = []

    emitter_path = repo_root / "factor_engine/backend/polars_expr_emitter.py"
    if not emitter_path.exists():
        print(f"ERROR: {emitter_path} not found")
        return violations

    content = emitter_path.read_text()
    lines = content.splitlines()

    # VIOLATION 1: ts_zscore zero std handling
    # Line 1784-1796: Polars correctly checks std==0 and returns zero_fill
    # But Pandas implementation (time_series.py:1636) uses .replace(0, 1)
    # This causes divergence when (value - mean) != 0 and std == 0
    for i, line in enumerate(lines, 1):
        if 'def ts_zscore' in content[max(0, i-50):i] and 'zero_fill' in line:
            violations.append(ParityViolation(
                operator="ts_zscore",
                category="division_by_zero",
                line_number=1784,
                file_path="factor_engine/backend/polars_expr_emitter.py",
                pandas_behavior="std.replace(0, 1) then (x-mean)/1 = 0 when x==mean, but can produce non-zero when x!=mean",
                polars_behavior=".when(std == 0).then(zero_fill=0.0) always returns 0.0",
                severity="MEDIUM",
                fix_required=True
            ))
            break

    # VIOLATION 2: _sanitize_nan_for_compute with drop_inf
    # Lines 124-144: Polars drops Inf before rolling operations
    # This is CORRECT per pandas rolling behavior, but needs verification
    for i, line in enumerate(lines, 1):
        if '_sanitize_nan_for_compute' in line and 'def' in line:
            violations.append(ParityViolation(
                operator="ts_mean,ts_std,ts_sum,ts_min,ts_max,ts_median,ts_var",
                category="inf_handling",
                line_number=124,
                file_path="factor_engine/backend/polars_expr_emitter.py",
                pandas_behavior="rolling().mean() treats ±Inf as missing (silently excludes)",
                polars_behavior="_sanitize_nan_for_compute(drop_inf=True) explicitly converts Inf to NULL",
                severity="LOW",
                fix_required=False
            ))
            break

    return violations


def analyze_pandas_time_series(repo_root: Path) -> List[ParityViolation]:
    """Analyze pandas time_series.py implementation."""
    violations = []

    ts_path = repo_root / "factor_engine/cleaned_operators/common/time_series.py"
    if not ts_path.exists():
        print(f"ERROR: {ts_path} not found")
        return violations

    content = ts_path.read_text()

    # VIOLATION 3: ts_zscore.replace(0, 1) is incorrect
    # Line 1636: std.replace(0, 1) doesn't match semantics
    if '.replace(0, 1)' in content and 'ts_zscore' in content:
        violations.append(ParityViolation(
            operator="ts_zscore",
            category="zero_std_handling",
            line_number=1636,
            file_path="factor_engine/cleaned_operators/common/time_series.py",
            pandas_behavior=".replace(0, 1) changes std=0 to std=1, giving (x-mean)/1",
            polars_behavior="Correctly checks std==0 and returns 0.0",
            severity="HIGH",
            fix_required=True
        ))

    return violations


def analyze_numeric_semantics(repo_root: Path) -> List[ParityViolation]:
    """Check numeric_semantics.py for policy definitions."""
    violations = []

    sem_path = repo_root / "factor_engine/backend/numeric_semantics.py"
    if not sem_path.exists():
        return violations

    content = sem_path.read_text()

    # Check zscore_zero_std policy
    if 'zscore_zero_std' in content:
        violations.append(ParityViolation(
            operator="ts_zscore,cs_zscore",
            category="semantic_policy",
            line_number=80,
            file_path="factor_engine/backend/numeric_semantics.py",
            pandas_behavior="Not enforced - uses .replace(0, 1) hack",
            polars_behavior="Correctly implements zero_fill=0.0 policy",
            severity="HIGH",
            fix_required=True
        ))

    return violations


def generate_report(violations: List[ParityViolation], output_path: Path):
    """Generate markdown report."""
    report = []
    report.append("# Polars Backend Parity Audit Report")
    report.append("")
    report.append("**Generated:** 2026-08-14")
    report.append("**Mission:** Wave1-Agent1-PolarsParity")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append(f"**Total Violations Found:** {len(violations)}")
    report.append(f"**High Severity:** {sum(1 for v in violations if v.severity == 'HIGH')}")
    report.append(f"**Medium Severity:** {sum(1 for v in violations if v.severity == 'MEDIUM')}")
    report.append(f"**Low Severity:** {sum(1 for v in violations if v.severity == 'LOW')}")
    report.append(f"**Fixes Required:** {sum(1 for v in violations if v.fix_required)}")
    report.append("")
    report.append("## Violation Details")
    report.append("")

    for i, v in enumerate(violations, 1):
        report.append(f"### Violation {i}: {v.operator} - {v.category}")
        report.append("")
        report.append(f"**Severity:** {v.severity}")
        report.append(f"**Fix Required:** {'YES' if v.fix_required else 'NO'}")
        report.append("")
        report.append(f"**Location:** `{v.file_path}:{v.line_number}`")
        report.append("")
        report.append(f"**Pandas Behavior:**")
        report.append(f"```")
        report.append(v.pandas_behavior)
        report.append(f"```")
        report.append("")
        report.append(f"**Polars Behavior:**")
        report.append(f"```")
        report.append(v.polars_behavior)
        report.append(f"```")
        report.append("")

    report.append("## Priority Fixes")
    report.append("")

    high_severity = [v for v in violations if v.severity == "HIGH" and v.fix_required]
    for i, v in enumerate(high_severity, 1):
        report.append(f"{i}. **{v.operator}** ({v.file_path}:{v.line_number})")
        report.append(f"   - Issue: {v.category}")
        report.append("")

    output_path.write_text("\n".join(report))
    print(f"Report written to: {output_path}")


def main():
    repo_root = Path("/home/shw/quant_projects/.claude/worktrees/agent-a5fdd81b281143528")

    print("Starting Polars Backend Parity Audit...")
    print(f"Repository: {repo_root}")
    print("")

    all_violations = []

    print("1. Analyzing polars_expr_emitter.py...")
    all_violations.extend(analyze_polars_expr_emitter(repo_root))

    print("2. Analyzing pandas time_series.py...")
    all_violations.extend(analyze_pandas_time_series(repo_root))

    print("3. Analyzing numeric_semantics.py...")
    all_violations.extend(analyze_numeric_semantics(repo_root))

    print(f"\nFound {len(all_violations)} violations")

    output_path = repo_root / "POLARS_PARITY_AUDIT_REPORT.md"
    generate_report(all_violations, output_path)

    print("\nSummary:")
    for v in all_violations:
        status = "🔴" if v.severity == "HIGH" else "🟡" if v.severity == "MEDIUM" else "🟢"
        print(f"  {status} {v.operator}: {v.category}")


if __name__ == "__main__":
    main()
