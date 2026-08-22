#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FE-DA-P0-002: Static architecture checker for DataAccess I/O boundary enforcement.

This checker ensures that direct physical I/O operations (duckdb.connect,
pd.read_parquet, pq.read_table, df.to_parquet, pl.read_parquet) are only
used in explicitly authorized modules that form the DataAccess abstraction
boundary.

Unauthorized usage bypasses:
- Point-in-time (PIT) validation
- Security/credential governance
- Schema versioning
- Resource accounting
- Audit trails

Exit codes:
    0: All checks passed
    1: Violations detected (ARCH-P0-001: always fails on violations)
    2: Internal error / scanner infrastructure failure (ARCH-P0-002)

Usage:
    python scripts/check_dataaccess_io_boundary.py
    python scripts/check_dataaccess_io_boundary.py --strict  # (deprecated: now default)

ARCH-P0-001: Violations always cause exit code 1 (fail-closed).
ARCH-P0-002: Scanner failures cause exit code 2 (infrastructure failure).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Import single authority policy (ARCH-P0-003)
try:
    from physical_io_authority_policy import (
        PHYSICAL_IO_AUTHORITY_POLICY,
        AUTOMATIC_EXEMPTION_PATTERNS,
        is_path_exempted,
        ExemptionCategory,
    )
except ImportError:
    # Fallback when running from different directory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from physical_io_authority_policy import (
        PHYSICAL_IO_AUTHORITY_POLICY,
        AUTOMATIC_EXEMPTION_PATTERNS,
        is_path_exempted,
        ExemptionCategory,
    )

#: Patterns that trigger violations
VIOLATION_PATTERNS = [
    "duckdb.connect",
    "pd.read_parquet",
    "pq.read_table",
    ".to_parquet",
    "pl.read_parquet",
    "pl.read_csv",
    "pl.scan_parquet",
]

# ---------------------------------------------------------------------------
# AST-based detection
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A detected I/O boundary violation."""

    file: str
    line: int
    column: int
    pattern: str
    context: str
    severity: str = "ERROR"


@dataclass
class ScanError:
    """A scanner infrastructure failure (ARCH-P0-002 requirement)."""

    file: str
    error: str
    exception_type: str


@dataclass
class CheckResult:
    """Overall check result."""

    violations: list[Violation] = field(default_factory=list)
    scan_errors: list[ScanError] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0

    @property
    def passed(self) -> bool:
        # ARCH-P0-002: Scanner failures are infrastructure failures
        return len(self.violations) == 0 and len(self.scan_errors) == 0

    @property
    def has_infrastructure_failure(self) -> bool:
        return len(self.scan_errors) > 0


class IOBoundaryVisitor(ast.NodeVisitor):
    """AST visitor that detects unauthorized I/O operations."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.violations: list[Violation] = []
        self._in_string_context = False

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for I/O operations."""
        call_repr = self._get_call_repr(node)

        for pattern in VIOLATION_PATTERNS:
            if pattern in call_repr:
                # Extract surrounding context
                context = call_repr[:80]
                self.violations.append(
                    Violation(
                        file=self.filepath,
                        line=node.lineno,
                        column=node.col_offset,
                        pattern=pattern,
                        context=context,
                    )
                )

        self.generic_visit(node)

    def _get_call_repr(self, node: ast.Call) -> str:
        """Reconstruct call representation for pattern matching."""
        parts = []

        # Handle attribute access: duckdb.connect, df.to_parquet
        func = node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                parts.append(func.value.id)
            parts.append(func.attr)
            return ".".join(parts)

        # Handle direct name: connect()
        if isinstance(func, ast.Name):
            return func.id

        return ""


def scan_file(filepath: Path) -> tuple[list[Violation], ScanError | None]:
    """Scan a single Python file for I/O boundary violations using AST.

    Returns:
        (violations, scan_error): violations found, or scan_error if scanner failed.

    ARCH-P0-002: Scanner failures are returned as ScanError, not empty violations.
    This distinguishes "scanned successfully, no violations" from "scan failed".
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))

        visitor = IOBoundaryVisitor(str(filepath))
        visitor.visit(tree)

        return visitor.violations, None

    except SyntaxError as exc:
        # ARCH-P0-002: Syntax errors are infrastructure failures, not "no violations"
        # A file with syntax errors cannot be scanned - this is fail-closed
        return [], ScanError(
            file=str(filepath),
            error=f"Syntax error at line {exc.lineno}: {exc.msg}",
            exception_type="SyntaxError",
        )
    except Exception as exc:
        # ARCH-P0-002: Any scanner failure is an infrastructure failure
        return [], ScanError(
            file=str(filepath),
            error=str(exc),
            exception_type=type(exc).__name__,
        )


def should_skip_path(path: Path) -> bool:
    """Check if path should be skipped from scanning."""
    # Get path components relative to factor_engine to avoid matching parent dirs
    parts = path.parts

    # Skip generated/build artifacts within the scanned tree
    skip_dirs = {
        "build",
        "dist",
        ".mypy_cache",
        "__pycache__",
        ".pytest_cache",
        ".egg-info",
        ".git",
        "cache",
        "evidence",
    }

    for part in parts:
        if part in skip_dirs or part.endswith(".egg-info"):
            return True

    return False


def is_authorized(filepath: Path, repo_root: Path) -> bool:
    """Check if file is exempted by the single authority policy (ARCH-P0-003)."""
    try:
        rel_path = filepath.relative_to(repo_root)
        rel_str = str(rel_path)

        # Use single authority policy
        is_exempted, category, reason = is_path_exempted(rel_str)
        return is_exempted

    except ValueError:
        # Not under repo_root
        return False


def find_python_files(repo_root: Path) -> Iterator[Path]:
    """Find all Python files to scan."""
    # Scan from repo_root directly (which should be factor_engine/ directory)
    for path in repo_root.rglob("*.py"):
        if not should_skip_path(path):
            yield path


def run_check(repo_root: Path, strict: bool = False) -> CheckResult:
    """Run the full boundary check."""
    result = CheckResult()

    for filepath in find_python_files(repo_root):
        result.files_scanned += 1

        # Skip authorized modules
        if is_authorized(filepath, repo_root):
            result.files_skipped += 1
            continue

        violations, scan_error = scan_file(filepath)

        if scan_error is not None:
            # ARCH-P0-002: Scanner failures are infrastructure failures
            result.scan_errors.append(scan_error)
        else:
            result.violations.extend(violations)

    return result


def print_report(result: CheckResult, repo_root: Path) -> None:
    """Print human-readable report."""
    print("=" * 80)
    print("FE-DA-P0-002: DataAccess I/O Boundary Check")
    print("=" * 80)
    print()
    print(f"Files scanned: {result.files_scanned}")
    print(f"Files skipped (authorized): {result.files_skipped}")
    print(f"Violations detected: {len(result.violations)}")
    print(f"Scanner failures (ARCH-P0-002): {len(result.scan_errors)}")
    print()

    # ARCH-P0-002: Report scanner failures first (infrastructure issues)
    if result.scan_errors:
        print("SCANNER INFRASTRUCTURE FAILURES:")
        print("-" * 80)
        print()
        for err in result.scan_errors:
            try:
                rel_path = Path(err.file).relative_to(repo_root)
            except ValueError:
                rel_path = Path(err.file)

            print(f"{rel_path}")
            print(f"  Type: {err.exception_type}")
            print(f"  Error: {err.error}")
            print()

        print("-" * 80)
        print("ARCH-P0-002: Scanner failures are CHECK_INFRASTRUCTURE_FAILURE.")
        print("These files could not be scanned and may contain violations.")
        print("Fix the syntax/parse errors and re-run the check.")
        print()

    if result.violations:
        print("VIOLATIONS:")
        print("-" * 80)

        # Group by file
        by_file: dict[str, list[Violation]] = {}
        for v in result.violations:
            by_file.setdefault(v.file, []).append(v)

        for filepath in sorted(by_file.keys()):
            violations = by_file[filepath]
            try:
                rel_path = Path(filepath).relative_to(repo_root)
            except ValueError:
                rel_path = Path(filepath)

            print(f"\n{rel_path}")
            for v in violations:
                print(f"  {v.line}:{v.column}  {v.pattern}")
                print(f"    Context: {v.context}")

        print()
        print("-" * 80)
        print("FIX: Move I/O operations to authorized DataAccess modules, or")
        print("     add module to PHYSICAL_IO_AUTHORITY_POLICY (scripts/physical_io_authority_policy.py)")
        print("     with appropriate ExemptionCategory and responsibility comment.")

    if not result.violations and not result.scan_errors:
        print("✓ All checks PASSED")

    print()
    print("Authorized modules (ARCH-P0-003 single authority):")
    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for module in PHYSICAL_IO_AUTHORITY_POLICY:
        by_category[module.category].append(module)

    for category in ExemptionCategory:
        modules = by_category.get(category, [])
        if not modules:
            continue
        print(f"\n  {category.value.upper()}:")
        for module in modules:
            print(f"    {module.path}")
            print(f"      → {module.responsibility}")


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check DataAccess I/O boundary enforcement (FE-DA-P0-002)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="(Deprecated: now default behavior) Exit code 1 on any violation",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect from script location)",
    )

    args = parser.parse_args()

    # Auto-detect repo root
    if args.repo_root is None:
        script_path = Path(__file__).resolve()
        repo_root = script_path.parent.parent
    else:
        repo_root = args.repo_root.resolve()

    if not repo_root.exists():
        print(f"ERROR: Repository root not found: {repo_root}", file=sys.stderr)
        return 2

    # Run check
    try:
        result = run_check(repo_root, strict=args.strict)
        print_report(result, repo_root)

        # ARCH-P0-002: Scanner failures are infrastructure failures (exit 2)
        if result.has_infrastructure_failure:
            return 2

        # ARCH-P0-001 fix: violations always cause hard failure (exit 1)
        # The --strict flag is now redundant but kept for backward compatibility
        if not result.passed:
            return 1

        return 0

    except Exception as exc:
        print(f"ERROR: Check failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
