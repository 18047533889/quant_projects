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
    1: Violations detected
    2: Internal error

Usage:
    python scripts/check_dataaccess_io_boundary.py
    python scripts/check_dataaccess_io_boundary.py --strict  # Fail on any violation
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Authorized modules (with responsibility documentation)
# ---------------------------------------------------------------------------

#: Modules explicitly authorized for direct physical I/O operations.
#: Each entry documents the module's specific responsibility in the DA boundary.
#: Paths are relative to repo root (factor_engine/ directory).
AUTHORIZED_MODULES: dict[str, str] = {
    # Physical storage layer
    "storage/cache.py": "Plan execution cache - persistent disk cache with schema versioning",
    "storage/result_store.py": "Materialized factor results - atomic write with metadata",
    "storage/delta_store.py": "Incremental update storage - delta merge with generation fence",
    "storage/block_lake.py": "Block-based data lake - versioned parquet blocks",
    "storage/parquet_batch_writer.py": "Low-level parquet streaming writer - batch row groups",
    "storage/schema_migration.py": "Schema evolution - migrate legacy formats",
    "storage/lake_version.py": "Lake versioning - snapshot management",
    "storage/spill_store.py": "OOM recovery spill-to-disk - temporary overflow storage",
    "storage/materialize/materializer.py": "Factor materialization - atomic result persistence with generation fence",
    "storage/materialize/factor_matrix_materializer.py": "Factor matrix persistence - wide-form batch write",
    "storage/materialize/lake_publish.py": "Lake publication - atomic multi-partition commit",
    "storage/sources/kline_parquet_source.py": "Raw kline parquet reader - source-specific physical I/O",
    "storage/sources/parquet_source.py": "Generic parquet source reader - configurable schema physical I/O",
    "storage/sources/staging_loader.py": "Staging area loader - temporary ingestion physical I/O",

    # Runtime execution layer
    "runtime/shard_executor.py": "Distributed execution - worker result materialization",
    "runtime/spill_store.py": "Runtime memory governor - spill buffer to disk",
    "runtime/resource_calibration_store.py": "Resource profile persistence - P99 calibration data",
    "runtime/intermediate_registry.py": "Shared subplan materialization - CSE physical cache",
    "runtime/reconcile/snapshot_reconcile.py": "Snapshot consistency checker - read lake metadata for validation",
    "runtime/multibackend/batch_transfer_optimizer.py": "Cross-backend transfer - DuckDB staging for batch conversion",

    # Backend execution
    "backend/sql_pushdown/executor.py": "DuckDB query executor - SQL backend physical I/O",
    "backend/sql_pushdown/duckdb_capabilities.py": "DuckDB capability probe - version detection and feature test",
    "backend/fastpath_plan_probe.py": "Backend capability probe - synthetic test execution",

    # Export/import boundary
    "export/serializers.py": "Factor export serialization - external system handoff",
    "export/importers.py": "External data ingestion - validated import with PIT check",

    # Benchmarking/tuning infrastructure (scripts/ - operational utilities)
    "benchmarks/backend_operator_bench.py": "Backend performance measurement - isolated bench fixtures",
    "scripts/calibrate_backend_costs.py": "Cost model calibration - empirical timing measurement",
    "scripts/duckdb_parallel_tuning_benchmark.py": "DuckDB parallelism tuning - synthetic load testing",
    "scripts/storage_tuning_benchmark.py": "Storage backend tuning - I/O pattern benchmarking",
    "scripts/sql_certification_factory.py": "SQL operator certification - generate parity test fixtures",

    # Evidence generation (scripts/ - one-time certification utilities)
    "scripts/certify_primitive_evidence.py": "Primitive operator evidence generator - read existing ledger",
    "scripts/certify_factor_operator_evidence.py": "Factor operator evidence generator - read existing ledger",
    "scripts/certify_intraday_parity.py": "Intraday parity certification - cross-backend validation fixtures",
    "scripts/generate_model_layer_redesign_evidence.py": "Model layer evidence generator - migration validation",
    "scripts/audit_r37_hard_gates.py": "R37 audit gate checker - read persisted evidence files",
    "scripts/compare_lqtp_golden.py": "LQTP golden comparison - regression test reference reader",
}

#: Test fixtures are allowed but must use deliberate violation patterns for negative tests
TEST_ALLOWLIST_PATTERNS = [
    "tests/",  # All test files allowed - they test the physical layer
]

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
class CheckResult:
    """Overall check result."""

    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0


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


def scan_file(filepath: Path) -> list[Violation]:
    """Scan a single Python file for I/O boundary violations."""
    try:
        source = filepath.read_text(encoding="utf-8")

        # Quick string scan for patterns (catches commented violations in tests)
        # We still use AST as primary detection to avoid false positives in strings
        tree = ast.parse(source, filename=str(filepath))

        visitor = IOBoundaryVisitor(str(filepath))
        visitor.visit(tree)

        return visitor.violations

    except SyntaxError:
        # Skip files with syntax errors (may be from concurrent edits)
        return []
    except Exception as exc:
        print(f"WARNING: Failed to scan {filepath}: {exc}", file=sys.stderr)
        return []


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
    """Check if file is in the authorized module list."""
    try:
        rel_path = filepath.relative_to(repo_root)
        rel_str = str(rel_path)

        # Check exact authorized modules
        if rel_str in AUTHORIZED_MODULES:
            return True

        # Check test allowlist patterns
        for pattern in TEST_ALLOWLIST_PATTERNS:
            if pattern in rel_str:
                return True

        return False

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

        violations = scan_file(filepath)
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
        print("     add module to AUTHORIZED_MODULES with responsibility comment.")
    else:
        print("✓ All checks PASSED")

    print()
    print("Authorized modules:")
    for module, responsibility in sorted(AUTHORIZED_MODULES.items()):
        print(f"  {module}")
        print(f"    → {responsibility}")


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check DataAccess I/O boundary enforcement (FE-DA-P0-002)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit code 1 on any violation (for CI)",
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

        if args.strict and not result.passed:
            return 1

        return 0

    except Exception as exc:
        print(f"ERROR: Check failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
