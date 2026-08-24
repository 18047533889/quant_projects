#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for FE-DA-P0-002 DataAccess I/O boundary checker.

Tests that the static checker correctly:
- Detects unauthorized I/O operations
- Allows authorized modules
- Skips test files
- Uses AST analysis to avoid string/comment false positives
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Import the checker module
import sys
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

from check_dataaccess_io_boundary import (
    run_check,
    scan_file,
    is_authorized,
)
from physical_io_authority_policy import (
    PHYSICAL_IO_AUTHORITY_POLICY,
)


class TestIOBoundaryChecker:
    """Test the DataAccess I/O boundary checker."""

    def test_detects_duckdb_connect(self):
        """Checker detects unauthorized duckdb.connect."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import duckdb\nconn = duckdb.connect(':memory:')\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)
            assert scan_error is None, "Should not have scan error"
            assert len(violations) >= 1
            assert any("duckdb.connect" in v.pattern for v in violations)
        finally:
            temp_path.unlink()

    def test_detects_pd_read_parquet(self):
        """Checker detects unauthorized pd.read_parquet."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import pandas as pd\ndf = pd.read_parquet('data.parquet')\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)
            assert scan_error is None
            assert len(violations) >= 1
            assert any("pd.read_parquet" in v.pattern for v in violations)
        finally:
            temp_path.unlink()

    def test_detects_to_parquet(self):
        """Checker detects unauthorized df.to_parquet."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("df.to_parquet('output.parquet')\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)
            assert scan_error is None
            assert len(violations) >= 1
            assert any(".to_parquet" in v.pattern for v in violations)
        finally:
            temp_path.unlink()

    def test_ignores_string_literals(self):
        """Checker ignores patterns in string literals (AST-based)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(
                'doc = "Use DataAccess instead of duckdb.connect or pd.read_parquet"\n'
            )
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)
            assert scan_error is None
            # Should not detect patterns in plain string literals
            assert len(violations) == 0
        finally:
            temp_path.unlink()

    def test_ignores_comments(self):
        """Checker ignores patterns in comments."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("# Don't use duckdb.connect directly\npass\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)
            assert scan_error is None
            assert len(violations) == 0
        finally:
            temp_path.unlink()

    def test_authorized_modules_recognized(self):
        """Authorized modules are correctly recognized."""
        # Test a known authorized module
        repo_root = Path(__file__).parent.parent
        authorized_file = repo_root / "storage" / "factor_engine.cache.py"

        assert is_authorized(authorized_file, repo_root)

    def test_test_files_are_allowlisted(self):
        """Test files in specific test directories are in the allowlist."""
        repo_root = Path(__file__).parent.parent
        # Use a test directory that's in TEST_ALLOWLIST_PATTERNS
        test_file = repo_root / "tests" / "storage" / "test_anything.py"

        # Create temporary test file
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# test file")

        try:
            assert is_authorized(test_file, repo_root)
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_unauthorized_module_detected(self):
        """Unauthorized modules are not in allowlist."""
        repo_root = Path(__file__).parent.parent
        # A module that should NOT have direct I/O access
        unauthorized_file = repo_root / "mining" / "direct_use_policy.py"

        assert not is_authorized(unauthorized_file, repo_root)

    def test_detects_pq_read_table(self):
        """Checker detects unauthorized pq.read_table (pyarrow)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import pyarrow.parquet as pq\ntable = pq.read_table('data.parquet')\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)
            assert scan_error is None
            assert len(violations) >= 1
            assert any("pq.read_table" in v.pattern for v in violations)
        finally:
            temp_path.unlink()

    def test_detects_pl_read_parquet(self):
        """Checker detects unauthorized pl.read_parquet (polars)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import polars as pl\ndf = pl.read_parquet('data.parquet')\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)
            assert scan_error is None
            assert len(violations) >= 1
            assert any("pl.read_parquet" in v.pattern for v in violations)
        finally:
            temp_path.unlink()

    def test_full_check_on_repo(self):
        """Full check runs without error on the repository."""
        repo_root = Path(__file__).parent.parent
        result = run_check(repo_root, strict=False)

        # Should scan files
        assert result.files_scanned > 100

        # Should skip authorized modules and tests
        assert result.files_skipped > 0

        # Report existing violations (known migration work)
        # The checker is designed to fail on violations; this test documents current state
        print(f"\nCurrent violations (to be migrated): {len(result.violations)}")
        if result.violations:
            print("Sample violations:")
            for v in result.violations[:5]:
                print(f"  {v.file}:{v.line} {v.pattern}")

    def test_catches_runtime_violation_pattern(self):
        """Checker catches runtime-style direct I/O violations outside tests/."""
        repo_root = Path(__file__).parent.parent
        # Use the runtime violation fixture that mirrors real runtime/shard_executor.py pattern
        runtime_fixture = repo_root / "tests" / "fixtures" / "runtime_direct_io_violation.py"

        # Copy to a temporary location outside tests/ (simulating production code)
        temp_dir = repo_root / "runtime"
        temp_violation_file = temp_dir / "temp_test_violation.py"

        try:
            # Copy the runtime-style violation to runtime/ (unauthorized)
            temp_violation_file.write_text(runtime_fixture.read_text())

            # Run check - should detect violations
            violations, scan_error = scan_file(temp_violation_file)
            assert scan_error is None

            # Should detect 2 violations (pd.read_parquet, .to_parquet)
            assert len(violations) >= 2, (
                f"Expected at least 2 violations, got {len(violations)}: {violations}"
            )

            # Verify specific patterns detected
            patterns = [v.pattern for v in violations]
            assert any("pd.read_parquet" in p for p in patterns), "Should detect pd.read_parquet"
            assert any(".to_parquet" in p for p in patterns), "Should detect .to_parquet"

            # Verify the fixture is NOT authorized (runtime/ is not in allowlist)
            assert not is_authorized(temp_violation_file, repo_root), (
                "Runtime modules should NOT be authorized for direct I/O"
            )

        finally:
            # Clean up
            if temp_violation_file.exists():
                temp_violation_file.unlink()

    def test_scanner_failure_returns_scan_error(self):
        """ARCH-P0-002: Scanner failures return ScanError, not empty violations."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            # Write file with syntax error
            f.write("import duckdb\nconn = duckdb.connect(':memory:'  # Missing closing paren\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            violations, scan_error = scan_file(temp_path)

            # Should return ScanError, not violations
            assert scan_error is not None, "Expected ScanError for syntax error"
            assert scan_error.exception_type == "SyntaxError"
            assert len(violations) == 0, "Violations should be empty when scan fails"
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
