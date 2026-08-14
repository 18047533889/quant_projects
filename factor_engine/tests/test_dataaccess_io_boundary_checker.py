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
    AUTHORIZED_MODULES,
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
            violations = scan_file(temp_path)
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
            violations = scan_file(temp_path)
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
            violations = scan_file(temp_path)
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
            violations = scan_file(temp_path)
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
            violations = scan_file(temp_path)
            assert len(violations) == 0
        finally:
            temp_path.unlink()

    def test_authorized_modules_recognized(self):
        """Authorized modules are correctly recognized."""
        # Test a known authorized module
        repo_root = Path(__file__).parent.parent
        authorized_file = repo_root / "storage" / "cache.py"

        assert is_authorized(authorized_file, repo_root)

    def test_test_files_are_allowlisted(self):
        """Test files are in the allowlist."""
        repo_root = Path(__file__).parent.parent
        test_file = repo_root / "tests" / "test_anything.py"

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
            violations = scan_file(temp_path)
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
            violations = scan_file(temp_path)
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

        # Production code should have no violations (all authorized)
        assert len(result.violations) == 0, (
            f"Found {len(result.violations)} violations in production code. "
            f"First 5: {result.violations[:5]}"
        )

    def test_catches_deliberate_violation_outside_tests(self):
        """Checker catches violations when deliberate violation moved outside tests/."""
        repo_root = Path(__file__).parent.parent
        deliberate_fixture = repo_root / "tests" / "fixtures" / "deliberate_io_violation.py"

        # Copy to a temporary location outside tests/
        temp_dir = repo_root / "mining"
        temp_violation_file = temp_dir / "temp_violation_for_test.py"

        try:
            # Copy the deliberate violation to mining/ (unauthorized)
            temp_violation_file.write_text(deliberate_fixture.read_text())

            # Run check - should detect violations
            violations = scan_file(temp_violation_file)

            # Should detect at least 3 violations (duckdb.connect, pd.read_parquet, .to_parquet)
            assert len(violations) >= 3, (
                f"Expected at least 3 violations, got {len(violations)}: {violations}"
            )

            # Verify specific patterns detected
            patterns = [v.pattern for v in violations]
            assert any("duckdb.connect" in p for p in patterns), "Should detect duckdb.connect"
            assert any("pd.read_parquet" in p for p in patterns), "Should detect pd.read_parquet"
            assert any(".to_parquet" in p for p in patterns), "Should detect .to_parquet"

        finally:
            # Clean up
            if temp_violation_file.exists():
                temp_violation_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
