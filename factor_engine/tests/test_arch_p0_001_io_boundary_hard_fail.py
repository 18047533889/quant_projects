#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARCH-P0-001: I/O boundary checker must hard-fail on violations.

Problem: check_dataaccess_io_boundary.py detects violations but exits with
code 0 unless --strict is passed. This is a fail-open architecture bug.

Requirements:
- violations > 0 → exit code 1 (always, not just with --strict)
- Production architecture gates must fail when violations detected
- No "detect but continue" mode in production
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestIOBoundaryHardFail:
    """Test that I/O boundary violations cause hard failures."""

    def test_violations_cause_nonzero_exit_without_strict_flag(self):
        """ARCH-P0-001: Violations must cause exit code 1 even without --strict.

        CURRENT BUG: Without --strict, checker prints violations but exits 0.
        REQUIRED: violations > 0 → exit code 1 (always).
        """
        # Create a temporary directory with a violation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a module with a violation (not in authorized list)
            violation_file = tmpdir_path / "runtime" / "bad_module.py"
            violation_file.parent.mkdir(parents=True, exist_ok=True)
            violation_file.write_text(
                "import duckdb\n"
                "conn = duckdb.connect(':memory:')\n"
            )

            # Run checker WITHOUT --strict flag
            repo_root = Path(__file__).parent.parent
            checker_script = repo_root / "scripts" / "check_dataaccess_io_boundary.py"

            result = subprocess.run(
                [sys.executable, str(checker_script), "--repo-root", str(tmpdir_path)],
                capture_output=True,
                text=True,
            )

            # Parse output to verify violation was detected
            assert "Violations detected:" in result.stdout
            assert "duckdb.connect" in result.stdout

            # BUG: Currently exits 0 without --strict
            # REQUIRED: Must exit 1 when violations detected
            if result.returncode == 0:
                pytest.fail(
                    "ARCH-P0-001 BUG CONFIRMED: Checker detected violations but "
                    "exited with code 0 (fail-open). This is an architecture bug. "
                    "Production gates that call this checker without --strict will "
                    "log violations but continue execution, bypassing DataAccess "
                    "governance."
                )

            assert result.returncode == 1, (
                f"Expected exit code 1 when violations detected, got {result.returncode}"
            )

    def test_no_violations_exits_zero(self):
        """Verify that clean code exits with 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a clean module
            clean_file = tmpdir_path / "utils.py"
            clean_file.write_text("# Clean code\nx = 1\n")

            repo_root = Path(__file__).parent.parent
            checker_script = repo_root / "scripts" / "check_dataaccess_io_boundary.py"

            result = subprocess.run(
                [sys.executable, str(checker_script), "--repo-root", str(tmpdir_path)],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "All checks PASSED" in result.stdout

    def test_strict_flag_redundant_after_fix(self):
        """After fix, --strict should have same behavior (violations → exit 1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            violation_file = tmpdir_path / "bad.py"
            violation_file.write_text("import pandas as pd\ndf = pd.read_parquet('x')\n")

            repo_root = Path(__file__).parent.parent
            checker_script = repo_root / "scripts" / "check_dataaccess_io_boundary.py"

            # With --strict
            result_strict = subprocess.run(
                [sys.executable, str(checker_script), "--repo-root", str(tmpdir_path), "--strict"],
                capture_output=True,
                text=True,
            )

            # Without --strict
            result_no_strict = subprocess.run(
                [sys.executable, str(checker_script), "--repo-root", str(tmpdir_path)],
                capture_output=True,
                text=True,
            )

            # Both should exit 1 when violations present
            assert result_strict.returncode == 1
            assert result_no_strict.returncode == 1

    def test_scanner_failure_is_infrastructure_failure(self):
        """ARCH-P0-002: Scanner failure must not be fail-open.

        When the checker scanner itself fails (SyntaxError, parse error),
        this must be treated as CHECK_INFRASTRUCTURE_FAILURE, not "no violations".
        """
        # This will be addressed in ARCH-P0-002
        # For now, verify current behavior: syntax errors are silently skipped
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a file with syntax error that would have violations
            syntax_error_file = tmpdir_path / "broken.py"
            syntax_error_file.write_text(
                "import duckdb\n"
                "conn = duckdb.connect(':memory:'  # Missing closing paren\n"
            )

            repo_root = Path(__file__).parent.parent
            checker_script = repo_root / "scripts" / "check_dataaccess_io_boundary.py"

            result = subprocess.run(
                [sys.executable, str(checker_script), "--repo-root", str(tmpdir_path)],
                capture_output=True,
                text=True,
            )

            # ARCH-P0-002: scanner failure should be infrastructure failure (exit 2)
            assert result.returncode == 2, (
                f"Expected exit code 2 for scanner failure, got {result.returncode}"
            )
            assert "SCANNER INFRASTRUCTURE FAILURES:" in result.stdout
            assert "SyntaxError" in result.stdout or "CHECK_INFRASTRUCTURE_FAILURE" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
