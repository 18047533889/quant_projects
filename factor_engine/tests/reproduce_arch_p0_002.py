#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARCH-P0-002: Scanner failure must not be fail-open."""

import subprocess
import sys
import tempfile
from pathlib import Path


def test_scanner_syntax_error_is_infrastructure_failure():
    """ARCH-P0-002: Syntax errors prevent scanning and must cause exit code 2.

    When scanner fails (SyntaxError, parse error), this must be:
    - CHECK_INFRASTRUCTURE_FAILURE (exit code 2)
    - NOT "no violations" (exit code 0)

    This distinguishes "scanned successfully, no violations" from "scan failed".
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a file with syntax error that would have violations if it could be scanned
        broken_file = tmpdir_path / "broken.py"
        broken_file.write_text(
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

        print("=" * 80)
        print("ARCH-P0-002: Scanner Failure Must Not Be Fail-Open")
        print("=" * 80)
        print()
        print("TEST: File with syntax error containing I/O violations")
        print()
        print("CHECKER OUTPUT:")
        print(result.stdout)
        print()
        if result.stderr:
            print("CHECKER STDERR:")
            print(result.stderr)
            print()
        print(f"EXIT CODE: {result.returncode}")
        print()

        # Check for scanner failure reporting
        has_scanner_failure = "Scanner failures" in result.stdout or "SCANNER" in result.stdout

        if result.returncode == 0:
            print("✗ BUG CONFIRMED: ARCH-P0-002")
            print()
            print("Scanner failed (SyntaxError) but checker exited with code 0.")
            print("This is fail-open: 'scan failed' is indistinguishable from 'no violations'.")
            print()
            print("IMPACT:")
            print("- Files with syntax errors are silently skipped")
            print("- Violations in broken files are not detected")
            print("- Concurrent edits or broken merges bypass the check")
            print()
            print("REQUIRED FIX:")
            print("- Scanner failure → exit code 2 (CHECK_INFRASTRUCTURE_FAILURE)")
            print("- Distinguish 'scan succeeded, no violations' from 'scan failed'")
            print()
            return False
        elif result.returncode == 2 and has_scanner_failure:
            print("✓ BUG FIXED: Scanner failures cause infrastructure failure (exit 2)")
            print()
            print("Scanner correctly reports:")
            print("- Exit code 2 (infrastructure failure)")
            print("- Clear indication of which files failed to scan")
            print("- Fail-closed behavior: broken files block the check")
            print()
            return True
        else:
            print(f"✗ UNEXPECTED: Exit code {result.returncode}, scanner_failure={has_scanner_failure}")
            return False


def test_io_error_is_infrastructure_failure():
    """ARCH-P0-002: File read errors must cause exit code 2."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a file that will cause read error (we'll make it unreadable)
        unreadable_file = tmpdir_path / "unreadable.py"
        unreadable_file.write_text("import duckdb\n")
        unreadable_file.chmod(0o000)  # Make unreadable

        repo_root = Path(__file__).parent.parent
        checker_script = repo_root / "scripts" / "check_dataaccess_io_boundary.py"

        result = subprocess.run(
            [sys.executable, str(checker_script), "--repo-root", str(tmpdir_path)],
            capture_output=True,
            text=True,
        )

        # Restore permissions for cleanup
        unreadable_file.chmod(0o644)

        print("=" * 80)
        print("ARCH-P0-002: File Read Error Test")
        print("=" * 80)
        print(f"Exit code: {result.returncode}")

        if result.returncode == 2:
            print("✓ File read errors cause infrastructure failure (exit 2)")
            return True
        else:
            print(f"✗ Expected exit 2, got {result.returncode}")
            return False


if __name__ == "__main__":
    test1 = test_scanner_syntax_error_is_infrastructure_failure()
    print()
    test2 = test_io_error_is_infrastructure_failure()

    if test1 and test2:
        print()
        print("=" * 80)
        print("✓ ARCH-P0-002: All tests passed")
        print("=" * 80)
        sys.exit(0)
    else:
        print()
        print("=" * 80)
        print("✗ ARCH-P0-002: Some tests failed")
        print("=" * 80)
        sys.exit(1)
