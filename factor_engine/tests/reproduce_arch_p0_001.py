#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARCH-P0-001: Reproduce I/O boundary checker fail-open bug."""

import subprocess
import sys
import tempfile
from pathlib import Path


def test_bug_reproduction():
    """Reproduce ARCH-P0-001: violations detected but exit code 0 without --strict."""

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

        print("=" * 80)
        print("ARCH-P0-001 BUG REPRODUCTION")
        print("=" * 80)
        print()
        print("CHECKER OUTPUT:")
        print(result.stdout)
        print()
        print("CHECKER STDERR:")
        if result.stderr:
            print(result.stderr)
        print()
        print(f"EXIT CODE: {result.returncode}")
        print()

        # Check if violation was detected
        violations_detected = "Violations detected:" in result.stdout and int(result.stdout.split("Violations detected:")[1].split()[0]) > 0

        if violations_detected and result.returncode == 0:
            print("✗ BUG CONFIRMED: ARCH-P0-001")
            print()
            print("The checker detected violations but exited with code 0.")
            print("This is a fail-open architecture bug.")
            print()
            print("IMPACT:")
            print("- Production gates calling this checker without --strict will")
            print("  log violations but continue execution")
            print("- Bypasses DataAccess governance (PIT, security, schema versioning)")
            print("- Silent architecture degradation")
            print()
            print("REQUIRED FIX:")
            print("- violations > 0 → exit code 1 (always, not just with --strict)")
            print("- --strict flag becomes redundant (kept for backward compat)")
            print()
            return False
        elif violations_detected and result.returncode == 1:
            print("✓ BUG FIXED: Checker properly fails on violations")
            return True
        else:
            print("✗ UNEXPECTED: No violations detected in test case")
            return False


if __name__ == "__main__":
    success = test_bug_reproduction()
    sys.exit(0 if success else 1)
