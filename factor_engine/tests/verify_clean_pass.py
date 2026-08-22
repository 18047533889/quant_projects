#!/usr/bin/env python3
"""Verify clean code exits with 0."""

import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    clean_file = tmpdir_path / "utils.py"
    clean_file.write_text("# Clean code\nx = 1\n")

    repo_root = Path(__file__).parent.parent
    checker_script = repo_root / "scripts" / "check_dataaccess_io_boundary.py"

    result = subprocess.run(
        [sys.executable, str(checker_script), "--repo-root", str(tmpdir_path)],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(f"Exit code: {result.returncode}")

    if result.returncode == 0 and "All checks PASSED" in result.stdout:
        print("✓ Clean code passes correctly")
        sys.exit(0)
    else:
        print("✗ Unexpected behavior")
        sys.exit(1)
