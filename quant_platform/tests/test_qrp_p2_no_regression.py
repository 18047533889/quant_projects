"""QRP-P2 — verify no pre-existing quant_platform tests are broken by the new
files. This guards against accidental suite regressions: any failure here other
than a config-limitation Skipped is a regression introduced by QRP-P2.

Run this file in isolation as a canary. If pytest reports the file collection as
skipped due to conftest import findings, the run shows only the suite-level
dependency gate, NOT a QRP-P2 regression.
"""

import subprocess
import sys
from pathlib import Path

import pytest

VICTIM = Path(__file__).resolve().parents[1]  # quant_platform/
FULL = VICTIM / "tests" if (VICTIM / "tests").exists() else VICTIM


def _paths(*names: str) -> list[str]:
    # The pre-existing tests live under quant_platform/tests; if a name is not
    # there (older layout copies), fall back to the repo-root tests dir.
    found: list[str] = []
    for name in names:
        candidate = VICTIM / "tests" / name
        if candidate.exists():
            found.append(str(candidate))
            continue
        alt = VICTIM.parent / "tests" / name
        if alt.exists():
            found.append(str(alt))
    return found


def test_existing_tracked_packages_tests_green():
    """The pre-existing test files (those present before QRP-P2) must pass."""
    pre = [
        "test_contracts_c1.py",
        "test_smoke.py",
    ]
    paths = _paths(*pre)
    if not paths:
        pytest.skip(
            "pre-existing canary test files not found (test layout changed); "
            "skipping the suite-level dependency gate"
        )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-p", "no:cacheprovider", "-q"],
        capture_output=True,
        text=True,
        cwd=str(VICTIM.parent),  # repo root (rootdir here = quant_platform config)
    )
    assert result.returncode == 0, (
        f"pre-existing tests regression:\n{result.stdout}\n{result.stderr}"
    )
    assert " passed" in result.stdout or "passed" in result.stdout