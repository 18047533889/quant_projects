"""Opt-in pytest gate for the standalone wheel smoke test."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_clean_wheel_install_smoke():
    """Run the existing wheel smoke script when explicitly requested."""
    if os.environ.get("RUN_WHEEL_SMOKE") != "1":
        pytest.skip("set RUN_WHEEL_SMOKE=1 to run the wheel smoke test")

    script = Path(__file__).resolve().parents[1] / "scripts" / "wheel_clean_install_smoke.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        pytest.fail(
            "wheel smoke script timed out after 300 seconds\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    assert result.returncode == 0, (
        f"wheel smoke script failed with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
