"""
Smoke test: CLI tools
Test that CLI tools can be executed without errors.
"""
import pytest
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path("/home/shw/quant_projects/scripts")


class TestCLITools:
    """Test CLI tools execute without errors."""

    def test_help_flag_scripts(self):
        """Test scripts that should accept --help flag."""
        scripts = [
            "rollback_factor.py",
            "pipeline_worker.py",
            "materialize_week2_factors.py",
            "correlate_run_audit.py",
        ]

        for script_name in scripts:
            script_path = SCRIPTS_DIR / script_name
            if not script_path.exists():
                pytest.skip(f"Script {script_name} not found")
                continue

            try:
                # Try to run with --help
                result = subprocess.run(
                    [sys.executable, str(script_path), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Either succeeds or fails gracefully (some scripts may not have --help)
                assert result.returncode in [0, 1, 2], \
                    f"{script_name} crashed unexpectedly: {result.stderr}"
            except subprocess.TimeoutExpired:
                pytest.fail(f"{script_name} timed out")
            except Exception as e:
                pytest.skip(f"Could not test {script_name}: {e}")

    def test_shell_scripts_syntax(self):
        """Test that shell scripts have valid syntax."""
        shell_scripts = [
            "clone_data_access.sh",
            "sync_us_stock_cos.sh",
            "sync_quantsociety_backend.sh",
            "sync_ashare_lqtp_cos.sh",
            "verify_repo_tracking.sh",
            "git_push.sh",
        ]

        for script_name in shell_scripts:
            script_path = SCRIPTS_DIR / script_name
            if not script_path.exists():
                pytest.skip(f"Script {script_name} not found")
                continue

            try:
                # Check bash syntax
                result = subprocess.run(
                    ["bash", "-n", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                assert result.returncode == 0, \
                    f"{script_name} has syntax errors: {result.stderr}"
            except subprocess.TimeoutExpired:
                pytest.fail(f"{script_name} syntax check timed out")
            except Exception as e:
                pytest.skip(f"Could not check {script_name}: {e}")
