from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    REPO_ROOT / "scripts" / "audit_fundamental_1288.py",
    REPO_ROOT / "factor_engine" / "scripts" / "audit_fundamental_1288.py",
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_audit_fundamental_script_imports_without_running_audit(script):
    spec = importlib.util.spec_from_file_location(f"audit_smoke_{script.parent.name}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.audit_row)
    assert callable(module.main)


def test_audit_fundamental_help_smoke_does_not_create_output(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS[0]), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--source" in result.stdout
    assert "--output" in result.stdout
    assert list(tmp_path.iterdir()) == []
