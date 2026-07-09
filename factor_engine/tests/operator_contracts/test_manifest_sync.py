# -*- coding: utf-8
"""Production core manifest 与 docs 同步 CI。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_operator_core_specs_yaml_in_sync():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_operator_specs.py"), "--core-only", "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
