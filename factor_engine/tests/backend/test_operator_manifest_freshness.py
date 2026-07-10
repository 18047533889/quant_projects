# -*- coding: utf-8
"""operator_manifest.json 与 build 同步 CI。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks" / "operator_manifest.json"


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_operator_manifest_json_fresh(_loaded):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_operator_manifest.py"),
            "--out",
            str(MANIFEST),
            "--check",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_operator_manifest_commit_sha_current(_loaded):
    import json
    import subprocess

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sha = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )
    assert data.get("generated_commit_sha") == sha, (
        f"manifest commit drift: {data.get('generated_commit_sha')!r} != {sha!r}"
    )
