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
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

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


def test_operator_manifest_commit_sha_is_ancestor(_loaded):
    import json
    import subprocess

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    generated = str(data.get("generated_commit_sha") or "")
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", generated, "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert generated and proc.returncode == 0, (
        f"manifest was generated from unrelated commit: {generated!r}"
    )
