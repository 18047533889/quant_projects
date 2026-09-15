from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_time_semantic_collection_import_cannot_claim_same_clock_lag():
    project_root = Path(__file__).resolve().parents[2]
    script = r'''
import factor_engine.cleaned_operators.time_semantic_ops  # collection-order trigger
from factor_engine.cleaned_operators.registry import OperatorRegistry

assert OperatorRegistry.get("same_clock_lag", "pandas_numpy", mode="any") is None

from factor_engine.cleaned_operators import load_all
load_all()
for backend in ("pandas_numpy", "polars"):
    operator = OperatorRegistry.get("same_clock_lag", backend, mode="any")
    assert operator is not None, backend
    assert list(operator.metadata.param_names) == ["x", "lag", "clock_unit"], (
        backend, operator.metadata.param_names
    )
    assert tuple(operator.metadata.panel_params) == ("x",), (
        backend, operator.metadata.panel_params
    )
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
