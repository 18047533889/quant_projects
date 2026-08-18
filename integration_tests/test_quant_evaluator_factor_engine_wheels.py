"""Isolated wheel test for the QuantEvaluator/FactorEngine adapter boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FACTOR_ENGINE_ROOT = ROOT / "factor_engine"
QUANT_EVALUATOR_ROOT = ROOT / "quant_evaluator"


def _isolated_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=_isolated_env(),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {command!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _build_wheel(project_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)],
        cwd=project_root,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel in {output_dir}, got {wheels}"
    return wheels[0]


def _assert_members(wheel: Path, required: set[str]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    assert required <= members, f"{wheel.name} missing {sorted(required - members)}"


def test_installed_wheels_construct_factor_engine_adapter():
    if not FACTOR_ENGINE_ROOT.is_dir() or not QUANT_EVALUATOR_ROOT.is_dir():
        pytest.skip("repository layout unavailable")

    with tempfile.TemporaryDirectory(prefix="qe_fe_wheel_boundary_") as raw:
        tmp = Path(raw)
        factor_engine_wheel = _build_wheel(FACTOR_ENGINE_ROOT, tmp / "factor_engine_dist")
        quant_evaluator_wheel = _build_wheel(QUANT_EVALUATOR_ROOT, tmp / "quant_evaluator_dist")

        _assert_members(
            factor_engine_wheel,
            {"api/factor.py", "runtime/__init__.py", "runtime/engine.py"},
        )
        _assert_members(
            quant_evaluator_wheel,
            {
                "quant_evaluator/adapters/factor_engine.py",
                "quant_evaluator/contracts/errors.py",
            },
        )

        venv = tmp / "venv"
        _run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            cwd=tmp,
        )
        python = venv / "bin" / "python"
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                str(factor_engine_wheel),
                str(quant_evaluator_wheel),
            ],
            cwd=tmp,
        )

        probe = """
import json
from pathlib import Path

import api.factor
import quant_evaluator
import quant_evaluator.adapters.factor_engine as adapter_module
import runtime
from quant_evaluator.adapters.factor_engine import FactorEngineAdapter

adapter = FactorEngineAdapter()
paths = {
    "api_factor": api.factor.__file__,
    "quant_evaluator": quant_evaluator.__file__,
    "adapter": adapter_module.__file__,
    "runtime": runtime.__file__,
}
for path in paths.values():
    resolved = Path(path).resolve()
    assert resolved.is_file()
    assert resolved.is_relative_to(Path(__import__("sys").prefix).resolve())
    assert "site-packages" in resolved.parts
assert adapter._fe_factor is api.factor
assert adapter._fe_engine is runtime.FactorEngine
assert not hasattr(adapter, "_fe_execute")
print(json.dumps(paths, sort_keys=True))
"""
        result = json.loads(_run([str(python), "-c", probe], cwd=tmp))
        assert all("site-packages" in path for path in result.values())
