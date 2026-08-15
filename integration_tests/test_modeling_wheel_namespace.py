"""Serialized wheel checks for the FactorEngine/adapter namespace boundary."""

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
FE_ROOT = ROOT / "factor_engine"
ADAPTER_ROOT = ROOT / "modeling"


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def _build_wheel(project_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)], cwd=project_root)
    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel in {output_dir}, got {wheels}"
    return wheels[0]


def _payload(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            name
            for name in archive.namelist()
            if not name.endswith("/") and ".dist-info/" not in name
        }


def _install_and_probe(first: Path, second: Path, *, root: Path, tmp: Path) -> dict[str, str]:
    venv = tmp / f"venv_{first.stem}_{second.stem}"
    _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=root)
    python = venv / "bin" / "python"
    _run([str(python), "-m", "pip", "install", "--no-deps", str(first), str(second)], cwd=root)
    probe = (
        "import json, modeling, modeling.trainer, modeling.predictor, modeling.walk_forward; "
        "import modeling_adapters, modeling_adapters.contracts, "
        "modeling_adapters.preprocess.fitted; "
        "print(json.dumps({'modeling': modeling.__file__, 'adapters': modeling_adapters.__file__}))"
    )
    output = _run([str(python), "-c", probe], cwd=root)
    return json.loads(output)


def test_two_wheels_have_disjoint_import_payloads_and_survive_both_orders():
    if not (ROOT / "pyproject.toml").exists():
        pytest.skip("repository layout unavailable")
    with tempfile.TemporaryDirectory(prefix="modeling_wheel_namespace_") as raw:
        tmp = Path(raw)
        fe_wheel = _build_wheel(FE_ROOT, tmp / "fe")
        adapter_wheel = _build_wheel(ADAPTER_ROOT, tmp / "adapters")

        fe_payload = _payload(fe_wheel)
        adapter_payload = _payload(adapter_wheel)
        assert not (fe_payload & adapter_payload)
        assert any(path.startswith("modeling/") for path in fe_payload)
        assert any(path.startswith("modeling_adapters/") for path in adapter_payload)
        assert not any(path.startswith("modeling/") for path in adapter_payload)

        first = _install_and_probe(fe_wheel, adapter_wheel, root=ROOT, tmp=tmp)
        second = _install_and_probe(adapter_wheel, fe_wheel, root=ROOT, tmp=tmp)
        for result in (first, second):
            assert "/site-packages/modeling/__init__.py" in result["modeling"]
            assert "/site-packages/modeling_adapters/__init__.py" in result["adapters"]
            assert result["modeling"] != result["adapters"]
