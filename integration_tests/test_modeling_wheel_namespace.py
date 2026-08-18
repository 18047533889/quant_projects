"""Serialized wheel checks for the FactorEngine/adapter namespace boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
FE_ROOT = ROOT / "factor_engine"
ADAPTER_ROOT = ROOT / "modeling"


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
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
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)], cwd=project_root)
    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel in {output_dir}, got {wheels}"
    return wheels[0]


def _build_legacy_wheel(output_dir: Path, *, root: Path) -> Path:
    source = output_dir.parent / "legacy_source"
    source.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "fba3aaf9^", "modeling"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(source)], input=archive.stdout, check=True)
    return _build_wheel(source / "modeling", output_dir)


def _payload(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            name
            for name in archive.namelist()
            if not name.endswith("/") and ".dist-info/" not in name
        }


def _required_distribution_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_paths) == 1, (
            f"expected one wheel METADATA entry in {wheel}, got {metadata_paths}"
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
    return {
        canonicalize_name(Requirement(value).name)
        for value in metadata.get_all("Requires-Dist", [])
    }


def _isolated_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONUSERBASE", None)
    env.pop("PYTHONNOUSERSITE", None)
    return env


def _install_and_probe(first: Path, second: Path, *, root: Path, tmp: Path) -> dict[str, str]:
    venv = tmp / f"venv_{first.stem}_{second.stem}"
    _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=root, env=_isolated_env())
    python = venv / "bin" / "python"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(first),
            str(second),
        ],
        cwd=root,
        env=_isolated_env(),
    )
    probe = (
        "import json, modeling, modeling.trainer, modeling.predictor, modeling.walk_forward; "
        "import modeling_adapters, modeling_adapters.contracts, "
        "modeling_adapters.preprocess.fitted; "
        "print(json.dumps({'modeling': modeling.__file__, 'adapters': modeling_adapters.__file__}))"
    )
    output = _run([str(python), "-c", probe], cwd=tmp, env=_isolated_env())
    return json.loads(output)


def _probe_legacy_modules(python: Path, *, cwd: Path) -> dict[str, object]:
    probe = (
        "import importlib.metadata as md, importlib.machinery as machinery, importlib.util, json, pathlib, sys; "
        "repo = pathlib.Path(sys.argv[1]).resolve(); "
        "sys.path[:] = [path for path in sys.path if not pathlib.Path(path or '.').resolve().is_relative_to(repo)]; "
        "sys.meta_path[:] = [machinery.BuiltinImporter, machinery.FrozenImporter, machinery.PathFinder]; "
        "names = ('modeling', 'modeling.contracts', 'modeling.trainer', 'modeling.predictor', "
        "'modeling.walk_forward', 'modeling.adapter', 'modeling.preprocess', "
        "'modeling.exposure'); "
        "root_spec = importlib.util.find_spec('modeling'); "
        "specs = {'modeling': root_spec}; "
        "specs.update({name: (importlib.util.find_spec(name) if root_spec is not None else None) "
        "for name in names[1:]}); "
        "print(json.dumps({'modules': {name: spec is not None for name, spec in specs.items()}, "
        "'origins': {name: (None if spec is None else spec.origin) for name, spec in specs.items()}, "
        "'owners': {name: md.packages_distributions().get(name, []) for name in ('modeling', 'modeling_adapters')}}))"
    )
    return json.loads(_run([str(python), "-c", probe, str(ROOT)], cwd=cwd, env=_isolated_env()))


def _dist_installed(python: Path, distribution: str, *, cwd: Path) -> bool:
    probe = (
        "import importlib.metadata as md, sys; "
        "name = sys.argv[1]; "
        "print('1' if any(dist.metadata['Name'] == name for dist in md.distributions()) else '0')"
    )
    return _run([str(python), "-c", probe, distribution], cwd=cwd, env=_isolated_env()).strip() == "1"


def _is_venv_file(origin: object, venv: Path) -> bool:
    return isinstance(origin, str) and Path(origin).is_file() and Path(origin).is_relative_to(venv)


def test_legacy_modeling_uninstall_requires_factor_engine_repair():
    if not (ROOT / "pyproject.toml").exists():
        pytest.skip("repository layout unavailable")
    with tempfile.TemporaryDirectory(prefix="legacy_modeling_migration_") as raw:
        tmp = Path(raw)
        legacy_wheel = _build_legacy_wheel(tmp / "legacy", root=ROOT)
        fe_wheel = _build_wheel(FE_ROOT, tmp / "fe")
        adapter_wheel = _build_wheel(ADAPTER_ROOT, tmp / "adapters")

        venv = tmp / "venv"
        _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=ROOT)
        python = venv / "bin" / "python"
        _run(
            [
                str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", str(legacy_wheel),
            ],
            cwd=tmp,
        )
        _run(
            [
                str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", str(fe_wheel),
            ],
            cwd=tmp,
        )
        _run(
            [
                str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", str(adapter_wheel),
            ],
            cwd=tmp,
        )
        before = _probe_legacy_modules(python, cwd=tmp)
        assert before["modules"]["modeling.trainer"] is True
        assert before["modules"]["modeling.adapter"] is True
        assert before["modules"]["modeling.preprocess"] is True
        assert before["modules"]["modeling.exposure"] is True
        assert "modeling" in before["owners"]["modeling"]
        assert "factor-engine" in before["owners"]["modeling"]
        assert before["owners"]["modeling_adapters"] == ["modeling-adapters"]
        assert "modeling-adapters" not in before["owners"]["modeling"]

        _run([str(python), "-m", "pip", "uninstall", "-y", "modeling"], cwd=tmp)
        after_uninstall = _probe_legacy_modules(python, cwd=tmp)
        assert after_uninstall["origins"]["modeling"] is None
        assert after_uninstall["modules"]["modeling.contracts"] is False
        assert after_uninstall["modules"]["modeling.trainer"] is True
        assert after_uninstall["modules"]["modeling.predictor"] is True
        assert after_uninstall["modules"]["modeling.walk_forward"] is True
        assert after_uninstall["modules"]["modeling.adapter"] is False
        assert after_uninstall["modules"]["modeling.preprocess"] is False
        assert after_uninstall["modules"]["modeling.exposure"] is False
        assert "factor-engine" in after_uninstall["owners"]["modeling"]
        assert _dist_installed(python, "modeling", cwd=tmp) is True
        assert _dist_installed(python, "factor-engine", cwd=tmp) is True
        assert _dist_installed(python, "modeling-adapters", cwd=tmp) is True

        _run(
            [
                str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", str(fe_wheel),
            ],
            cwd=tmp,
        )
        repaired = _probe_legacy_modules(python, cwd=tmp)
        assert _is_venv_file(repaired["origins"]["modeling"], venv)
        assert _is_venv_file(repaired["origins"]["modeling.contracts"], venv)
        assert repaired["modules"]["modeling.trainer"] is True
        assert repaired["modules"]["modeling.predictor"] is True
        assert repaired["modules"]["modeling.walk_forward"] is True
        assert repaired["modules"]["modeling.adapter"] is False
        assert repaired["modules"]["modeling.preprocess"] is False
        assert repaired["modules"]["modeling.exposure"] is False


def test_factor_engine_uninstall_requires_legacy_repair_for_shared_namespace():
    """Document and verify recovery from reverse shared-namespace uninstall."""
    if not (ROOT / "pyproject.toml").exists():
        pytest.skip("repository layout unavailable")
    with tempfile.TemporaryDirectory(prefix="factor_engine_uninstall_migration_") as raw:
        tmp = Path(raw)
        legacy_wheel = _build_legacy_wheel(tmp / "legacy", root=ROOT)
        fe_wheel = _build_wheel(FE_ROOT, tmp / "fe")

        for index, wheels in enumerate(
            ((legacy_wheel, fe_wheel), (fe_wheel, legacy_wheel)),
            start=1,
        ):
            venv = tmp / f"venv_{index}"
            _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=tmp)
            python = venv / "bin" / "python"
            for wheel in wheels:
                _run(
                    [
                        str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                        "--force-reinstall", str(wheel),
                    ],
                    cwd=tmp,
                    env=_isolated_env(),
                )

            _run([str(python), "-m", "pip", "uninstall", "-y", "factor-engine"], cwd=tmp)
            damaged = _probe_legacy_modules(python, cwd=tmp)
            assert _dist_installed(python, "modeling", cwd=tmp) is True
            assert damaged["modules"]["modeling.adapter"] is True
            assert damaged["modules"]["modeling.preprocess"] is True
            assert damaged["modules"]["modeling.exposure"] is True
            assert damaged["modules"]["modeling.contracts"] is False
            assert damaged["modules"]["modeling.trainer"] is False
            assert damaged["modules"]["modeling.predictor"] is False
            assert damaged["modules"]["modeling.walk_forward"] is False

            _run(
                [
                    str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                    "--force-reinstall", str(legacy_wheel),
                ],
                cwd=tmp,
                env=_isolated_env(),
            )
            repaired = _probe_legacy_modules(python, cwd=tmp)
            assert _is_venv_file(repaired["origins"]["modeling"], venv)
            assert _is_venv_file(repaired["origins"]["modeling.contracts"], venv)
            assert repaired["modules"]["modeling.adapter"] is True
            assert repaired["modules"]["modeling.preprocess"] is True
            assert repaired["modules"]["modeling.exposure"] is True
            assert repaired["modules"]["modeling.trainer"] is False
            assert repaired["modules"]["modeling.predictor"] is False
            assert repaired["modules"]["modeling.walk_forward"] is False


def test_two_wheels_have_disjoint_import_payloads_and_survive_both_orders():
    if not (ROOT / "pyproject.toml").exists():
        pytest.skip("repository layout unavailable")
    with tempfile.TemporaryDirectory(prefix="modeling_wheel_namespace_") as raw:
        tmp = Path(raw)
        fe_wheel = _build_wheel(FE_ROOT, tmp / "fe")
        adapter_wheel = _build_wheel(ADAPTER_ROOT, tmp / "adapters")
        assert {"numpy", "pandas", "scipy"} <= _required_distribution_names(adapter_wheel)

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

        venv = tmp / "venv_uninstall_adapter"
        _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], cwd=ROOT, env=_isolated_env())
        python = venv / "bin" / "python"
        _run(
            [
                str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", str(fe_wheel), str(adapter_wheel),
            ],
            cwd=tmp,
            env=_isolated_env(),
        )
        _run(
            [str(python), "-m", "pip", "uninstall", "-y", "modeling-adapters"],
            cwd=tmp,
            env=_isolated_env(),
        )
        remaining = _probe_legacy_modules(python, cwd=tmp)
        assert remaining["modules"]["modeling"] is True
        assert remaining["modules"]["modeling.trainer"] is True
        assert remaining["owners"]["modeling_adapters"] == []
        assert _dist_installed(python, "factor-engine", cwd=tmp) is True
