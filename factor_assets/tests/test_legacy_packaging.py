"""Regression coverage for legacy setuptools archive commands."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_dependencies_are_declared_in_pyproject() -> None:
    dependencies = (PROJECT_ROOT / "pyproject.toml").read_text()

    assert '"numpy>=1.24"' in dependencies
    assert '"scipy>=1.10"' in dependencies


def _source_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "factor_assets"
    shutil.copytree(
        PROJECT_ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    return destination


def test_legacy_archives_preserve_factor_assets_namespace(tmp_path: Path) -> None:
    source = _source_copy(tmp_path)
    dist = tmp_path / "dist"

    for command in ("bdist_wheel", "sdist"):
        subprocess.run(
            [sys.executable, "setup.py", command, "--dist-dir", str(dist)],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        )

    wheel_paths = list(dist.glob("*.whl"))
    sdist_paths = list(dist.glob("*.tar.gz"))
    assert len(wheel_paths) == 1
    assert len(sdist_paths) == 1

    with zipfile.ZipFile(wheel_paths[0]) as archive:
        names = set(archive.namelist())
        assert "factor_assets/__init__.py" in names
        assert "factor_assets/registry/__init__.py" in names
        assert "registry/__init__.py" not in names

        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")
        assert "Name: factor_assets" in metadata
        assert "Version: 0.1.0" in metadata
        assert "Requires-Python: >=3.10" in metadata
        assert "Requires-Dist: typing-extensions>=4.5.0" in metadata
        assert "Requires-Dist: dataclasses-json>=0.5.13" in metadata
        assert "Requires-Dist: numpy>=1.24" in metadata
        assert "Requires-Dist: scipy>=1.10" in metadata

    with tarfile.open(sdist_paths[0], "r:gz") as archive:
        names = set(archive.getnames())
        prefix = sdist_paths[0].stem.removesuffix(".tar")
        # The source tree is intentionally mapped from the archive root into
        # the factor_assets package when the sdist is rebuilt.
        assert f"{prefix}/__init__.py" in names
        assert f"{prefix}/registry/__init__.py" in names
        assert f"{prefix}/factor_assets/__init__.py" not in names
