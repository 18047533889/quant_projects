#!/usr/bin/env python3
"""
Clean wheel installation smoke test for modeling_adapters.

Validates package installation and basic functionality in isolation.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def run_command(cmd, cwd=None, check=True, *, disable_user_site=True):
    """Run an argument-vector command and return output."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONUSERBASE", None)
    if disable_user_site:
        env["PYTHONNOUSERSITE"] = "1"
    else:
        env.pop("PYTHONNOUSERSITE", None)
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=check
    )
    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 70)
    print("modeling_adapters: Clean Wheel Installation Smoke Test")
    print("=" * 70)

    package_root = Path(__file__).resolve().parent.parent
    print(f"\nPackage root: {package_root}")

    # The active project legitimately generates modeling_adapters.egg-info;
    # only legacy modeling metadata is stale for this distribution.
    stale_metadata = package_root / "modeling.egg-info"
    if stale_metadata.exists():
        print(f"FAILED: stale package metadata is present: {stale_metadata.name}")
        return 1

    with tempfile.TemporaryDirectory(prefix="modeling_wheel_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Test directory: {tmpdir}")

        # Ensure stale build output cannot contribute interpreter-specific files.
        build_dir = package_root / "build"
        if build_dir.is_dir():
            shutil.rmtree(build_dir)

        # Build wheel
        print("\n[1/5] Building wheel...")
        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()

        code, _, stderr = run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--outdir",
                str(dist_dir),
                str(package_root),
            ],
            cwd=tmpdir,
            disable_user_site=False,
        )
        if code != 0:
            print(f"FAILED: Wheel build\n{stderr}")
            return 1

        wheels = sorted(dist_dir.glob("*.whl"))
        if len(wheels) != 1:
            print(f"FAILED: expected exactly one wheel, found {len(wheels)}")
            return 1
        wheel_path = wheels[0]
        with zipfile.ZipFile(wheel_path) as wheel_zip:
            names = set(wheel_zip.namelist())
            dist_info = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            metadata = wheel_zip.read(dist_info).decode("utf-8")
            assert "Name: modeling-adapters" in metadata
            assert "Version: 0.1.0" in metadata
            package_roots = {
                name.split("/", 1)[0]
                for name in names
                if ".dist-info/" not in name
                and not name.endswith(".dist-info")
                and name
            }
            assert package_roots == {"modeling_adapters"}, package_roots
            assert any(name.startswith("modeling_adapters/") for name in names)
            assert not any(name.startswith("modeling/") for name in names)
            assert not any(
                "__pycache__/" in name
                or name.endswith((".pyc", ".pyo"))
                for name in names
            )
        print(f"✓ Built and layout-checked: {wheel_path.name}")

        # Create venv
        print("\n[2/5] Creating venv...")
        venv_dir = tmpdir / "venv"
        code, _, stderr = run_command(
            [sys.executable, "-m", "venv", str(venv_dir)]
        )
        if code != 0:
            print(f"FAILED: venv\n{stderr}")
            return 1
        pip_exe = venv_dir / "bin" / "pip"
        python_exe = venv_dir / "bin" / "python3"
        print("✓ Created venv")

        # Install wheel
        print("\n[3/5] Installing wheel...")
        code, _, stderr = run_command(
            [str(pip_exe), "install", str(wheel_path)], check=False
        )
        if code != 0:
            print(f"FAILED: Install\n{stderr}")
            return 1
        print("✓ Installed")

        # Import test
        print("\n[4/5] Import test...")
        import_script = tmpdir / "test_imports.py"
        import_script.write_text("""
import importlib.metadata as importlib_metadata
import importlib.util
import sys
sys.path = [p for p in sys.path if 'quant_projects' not in p]

installed = {
    distribution.metadata.get('Name')
    for distribution in importlib_metadata.distributions()
    if distribution.metadata.get('Name') in {'modeling', 'modeling-adapters'}
}
if installed != {'modeling-adapters'}:
    raise AssertionError(f'unexpected modeling distributions: {sorted(installed)}')

import modeling_adapters
from modeling_adapters import __version__
from modeling_adapters.contracts import FitWindow, SplitSpec
from modeling_adapters.errors import FutureLeakageError

if importlib.util.find_spec('modeling') is not None:
    raise AssertionError('standalone adapter wheel must not provide modeling')

print(f'modeling_adapters version: {__version__}')
print('✓ Imports OK')
""")
        code, stdout, stderr = run_command(
            [str(python_exe), str(import_script)], check=False
        )
        if code != 0:
            print(f"FAILED: Import\n{stderr}")
            return 1
        print(stdout)

        # Smoke test
        print("\n[5/5] Smoke test...")
        smoke_script = tmpdir / "test_smoke.py"
        smoke_script.write_text("""
import numpy as np

from modeling_adapters.contracts import FitWindow
from modeling_adapters.preprocess import rank_transform
from datetime import datetime

window = FitWindow(
    fit_start=datetime(2020, 1, 1),
    fit_end=datetime(2020, 12, 31),
)
print(f'✓ Created FitWindow: {window.fit_start} to {window.fit_end}')

ranked = rank_transform(
    np.array([[1.0, 2.0, 2.0, 4.0], [4.0, np.nan, 1.0, 2.0]]),
    axis=1,
    pct=True,
)
assert np.allclose(ranked[0], [0.0, 0.5, 0.5, 1.0])
assert np.isnan(ranked[1, 1])
print('✓ rank_transform path executed')
""")
        code, stdout, stderr = run_command(
            [str(python_exe), str(smoke_script)], check=False
        )
        if code != 0:
            print(f"FAILED: Smoke\n{stderr}")
            return 1
        print(stdout)

    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
