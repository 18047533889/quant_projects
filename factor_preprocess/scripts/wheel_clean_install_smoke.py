#!/usr/bin/env python3
"""
Clean wheel installation smoke test for factor_preprocess.

Validates package installation and basic functionality in isolation.
"""

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run an argv command in a sanitized Python environment."""
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=check
    )
    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 70)
    print("factor_preprocess: Clean Wheel Installation Smoke Test")
    print("=" * 70)

    package_root = Path(__file__).resolve().parent.parent
    print(f"\nPackage root: {package_root}")

    with tempfile.TemporaryDirectory(prefix="fp_wheel_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Test directory: {tmpdir}")

        # Build wheel
        print("\n[1/5] Building wheel...")
        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()

        code, _, stderr = run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(dist_dir),
            ],
            cwd=package_root,
        )
        if code != 0:
            print(f"FAILED: Wheel build\n{stderr}")
            return 1

        wheels = sorted(dist_dir.glob("*.whl"))
        if len(wheels) != 1:
            print(
                "FAILED: Expected exactly one wheel, "
                f"found {len(wheels)}: {[path.name for path in wheels]}"
            )
            return 1
        wheel_path = wheels[0]
        print(f"✓ Built: {wheel_path.name}")

        source_files = {
            path.relative_to(package_root / "factor_preprocess").as_posix()
            for path in (package_root / "factor_preprocess").rglob("*.py")
        }
        with zipfile.ZipFile(wheel_path) as archive:
            wheel_files = {
                name.removeprefix("factor_preprocess/")
                for name in archive.namelist()
                if name.startswith("factor_preprocess/") and name.endswith(".py")
            }
            bytecode = sorted(
                name
                for name in archive.namelist()
                if "__pycache__" in name or name.endswith((".pyc", ".pyo"))
            )
        missing = sorted(source_files - wheel_files)
        extra = sorted(wheel_files - source_files)
        if missing or extra or bytecode:
            print(
                "FAILED: Wheel manifest mismatch "
                f"(missing={missing}, extra={extra}, bytecode={bytecode})"
            )
            return 1
        print(f"✓ Source manifest matched: {len(source_files)} Python files")

        # Create venv
        print("\n[2/5] Creating venv...")
        venv_dir = tmpdir / "venv"
        code, _, stderr = run_command(
            [sys.executable, "-m", "venv", str(venv_dir)]
        )
        if code != 0:
            print(f"FAILED: venv\n{stderr}")
            return 1
        python_exe = venv_dir / "bin" / "python3"
        print("✓ Created venv")

        # Install wheel
        print("\n[3/5] Installing wheel...")
        code, _, stderr = run_command(
            [str(python_exe), "-m", "pip", "install", str(wheel_path)],
            check=False,
        )
        if code != 0:
            print(f"FAILED: Install\n{stderr}")
            return 1
        print("✓ Installed")

        # Import test
        print("\n[4/5] Import test...")
        import_script = tmpdir / "test_import.py"
        import_script.write_text("""
import importlib.metadata
import sysconfig
from pathlib import Path

import factor_preprocess
import factor_preprocess.adapters as adapters
import factor_preprocess.backends as backends
import factor_preprocess.kernels.fast as fast_kernels
import factor_preprocess.kernels.numba_transforms as numba_transforms
import factor_preprocess.neutralization as neutralization
import factor_preprocess.neutralization.advanced as advanced_neutralization
import factor_preprocess.regime as regime
import factor_preprocess.representation as representation
import factor_preprocess.transforms as transforms
import factor_preprocess.transforms.decomposition as decomposition
from factor_preprocess import __version__
from factor_preprocess.transforms import cs_rank, cs_zscore, rolling_mean
from factor_preprocess.transforms.decomposition import stl_decompose
from factor_preprocess.neutralization import ols_neutralize

expected_version = "0.1.0"
distribution = importlib.metadata.distribution("factor-preprocess")
assert distribution.version == expected_version, distribution.version
assert importlib.metadata.version("factor-preprocess") == expected_version
assert __version__ == expected_version, __version__

site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
imported_modules = (
    factor_preprocess,
    adapters,
    backends,
    fast_kernels,
    numba_transforms,
    neutralization,
    advanced_neutralization,
    regime,
    representation,
    transforms,
    decomposition,
)
for module in imported_modules:
    if module.__file__ is None:
        continue
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to(site_packages), (
        f"{module.__name__} loaded from {module_path}, outside {site_packages}"
    )
assert stl_decompose.__module__.startswith("factor_preprocess")
print(f"factor_preprocess version: {__version__}")
print("✓ Imports OK")
""")
        code, stdout, stderr = run_command(
            [str(python_exe), "-I", str(import_script)],
            cwd=tmpdir,
            check=False,
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
from factor_preprocess.transforms import cs_rank


data = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
ranked = cs_rank(data)
expected = np.array([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
assert ranked.shape == (2, 3), ranked.shape
assert np.isfinite(ranked).all()
np.testing.assert_array_equal(ranked, expected)
print(f"✓ cs_rank: {ranked.shape}")
""")
        code, stdout, stderr = run_command(
            [str(python_exe), "-I", str(smoke_script)],
            cwd=tmpdir,
            check=False,
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
