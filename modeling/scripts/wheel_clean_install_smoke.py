#!/usr/bin/env python3
"""
Clean wheel installation smoke test for modeling_adapters.

Validates package installation and basic functionality in isolation.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=check
    )
    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 70)
    print("modeling_adapters: Clean Wheel Installation Smoke Test")
    print("=" * 70)

    package_root = Path(__file__).resolve().parent.parent
    print(f"\nPackage root: {package_root}")

    with tempfile.TemporaryDirectory(prefix="modeling_wheel_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Test directory: {tmpdir}")

        # Build wheel
        print("\n[1/5] Building wheel...")
        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()

        code, _, stderr = run_command(
            f"python3 -m build --wheel --outdir {dist_dir}", cwd=package_root
        )
        if code != 0:
            print(f"FAILED: Wheel build\n{stderr}")
            return 1

        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            print("FAILED: No wheel generated")
            return 1
        wheel_path = wheels[0]
        print(f"✓ Built: {wheel_path.name}")

        # Create venv
        print("\n[2/5] Creating venv...")
        venv_dir = tmpdir / "venv"
        code, _, stderr = run_command(f"python3 -m venv {venv_dir}")
        if code != 0:
            print(f"FAILED: venv\n{stderr}")
            return 1
        pip_exe = venv_dir / "bin" / "pip"
        python_exe = venv_dir / "bin" / "python3"
        print("✓ Created venv")

        # Install wheel
        print("\n[3/5] Installing wheel...")
        code, _, stderr = run_command(f"{pip_exe} install {wheel_path}", check=False)
        if code != 0:
            print(f"FAILED: Install\n{stderr}")
            return 1
        print("✓ Installed")

        # Import test
        print("\n[4/5] Import test...")
        import_script = tmpdir / "test_imports.py"
        import_script.write_text("""
import sys
sys.path = [p for p in sys.path if 'quant_projects' not in p]

import modeling_adapters
from modeling_adapters import __version__
from modeling_adapters.contracts import FitWindow, SplitSpec
from modeling_adapters.errors import FutureLeakageError

print(f'modeling_adapters version: {__version__}')
print('✓ Imports OK')
""")
        code, stdout, stderr = run_command(f"{python_exe} {import_script}", check=False)
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
        code, stdout, stderr = run_command(f"{python_exe} {smoke_script}", check=False)
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
