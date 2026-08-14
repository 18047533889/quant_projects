#!/usr/bin/env python3
"""
Clean wheel installation smoke test for factor_optimizer.

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
    print("factor_optimizer: Clean Wheel Installation Smoke Test")
    print("=" * 70)

    package_root = Path(__file__).resolve().parent.parent
    print(f"\nPackage root: {package_root}")

    with tempfile.TemporaryDirectory(prefix="fo_wheel_test_") as tmpdir:
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
        import_test = """
import sys
sys.path = [p for p in sys.path if 'quant_projects' not in p]

import factor_optimizer
from factor_optimizer import __version__
from factor_optimizer.search import SearchPolicy
from factor_optimizer.diagnosis import DiagnosisMapper

print(f'factor_optimizer version: {__version__}')
print('✓ Imports OK')
"""
        code, stdout, stderr = run_command(f"{python_exe} -c '{import_test}'", check=False)
        if code != 0:
            print(f"FAILED: Import\n{stderr}")
            return 1
        print(stdout)

        # Smoke test
        print("\n[5/5] Smoke test...")
        smoke_test = """
from factor_optimizer.search import SearchPolicy

policy = SearchPolicy(
    max_iterations=10,
    exploration_rate=0.2,
)
print(f"✓ Created policy: max_iter={policy.max_iterations}")
"""
        code, stdout, stderr = run_command(f"{python_exe} -c '{smoke_test}'", check=False)
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
