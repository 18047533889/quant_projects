#!/usr/bin/env python3
"""
Clean wheel installation smoke test for factor_assets.

Validates that the package can be installed and imported from a clean wheel
in an isolated environment.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 70)
    print("factor_assets: Clean Wheel Installation Smoke Test")
    print("=" * 70)

    package_root = Path(__file__).resolve().parent.parent
    print(f"\nPackage root: {package_root}")

    with tempfile.TemporaryDirectory(prefix="fa_wheel_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Test directory: {tmpdir}")

        # Build wheel
        print("\n[1/5] Building wheel...")
        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()

        code, stdout, stderr = run_command(
            f"python3 -m build --wheel --outdir {dist_dir}",
            cwd=package_root,
        )
        if code != 0:
            print(f"FAILED: Wheel build failed\n{stderr}")
            return 1

        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            print("FAILED: No wheel file generated")
            return 1

        wheel_path = wheels[0]
        print(f"✓ Built: {wheel_path.name}")

        # Create clean venv
        print("\n[2/5] Creating clean virtual environment...")
        venv_dir = tmpdir / "venv"

        code, _, stderr = run_command(f"python3 -m venv {venv_dir}")
        if code != 0:
            print(f"FAILED: venv creation failed\n{stderr}")
            return 1

        python_exe = venv_dir / "bin" / "python3"
        pip_exe = venv_dir / "bin" / "pip"
        print(f"✓ Created venv")

        # Install wheel
        print("\n[3/5] Installing wheel...")
        code, stdout, stderr = run_command(
            f"{pip_exe} install {wheel_path}",
            check=False,
        )
        if code != 0:
            print(f"FAILED: Installation failed\n{stderr}")
            return 1
        print("✓ Installed")

        # Import public API
        print("\n[4/5] Importing public API...")
        import_test = """
import sys
sys.path = [p for p in sys.path if 'quant_projects' not in p]

import factor_assets
from factor_assets import __version__
from factor_assets.registry import FactorAsset, AssetRegistry
from factor_assets.selection import SelectionGate, CompositeGate
from factor_assets.similarity import CosineSimilarity
from factor_assets.novelty import NoveltyOrchestrator

print(f'factor_assets version: {__version__}')
print(f'AssetRegistry: {AssetRegistry}')
print('✓ All imports successful')
"""

        code, stdout, stderr = run_command(
            f"{python_exe} -c '{import_test}'",
            check=False,
        )
        if code != 0:
            print(f"FAILED: Import failed\n{stderr}")
            return 1
        print(stdout)

        # Smoke test
        print("\n[5/5] Running smoke test...")
        smoke_test = """
from factor_assets.registry import FactorAsset, AssetRegistry

registry = AssetRegistry()
asset = FactorAsset(
    asset_id="test_factor",
    expression="close / ts_delay(close, 1) - 1",
    metadata={"source": "smoke_test"},
)
registry.register(asset)

retrieved = registry.get("test_factor")
assert retrieved.asset_id == "test_factor"
print(f"✓ Registered and retrieved asset: {retrieved.asset_id}")
"""

        code, stdout, stderr = run_command(
            f"{python_exe} -c '{smoke_test}'",
            check=False,
        )
        if code != 0:
            print(f"FAILED: Smoke test failed\n{stderr}")
            return 1
        print(stdout)

    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
