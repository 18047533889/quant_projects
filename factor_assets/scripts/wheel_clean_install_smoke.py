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
        import_script = tmpdir / "test_imports.py"
        import_script.write_text("""
import sys
sys.path = [p for p in sys.path if 'quant_projects' not in p]

import factor_assets
import importlib
import pkgutil
from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.registry import AssetRepository
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef

failures = {}
modules = [factor_assets.__name__]
modules.extend(
    info.name
    for info in pkgutil.walk_packages(
        factor_assets.__path__, prefix=f"{factor_assets.__name__}."
    )
    if not info.name.startswith("factor_assets.tests")
    and info.name not in {"factor_assets.setup", "factor_assets.examples_usage"}
)
for module_name in sorted(modules):
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        failures[module_name] = f"{type(exc).__name__}: {exc}"
if failures:
    raise AssertionError(f"Broken shipped imports: {failures}")

print('factor_assets imported')
print(f'FactorAsset: {FactorAsset}')
print(f'AssetRepository: {AssetRepository}')
print(f'LifecycleState: {LifecycleState}')
print('✓ All imports successful')
""")

        code, stdout, stderr = run_command(
            f"{python_exe} {import_script}",
            check=False,
        )
        if code != 0:
            print(f"FAILED: Import failed\n{stderr}")
            return 1
        print(stdout)

        # Smoke test
        print("\n[5/5] Running smoke test...")
        smoke_script = tmpdir / "test_smoke.py"
        smoke_script.write_text("""
from datetime import datetime
from factor_assets.contracts.asset import AssetMetadata
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.registry import AssetRepository

# Create repository
repo = AssetRepository()

# Create asset metadata
metadata = AssetMetadata(
    factor_id="test_factor",
    canonical_repr="close / ts_delay(close, 1) - 1",
    canonical_hash="test_hash_12345",
    frequency="daily",
    domains=("equity",),
    timing="daily",
    description="Test momentum factor"
)

# Create lineage (no parents for this test factor)
lineage = LineageRef(
    factor_id="test_factor",
    parents=(),
    campaign_id="smoke_test",
    created_at=datetime.utcnow().isoformat(),
)

# Register (returns FactorAsset)
asset = repo.register(metadata, lineage)

# Retrieve
retrieved = repo.get(asset.factor_id)
assert retrieved.factor_id == "test_factor"
assert retrieved.lifecycle_state == LifecycleState.REGISTERED
print(f"✓ Registered and retrieved asset: {retrieved.factor_id}")
""")

        code, stdout, stderr = run_command(
            f"{python_exe} {smoke_script}",
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
