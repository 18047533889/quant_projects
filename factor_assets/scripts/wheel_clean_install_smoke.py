#!/usr/bin/env python3
"""
Clean wheel installation smoke test for factor_assets.

Validates that the package can be installed and imported from a clean wheel
in an isolated environment.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def isolated_env():
    """Remove ambient import and user-site configuration from child processes."""
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_command(cmd, cwd=None, check=True, env=None):
    """Run an argument-vector command in an isolated import environment."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=isolated_env() if env is None else env,
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

        # Build from a clean source copy so generated tree state cannot leak into
        # the release payload.
        print("\n[1/5] Building wheel...")
        source_dir = tmpdir / "source"
        shutil.copytree(
            package_root,
            source_dir,
            ignore=shutil.ignore_patterns(
                "build", "dist", "*.egg-info", "__pycache__", "*.pyc"
            ),
        )
        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()

        env = isolated_env()
        code, _, stderr = run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(dist_dir),
            ],
            cwd=source_dir,
            check=False,
            env=env,
        )
        if code != 0:
            print(f"FAILED: Wheel build failed\n{stderr}")
            return 1

        wheels = sorted(dist_dir.glob("*.whl"))
        if len(wheels) != 1:
            print(f"FAILED: Expected one wheel, found {len(wheels)}")
            return 1

        wheel_path = wheels[0]
        required_members = {
            "factor_assets/registry/factory.py",
            "factor_assets/registry/sqlite_repository.py",
            "factor_assets/registry/migrations.py",
            "factor_assets/registry/serialization.py",
        }
        with zipfile.ZipFile(wheel_path) as archive:
            members = set(archive.namelist())
        missing = sorted(required_members - members)
        bytecode = sorted(
            name for name in members if "__pycache__" in name or name.endswith(".pyc")
        )
        if missing or bytecode:
            print(
                "FAILED: Invalid wheel payload "
                f"(missing={missing}, bytecode={bytecode})"
            )
            return 1
        print(f"✓ Built and payload-checked: {wheel_path.name}")

        # Create clean venv
        print("\n[2/5] Creating clean virtual environment...")
        venv_dir = tmpdir / "venv"

        code, _, stderr = run_command(
            [sys.executable, "-m", "venv", str(venv_dir)],
            cwd=tmpdir,
            env=env,
        )
        if code != 0:
            print(f"FAILED: venv creation failed\n{stderr}")
            return 1

        python_exe = venv_dir / "bin" / "python3"
        print(f"✓ Created venv")

        # Install wheel
        print("\n[3/5] Installing wheel...")
        code, _, stderr = run_command(
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                str(wheel_path),
            ],
            cwd=tmpdir,
            check=False,
            env=env,
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
from factor_assets import SQLiteLifecycleRepository, create_repository
from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.registry import AssetRepository
import factor_assets.registry.migrations
import factor_assets.registry.serialization
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
print(f'SQLiteLifecycleRepository: {SQLiteLifecycleRepository}')
print(f'create_repository: {create_repository}')
print(f'LifecycleState: {LifecycleState}')
print('✓ All imports successful')
""")

        code, stdout, stderr = run_command(
            [str(python_exe), str(import_script)],
            cwd=tmpdir,
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
from factor_assets import SQLiteLifecycleRepository, create_repository
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

# Exercise the durable public factory from the installed wheel, then reopen the
# database to prove migration and serialization modules are present and usable.
sqlite_path = __import__("pathlib").Path(__file__).with_name("registry.db")
durable = create_repository(db_path=sqlite_path)
assert isinstance(durable, SQLiteLifecycleRepository)
durable.register(
    AssetMetadata(
        factor_id="durable_factor",
        canonical_repr="close",
        canonical_hash="durable_hash_12345",
        frequency="daily",
        domains=("equity",),
        timing="daily",
    ),
    LineageRef(factor_id="durable_factor", parents=()),
)
reopened = SQLiteLifecycleRepository(sqlite_path)
assert reopened.get("durable_factor").lifecycle_state == LifecycleState.REGISTERED
assert len(reopened.get_events("durable_factor")) == 1
print("✓ Created, persisted, and reopened SQLite lifecycle repository")
""")

        code, stdout, stderr = run_command(
            [str(python_exe), str(smoke_script)],
            cwd=tmpdir,
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
