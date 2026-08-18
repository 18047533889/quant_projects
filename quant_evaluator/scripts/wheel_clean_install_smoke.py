#!/usr/bin/env python3
"""
Clean wheel installation smoke test for quant_evaluator.

Validates that the package can be installed and imported from a clean wheel
in an isolated environment (no editable install, no monorepo dependencies).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def run_command(cmd, cwd=None, check=True, env=None):
    """Run a subprocess without invoking a shell."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.returncode, result.stdout, result.stderr


def isolated_env():
    """Remove ambient import and user-site configuration from child processes."""
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def main():
    print("=" * 70)
    print("quant_evaluator: Clean Wheel Installation Smoke Test")
    print("=" * 70)

    # Get package root
    package_root = Path(__file__).resolve().parent.parent
    print(f"\nPackage root: {package_root}")

    # Create temporary directory for isolated test
    with tempfile.TemporaryDirectory(prefix="qe_wheel_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Test directory: {tmpdir}")

        # Step 1: Build wheel
        print("\n[1/5] Building wheel...")
        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()

        env = isolated_env()
        build_env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
            build_env.pop(name, None)
        code, stdout, stderr = run_command(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(dist_dir),
                str(package_root),
            ],
            cwd=tmpdir,
            check=False,
            env=build_env,
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

        required_members = {
            "quant_evaluator/backends/__init__.py",
            "quant_evaluator/backends/selector.py",
            "quant_evaluator/backends/registry.py",
        }
        with ZipFile(wheel_path) as wheel:
            members = set(wheel.namelist())
        missing_members = sorted(required_members - members)
        if missing_members:
            print(f"FAILED: Wheel is missing required files: {missing_members}")
            return 1

        # Step 2: Create clean venv
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
        pip_exe = venv_dir / "bin" / "pip"
        print(f"✓ Created venv at {venv_dir}")

        # Step 3: Install wheel (wheel-only, no editable, no monorepo)
        print("\n[3/5] Installing wheel in isolated environment...")
        code, stdout, stderr = run_command(
            [str(python_exe), "-m", "pip", "install", str(wheel_path)],
            cwd=tmpdir,
            check=False,
            env=env,
        )
        if code != 0:
            print(f"FAILED: Wheel installation failed\n{stderr}")
            return 1
        print("✓ Wheel installed")

        # Step 4: Import public API
        print("\n[4/5] Importing public API...")
        import_script = tmpdir / "test_imports.py"
        import_script.write_text("""
import os
import sys
from pathlib import Path
import numpy as np
os.environ.pop('PYTHONPATH', None)
os.environ.pop('PYTHONHOME', None)
sys.path = [p for p in sys.path if 'quant_projects' not in p]  # Ensure no monorepo leakage

# Import public API
import quant_evaluator
assert 'site-packages' in str(Path(quant_evaluator.__file__).resolve())
from quant_evaluator import __version__
from quant_evaluator.contracts import FactorBatch, LabelBundle, AxisRef
from quant_evaluator.metrics import ic, quality, turnover
from quant_evaluator.planner import create_batch_plan
from quant_evaluator.registry import get_metric, list_metrics
from quant_evaluator.backends import get_available_backends, get_default_backend
from quant_evaluator.backends.selector import get_backend_capabilities

# Verify nested packages
from quant_evaluator.metrics import interactions, risk, stats
from quant_evaluator.metrics.stats import (
    GaussianHMM,
    detect_regimes,
    granger_causality_test,
    johansen_test,
    engle_granger_test,
    chow_test,
)

# Exercise representative modeling APIs, not only namespace imports.
series = np.linspace(-1.0, 1.0, 40)
assert callable(GaussianHMM)
assert callable(detect_regimes)
assert callable(granger_causality_test)
assert callable(johansen_test)
assert callable(engle_granger_test)
assert callable(chow_test)
model = GaussianHMM(n_states=2, n_iter=2, random_state=0)
model.fit(series.reshape(-1, 1))
assert model.predict(series.reshape(-1, 1)).shape == (40,)
print(f'FactorBatch: {FactorBatch}')
print(f'Metrics available: {len(list_metrics())}')
print(f'Backends available: {get_available_backends()} (default={get_default_backend()})')
print(f'Backend capabilities: {len(get_backend_capabilities())}')
print('✓ All imports successful')
""")

        code, stdout, stderr = run_command(
            [str(python_exe), str(import_script)],
            cwd=tmpdir,
            check=False,
            env=env,
        )
        if code != 0:
            print(f"FAILED: Import test failed\n{stderr}")
            return 1
        print(stdout)

        # Step 5: Run basic smoke evaluation
        print("\n[5/5] Running smoke evaluation...")
        smoke_script = tmpdir / "test_smoke.py"
        smoke_script.write_text("""
import os
import sys
from pathlib import Path
import numpy as np
os.environ.pop('PYTHONPATH', None)
os.environ.pop('PYTHONHOME', None)
sys.path = [p for p in sys.path if 'quant_projects' not in p]  # Ensure no monorepo leakage

from quant_evaluator.contracts import FactorBatch, LabelBundle, AxisRef
from quant_evaluator.metrics.ic import compute_daily_ic

# Create minimal test data
T, N, F = 10, 20, 2
time_values = np.arange(T)
asset_values = np.arange(N)
time_axis = AxisRef("time", "int64", T, values=time_values)
asset_axis = AxisRef("asset", "int64", N, values=asset_values)
factor_values = np.random.randn(T, N, F)

batch = FactorBatch(
    factor_ids=("factor_a", "factor_b"),
    time_axis=time_axis,
    asset_axis=asset_axis,
    values=factor_values,
)

labels = LabelBundle(
    target_id="forward_return_1d",
    values=np.random.randn(T, N),
    horizon=1,
    execution_delay=0,
    decision_time=tuple(time_values),
    execution_time=tuple(time_values),
    label_start_time=tuple(time_values),
    label_end_time=tuple(time_values + 1),
)

# Compute IC
ic_series, valid_counts = compute_daily_ic(batch, labels)
print(f"IC computed: shape={ic_series.shape}, mean={np.nanmean(ic_series):.4f}")
print("✓ Smoke evaluation passed")
""")

        code, stdout, stderr = run_command(
            [str(python_exe), str(smoke_script)],
            cwd=tmpdir,
            check=False,
            env=env,
        )
        if code != 0:
            print(f"FAILED: Smoke evaluation failed\n{stderr}")
            return 1
        print(stdout)

    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED")
    print("=" * 70)
    print("\nThe wheel can be installed and used in isolation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
