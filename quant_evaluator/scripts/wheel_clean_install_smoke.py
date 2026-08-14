#!/usr/bin/env python3
"""
Clean wheel installation smoke test for quant_evaluator.

Validates that the package can be installed and imported from a clean wheel
in an isolated environment (no editable install, no monorepo dependencies).
"""

import subprocess
import sys
import tempfile
import shutil
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

        # Step 2: Create clean venv
        print("\n[2/5] Creating clean virtual environment...")
        venv_dir = tmpdir / "venv"

        code, _, stderr = run_command(f"python3 -m venv {venv_dir}")
        if code != 0:
            print(f"FAILED: venv creation failed\n{stderr}")
            return 1

        python_exe = venv_dir / "bin" / "python3"
        pip_exe = venv_dir / "bin" / "pip"
        print(f"✓ Created venv at {venv_dir}")

        # Step 3: Install wheel (wheel-only, no editable, no monorepo)
        print("\n[3/5] Installing wheel in isolated environment...")
        code, stdout, stderr = run_command(
            f"{pip_exe} install {wheel_path}",
            check=False,
        )
        if code != 0:
            print(f"FAILED: Wheel installation failed\n{stderr}")
            return 1
        print("✓ Wheel installed")

        # Step 4: Import public API
        print("\n[4/5] Importing public API...")
        import_script = tmpdir / "test_imports.py"
        import_script.write_text("""
import sys
sys.path = [p for p in sys.path if 'quant_projects' not in p]  # Ensure no monorepo leakage

# Import public API
import quant_evaluator
from quant_evaluator import __version__
from quant_evaluator.contracts import FactorBatch, LabelBundle, AxisRef
from quant_evaluator.metrics import ic, quality, turnover
from quant_evaluator.planner import create_batch_plan
from quant_evaluator.registry import get_metric, list_metrics

# Verify nested packages
from quant_evaluator.metrics import interactions, risk, stats

print(f'quant_evaluator version: {__version__}')
print(f'FactorBatch: {FactorBatch}')
print(f'Metrics available: {len(list_metrics())}')
print('✓ All imports successful')
""")

        code, stdout, stderr = run_command(
            f'{python_exe} {import_script}',
            check=False,
        )
        if code != 0:
            print(f"FAILED: Import test failed\n{stderr}")
            return 1
        print(stdout)

        # Step 5: Run basic smoke evaluation
        print("\n[5/5] Running smoke evaluation...")
        smoke_script = tmpdir / "test_smoke.py"
        smoke_script.write_text("""
import numpy as np
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
            f'{python_exe} {smoke_script}',
            check=False,
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
