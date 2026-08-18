#!/usr/bin/env python3
"""
Clean wheel installation smoke test for factor_optimizer.

Validates package installation and basic functionality in isolation.
"""

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path


def run_command(cmd, cwd=None, check=True, *, disable_user_site=True):
    """Run an argument-vector command in an isolated import environment."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONUSERBASE", None)
    if disable_user_site:
        env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=check
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
            check=False,
        )
        if code != 0:
            print(f"FAILED: Wheel build\n{stderr}")
            return 1

        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            print("FAILED: No wheel generated")
            return 1
        if len(wheels) != 1:
            print(f"FAILED: Expected one wheel, found {len(wheels)}")
            return 1
        wheel_path = wheels[0]
        required_members = {
            "factor_optimizer/capabilities.py",
            "factor_optimizer/errors.py",
            "factor_optimizer/complexity/budget.py",
            "factor_optimizer/contracts/splits.py",
            "factor_optimizer/grammar/__init__.py",
            "factor_optimizer/contracts/__init__.py",
            "factor_optimizer/seen/__init__.py",
            "factor_optimizer/policy/__init__.py",
            "factor_optimizer/search/__init__.py",
            "factor_optimizer/llm/__init__.py",
            "factor_optimizer/llm/prompts.py",
            "factor_optimizer/llm/proposal.py",
            "factor_optimizer/llm/records.py",
            "factor_optimizer/adapters/__init__.py",
        }
        with zipfile.ZipFile(wheel_path) as archive:
            members = set(archive.namelist())
            metadata_members = [
                name
                for name in members
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_members) != 1:
                print(
                    "FAILED: Expected one dist-info/METADATA file, "
                    f"found {len(metadata_members)}"
                )
                return 1
            metadata = Parser().parsestr(
                archive.read(metadata_members[0]).decode("utf-8")
            )
            requires_dist = {
                re.sub(r"[-_.]+", "-", match.group(1).lower())
                for requirement in metadata.get_all("Requires-Dist", [])
                if (match := re.match(
                    r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)",
                    requirement.split(";", 1)[0],
                ))
            }
            assert {"pyyaml", "numpy"} <= requires_dist, (
                "Wheel metadata is missing declared Requires-Dist entries: "
                f"{sorted({'pyyaml', 'numpy'} - requires_dist)}"
            )
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

        # Create venv
        print("\n[2/5] Creating venv...")
        venv_dir = tmpdir / "venv"
        code, _, stderr = run_command([sys.executable, "-m", "venv", str(venv_dir)])
        if code != 0:
            print(f"FAILED: venv\n{stderr}")
            return 1
        pip_exe = venv_dir / "bin" / "pip"
        python_exe = venv_dir / "bin" / "python3"
        print("✓ Created venv")

        # Install wheel
        print("\n[3/5] Installing wheel...")
        code, _, stderr = run_command(
            [str(pip_exe), "install", "--no-index", "--no-deps", str(wheel_path)],
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
import sys
sys.path = [p for p in sys.path if 'quant_projects' not in p]

import factor_optimizer
from factor_optimizer import CapabilityError, ExecutionMode
from factor_optimizer import adapters, contracts, grammar, llm, policy, search, seen
from factor_optimizer.complexity import ComplexityBudget
from factor_optimizer.contracts.splits import SealedTestResult
from factor_optimizer.llm.prompts import PromptTemplate
from factor_optimizer.llm.proposal import ProposalRequest
from factor_optimizer.llm.records import compute_hash

assert ExecutionMode.RESEARCH_ONLY.value == 'research_only'
assert issubclass(CapabilityError, Exception)
assert SealedTestResult.__name__ == 'SealedTestResult'
assert PromptTemplate.__name__ == 'PromptTemplate'
assert ProposalRequest.__name__ == 'ProposalRequest'
assert len(compute_hash('wheel-smoke')) == 64
assert ComplexityBudget(strict=True).is_within_budget(
    __import__('factor_optimizer.complexity.profile', fromlist=['ComplexityProfile'])
    .ComplexityProfile()
)

print(f'factor_optimizer version: {factor_optimizer.__version__}')
print('✓ Three public imports and current module contracts are available')
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
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.search import SearchConfig

budget = SearchBudget(max_trials=10, max_evaluations=50)
config = SearchConfig(budget=budget, plateau_window=20)
print(f"✓ Created config with budget: {budget.max_trials} trials")
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
