#!/usr/bin/env python3
"""
Test runner for all examples.
Runs each example and verifies it completes successfully.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_example(example_file: Path) -> tuple[bool, str, float]:
    """
    Run an example and return (success, output, runtime).
    """
    start = datetime.now()

    try:
        result = subprocess.run(
            ["python3", str(example_file)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=example_file.parent,
        )

        elapsed = (datetime.now() - start).total_seconds()

        if result.returncode == 0:
            return True, result.stdout, elapsed
        else:
            return False, result.stderr, elapsed

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start).total_seconds()
        return False, "Timeout after 30 seconds", elapsed
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        return False, str(e), elapsed


def main():
    """Run all examples and report results."""

    examples_dir = Path(__file__).parent

    examples = [
        ("01_basic_evaluation.py", "Factor Evaluation (QE)"),
        ("02_preprocessing_pipeline.py", "Preprocessing Pipeline (FP)"),
        ("03_factor_selection.py", "Factor Selection (FA)"),
        ("04_optimization_search.py", "Optimization Search (FO)"),
        ("05_complete_workflow.py", "Complete Workflow (All)"),
    ]

    print("=" * 70)
    print("Running All Examples")
    print("=" * 70)
    print()

    results = []

    for example_file, description in examples:
        example_path = examples_dir / example_file

        print(f"Running: {example_file}")
        print(f"  Description: {description}")

        if not example_path.exists():
            print(f"  ✗ File not found")
            results.append((example_file, False, 0))
            print()
            continue

        success, output, elapsed = run_example(example_path)

        if success:
            print(f"  ✓ Success ({elapsed:.1f}s)")
            results.append((example_file, True, elapsed))
        else:
            print(f"  ✗ Failed ({elapsed:.1f}s)")
            print(f"  Error: {output[:200]}")
            results.append((example_file, False, elapsed))

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    total_time = sum(elapsed for _, _, elapsed in results)

    print(f"Results: {passed}/{total} passed")
    print(f"Total runtime: {total_time:.1f}s")
    print()

    print("Individual results:")
    for example_file, success, elapsed in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {example_file:<35} ({elapsed:.1f}s)")

    print()

    if passed == total:
        print("All examples completed successfully!")
        return 0
    else:
        print(f"Warning: {total - passed} example(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
