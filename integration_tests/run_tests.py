#!/usr/bin/env python3
"""
Quick test runner for integration tests.
Run specific test categories or all tests.
"""

import subprocess
import sys

def run_tests(pattern=None, verbose=True):
    """Run pytest with optional pattern filter."""
    cmd = ["python3", "-m", "pytest"]

    if pattern:
        cmd.append(pattern)
    else:
        cmd.append(".")

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    cmd.extend(["--tb=short"])

    result = subprocess.run(cmd, cwd="/home/shw/quant_projects/integration_tests")
    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        print(f"Running tests matching: {pattern}")
        exit_code = run_tests(pattern)
    else:
        print("Running all integration tests...")
        exit_code = run_tests()

    sys.exit(exit_code)
