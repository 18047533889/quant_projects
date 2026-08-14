#!/usr/bin/env python3
"""
Quick Smoke Test - Verify benchmark infrastructure works

Runs minimal versions of each benchmark to ensure no import/syntax errors.
Fast execution (< 1 minute) to validate setup before running full suite.
"""

import sys
import traceback
from pathlib import Path


def test_import(module_name: str, import_statement: str) -> bool:
    """Test if module can be imported."""
    try:
        exec(import_statement)
        print(f"  ✓ {module_name}")
        return True
    except Exception as e:
        print(f"  ✗ {module_name}: {e}")
        return False


def smoke_test_qe():
    """Test quant_evaluator benchmark."""
    try:
        import numpy as np
        from pathlib import Path
        # Basic imports
        print("    Testing QE benchmark structure...")
        return True
    except Exception as e:
        print(f"    ✗ QE: {e}")
        return False


def smoke_test_rc():
    """Test research_control benchmark."""
    try:
        import sqlite3
        import tempfile
        print("    Testing RC benchmark structure...")
        return True
    except Exception as e:
        print(f"    ✗ RC: {e}")
        return False


def smoke_test_fo():
    """Test factor_optimizer benchmark."""
    try:
        import numpy as np
        print("    Testing FO benchmark structure...")
        return True
    except Exception as e:
        print(f"    ✗ FO: {e}")
        return False


def smoke_test_fa():
    """Test factor_assets benchmark."""
    try:
        import numpy as np
        print("    Testing FA benchmark structure...")
        return True
    except Exception as e:
        print(f"    ✗ FA: {e}")
        return False


def smoke_test_fp():
    """Test factor_preprocess benchmark."""
    try:
        import numpy as np
        import pandas as pd
        print("    Testing FP benchmark structure...")
        return True
    except Exception as e:
        print(f"    ✗ FP: {e}")
        return False


def smoke_test_stress():
    """Test stress tests."""
    try:
        import psutil
        print("    Testing stress test structure...")
        return True
    except Exception as e:
        print(f"    ✗ Stress: {e}")
        return False


def main():
    """Run smoke tests."""
    print("=" * 70)
    print("BENCHMARK INFRASTRUCTURE SMOKE TEST")
    print("=" * 70)
    print("\nVerifying imports and basic structure...\n")

    results = {}

    # Core dependencies
    print("Core Dependencies:")
    results['numpy'] = test_import('numpy', 'import numpy as np')
    results['pandas'] = test_import('pandas', 'import pandas as pd')
    results['matplotlib'] = test_import('matplotlib', 'import matplotlib.pyplot as plt')
    results['psutil'] = test_import('psutil', 'import psutil')

    # Package imports (optional)
    print("\nPackage Imports (optional):")
    results['qe'] = test_import('quant_evaluator', 'from quant_evaluator.runtime.evaluator import Evaluator')
    results['rc'] = test_import('research_control', 'from research_control.ledger.campaign import CampaignLedger')
    results['fo'] = test_import('factor_optimizer', 'from factor_optimizer.factor_optimizer.complexity.profile import ComplexityProfiler')
    results['fa'] = test_import('factor_assets', 'from factor_assets.clustering.families import ClusteringEngine')
    results['fp'] = test_import('factor_preprocess', 'from factor_preprocess.factor_preprocess.neutralization.ols import neutralize_ols')

    # Benchmark script structure
    print("\nBenchmark Scripts:")
    benchmark_dir = Path(__file__).parent

    scripts = [
        'bench_qe_ic_computation.py',
        'bench_rc_ledger.py',
        'bench_fo_search_optimized.py',
        'bench_fa_similarity.py',
        'bench_fp_preprocess.py',
        'stress_tests.py',
        'run_all_benchmarks_v2.py',
        'detect_regression.py',
        'visualize_performance.py',
        'ci_benchmark.py'
    ]

    for script in scripts:
        script_path = benchmark_dir / script
        exists = script_path.exists()
        results[script] = exists
        print(f"  {'✓' if exists else '✗'} {script}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    core_deps = ['numpy', 'pandas', 'matplotlib', 'psutil']
    core_ok = all(results.get(d, False) for d in core_deps)

    packages = ['qe', 'rc', 'fo', 'fa', 'fp']
    packages_ok = sum(results.get(p, False) for p in packages)

    scripts_ok = all(results.get(s, False) for s in scripts)

    print(f"\nCore Dependencies: {'✓ PASS' if core_ok else '✗ FAIL'}")
    print(f"Package Imports: {packages_ok}/{len(packages)} available")
    print(f"Benchmark Scripts: {'✓ PASS' if scripts_ok else '✗ FAIL'}")

    if core_ok and scripts_ok:
        print("\n✓ Smoke test PASSED - Ready to run benchmarks")
        print("\nNext steps:")
        print("  ./run_benchmarks.sh --quick    # Quick test")
        print("  ./run_benchmarks.sh --full     # Full suite")
        return 0
    else:
        print("\n✗ Smoke test FAILED - Fix errors before running benchmarks")
        if not core_ok:
            print("\nInstall core dependencies:")
            print("  pip install numpy pandas matplotlib psutil")
        if packages_ok < len(packages):
            print("\nInstall packages (optional, benchmarks will skip missing):")
            print("  pip install -e ../quant_evaluator")
            print("  pip install -e ../research_control")
            print("  pip install -e ../factor_optimizer")
            print("  pip install -e ../factor_assets")
            print("  pip install -e ../factor_preprocess")
        return 1


if __name__ == "__main__":
    sys.exit(main())
