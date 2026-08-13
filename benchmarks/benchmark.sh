#!/bin/bash
# Quick reference for running benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================="
echo "Benchmark Quick Reference"
echo "==================================="
echo ""
echo "Available benchmarks:"
echo "  1. standalone_benchmark.py   - Complete self-contained suite (recommended)"
echo "  2. bench_fo_search.py        - Factor Optimization only (~1 second)"
echo "  3. bench_fa_operations.py    - Fundamental Analysis only (~1 minute)"
echo "  4. bench_fp_transforms.py    - Factor Processing only (~2-5 minutes, requires factor_engine)"
echo "  5. bench_qe_metrics.py       - Quant Evaluator only (~5-10 minutes, requires quant_evaluator)"
echo "  6. run_all_benchmarks.py     - Orchestrated suite (requires all dependencies)"
echo ""
echo "Usage:"
echo "  ./benchmark.sh <number>    - Run specific benchmark"
echo "  ./benchmark.sh all         - Run standalone suite"
echo ""

if [ $# -eq 0 ]; then
    echo "No arguments provided. Use './benchmark.sh all' or './benchmark.sh <1-6>'"
    exit 0
fi

case "$1" in
    1|standalone)
        echo "Running standalone benchmark suite..."
        python3 standalone_benchmark.py
        ;;
    2|fo)
        echo "Running FO (Factor Optimization) benchmark..."
        python3 bench_fo_search.py
        ;;
    3|fa)
        echo "Running FA (Fundamental Analysis) benchmark..."
        python3 bench_fa_operations.py
        ;;
    4|fp)
        echo "Running FP (Factor Processing) benchmark..."
        python3 bench_fp_transforms.py
        ;;
    5|qe)
        echo "Running QE (Quant Evaluator) benchmark..."
        python3 bench_qe_metrics.py
        ;;
    6|integrated)
        echo "Running integrated benchmark suite..."
        python3 run_all_benchmarks.py
        ;;
    all)
        echo "Running standalone benchmark suite..."
        python3 standalone_benchmark.py
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use: ./benchmark.sh <1-6|all|standalone|fo|fa|fp|qe|integrated>"
        exit 1
        ;;
esac

echo ""
echo "==================================="
echo "Results written to baseline.json"
echo "==================================="
