#!/bin/bash
# Comprehensive Benchmark Suite Runner
#
# Usage:
#   ./run_benchmarks.sh [options]
#
# Options:
#   --quick     Run only small scales (fast)
#   --full      Run all scales including stress tests (slow)
#   --visualize Generate performance charts
#   --regress   Check for performance regressions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
RUN_QUICK=0
RUN_FULL=1
RUN_VISUALIZE=0
RUN_REGRESS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            RUN_QUICK=1
            RUN_FULL=0
            shift
            ;;
        --full)
            RUN_FULL=1
            shift
            ;;
        --visualize)
            RUN_VISUALIZE=1
            shift
            ;;
        --regress)
            RUN_REGRESS=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--full] [--visualize] [--regress]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "COMPREHENSIVE BENCHMARK SUITE"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Quick mode: $RUN_QUICK"
echo "  Full mode: $RUN_FULL"
echo "  Visualize: $RUN_VISUALIZE"
echo "  Regression check: $RUN_REGRESS"
echo ""

# Ensure results directory exists
mkdir -p results/charts

if [ $RUN_QUICK -eq 1 ]; then
    echo "Running QUICK benchmarks (small scales only)..."
    echo ""

    echo "1. Quant Evaluator IC Computation..."
    python bench_qe_ic_computation.py || echo "  [WARNING] Failed"

    echo ""
    echo "2. Research Control Ledger..."
    python bench_rc_ledger.py || echo "  [WARNING] Failed"

    echo ""
    echo "3. Factor Optimizer Search..."
    python bench_fo_search_optimized.py || echo "  [WARNING] Failed"

    echo ""
    echo "4. Factor Assets Similarity..."
    python bench_fa_similarity.py || echo "  [WARNING] Failed"

    echo ""
    echo "5. Factor Preprocess Transforms..."
    python bench_fp_preprocess.py || echo "  [WARNING] Failed"

    echo ""
    echo "✓ Quick benchmarks completed"

elif [ $RUN_FULL -eq 1 ]; then
    echo "Running FULL benchmark suite..."
    echo ""

    python run_all_benchmarks_v2.py

    echo ""
    echo "✓ Full benchmarks completed"
fi

if [ $RUN_VISUALIZE -eq 1 ]; then
    echo ""
    echo "========================================================================"
    echo "GENERATING PERFORMANCE VISUALIZATIONS"
    echo "========================================================================"
    echo ""

    python visualize_performance.py

    echo ""
    echo "✓ Visualizations completed"
fi

if [ $RUN_REGRESS -eq 1 ]; then
    echo ""
    echo "========================================================================"
    echo "CHECKING FOR PERFORMANCE REGRESSIONS"
    echo "========================================================================"
    echo ""

    python detect_regression.py
    REGRESS_EXIT=$?

    if [ $REGRESS_EXIT -ne 0 ]; then
        echo ""
        echo "⚠️  PERFORMANCE REGRESSIONS DETECTED"
        echo ""
        exit 1
    else
        echo ""
        echo "✓ No regressions detected"
    fi
fi

echo ""
echo "========================================================================"
echo "BENCHMARK SUITE COMPLETED"
echo "========================================================================"
echo ""
echo "Results available in: results/"
echo ""
echo "Next steps:"
echo "  - Review results/latest_results.json"
echo "  - Check results/charts/ for visualizations"
echo "  - Run './run_benchmarks.sh --regress' to check for regressions"
echo ""
