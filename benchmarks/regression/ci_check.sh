#!/bin/bash
# CI integration script for performance regression tracking
#
# Usage in CI:
#   ./ci_check.sh [--threshold PERCENT] [--baseline COMMIT]
#
# Environment variables:
#   CI_COMMIT_SHA: Current commit SHA (auto-detected if not set)
#   CI_BASELINE_SHA: Baseline commit SHA (optional)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARKS_DIR="$PROJECT_ROOT/benchmarks"
RESULTS_DIR="$BENCHMARKS_DIR/results"
REGRESSION_DIR="$BENCHMARKS_DIR/regression"

# Parse arguments
THRESHOLD=10.0
BASELINE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use environment variable if baseline not specified
if [[ -z "$BASELINE" && -n "$CI_BASELINE_SHA" ]]; then
    BASELINE="$CI_BASELINE_SHA"
fi

echo "=================================="
echo "Performance Regression CI Check"
echo "=================================="
echo "Threshold: ${THRESHOLD}%"
[[ -n "$BASELINE" ]] && echo "Baseline: $BASELINE"
echo ""

# Step 1: Run benchmarks
echo "[1/4] Running benchmark suite..."
cd "$BENCHMARKS_DIR"

CURRENT_RESULTS="$RESULTS_DIR/current_run.json"
python3 run_all_benchmarks.py --output "$CURRENT_RESULTS"

if [[ ! -f "$CURRENT_RESULTS" ]]; then
    echo "❌ Benchmark run failed - no results generated"
    exit 1
fi

echo "✓ Benchmarks completed"
echo ""

# Step 2: Store results
echo "[2/4] Storing results..."
cd "$REGRESSION_DIR"

python3 tracker.py "$CURRENT_RESULTS" --results-dir "$RESULTS_DIR"

if [[ $? -ne 0 ]]; then
    echo "❌ Failed to store results"
    exit 1
fi

echo "✓ Results stored"
echo ""

# Step 3: Compare with baseline
echo "[3/4] Comparing with baseline..."

COMPARISON_JSON="$RESULTS_DIR/comparison.json"
BASELINE_ARG=""
[[ -n "$BASELINE" ]] && BASELINE_ARG="--baseline $BASELINE"

python3 compare.py \
    "$CURRENT_RESULTS" \
    --threshold "$THRESHOLD" \
    --results-dir "$RESULTS_DIR" \
    --json-output "$COMPARISON_JSON" \
    --fail-on-regression \
    $BASELINE_ARG

COMPARE_EXIT=$?

if [[ $COMPARE_EXIT -eq 0 ]]; then
    echo "✓ No regressions detected"
else
    echo "⚠️  Regressions detected (exit code: $COMPARE_EXIT)"
fi

echo ""

# Step 4: Generate reports
echo "[4/4] Generating reports..."

if [[ -f "$COMPARISON_JSON" ]]; then
    HTML_REPORT="$RESULTS_DIR/regression_report.html"
    MD_REPORT="$RESULTS_DIR/regression_report.md"

    python3 report.py \
        "$COMPARISON_JSON" \
        --html "$HTML_REPORT" \
        --markdown "$MD_REPORT" \
        --results-dir "$RESULTS_DIR"

    if [[ $? -eq 0 ]]; then
        echo "✓ Reports generated:"
        echo "  - HTML: $HTML_REPORT"
        echo "  - Markdown: $MD_REPORT"
    else
        echo "⚠️  Report generation failed"
    fi
else
    echo "⚠️  No comparison data available - skipping report generation"
fi

echo ""
echo "=================================="

# Exit with comparison result
exit $COMPARE_EXIT
