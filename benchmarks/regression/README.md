# Performance Regression Tracking

Automated performance regression detection system for benchmark suites.

## Overview

This system tracks benchmark results across git commits, compares performance against baselines, and generates detailed reports with visualizations. It integrates with CI to automatically fail builds when performance regressions exceed a configurable threshold.

## Components

### 1. `tracker.py` - Result Storage
Stores benchmark results with git metadata for historical tracking.

**Features:**
- Automatic git commit metadata extraction (SHA, date, branch, author)
- JSONL storage format for efficient append operations
- Support for multiple benchmark suites (QE, FP, FA, FO)
- Per-benchmark result files for fast lookup

**Usage:**
```bash
# Store results from benchmark run
python tracker.py /path/to/benchmark_results.json

# Specify custom results directory
python tracker.py baseline.json --results-dir ./custom_results/
```

**Storage Format:**
Results are stored in `benchmarks/results/{benchmark_name}.jsonl` with one JSON object per line:
```json
{
  "commit_sha": "abc123...",
  "commit_date": "2026-08-14 10:00:00 +0000",
  "branch": "main",
  "author": "Developer",
  "benchmark_name": "qe_100",
  "metric_name": "mean_time",
  "value": 2.5,
  "unit": "seconds",
  "timestamp": "2026-08-14T10:00:00.123456",
  "metadata": {"throughput": 40.0}
}
```

### 2. `compare.py` - Regression Detection
Compares current results against baseline to detect regressions.

**Features:**
- Configurable regression threshold (default: 10%)
- Automatic metric type detection (time-based vs throughput)
- Baseline selection (most recent or specific commit)
- JSON output for downstream processing
- CI-friendly exit codes

**Usage:**
```bash
# Compare against most recent baseline
python compare.py current_results.json

# Compare against specific commit
python compare.py current_results.json --baseline abc123

# Custom threshold and fail on regression
python compare.py current_results.json \
  --threshold 5.0 \
  --fail-on-regression \
  --json-output comparison.json
```

**Detection Rules:**
- **Time metrics** (elapsed_time, mean_time): Higher is worse
- **Throughput metrics**: Lower is worse
- **Regression**: Performance degrades by > threshold%
- **Improvement**: Performance improves (opposite direction)

### 3. `report.py` - Visualization & Reports
Generates HTML and Markdown reports with performance trend charts.

**Features:**
- Interactive HTML dashboard with Chart.js visualizations
- Time series charts showing performance trends
- Markdown reports for CI/PR comments
- Dark theme consistent with design system
- Responsive layout

**Usage:**
```bash
# Generate HTML report
python report.py comparison.json --html report.html

# Generate Markdown for PR comments
python report.py comparison.json --markdown report.md

# Generate both formats
python report.py comparison.json \
  --html report.html \
  --markdown report.md
```

**Report Contents:**
- Summary statistics (total comparisons, regressions, improvements)
- Detailed regression table with percentage changes
- Performance improvement highlights
- Time series charts for each benchmark
- Commit metadata

### 4. `ci_check.sh` - CI Integration
Bash script that orchestrates the complete regression check workflow.

**Features:**
- Runs full benchmark suite
- Stores results with git metadata
- Compares against baseline
- Generates reports
- Exits with non-zero code on regression

**Usage:**
```bash
# Basic CI check (10% threshold)
./ci_check.sh

# Custom threshold
./ci_check.sh --threshold 5.0

# Compare against specific baseline
./ci_check.sh --baseline abc123def

# CI environment variables
CI_BASELINE_SHA=abc123 ./ci_check.sh
```

**CI Integration Example (GitHub Actions):**
```yaml
name: Performance Regression Check

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Need history for baseline
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install pytest numpy pandas polars
      
      - name: Run regression check
        run: |
          cd benchmarks/regression
          ./ci_check.sh --threshold 10.0
        env:
          CI_BASELINE_SHA: ${{ github.event.before }}
      
      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: performance-reports
          path: |
            benchmarks/results/regression_report.html
            benchmarks/results/regression_report.md
      
      - name: Comment PR
        if: github.event_name == 'pull_request' && always()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync(
              'benchmarks/results/regression_report.md',
              'utf8'
            );
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

## Directory Structure

```
benchmarks/
├── regression/
│   ├── __init__.py           # Package initialization
│   ├── tracker.py            # Result storage
│   ├── compare.py            # Regression detection
│   ├── report.py             # Report generation
│   ├── ci_check.sh           # CI integration script
│   ├── test_regression.py    # Test suite
│   └── README.md             # This file
├── results/
│   ├── *.jsonl               # Per-benchmark historical results
│   ├── current_run.json      # Latest benchmark run
│   ├── comparison.json       # Latest comparison results
│   ├── regression_report.html
│   └── regression_report.md
├── run_all_benchmarks.py     # Main benchmark runner
└── baseline.json             # Current baseline results
```

## Workflow

### Local Development
```bash
# 1. Run benchmarks
cd /home/shw/quant_projects/benchmarks
python run_all_benchmarks.py --output baseline.json

# 2. Store results
cd regression
python tracker.py ../baseline.json

# 3. Make code changes...

# 4. Run benchmarks again
cd ..
python run_all_benchmarks.py --output current.json

# 5. Compare
cd regression
python compare.py ../current.json --fail-on-regression

# 6. Generate report
python report.py ../results/comparison.json \
  --html ../results/report.html \
  --markdown ../results/report.md
```

### CI/CD Pipeline
```bash
# Single command runs entire workflow
cd benchmarks/regression
./ci_check.sh --threshold 10.0
```

## Configuration

### Regression Threshold
Adjust sensitivity by changing the threshold percentage:
- **5%**: Strict - catch small regressions
- **10%**: Default - balanced sensitivity
- **20%**: Relaxed - only major regressions

### Baseline Selection
- **Most recent** (default): Compare against last stored result
- **Specific commit**: Compare against known good baseline
- **Parent commit**: CI can use `git merge-base` or `$CI_COMMIT_BEFORE_SHA`

### Custom Results Directory
Store results in project-specific locations:
```bash
python tracker.py results.json --results-dir /custom/path/
python compare.py results.json --results-dir /custom/path/
```

## Testing

Run the test suite:
```bash
cd /home/shw/quant_projects/benchmarks/regression
pytest test_regression.py -v
```

Test coverage:
- Result storage and retrieval
- Git metadata extraction
- Comparison logic (regression/improvement detection)
- Time series data building
- Report generation (HTML/Markdown)

## Interpreting Results

### Comparison Output
```
🔴 REGRESSIONS DETECTED
  🔴 qe_100/mean_time: 3.000 vs 2.500 (+20.0%)
  🔴 fp_small_zscore/elapsed_time: 0.600 vs 0.500 (+20.0%)

🟢 IMPROVEMENTS
  🟢 fa_large_filter/elapsed_time: 0.800 vs 1.000 (-20.0%)
```

### HTML Report Features
- **Summary cards**: Quick overview with color-coded statistics
- **Regression table**: Detailed breakdown of all regressions
- **Time series charts**: Visual trends over commits
- **Responsive design**: Works on mobile and desktop
- **Dark theme**: Consistent with platform design system

## Troubleshooting

### No baseline found
First run will have no baseline to compare against. This is expected.
```bash
# Solution: Run benchmarks twice
python tracker.py baseline.json  # First run establishes baseline
# ... make changes ...
python compare.py current.json   # Second run compares
```

### Git metadata unavailable
Tracker falls back to "unknown" values in non-git environments.
```bash
# Solution: Run in git repository or set fallback values
git init  # Initialize if needed
```

### Noisy benchmarks
High variance between runs causes false positives.
```bash
# Solution: Increase threshold or improve benchmark stability
./ci_check.sh --threshold 15.0  # More tolerant
```

### Missing dependencies
```bash
# Install required packages
pip install pytest  # For tests
# Chart.js is loaded via CDN in HTML reports (no install needed)
```

## Future Enhancements

- **Statistical significance testing**: Mann-Whitney U test for variance
- **Multi-commit comparison**: Detect trends over N commits
- **Percentile tracking**: P50/P95/P99 instead of mean
- **Automatic bisection**: Git bisect to find regression commit
- **Slack/Email notifications**: Alert on regressions
- **Historical analysis**: Detect long-term trends and patterns
