# Performance Regression Tracking - Quick Start

Complete performance regression detection system for benchmark suites.

## Installation

No external dependencies required beyond standard Python libraries and existing project dependencies.

```bash
cd /home/shw/quant_projects/benchmarks/regression
chmod +x ci_check.sh
```

## Quick Start

### Option 1: Run Complete CI Check (Recommended)

```bash
cd /home/shw/quant_projects/benchmarks/regression
./ci_check.sh
```

This will:
1. Run all benchmarks
2. Store results with git metadata
3. Compare against baseline
4. Generate HTML and Markdown reports
5. Exit with error if regressions > 10% detected

### Option 2: Manual Workflow

```bash
# Step 1: Run benchmarks
cd /home/shw/quant_projects/benchmarks
python3 run_all_benchmarks.py --output baseline.json

# Step 2: Store results
cd regression
python3 tracker.py ../baseline.json

# Step 3: Make code changes, then run benchmarks again
cd ..
python3 run_all_benchmarks.py --output current.json

# Step 4: Compare against baseline
cd regression
python3 compare.py ../current.json --fail-on-regression --json-output ../results/comparison.json

# Step 5: Generate reports
python3 report.py ../results/comparison.json \
  --html ../results/report.html \
  --markdown ../results/report.md
```

## Usage Examples

### Run Examples
```bash
cd /home/shw/quant_projects/benchmarks/regression
python3 example_usage.py
```

### Custom Threshold
```bash
# More strict (5% threshold)
./ci_check.sh --threshold 5.0

# More relaxed (20% threshold)
./ci_check.sh --threshold 20.0
```

### Compare Against Specific Baseline
```bash
# Compare against specific commit
./ci_check.sh --baseline abc123

# Compare against tag
./ci_check.sh --baseline v1.0.0
```

### View Historical Results
```bash
# List all stored benchmarks
ls ../results/*.jsonl

# View specific benchmark history
cat ../results/qe_100.jsonl | python3 -m json.tool
```

## CI Integration

### GitHub Actions

Copy `.github_workflows_example.yml` to `.github/workflows/performance.yml`:

```bash
mkdir -p .github/workflows
cp /home/shw/quant_projects/benchmarks/regression/.github_workflows_example.yml \
   .github/workflows/performance.yml
```

Features:
- ✅ Automatic benchmark execution on push/PR
- ✅ Compare against baseline commit
- ✅ Fail CI if regression > threshold
- ✅ Upload reports as artifacts
- ✅ Comment on PRs with performance summary

### GitLab CI

```yaml
performance:
  stage: test
  script:
    - cd benchmarks/regression
    - ./ci_check.sh --threshold 10.0 --baseline $CI_MERGE_REQUEST_DIFF_BASE_SHA
  artifacts:
    when: always
    paths:
      - benchmarks/results/regression_report.html
      - benchmarks/results/regression_report.md
    expire_in: 30 days
  only:
    - merge_requests
    - main
```

## Directory Structure

```
benchmarks/
├── regression/
│   ├── __init__.py              # Package initialization
│   ├── tracker.py               # Store results (CLI + library)
│   ├── compare.py               # Detect regressions (CLI + library)
│   ├── report.py                # Generate reports (CLI + library)
│   ├── ci_check.sh              # Complete CI workflow
│   ├── test_regression.py       # Test suite (15 tests)
│   ├── example_usage.py         # Usage examples
│   ├── README.md                # Full documentation
│   ├── QUICKSTART.md            # This file
│   └── .github_workflows_example.yml
├── results/
│   ├── *.jsonl                  # Per-benchmark history
│   ├── current_run.json         # Latest benchmark run
│   ├── comparison.json          # Latest comparison
│   ├── regression_report.html   # Interactive dashboard
│   └── regression_report.md     # PR comment format
└── run_all_benchmarks.py        # Main benchmark runner
```

## Output Files

### Stored Results (`results/*.jsonl`)
One JSON object per line with git metadata:
```json
{
  "commit_sha": "797c21a6...",
  "commit_date": "2026-08-14 01:13:01 +0800",
  "branch": "main",
  "author": "Developer",
  "benchmark_name": "qe_100",
  "metric_name": "mean_time",
  "value": 2.5,
  "unit": "seconds",
  "timestamp": "2026-08-14T01:17:44.373136",
  "metadata": {}
}
```

### Comparison Results (`results/comparison.json`)
```json
{
  "threshold_pct": 10.0,
  "total_comparisons": 15,
  "regressions": 2,
  "improvements": 3,
  "comparisons": [...]
}
```

### HTML Report (`results/regression_report.html`)
- Interactive dashboard with Chart.js
- Time series performance trends
- Regression/improvement tables
- Dark theme, responsive layout

### Markdown Report (`results/regression_report.md`)
- Summary statistics
- Regression/improvement tables
- Perfect for PR comments

## Testing

Run the test suite:
```bash
cd /home/shw/quant_projects/benchmarks/regression
python3 -m pytest test_regression.py -v
```

Expected output: **15 passed**

## Troubleshooting

### "No baseline found for comparison"
This is expected on first run. Solution:
```bash
# Run twice to establish baseline
./ci_check.sh  # First run
# ... make changes ...
./ci_check.sh  # Second run will compare
```

### Git metadata shows "unknown"
Tracker requires git repository. Solution:
```bash
git init  # If not in a git repo
git add .
git commit -m "Initial commit"
```

### High false positive rate
Increase threshold or improve benchmark stability:
```bash
./ci_check.sh --threshold 15.0  # More tolerant
```

## Performance

- **Storage**: ~300 bytes per result, append-only JSONL
- **Lookup**: O(1) file lookup, O(n) scan for baseline
- **Comparison**: O(n) where n = number of metrics
- **Report generation**: O(n*m) where n = metrics, m = historical points

## API Usage

Can be used as a library:

```python
from regression import BenchmarkTracker, BenchmarkComparator, RegressionReporter

# Store results
tracker = BenchmarkTracker()
tracker.store_result("my_bench", "time", 1.5, "seconds")

# Compare
comparator = BenchmarkComparator(tracker, regression_threshold_pct=10.0)
comparisons = comparator.compare_suite(current_data)

# Generate report
reporter = RegressionReporter(tracker)
reporter.generate_html_report(comparisons, Path("report.html"))
```

## Next Steps

1. Run the example: `python3 example_usage.py`
2. Run tests: `pytest test_regression.py -v`
3. Try CI check: `./ci_check.sh`
4. View the full README.md for detailed documentation
5. Integrate with your CI pipeline

## Support

- Full documentation: [README.md](README.md)
- Test suite: `test_regression.py`
- Examples: `example_usage.py`
- CI template: `.github_workflows_example.yml`
