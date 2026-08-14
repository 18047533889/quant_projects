# Benchmark Suite README

## Overview

Comprehensive performance benchmarks and stress tests for all quant_projects packages.

## Quick Start

```bash
cd /home/shw/quant_projects/benchmarks

# Run all benchmarks
./run_benchmarks.sh --full

# Run quick benchmarks only (small scales)
./run_benchmarks.sh --quick

# Run with visualization
./run_benchmarks.sh --full --visualize

# Check for regressions
./run_benchmarks.sh --regress
```

## Benchmark Coverage

### 1. Quant Evaluator (`bench_qe_ic_computation.py`)
- IC computation at multiple scales
- Tests: 100 / 1,000 / 10,000 / 10,000×5 factors
- Metrics: throughput (factors/s), latency (ms/factor)

### 2. Research Control (`bench_rc_ledger.py`)
- Campaign and trial write performance
- Query performance (full scan, top-K, filtered)
- Concurrent write simulation
- Scales: 100 / 1K / 10K / 100K records

### 3. Factor Optimizer (`bench_fo_search_optimized.py`)
- Mutation generation
- Deduplication performance
- Complexity profiling
- Validation throughput
- Scales: 1K / 10K / 100K operations

### 4. Factor Assets (`bench_fa_similarity.py`)
- Exact similarity search
- Clustering (k-means)
- Representative selection
- Batch deduplication
- Scales: 1K / 10K / 100K factors

### 5. Factor Preprocess (`bench_fp_preprocess.py`)
- Zscore standardization (pandas, polars)
- Rank transformation
- Industry neutralization
- Winsorization
- Scales: 100 / 1K / 3K assets × 252-504 dates

### 6. Stress Tests (`stress_tests.py`)
- Memory pressure (large allocations)
- Repeated allocations (leak detection)
- Long-running stability (60s)
- High concurrency simulation
- Large dataset processing

## Directory Structure

```
benchmarks/
├── bench_qe_ic_computation.py      # Quant Evaluator
├── bench_rc_ledger.py              # Research Control
├── bench_fo_search_optimized.py    # Factor Optimizer
├── bench_fa_similarity.py          # Factor Assets
├── bench_fp_preprocess.py          # Factor Preprocess
├── stress_tests.py                 # Stress tests
├── run_all_benchmarks_v2.py        # Master runner
├── detect_regression.py            # Regression detection
├── visualize_performance.py        # Chart generation
├── ci_benchmark.py                 # CI/CD integration
├── run_benchmarks.sh               # Shell runner
├── README_BENCHMARKS.md            # This file
└── results/
    ├── latest_results.json         # Latest run
    ├── baseline.json               # Baseline for comparison
    ├── benchmark_results_*.json    # Historical results
    ├── benchmark_summary_*.txt     # Human-readable summaries
    └── charts/                     # Performance visualizations
        ├── throughput_comparison.png
        ├── *_scaling.png
        ├── memory_usage.png
        └── latency_distribution.png
```

## Running Individual Benchmarks

```bash
# Quant Evaluator
python bench_qe_ic_computation.py

# Research Control
python bench_rc_ledger.py

# Factor Optimizer
python bench_fo_search_optimized.py

# Factor Assets
python bench_fa_similarity.py

# Factor Preprocess
python bench_fp_preprocess.py

# Stress Tests
python stress_tests.py
```

## Running All Benchmarks

```bash
# Python script (recommended)
python run_all_benchmarks_v2.py

# Shell script with options
./run_benchmarks.sh --full --visualize
```

## Regression Detection

```bash
# Check for performance regressions
python detect_regression.py

# Expected output:
# - No regressions: exit code 0
# - Regressions detected: exit code 1, lists affected metrics
```

**Regression Thresholds:**
- Throughput: > 10% decrease
- Latency: > 10% increase

## Visualization

```bash
# Generate performance charts
python visualize_performance.py

# Output: results/charts/*.png
```

**Generated Charts:**
- `throughput_comparison.png` - Throughput across all benchmarks
- `*_scaling.png` - Scaling behavior for each benchmark
- `memory_usage.png` - Memory usage from stress tests
- `latency_distribution.png` - Operation latency distribution

## CI/CD Integration

```bash
# Run in CI pipeline
python ci_benchmark.py

# Workflow:
# 1. Run benchmarks
# 2. Check for regressions (if baseline exists)
# 3. Generate visualizations
# 4. Report results
# 5. Exit with code 0 (pass) or 1 (fail)
```

### GitHub Actions Example

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e ./quant_evaluator
          pip install -e ./research_control
          pip install -e ./factor_optimizer
          pip install -e ./factor_assets
          pip install -e ./factor_preprocess
          pip install matplotlib psutil
      
      - name: Run benchmarks
        run: |
          cd benchmarks
          python ci_benchmark.py
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmarks/results/
```

## Establishing Baseline

First run establishes the baseline:

```bash
# Run benchmarks
python run_all_benchmarks_v2.py

# Copy to baseline
cp results/latest_results.json results/baseline.json
```

## Updating Baseline

After intentional performance improvements:

```bash
# Run benchmarks
python run_all_benchmarks_v2.py

# Verify improvements
python detect_regression.py

# Update baseline
cp results/latest_results.json results/baseline.json
```

## Expected Runtime

**Quick Mode (~5-10 minutes):**
- Small scales only
- Fast iteration

**Full Mode (~15-30 minutes):**
- All scales including large
- Stress tests
- Comprehensive coverage

**Hardware:**
- 8-core CPU, 32GB RAM
- Times will vary with hardware

## Interpreting Results

### JSON Output Structure

```json
{
  "schema_version": "2.0",
  "generated_at": "2026-08-14T...",
  "platform": "quant_projects",
  "benchmarks": {
    "bench_qe_ic_computation": {
      "status": "SUCCESS",
      "wall_time_s": 123.45,
      "results": {
        "small": {
          "n_factors": 100,
          "elapsed_seconds": 2.5,
          "throughput_factors_per_sec": 40.0
        }
      }
    }
  },
  "stress_tests": {...},
  "total_wall_time_s": 1234.56
}
```

### Key Metrics

**Throughput:**
- Higher is better
- ops/s, factors/s, queries/s

**Latency:**
- Lower is better
- Milliseconds per operation

**Memory:**
- Stable usage (no leaks)
- Proper cleanup after operations

### Success Criteria

✓ All benchmarks complete without errors  
✓ Throughput within target ranges  
✓ No memory leaks detected  
✓ No regressions vs baseline

## Troubleshooting

### Import Errors

```bash
# Ensure packages installed
cd /home/shw/quant_projects
pip install -e ./quant_evaluator
pip install -e ./research_control
pip install -e ./factor_optimizer
pip install -e ./factor_assets
pip install -e ./factor_preprocess
```

### Memory Errors

- Reduce stress test scales in `stress_tests.py`
- Close other applications
- Ensure sufficient RAM (recommended: 16GB+)

### Timeouts

- Expected for large scales
- Adjust timeout in runner scripts
- Use `--quick` mode for faster iteration

### Inconsistent Results

- Run on idle system (minimal background load)
- Disable CPU frequency scaling
- Run multiple times and average

## Performance Targets

See [PERFORMANCE_BASELINE.md](../PERFORMANCE_BASELINE.md) for detailed targets.

**Quick Reference:**
- QE: 30-100 factors/s
- RC: 1K-3K writes/s, < 50ms queries
- FO: 10K-100K mutations/s
- FA: 3-100 queries/s (scale dependent)
- FP: 0.1-5 Mcells/s

## Adding New Benchmarks

1. Create benchmark script following naming convention:
   - `bench_<package>_<feature>.py`

2. Implement standard interface:
   ```python
   def main():
       """Run benchmark and return results dict."""
       return {
           "benchmark": "my_benchmark",
           "description": "...",
           "results": {...},
           "total_time_s": elapsed
       }
   
   if __name__ == "__main__":
       import json
       result = main()
       print(json.dumps(result, indent=2))
   ```

3. Add to `run_all_benchmarks_v2.py`:
   ```python
   BENCHMARKS = [
       ...,
       ("bench_my_new_benchmark.py", "My New Benchmark"),
   ]
   ```

4. Run and establish baseline

## Maintenance

**Weekly:**
- Review latest benchmark results
- Check for anomalies

**After Major Changes:**
- Run full benchmark suite
- Check for regressions
- Update baseline if intended

**Quarterly:**
- Review and update performance targets
- Optimize underperforming areas
- Document improvements

## References

- Main documentation: [PERFORMANCE_BASELINE.md](../PERFORMANCE_BASELINE.md)
- Project root: `/home/shw/quant_projects`
- Benchmark directory: `/home/shw/quant_projects/benchmarks`

## Contact

For questions or issues with benchmarks, consult project documentation or team.
