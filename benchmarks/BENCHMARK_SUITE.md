# Quantitative Platform Benchmark Suite

Comprehensive performance benchmarking for quantitative analysis components.

## Overview

This benchmark suite measures performance characteristics across four key areas of the quantitative platform:

### 1. QE (Quant Evaluator) - Metric Computation at Scale
Tests the performance of computing quantitative metrics used for factor evaluation.

**Operations tested:**
- Rank IC (Information Coefficient) computation
- Rank ICIR (IC Information Ratio) computation
- Correlation analysis between factors and forward returns

**Scales:**
- Small: 100 factors
- Medium: 1,000 factors  
- Large: 10,000 factors

**Fixed dimensions:**
- 252 dates (1 trading year)
- 1,000 assets per cross-section

**Metrics reported:**
- Elapsed time (seconds)
- Throughput (factors/second)
- Per-factor latency (milliseconds)

### 2. FP (Factor Processing) - Panel Transforms
Tests performance of panel data transformations commonly used in factor engineering.

**Operations tested:**
- Time-series: rolling mean, rolling std, rolling z-score (window=20)
- Cross-sectional: rank, z-score normalization

**Scales:**
- Small: 100 assets × 252 dates = 25,200 cells
- Medium: 1,000 assets × 756 dates = 756,000 cells
- Large: 10,000 assets × 2,520 dates = 25,200,000 cells

**Metrics reported:**
- Elapsed time per transform (seconds)
- Throughput (millions of cells/second)

### 3. FA (Fundamental Analysis) - Financial Operations
Tests performance of fundamental data processing and point-in-time operations.

**Operations tested:**
- Financial ratio computation (margins, ROA, ROE, asset turnover)
- Point-in-time joins (daily prices + quarterly fundamentals)

**Scales:**
- Small: 1,000 assets × 20 quarters = 20,000 records
- Medium: 10,000 assets × 20 quarters = 200,000 records
- Large: 100,000 assets × 20 quarters = 2,000,000 records

**Metrics reported:**
- Elapsed time per operation (seconds)
- Throughput (thousands of rows/second)

### 4. FO (Factor Optimization) - Search Operations
Tests performance of factor search and optimization operations.

**Operations tested:**
- Candidate generation (mutation)
- Deduplication (seen cache lookups)

**Scales:**
- Small: 100 trials
- Medium: 1,000 trials
- Large: 10,000 trials

**Metrics reported:**
- Elapsed time per operation (seconds)
- Throughput (trials/second)
- Deduplication rate (%)
- Unique candidate count

## Files

### Core Benchmarks
- `standalone_benchmark.py` - Complete self-contained benchmark suite
- `bench_qe_metrics.py` - QE metrics benchmark (requires quant_evaluator)
- `bench_fp_transforms.py` - FP transforms benchmark (requires factor_engine)
- `bench_fa_operations.py` - FA operations benchmark (standalone)
- `bench_fo_search.py` - FO search benchmark (standalone)

### Runners
- `run_all_benchmarks.py` - Orchestrates all four benchmark modules

### Output
- `baseline.json` - Structured benchmark results
- `baseline.txt` - Human-readable summary report

## Usage

### Run Complete Standalone Suite
```bash
cd /home/shw/quant_projects/benchmarks
python3 standalone_benchmark.py
```

This runs all four benchmark categories and generates `baseline.json`.

### Run Individual Benchmarks
```bash
# Factor Optimization (fastest, ~1 second)
python3 bench_fo_search.py

# Fundamental Analysis (~30-60 seconds)
python3 bench_fa_operations.py

# Factor Processing (~2-5 minutes)
python3 bench_fp_transforms.py

# Quant Evaluator (~5-10 minutes for 10K factors)
python3 bench_qe_metrics.py
```

### Run Integrated Suite (requires all dependencies)
```bash
python3 run_all_benchmarks.py
```

## Benchmark Results Structure

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-08-14T...",
  "platform": "quantitative_analysis",
  "benchmarks": {
    "qe": {
      "benchmark": "qe_metrics",
      "description": "Rank IC and ICIR computation at scale",
      "total_time_s": 123.45,
      "results": [
        {
          "n_factors": 100,
          "n_dates": 252,
          "n_assets": 1000,
          "elapsed_s": 12.5,
          "throughput_factors_per_s": 8.0,
          "per_factor_ms": 125.0
        }
      ]
    },
    "fp": {...},
    "fa": {...},
    "fo": {...}
  }
}
```

## Expected Performance

Reference hardware: 8-core CPU, 32GB RAM

| Benchmark | Scale | Expected Time | Throughput |
|-----------|-------|---------------|------------|
| **QE** | 100 factors | 10-30s | 3-10 factors/s |
| **QE** | 1K factors | 100-300s | 3-10 factors/s |
| **QE** | 10K factors | 1000-3000s | 3-10 factors/s |
| **FP** | Small (25K cells) | 0.1-0.5s | 50-250K cells/s |
| **FP** | Medium (756K cells) | 1-5s | 150K-750K cells/s |
| **FP** | Large (25M cells) | 30-150s | 170K-850K cells/s |
| **FA** | 1K assets | 0.1-0.5s | 20-100K rows/s |
| **FA** | 10K assets | 1-5s | 20-100K rows/s |
| **FA** | 100K assets | 10-50s | 20-100K rows/s |
| **FO** | 100 trials | <0.1s | 5K-50K trials/s |
| **FO** | 1K trials | 0.1-0.5s | 2K-10K trials/s |
| **FO** | 10K trials | 1-5s | 2K-10K trials/s |

## Performance Characteristics

### QE: Computation-Bound
- Dominated by rank computation and correlation analysis
- Scales linearly with number of factors
- Each factor requires O(n_dates × n_assets × log(n_assets)) operations
- Memory: O(n_assets) per date slice

### FP: Memory-Bandwidth Bound
- Time-series operations (rolling): O(n_assets × n_dates)
- Cross-sectional operations: O(n_dates × n_assets × log(n_assets))
- Benefits from vectorization (NumPy/Pandas)
- Large datasets may exceed L3 cache

### FA: I/O and Join-Bound
- Financial ratios: trivially parallelizable, very fast
- PIT joins: O(n_assets × n_dates × log(n_quarters))
- Dominated by sort operations
- Memory: O(n_assets × n_dates) for daily data

### FO: Hash-Table Bound
- Generation: O(1) per candidate
- Deduplication: O(1) average case with good hash function
- High deduplication rates (>80%) indicate search space saturation
- Very fast; typically completes in seconds

## Dependencies

### Standalone Benchmark
- numpy
- pandas

### Full Suite
- numpy
- pandas
- quant_evaluator (for QE)
- factor_engine (for FP)
- dataaccess (for FA, optional)
- factor_optimizer (for FO)

## Notes

- All benchmarks include warmup runs to ensure JIT compilation
- Garbage collection is explicitly called between runs
- Large-scale benchmarks (10K+ assets) require significant RAM
- PIT join benchmark skips large scale due to memory constraints
- Backend comparison (pandas vs polars) available in FP benchmark

## Interpreting Results

### Good Performance Indicators
- QE: >5 factors/s
- FP: >100K cells/s for rolling operations
- FA: >50K rows/s for financial ratios
- FO: >1K trials/s with <90% dedup rate

### Red Flags
- Throughput decreases with scale (should be roughly constant)
- Very high dedup rates (>95%) indicate search exhaustion
- Large variance between runs indicates system contention
- Memory growth during benchmarks indicates memory leaks

## Future Enhancements

- GPU acceleration benchmarks
- Multi-process parallelization tests
- Cache efficiency profiling
- Memory usage tracking
- Comparison with alternative backends (Polars, DuckDB)
- Network I/O benchmarks for distributed computation
