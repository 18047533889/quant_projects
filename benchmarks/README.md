# Benchmarks

Comprehensive benchmark suite for quantitative platform components.

## Overview

Four benchmark modules:

1. **QE (Quant Evaluator)** - Metric computation at scale
2. **FP (Factor Processing)** - Panel transforms and backend comparison
3. **FA (Fundamental Analysis)** - Point-in-time operations and financial computations
4. **FO (Factor Optimization)** - Search, mutation, and validation

## Quick Start

```bash
cd /home/shw/quant_projects/benchmarks

# Run all benchmarks
python run_all_benchmarks.py

# Run individual benchmarks
python bench_qe_metrics.py
python bench_fp_transforms.py
python bench_fa_operations.py
python bench_fo_search.py
```

## Benchmark Details

### QE: Quant Evaluator Metrics
- **Scales**: 100 / 1,000 / 10,000 factors
- **Metrics**: rank_ic, rank_icir, coverage, turnover
- **Data**: 252 dates × 1,000 assets per factor
- **Output**: Throughput (factors/s), latency (ms/factor)

### FP: Factor Processing Transforms
- **Small**: 100 assets × 252 dates (25.2K cells)
- **Medium**: 1,000 assets × 756 dates (756K cells)
- **Large**: 10,000 assets × 2,520 dates (25.2M cells)
- **Operations**: ts_mean, ts_std, ts_zscore, ts_corr, rank, zscore, vwap, volatility
- **Backends**: pandas, polars, polars_long
- **Output**: Throughput (Mcells/s), elapsed time per transform

### FA: Fundamental Analysis Operations
- **Scales**: 1K / 10K / 100K assets × 20 quarters
- **Operations**:
  - Financial ratios (margins, ROA, ROE, leverage)
  - Growth rates (YoY, QoQ)
  - Rolling aggregations (TTM, trailing averages)
  - Point-in-time joins (price + fundamentals)
- **Output**: Throughput (Krows/s), operation latency

### FO: Factor Optimization Search
- **Scales**: 100 / 1,000 / 10,000 trials
- **Operations**:
  - Mutation generation
  - Deduplication (seen cache)
  - Validation
  - Complexity profiling
- **Output**: Throughput (trials/s), deduplication rate, validation rate

## Output

Running `run_all_benchmarks.py` generates:

- `baseline.json` - Complete benchmark results in structured JSON
- `baseline.txt` - Human-readable summary report

### baseline.json Structure

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-08-14T...",
  "platform": "quantitative_analysis",
  "benchmarks": {
    "qe": {
      "benchmark": "qe_metrics",
      "description": "...",
      "total_time_s": 123.45,
      "results": [...]
    },
    "fp": {...},
    "fa": {...},
    "fo": {...}
  }
}
```

## Performance Characteristics

Expected performance (reference hardware: 8-core CPU, 32GB RAM):

| Benchmark | Scale | Expected Time | Throughput |
|-----------|-------|---------------|------------|
| QE | 1K factors | ~10-30s | 30-100 factors/s |
| QE | 10K factors | ~100-300s | 30-100 factors/s |
| FP | Small | ~1-5s | 5-25 Mcells/s |
| FP | Medium | ~5-30s | 25-150 Kcells/s |
| FP | Large | ~50-300s | 50-500 Kcells/s |
| FA | 1K assets | ~0.1-0.5s | 20-100 Krows/s |
| FA | 100K assets | ~10-50s | 20-100 Krows/s |
| FO | 1K trials | ~0.1-1s | 1K-10K trials/s |
| FO | 10K trials | ~1-10s | 1K-10K trials/s |

## Dependencies

- numpy
- pandas
- quant_evaluator (for QE benchmarks)
- factor_engine + cleaned_operators (for FP benchmarks)
- dataaccess (optional, for FA PIT operations)
- factor_optimizer (for FO benchmarks)

## Notes

- Benchmarks include warmup runs to ensure JIT compilation and cache warming
- GC is explicitly called between runs for consistent measurements
- Large-scale benchmarks may require significant memory (>16GB for 100K assets)
- Backend comparison (FP) tests pandas vs polars performance
- PIT join benchmark (FA) only runs for small/medium scales due to memory constraints
