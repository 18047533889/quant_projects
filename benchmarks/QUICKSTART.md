# Benchmark Suite - Quick Start

## What We Built

Comprehensive benchmark suite for quantitative platform with 4 test categories:

1. **QE (Quant Evaluator)**: Metric computation (IC, ICIR) at 100/1K/10K factor scales
2. **FP (Factor Processing)**: Panel transforms at 25K/756K/25M cell scales  
3. **FA (Fundamental Analysis)**: Financial ops at 1K/10K/100K asset scales
4. **FO (Factor Optimization)**: Search operations at 100/1K/10K trial scales

## Running Benchmarks

### Recommended: Standalone Suite
```bash
cd /home/shw/quant_projects/benchmarks
python3 standalone_benchmark.py
```

This is self-contained, requires only numpy and pandas, and generates `baseline.json`.

### Using Helper Script
```bash
./benchmark.sh all              # Run standalone suite
./benchmark.sh fo               # Factor Optimization only (fast)
./benchmark.sh fa               # Fundamental Analysis only
./benchmark.sh fp               # Factor Processing (requires factor_engine)
./benchmark.sh qe               # Quant Evaluator (requires quant_evaluator)
```

### Individual Benchmarks
```bash
python3 bench_fo_search.py      # ~1 second
python3 bench_fa_operations.py  # ~30-60 seconds
python3 bench_fp_transforms.py  # ~2-5 minutes (needs dependencies)
python3 bench_qe_metrics.py     # ~5-10 minutes (needs dependencies)
```

## Output Files

- `baseline.json` - Structured benchmark results with all metrics
- `baseline.txt` - Human-readable summary (if using run_all_benchmarks.py)

## Benchmark Scales

### QE: Factors × Dates × Assets
- Small: 100 × 252 × 1000 = 25.2M operations
- Medium: 1,000 × 252 × 1000 = 252M operations
- Large: 10,000 × 252 × 1000 = 2.52B operations

### FP: Assets × Dates (panel cells)
- Small: 100 × 252 = 25,200 cells
- Medium: 1,000 × 756 = 756,000 cells
- Large: 10,000 × 2,520 = 25,200,000 cells

### FA: Assets × Quarters
- Small: 1,000 × 20 = 20,000 records
- Medium: 10,000 × 20 = 200,000 records
- Large: 100,000 × 20 = 2,000,000 records

### FO: Search Trials
- Small: 100 trials
- Medium: 1,000 trials
- Large: 10,000 trials

## Performance Metrics

### QE
- **Throughput**: factors/second
- **Latency**: milliseconds/factor
- Typical: 3-10 factors/s (compute-bound)

### FP
- **Throughput**: million cells/second
- Typical: 0.1-1.0 Mcells/s (varies by operation)

### FA
- **Throughput**: thousand rows/second
- Typical: 20-100 Krows/s (memory-bound)

### FO
- **Throughput**: trials/second
- **Dedup rate**: % of duplicate candidates
- Typical: 1K-50K trials/s (hash-bound)

## Files Created

```
benchmarks/
├── standalone_benchmark.py      # Main self-contained suite ⭐
├── bench_qe_metrics.py          # QE benchmark
├── bench_fp_transforms.py       # FP benchmark
├── bench_fa_operations.py       # FA benchmark
├── bench_fo_search.py           # FO benchmark
├── run_all_benchmarks.py        # Orchestrator for all modules
├── benchmark.sh                 # Helper script
├── BENCHMARK_SUITE.md           # Detailed documentation
├── README.md                    # Original README
└── QUICKSTART.md                # This file
```

## Expected Runtime

| Benchmark | Approximate Time |
|-----------|-----------------|
| FO only | 1 second |
| FA only | 30-60 seconds |
| FP only | 2-5 minutes |
| QE only | 5-10 minutes |
| **Standalone (all)** | **10-15 minutes** |

## Reading Results

Example `baseline.json` structure:

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-08-14T...",
  "benchmarks": {
    "qe": {
      "results": [
        {
          "n_factors": 100,
          "elapsed_s": 12.5,
          "throughput_factors_per_s": 8.0
        }
      ]
    },
    "fp": { "results": {...} },
    "fa": { "results": {...} },
    "fo": { "results": {...} }
  }
}
```

## Troubleshooting

### Out of Memory
- Large-scale benchmarks require 4-8GB RAM
- Reduce scales in the benchmark scripts if needed

### Missing Dependencies
- Use `standalone_benchmark.py` - only needs numpy/pandas
- Individual benchmarks may need factor_engine, quant_evaluator, etc.

### Slow Performance
- Expected for large scales (10K factors = ~10 minutes)
- Check system load with `htop` or `top`
- Consider running smaller scales first

## Next Steps

1. Run `python3 standalone_benchmark.py` to get baseline
2. Review `baseline.json` for performance characteristics
3. Compare results across different hardware/configurations
4. Use metrics to identify bottlenecks in production workloads
5. Track performance regression over time

## Documentation

- `BENCHMARK_SUITE.md` - Comprehensive documentation
- `README.md` - Original overview
- Inline comments in each benchmark script

## Support

For questions or issues:
- Check script docstrings: `python3 <script> --help`
- Review error messages in console output
- Examine `baseline.json` for partial results
