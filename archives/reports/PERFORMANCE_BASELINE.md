# Performance Baseline and Analysis

## Overview

This document establishes the performance baseline for all quant_projects packages and provides analysis of benchmark results.

**Last Updated:** 2026-08-14  
**Baseline Version:** 2.0

---

## Benchmark Suite

### 1. Quant Evaluator (IC Computation)

**Package:** `quant_evaluator`  
**Focus:** IC computation speed at scale

| Scale | Factors | Dates | Assets | Target Time | Target Throughput |
|-------|---------|-------|--------|-------------|-------------------|
| Small | 100 | 252 | 1,000 | 1-5s | 20-100 factors/s |
| Medium | 1,000 | 252 | 1,000 | 10-30s | 30-100 factors/s |
| Large | 10,000 | 252 | 1,000 | 100-300s | 30-100 factors/s |
| XLarge | 10,000 | 1,260 | 3,000 | 500-1500s | 20-50 factors/s |

**Key Metrics:**
- `throughput_factors_per_sec`: Number of factors evaluated per second
- `latency_ms_per_factor`: Milliseconds to evaluate one factor
- `throughput_obs_per_sec`: Raw observations processed per second

**Expected Behavior:**
- Linear scaling with number of factors
- Cache hit rate > 0% on repeated evaluations
- Memory usage scales linearly with data size

---

### 2. Research Control (Ledger Operations)

**Package:** `research_control`  
**Focus:** Write and query performance for experiment tracking

#### Campaign Writes
| Scale | Records | Target Time | Target Throughput |
|-------|---------|-------------|-------------------|
| Small | 100 | < 0.5s | > 200 writes/s |
| Medium | 1,000 | < 2s | > 500 writes/s |
| Large | 10,000 | < 10s | > 1,000 writes/s |

#### Trial Writes
| Scale | Records | Target Time | Target Throughput |
|-------|---------|-------------|-------------------|
| Small | 1,000 | < 1s | > 1,000 writes/s |
| Medium | 10,000 | < 5s | > 2,000 writes/s |
| Large | 100,000 | < 30s | > 3,000 writes/s |

#### Query Performance
| Scale | DB Size | Query Type | Target Latency |
|-------|---------|------------|----------------|
| Small | 1K trials | Full scan | < 10ms |
| Medium | 10K trials | Top-K | < 20ms |
| Large | 100K trials | Filtered range | < 50ms |

**Key Metrics:**
- `throughput_per_sec`: Write/query operations per second
- `latency_ms_per_write`: Milliseconds per write
- `median_latency_ms`: Query latency (median of 5 runs)

**Expected Behavior:**
- Write throughput remains stable across scales
- Query latency scales logarithmically with database size
- No memory leaks on repeated writes

---

### 3. Factor Optimizer (Search Operations)

**Package:** `factor_optimizer`  
**Focus:** Mutation generation and search efficiency

| Operation | Scale | Target Time | Target Throughput |
|-----------|-------|-------------|-------------------|
| Mutation Generation | 1K | < 0.1s | > 10,000 mutations/s |
| Mutation Generation | 100K | < 5s | > 20,000 mutations/s |
| Deduplication | 1K | < 0.01s | > 100,000 checks/s |
| Deduplication | 100K | < 0.5s | > 200,000 checks/s |
| Complexity Profiling | 1K | < 0.1s | > 10,000 profiles/s |
| Validation | 100K | < 1s | > 100,000 validations/s |

**Key Metrics:**
- `throughput_per_sec`: Operations per second
- `duplicate_rate`: Fraction of duplicates detected
- `latency_ms_per_mutation`: Milliseconds per mutation

**Expected Behavior:**
- Near-linear scaling for mutation generation
- O(1) deduplication using hash-based cache
- Duplicate rate matches input distribution

---

### 4. Factor Assets (Similarity Search)

**Package:** `factor_assets`  
**Focus:** Similarity search and clustering performance

#### Exact Similarity Search
| Scale | Database | Queries | k | Target Time | Target Throughput |
|-------|----------|---------|---|-------------|-------------------|
| Small | 1K | 100 | 10 | < 1s | > 100 queries/s |
| Medium | 10K | 100 | 10 | < 5s | > 20 queries/s |
| Large | 100K | 100 | 10 | < 30s | > 3 queries/s |

#### Clustering
| Scale | Factors | Clusters | Target Time |
|-------|---------|----------|-------------|
| Small | 1K | 10 | < 1s |
| Medium | 10K | 50 | < 10s |
| Large | 100K | 200 | < 60s |

**Key Metrics:**
- `throughput_queries_per_sec`: Similarity queries per second
- `latency_ms_per_query`: Query latency
- `throughput_comparisons_per_sec`: Raw comparison rate
- `mean_cluster_size`: Average cluster size

**Expected Behavior:**
- Query time scales linearly with database size (brute force)
- Clustering converges in < 20 iterations
- Balanced cluster sizes (no degenerate clusters)

---

### 5. Factor Preprocess (Transforms)

**Package:** `factor_preprocess`  
**Focus:** Preprocessing transform throughput

| Scale | Assets | Dates | Factors | Total Cells | Target Time |
|-------|--------|-------|---------|-------------|-------------|
| Small | 100 | 252 | 3 | 75.6K | < 5s |
| Medium | 1,000 | 252 | 3 | 756K | < 10s |
| Large | 3,000 | 504 | 5 | 7.56M | < 60s |

**Operations Benchmarked:**
- Zscore (cross-sectional)
- Rank (cross-sectional)
- Industry neutralization
- Winsorization

**Key Metrics:**
- `throughput_cells_per_sec`: Data cells processed per second
- Backend comparison (pandas vs polars)

**Expected Behavior:**
- Polars 2-5x faster than pandas for large scales
- Throughput: 0.1-1 Mcells/s (pandas), 0.5-5 Mcells/s (polars)
- Memory usage scales linearly with data size

---

## Stress Tests

### 1. Memory Pressure

**Test:** Large array allocations  
**Target:** Successfully allocate and process up to 2GB  
**Pass Criteria:**
- No memory errors up to target size
- Memory properly freed after cleanup
- Memory leak < 50 MB after 1000 allocations

### 2. Long-Running Stability

**Test:** Continuous computation for 60 seconds  
**Pass Criteria:**
- No crashes or hangs
- Memory remains stable (< 100 MB growth)
- Consistent iteration throughput

### 3. High Concurrency Simulation

**Test:** 100-200 simulated concurrent tasks  
**Pass Criteria:**
- All tasks complete successfully
- P95 latency < 2x mean latency
- P99 latency < 5x mean latency
- No resource contention deadlocks

### 4. Large Dataset Processing

**Test:** 50,000 factors × 1,000 observations  
**Target:** Process in < 60s  
**Pass Criteria:**
- Successfully generates and processes dataset
- Memory properly cleaned up (> 50% of data size freed)
- Throughput > 500K elements/s

---

## Performance Regression Thresholds

Regressions are flagged when:

- **Throughput metrics:** > 10% decrease
- **Latency metrics:** > 10% increase
- **Memory usage:** > 20% increase

**Regression Detection:**
```bash
python benchmarks/detect_regression.py
```

**CI Integration:**
- Run benchmarks on main branch commits
- Compare against last stable baseline
- Block merge if regressions detected

---

## Running Benchmarks

### Individual Benchmarks
```bash
cd /home/shw/quant_projects/benchmarks

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

### All Benchmarks
```bash
python run_all_benchmarks_v2.py
```

### Regression Detection
```bash
python detect_regression.py
```

---

## Baseline Results

### Hardware Reference
- **CPU:** 8-core, 3.0 GHz
- **Memory:** 32 GB RAM
- **Storage:** SSD
- **OS:** Linux 5.15

### Expected Total Runtime
- **Benchmarks:** 10-20 minutes
- **Stress Tests:** 5-10 minutes
- **Total:** 15-30 minutes

---

## Interpreting Results

### Good Performance Indicators
✓ Throughput within target ranges  
✓ Linear scaling with data size  
✓ Low P95/P99 latency variance  
✓ Stable memory usage  
✓ High cache hit rates (when applicable)

### Warning Signs
⚠️ Throughput < 50% of target  
⚠️ Non-linear scaling  
⚠️ Memory leaks (growing over time)  
⚠️ High P99 latency (> 10x mean)  
⚠️ Crashes on large datasets

### Critical Issues
🔴 Benchmark failures  
🔴 Memory errors on medium scales  
🔴 Deadlocks or hangs  
🔴 Correctness errors  
🔴 > 20% performance regression

---

## Optimization Priorities

### High Impact
1. **Quant Evaluator:** Numba JIT for IC computation
2. **Factor Preprocess:** Polars backend for large datasets
3. **Factor Assets:** Approximate nearest neighbor (FAISS/Annoy)
4. **Research Control:** Bulk insert APIs

### Medium Impact
1. Caching strategies
2. Memory pooling
3. Parallel execution
4. Compression

### Low Impact
1. Micro-optimizations
2. Code cleanup
3. Documentation

---

## Maintenance

### When to Update Baseline
- After major performance improvements
- After infrastructure changes
- Quarterly review
- Before major releases

### How to Update Baseline
```bash
# Run benchmarks
python run_all_benchmarks_v2.py

# Copy to baseline
cp benchmarks/results/latest_results.json benchmarks/results/baseline.json
```

### Reviewing Results
1. Check for any failures
2. Verify throughput targets met
3. Review memory usage
4. Check stress test pass rates
5. Document any anomalies

---

## Troubleshooting

### Benchmark Failures

**Import errors:**
- Ensure packages installed: `pip install -e .`
- Check Python path

**Memory errors:**
- Reduce scale parameters
- Close other applications
- Check available RAM

**Timeout:**
- Expected for large scales
- Increase timeout in runner script

### Inconsistent Results

**High variance:**
- Run on idle system
- Disable CPU throttling
- Increase warmup iterations

**Regression false positives:**
- Check for system load
- Verify baseline is current
- Review threshold settings

---

## Future Enhancements

### Planned Additions
- [ ] GPU backend benchmarks
- [ ] Distributed execution benchmarks
- [ ] Real data benchmarks
- [ ] End-to-end workflow benchmarks
- [ ] Power consumption metrics

### Infrastructure
- [ ] Automated CI/CD integration
- [ ] Performance dashboard
- [ ] Historical trending
- [ ] Alerting on regressions
- [ ] Benchmark result database

---

## References

- **Benchmark Scripts:** `/home/shw/quant_projects/benchmarks/`
- **Results Directory:** `/home/shw/quant_projects/benchmarks/results/`
- **Latest Results:** `benchmarks/results/latest_results.json`
- **Baseline:** `benchmarks/results/baseline.json`
