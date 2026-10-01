# F32 coverage singleton: current auto replay (2026-10-01)

The six-round public-entry replay used 32 COS factors, 2,586 dates and 5,461 stocks, with float64 inputs and the exact metric request `("coverage",)`. Both `auto` rounds selected CUDA with reason `certified_single_metric_real_cos_f32_coverage`. Complete CPU/CUDA/auto comparisons passed. This confirms the current route for this request; it does not certify other shapes, metric combinations or devices.

| Round | Requested backend | Used | Cold seconds | Warm seconds |
|---:|---|---|---:|---:|
| 1 | cpu | cpu | 17.6645 | 16.7385 |
| 2 | cuda_strict | cuda | 16.7077 | 15.4386 |
| 3 | auto | cuda | 16.8525 | 15.5767 |
| 4 | auto | cuda | 16.8368 | 15.5728 |
| 5 | cuda_strict | cuda | 16.6572 | 15.5677 |
| 6 | cpu | cpu | 17.6669 | 16.6606 |

Warm medians: CPU **16.6995 s**, explicit CUDA **15.5031 s**, auto **15.5748 s**. Auto reduced elapsed time by about **6.74%** (CPU/auto ratio **1.0722**). This singleton gain is smaller than the separately measured default-five batch gain. Each child ran two evaluations; the first is cold and the second is warm. The order brackets the GPU rounds with CPU rounds, but six rounds cannot characterize host-load variability across deployments.

The GPU rounds used eight-factor tiles, four tiles per request, and peaked at **1,256,872,448 bytes** of pool allocations (about **1.17 GiB**). The replay used the existing rank-shaped tile estimator; the proposed coverage-specific estimator was not present. Source loading took **120.2901 s**, excluded from evaluation timings. Callers that reload COS data for each evaluation must include that load time in their own end-to-end measurements.

## Correctness and source boundaries

Every comparison checked configuration identity, artifact descriptor, values (`rtol=1e-8`, `atol=1e-10`), finite and valid masks, counts, provenance observation counts, provenance and scalar metric values. The common configuration hash was `c2310e2f8c4a409b435d3cfa8ea22f0d45bdae9c10a0b0e39d0f7f6bb7a53435`.

CUDA and auto artifact hashes matched: `aec8ca8664b0e792fa8035bf3a4ac5f60b5a9caee8062aed5da2b508e9316298`. CPU artifact hash was `014d8f5a2f40024cfcd5eed7360af0a57bc3602c5af58b4a9cb53cb488576a2b`. CPU and GPU are tolerance-equivalent here, not bit-identical.

The bound manifest SHA-256 is `b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`. Dates span 2016-01-04 through 2026-08-25. Factor finite ratio is 0.76619; label finite ratio is 0.77386. Labels use decision t, execution t+1, and `AdjVwap(t+2)/AdjVwap(t+1)-1`. The COS lineage is research data with no upstream PIT/investability certification. The replay followed the precision-guard commit `2aa9f09b1`; the JSON retains the benchmark script hash and execution receipts.

## Reproduce

Use the configured COS environment and the project virtual environment on server-c. The command preflights host/GPU memory before loading data. Keep performance jobs serial and preserve the memory reserve.

```bash
.venv/bin/python -u quant_evaluator/scripts/benchmark_real_cos_metric_batch.py \
  --factors 32 --metrics coverage --run --timeout-s 120 --compact \
  --output quant_evaluator/docs/benchmarks/real_cos_f32_coverage_single_auto_post_v28_20261001.json
```

The [raw receipt](real_cos_f32_coverage_single_auto_post_v28_20261001.json) records the six runs and full comparison outcomes. Do not overwrite it when measuring a future estimator change.
