# F32 coverage: working-set estimator A/B (2026-10-01)

We measured the public entry point on server-c at commit `56d0945136f9a4c8ac7f1112af57270834b306f0`, including estimator commit `fe25bf314`. The request contained 32 COS factors, 2,586 dates, 5,461 stocks, float64 inputs and the exact metric tuple `("coverage",)`. Both auto rounds selected CUDA with reason `certified_single_metric_real_cos_f32_coverage`.

| Round | Requested | Cold seconds | Warm seconds |
|---:|---|---:|---:|
| 1 | CPU | 17.5140 | 16.5411 |
| 2 | CUDA strict | 15.4606 | 13.6529 |
| 3 | auto | 15.2913 | 13.6787 |
| 4 | auto | 15.4550 | 13.8122 |
| 5 | CUDA strict | 15.2715 | 13.6362 |
| 6 | CPU | 17.4042 | 16.5887 |

Warm medians: CPU **16.5649 s**, CUDA **13.6446 s**, auto **13.7454 s**. Auto used **17.02% less elapsed time than CPU**. Compared with the [preceding estimator replay](real_cos_f32_coverage_single_auto_post_v28_20261001.md), auto used **11.75% less elapsed time** (15.5748 s to 13.7454 s). We ran old and new replays in separate serial runs; host variation limits causal attribution. Each replay bracketed GPU rounds with CPU rounds and used the same configuration hash and CPU artifact hash.

## Memory model and tradeoff

For T dates, N assets and f factors per tile, the coverage helper estimates bytes as:

`B = T*N*f*d_factor + T*N*d_label + 3*T*N*f + 16*T*f + 16*f + T*N`

`estimate = ceil(1.5*B) + 64 MiB`

The session adds existing pool allocations. Label item size is at least eight bytes; FP64 precision modes promote factor item size to at least eight bytes. Only coverage singleton requests without metric parameters use this model. Mixed requests keep their existing estimator and OOM retry path.

At this shape, the eight-factor estimate is 2,122,377,811 bytes, below the certified 2 GiB effective-budget floor. The prior rank estimator exceeded that floor even for one factor. Under the measured larger budget, the new execution used one 32-factor tile and peaked at **4,646,187,520 GPU pool bytes (4.33 GiB)**. The old execution used four eight-factor tiles and peaked at 1,256,872,448 bytes (1.17 GiB). New child peak RSS reached about **13.8 GiB**, versus about 7.8 GiB before. Smaller budgets retain smaller tiles. Callers must keep the host and GPU memory reserve; the speed improvement has a memory cost.

## Correctness and evidence boundaries

All six runs completed and all comparisons passed: configuration identity, descriptors, values (`rtol=1e-8`, `atol=1e-10`), finite/valid masks, counts, provenance and scalar metric values. CPU artifact hash remained `014d8f5a2f40024cfcd5eed7360af0a57bc3602c5af58b4a9cb53cb488576a2b`. New CUDA/auto hash is `13fc016d79ac62d8d7e828e50bd2454da2794cef3d517ce6fefca83435dc8a19`; the old GPU hash differs. Tile layout changes reduction rounding, so GPU results are tolerance-equivalent, not bit-identical.

The parent ran the QE suite: **4,490 passed, 26 skipped**. After adding a real CUDA bounded-pool coverage test, the boundary/OOM/resource group passed **16 tests**. That supplemental test checks independent NumPy pair-mask results, NaN/Inf, validity masks and a singleton tail with real estimator-selected `[2,2,1]` tiles. The full-suite count predates this added test.

Dates span 2016-01-04 through 2026-08-25. Manifest SHA-256: `b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`. Labels use decision t, execution t+1 and `AdjVwap(t+2)/AdjVwap(t+1)-1`. Research lineage does not certify upstream PIT/investability. Source loading took **119.1715 s**, excluded from evaluation timings. The [raw receipt](real_cos_f32_coverage_working_set_ab_20261001.json) includes the script hash and comparisons, not a full source-closure manifest. This evidence covers the stated request and device, not arbitrary metrics or shapes.

Reproduce with the configured COS environment and project `.venv/bin/python`, running performance jobs serially:

```bash
.venv/bin/python -u quant_evaluator/scripts/benchmark_real_cos_metric_batch.py --factors 32 --metrics coverage --run --timeout-s 120 --compact --output quant_evaluator/docs/benchmarks/real_cos_f32_coverage_working_set_ab_NEW.json
```
