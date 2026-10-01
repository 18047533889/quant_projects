# F32 coverage: exact hash-cache replay (2026-10-01)

We replayed the public evaluator at main `b8ad29c76c3ae4a1ca5539fc687d3e2868ed0e2d`, including immutable-array cache commit `37f8eda8c`. Inputs were the same 32 COS factors, 2,586 dates, 5,461 stocks and float64 coverage singleton used in the [preceding working-set replay](real_cos_f32_coverage_working_set_ab_20261001.md). We kept performance jobs serial and made no source or HEAD changes during measurement.

| Round | Requested | Cold seconds | Warm seconds |
|---:|---|---:|---:|
| 1 | CPU | 17.5837 | 10.6776 |
| 2 | CUDA strict | 15.6583 | 7.8069 |
| 3 | auto | 15.5227 | 7.7424 |
| 4 | auto | 16.0722 | 8.2167 |
| 5 | CUDA strict | 15.9168 | 7.6743 |
| 6 | CPU | 17.4989 | 10.6508 |

Warm medians: CPU **10.6642 s**, explicit CUDA **7.7406 s**, auto **7.9795 s**. Both auto rounds selected CUDA with reason `certified_single_metric_real_cos_f32_coverage`; auto used **25.17% less elapsed time than CPU**. Compared with the preceding replay, warm elapsed time fell **35.62% on CPU**, **43.27% on CUDA**, and **41.95% on auto**. These old/new measurements came from separate serial runs, so host variation limits attribution. The bracketed order reduces order bias but six rounds cannot establish performance across deployments.

Cold timings remain near the preceding replay because each child must encode the array on its first evaluation. The second evaluation reuses the same immutable FactorBatch. Reconstructing arrays or reloading data forfeits this identity-based cache benefit. The [cache contract and microtest](immutable_array_hash_cache_20261001.md) explain the bounded weak-reference design.

GPU execution retained one 32-factor tile and peak pool allocation **4,646,187,520 bytes (4.33 GiB)**. GPU child peak RSS reached 14,408,500 KiB (about 13.74 GiB), and CPU child peak reached 6,703,872 KiB (about 6.39 GiB). Source loading took **121.7477 s**, excluded from evaluation timings. Include that cost when measuring an application that loads data per request.

## Correctness and lineage

All six runs completed. We checked configuration identity, descriptors, values (`rtol=1e-8`, `atol=1e-10`), finite/valid masks, counts, provenance observation counts, provenance and scalar metric values. Every check passed. Source summaries and configuration hash match the preceding replay: `c2310e2f8c4a409b435d3cfa8ea22f0d45bdae9c10a0b0e39d0f7f6bb7a53435`.

Same-backend artifact hashes are unchanged: CPU `014d8f5a2f40024cfcd5eed7360af0a57bc3602c5af58b4a9cb53cb488576a2b`, CUDA/auto `13fc016d79ac62d8d7e828e50bd2454da2794cef3d517ce6fefca83435dc8a19`. CPU and GPU remain tolerance-equivalent, not bit-identical. The QE full suite passed 4,501 tests with 26 skips before the strengthened metadata-cache test; afterward the focused hash suite passed 22 tests.

Dates span 2016-01-04 through 2026-08-25. Manifest SHA-256 is `b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`. Labels use decision t, execution t+1 and `AdjVwap(t+2)/AdjVwap(t+1)-1`. The research lineage does not certify upstream PIT/investability. This replay covers this device and request, not all metrics or batch shapes. The [raw receipt](real_cos_f32_coverage_hash_cache_ab_20261001.json) records script identity and comparisons; it is not a full repository closure manifest.

To reproduce, use the configured COS environment and `.venv/bin/python -u quant_evaluator/scripts/benchmark_real_cos_metric_batch.py --factors 32 --metrics coverage --run --timeout-s 120 --compact --output quant_evaluator/docs/benchmarks/real_cos_f32_coverage_hash_cache_ab_NEW.json`. Preserve existing receipts and memory reserves.
