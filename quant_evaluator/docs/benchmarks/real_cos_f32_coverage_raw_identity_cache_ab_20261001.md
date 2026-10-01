# F32 coverage: raw identity cache replay

We measured main `752945bae4a8304dd4e51b1f0189159c386a4e6d` on 2026-10-01.
The request used 32 existing COS factors, 2,586 dates, 5,461 stocks and float64
inputs. We ran performance jobs serially and kept source and HEAD unchanged.
Each child evaluated the same immutable batch twice.

| Round | Backend | Cold seconds | Warm seconds |
|---:|---|---:|---:|
| 1 | CPU | 15.0517 | 5.8353 |
| 2 | CUDA strict | 16.1561 | 5.3362 |
| 3 | auto | 16.4092 | 5.3121 |
| 4 | auto | 16.3357 | 5.3013 |
| 5 | CUDA strict | 16.2233 | 5.2505 |
| 6 | CPU | 15.3962 | 6.0702 |

Warm medians were CPU 5.9527 s, CUDA 5.2933 s and auto 5.3067 s. Both auto
rounds selected CUDA. Auto used about 10.9% less warm elapsed time than CPU,
and about 33.5% less than the preceding JSON-cache replay's 7.9795 s.
These separate old/new runs include host variation and two changes: CPU
provenance reuse and the raw-array identity cache. They do not isolate attribution.

Cold medians were CPU 15.2239 s, CUDA 16.1897 s and auto 16.3725 s. CPU won
cold calls. The existing static GPU route therefore does not prove the fastest
first evaluation; cold/warm-aware selection remains unfinished.

All six runs passed descriptor, value tolerance, finite/valid mask, count,
provenance and configuration comparisons. Same-backend artifact hashes match
the preceding replay. GPU peak pool allocation remained 4,646,187,520 bytes.
Source loading took 122.8073 s, excluded from evaluation timings. The QE suite
passed 4,519 tests with 26 skips before this run. See the [raw receipt](real_cos_f32_coverage_raw_identity_cache_ab_20261001.json)
and [cache contract](../ARRAY_IDENTITY_CACHE.md). This evidence covers coverage
on this request and device, not all metrics, shapes or deployments.
