# F12 real-COS Pearson IC single-metric A/B (2026-09-28)

The public `evaluate()` request used the verified DataAccess/COS landing manifest
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`.
The bound panel was 2,586 sessions × 5,461 stocks × 12 factors, `float64`,
spanning 2016-01-04 through 2026-08-25. Labels used decision `t`,
execution `t+1`, and `AdjVwap(t+2)/AdjVwap(t+1)-1`.

Each run used isolated workers in the order CPU → strict CUDA → auto → auto →
strict CUDA → CPU; each worker made a cold and warm full public request.
The NVIDIA L20 had 39,094 MiB free before testing. The bound source objects
totaled 194,887,083 bytes, below the loader's 256 MiB limit.

| Phase | Backend | Cold runs (s) | Warm runs (s) | Actual route | Peak GPU pool |
| --- | --- | --- | --- | --- | ---: |
| Raw A/B, before route | CPU | 13.885 / 14.462 | 12.307 / 12.209 | CPU | — |
| Raw A/B, before route | strict CUDA | 12.249 / 12.193 | 9.219 / 9.548 | CUDA | 3,968,325,632 B |
| Raw A/B, before route | auto | 14.057 / 14.195 | 12.258 / 12.490 | CPU, `metric_not_certified` | — |
| After exact route | CPU | 13.823 / 14.000 | 12.146 / 11.976 | CPU | — |
| After exact route | strict CUDA | 11.861 / 11.863 | 9.105 / 9.190 | CUDA | 3,968,325,632 B |
| After exact route | auto | 11.915 / 11.838 | 9.108 / 9.095 | CUDA, `certified_single_metric_real_cos_f12_pearson_ic` | 3,968,325,632 B |

All six comparisons in each phase passed against the first CPU result:
artifact descriptor, numeric values (`rtol=1e-8`, `atol=1e-10`), finite and
valid masks, counts, full provenance, per-factor `MetricValue`, and one
semantic configuration hash per phase. The hash stayed
`c7116d6c4fe3515119372ddbb16009f7261e45d874bc0acf1dbd8bce50b8875f`
across both phases. Raw and after-route details, including source digests,
worker RSS, and every comparison field, are in the companion compact JSONs:
`real_cos_f12_pearson_ic_20260928.json` and
`real_cos_f12_pearson_ic_auto_20260928.json`.

The auto rule applies only to the exact shape, default parameters, float64
factor and label arrays, no special inputs, and one NVIDIA L20. Both actual
and policy-effective free VRAM must be at least 8 GiB, about 2.16 times
the measured GPU pool peak. Adjacent shapes and other Pearson metrics remain
on CPU. Host load may change the timing; this is one bounded research panel.
