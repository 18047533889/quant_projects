# F8 real-COS mixed public evaluation benchmark (2026-09-27)

This is a whole-request A/B for the public `evaluate()` call with
`rank_ic`, `quantile_spread`, and `factor_turnover_rate`. The loader verified
eight distinct factor objects against the DataAccess landing manifest
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864` and
their object SHA256/byte identities. The common `float64` panel is
2,586 sessions × 5,461 assets × 8 factors, from 2016-01-04 through 2026-08-25.
The source IDs, per-object digests, byte counts, ETags, and full loader
provenance are in the companion JSON. No source panel values or credentials
are included.

The run order was `cpu → cuda_strict → auto → auto → cuda_strict → cpu`.
Each isolated worker made two back-to-back complete public evaluations; each
individual evaluation had a 180-second timeout. The two outer passes reverse
backend order to reduce order bias.

| Backend | Cold mean | Warm mean | Route / reason | Peak GPU memory |
| --- | ---: | ---: | --- | ---: |
| CPU | 38.19 s | 34.42 s | CPU | — |
| `cuda_strict` | 12.44 s | 9.09 s | CUDA | 9,332,757,504 bytes |
| `auto` | 38.53 s | 34.53 s | CPU / `shape_outside_certified_range` | — |

The NVIDIA L20 reported 46,068 MiB total VRAM. Child peak RSS was 12,114,208
KiB for CPU, 10,873,272 KiB for `cuda_strict`, and 12,114,792 KiB for auto.
The parent peak RSS during loading was 7,119,512 KiB. Other CPU and GPU work
was active on the server during the run, so these timings describe that host
and observed load.

All six artifacts matched the first CPU result for all three metrics.
Artifact descriptors, finite and valid masks, optional artifact counts, full
provenance and per-factor `MetricValue` fields matched; all eight
`MetricValue.observation_count` values per metric were identical. Artifact
and `MetricValue` numeric values matched at `rtol=1e-8, atol=1e-10`. All six requests had the same
`config_hash`:
`867b381441364470a53787c86bf25200d00b106a2267cbf52977c74e6213fd8c`.
`cuda_strict` was about 3.8× faster on warm requests than CPU in this run.
Auto still selected CPU because this F8 shape was outside the certified auto
shape range at the time of this benchmark; the run metadata records the
original v4 route.

Follow-up: strategy v7 validated auto on the same real F8 payload. It routed
all three metrics to CUDA with reason
`certified_batch_real_cos_f8_mixed_three`; the request retained
`config_hash=867b381441364470a53787c86bf25200d00b106a2267cbf52977c74e6213fd8c`.

The lineage is research-only: labels use decision `t`, execute at `t+1`, and
measure `AdjVwap(t+2)/AdjVwap(t+1)-1`. It has no upstream point-in-time or
investability certification.

Reproduce from the server-c main tree with the existing DataAccess/COS setup:

```bash
cd /home/sunhaiwei/quant_projects
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u quant_evaluator/scripts/benchmark_real_cos_metric_batch.py \
  --timeout-s 180 \
  --output quant_evaluator/docs/benchmarks/real_cos_f8_mixed_20260927.json
```

The JSON report contains every round's timing and route metadata plus
per-metric parity results, factor/source provenance, peak RSS/VRAM, and the
benchmark script SHA256.
