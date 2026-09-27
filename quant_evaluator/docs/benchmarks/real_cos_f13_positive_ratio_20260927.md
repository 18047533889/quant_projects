# F13 real-COS `rank_ic_positive_ratio` benchmark (2026-09-27)

This bounded public-runtime A/B requested exactly one metric,
`rank_ic_positive_ratio`, over 13 verified factors and the full-history panel:
2,586 sessions × 5,461 assets, `float64`, 2016-01-04 through 2026-08-25.
The loader checked the factors against manifest SHA256
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`.
The companion JSON retains each source object identity, byte count, ETag, full
artifact values, masks, counts, provenance, timings, routes, and parity results.
No credentials or panel values beyond the metric artifacts are included.

The order was `cpu → cuda_strict → auto → auto → cuda_strict → cpu`; each
isolated child made two complete public evaluations, with a 180-second child
timeout. Results below show the mean cold time and mean warm time across the
two requests for each backend.

| Request | Cold mean | Warm mean | Route / reason | Peak child RSS | Peak VRAM |
| --- | ---: | ---: | --- | ---: | ---: |
| CPU | 34.94 s | 32.59 s | CPU | 12,447,668 KiB | — |
| `cuda_strict` | 14.72 s | 11.77 s | CUDA | 14,218,636 KiB | 9,332,757,504 bytes |
| `auto` | 34.55 s | 32.66 s | CPU / `metric_not_certified` | 12,448,020 KiB | — |

Every one of the six requests had config hash
`ff3127bc82669088ac2d1894852cbea7d7d89cb1a85fc01a25dcdb4358720901`.
All artifact descriptor, numeric values (`rtol=1e-8, atol=1e-10`), finite
and valid masks, counts, provenance observation counts, full provenance, and
per-factor `MetricValue` checks passed against the first CPU request.
Overall `parity_pass=true`. The measured CUDA warm mean was about 2.77× faster
than CPU on this host and observed load. This initial run alone did not justify
a broad auto rule; the exact-shape route received an independent retest below.

Preflight before loading reported 193 GiB free disk, 54,503,336 kB
`MemAvailable`, and 39,964 MiB free on the NVIDIA L20. The run's parent peak
RSS was 7,029,224 KiB. The data lineage remains research-only and has no
upstream point-in-time or investability certification.

Reproduce from the server-c main tree with the configured DataAccess/COS roots:

```bash
cd /home/sunhaiwei/quant_projects
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -u quant_evaluator/scripts/benchmark_real_cos_metric_batch.py \
  --factors 13 --metrics rank_ic_positive_ratio --timeout-s 180 \
  --output quant_evaluator/docs/benchmarks/real_cos_f13_positive_ratio_20260927.json
```

## Exact-shape `auto` retest

After adding only the certified F13 singleton route, the same immutable
manifest, shape, metric and six-child CPU/CUDA/auto interleaving were run
again. The report is `real_cos_f13_positive_ratio_auto_20260927.json`.
Both `auto` children actually used CUDA, with route reason
`certified_single_metric_real_cos_f13_rank_ic_positive_ratio`.
Auto cold times were 14.930/14.891 s and warm times 11.824/11.828 s;
CPU endpoints were 35.092/35.174 s cold and 32.638/32.890 s warm.
All six requests retained one config hash and passed the complete artifact,
mask, counts, provenance and per-factor `MetricValue` comparison.

The route is limited to the exact 2,586 × 5,461 × 13 float64 panel, this
single metric with default parameters, no special inputs, one NVIDIA L20,
and at least 14 GiB of both actual and policy-effective free VRAM.
Other shapes, dtypes, metric sets, parameter overrides, hardware or
insufficient VRAM remain on CPU. The measured GPU pool peak was
9,332,757,504 bytes; the 14 GiB gate leaves headroom.
