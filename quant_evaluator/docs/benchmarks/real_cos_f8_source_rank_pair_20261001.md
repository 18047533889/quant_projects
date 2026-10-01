# F8 source rank pair: measured auto route

We evaluated eight existing COS factors over 2,586 dates and 5,461 stocks
with float64 inputs, `rank_ic` and `rank_ic_series`. The source adapter used
bounded auto prefetch, a two-object window and requested tile width 2.

| Order | CPU seconds | CUDA seconds |
|---|---:|---:|
| CPU first | 23.2008 | 8.3393 |
| CUDA first | 22.9402 | 8.3391 |

Both orders passed comparisons of 20,696 values, finite masks and observation
counts. Maximum absolute error was 2.2204e-16. Each backend produced matching
output hashes across orders. Timings include source reads inside evaluation;
they exclude initial axis and label preparation.

We enabled the measured source-auto registry entry only for this shape,
float64 inputs, this metric pair and requested tile width 2. Existing resource
and precision guards still apply. Default tile cap 8 has no certification from
these runs; callers use `max_tile_size=2` in the public source API for this route.

The public auto replay selected CUDA with reason `bounded_f8_rank_pair_gpu`
and completed in 8.8556 seconds. It matched the certified CUDA outputs and a
fresh explicit CUDA request across all 20,696 values and observation counts.
See `real_cos_f8_source_rank_pair_cpu_first_20261001.json`,
`real_cos_f8_source_rank_pair_cuda_first_20261001.json` and
`real_cos_f8_source_rank_pair_auto_20261001.json`. This is research-source
evidence on server-c, not a PIT or production certification.
