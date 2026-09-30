# Real COS F61: consuming assembly and evidence-router validation

Evidence: `real_cos_f61_consuming_source_pearson_ab_20260930.json`.
Code checkpoint: `08d49512c`; 2586 dates, 5461 stocks, 61 float64 factors.
Four Pearson metrics run together: IC, IC series, standard deviation and IR.
Manifest SHA256: `b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`.
Source adapter COS, two prefetch workers, tile cap 16, explicit source cap 12288 MiB.
Logical source peak estimate: 10,930,770,368 bytes; this is not a hard RSS bound.

| Backend | Whole request seconds | Source-read seconds |
| --- | ---: | ---: |
| CPU | 134.4443 | 88.5440 |
| CUDA | 90.4782 | 87.7460 |
| auto (CUDA) | 89.8809 | 87.9541 |

Run order was CPU, CUDA, auto, not simultaneous or randomized repeated trials.
Both comparisons passed all 157,929 output elements, finite masks and observation counts.
Maximum CPU/CUDA absolute error was 8.3267e-16.
Auto retained `bounded_f61_pearson_chain_gpu_tile16`; all factors completed in four tiles.
Cumulative process peak RSS reached 13,133,244 KiB (~12.53 GiB).
These are not isolated per-backend memory peaks or a controlled memory-improvement A/B.
The small auto-versus-CUDA time difference may reflect IO/cache variation, not faster kernels.

This validates the changed assembly/router on this exact real workload. It does not prove
every metric or input is fastest, authorize wider profiles, or establish production/PIT certification.
Most CUDA wall time is source IO; bounded reader concurrency is the next measured optimization target.
