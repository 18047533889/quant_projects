# Real COS F61 Pearson selective-repair validation

Measured on server-c with the candidate device-side selective Pearson repair,
using the existing verified COS adapter, two-object prefetch and tile width 16.
The source panel was 2,586 dates × 5,461 stocks × 61 factors, float64.
All four metrics were requested together: `pearson_ic`, `pearson_ic_series`,
`pearson_ic_std`, `pearson_ic_ir`. No production factor was published.

| Backend | Whole request | Source reads |
| --- | ---: | ---: |
| CPU | 131.1358 s | 85.7468 s |
| Explicit CUDA | 94.4229 s | 90.3428 s |
| Auto (selected CUDA, tile 16) | 88.3663 s | 85.6431 s |

Actual execution order was CUDA, CPU, then auto. The same 157,929 output
elements were compared, with identical finite masks and observation counts.
CPU/CUDA and CPU/auto comparisons passed; maximum absolute error was
8.3267e-16. Auto used reason `bounded_f61_pearson_chain_gpu_tile16`.

The process-wide recorded high-water RSS reached 15,423,272 KiB (~14.7 GiB).
RSS is cumulative process evidence, not an isolated per-backend peak. Each
preflight passed a 32 GiB available-RAM floor; source assembly was bounded
to 4 GiB and individual tiles were released instead of materializing F61.

Source reads dominate GPU end-to-end wall time. Auto's lower wall time does
not mean a faster CUDA kernel: read/cache/service state differed between
passes. This run verifies the candidate on real data and confirms CUDA beats
CPU for this request; it is not a controlled old/new kernel speedup claim,
nor evidence for arbitrary shapes, contents, metrics or GPU devices.

Raw report: [JSON](real_cos_f61_pearson_device_repair_cos_ab_20260930.json).
This evidence alone does not widen existing auto certification gates.
