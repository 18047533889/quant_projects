# Quantile rank stability A/B — 2026-09-27

Dense common-support input, one factor, 2,586 days × 10 quantile buckets (3,342,405 date pairs), with OPENBLAS_NUM_THREADS=1:

| Path | Cold | Warm | Peak RSS | Mean rank correlation |
| --- | ---: | ---: | ---: | ---: |
| Pairwise re-rank baseline | 3.277 s | — | 249.89 MiB | -0.00013665536515101436 |
| Shared-support Gram tiles | 0.1029 s | 0.0887 s median (3 runs) | 217.70 MiB | -0.00013665536515101436 |

The dense case was 31.7× faster on the cold call; output values matched exactly. RSS is process peak (ru_maxrss) and includes the Python, NumPy, SciPy, and input baseline.

The optimization applies when all valid days for a factor share the same finite-bucket mask. It ranks each day once and evaluates pair correlations in bounded Gram tiles. Factors with date-varying supports continue through the existing pairwise re-ranking path, because each pair’s joint support changes its ranks. The varying-support and insufficient-joint-support oracle regressions passed (4 passed).
