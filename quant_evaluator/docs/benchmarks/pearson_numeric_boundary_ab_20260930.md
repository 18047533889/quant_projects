# Pearson numeric-boundary GPU A/B

Environment: server-c L20; CUDA events; shape (T,F,N) = (128,8,5000) and labels (128,5000). Each path received two warmups, then old/new were measured in alternating order. Values are medians; timings include the kernel's reductions and, on the new float64 path, its safety check.

| Input case | Old path | New path | Relative change |
| --- | ---: | ---: | ---: |
| float32 standard normal, 7 pairs | 0.825 ms | 0.846 ms | +2.5% |
| float64 standard normal, 5 pairs | 0.754 ms | 0.944 ms | +25.2% |
| float64 large offset (1e16 + 2*k) fallback, 5 pairs | 0.690 ms | 4.424 ms | 6.4x |

The float32 fast path avoids a host synchronization and stays close to baseline. CPU and GPU use the same fallback threshold based on sample standard deviation (ddof=1). Ordinary float64 pays for one unsafe-row check. The large-offset case invokes the slower origin-centered fallback for all rows; this is the cost of recovering precision that the previous path lost. The old path in this comparison is the prior centered-sum/product formula reproduced in the benchmark harness; it is a timing baseline, not a correctness oracle.
