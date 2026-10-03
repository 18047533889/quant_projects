# GPU rank scatter and distinct allocation A/B (2026-10-03)

Compared the pre-change rank source at `249113d554f2a88c03879e0a776042af26db002e` with the current `rank.py` on an NVIDIA L20. Both paths produced exactly equal rank values and distinct counts for the benchmark input. The batch was `(T,F,N)=(512,48,1000)` float32. CUDA event times include warmed stream work and synchronize the stop event per sample; they are not CPU wall-clock measurements. Each mode had three warmups per implementation across validation and the post-memory warmup, then three ABBA cycles.

| Mode | Baseline median | Current median | Change |
|---|---:|---:|---:|
| Rank only | 33.226 ms | 32.588 ms | −0.637 ms (−1.92%) |
| Rank + distinct count | 33.443 ms | 33.187 ms | −0.257 ms (−0.77%) |

The six samples per side were tightly grouped. The separate CuPy pool-growth pass measured 660,609,536 bytes for both versions. This metric is pool reservation growth, not process-wide or device peak VRAM, so it does not establish a peak-memory reduction. The conservative working-set estimate was 1,368,653,824 bytes against the 4 GiB hard limit; preflight free VRAM was 40,152,858,624 bytes. Driver 580.126.20, CUDA runtime 13020, CuPy 14.2.0, NumPy 2.2.6.

The targeted GPU rank and pairwise-rank regression set passed 17 tests in 0.59 seconds, including ties, NaN and infinities, all-invalid rows, non-last axes, forced chunks, pct ranks, distinct counts, and pairwise-mask semantics. This benchmark isolates `batched_rank`; it is not a full Spearman IC end-to-end measurement. The observed timing improvement is modest. Full QE and real F48 validation remain separate integration gates.

Source SHA256: baseline rank `5ca3807ec1e56a61830dd70eeb036dee43e9d42fcd822f7ed668edc5a1a08ae8`; current rank `e3a67854126e5e63ffadb79c4605c18d8ac1e623bc3ce2764b2d99148133b141`; `sorted_rank_runs.py` `77736a528d0da660b34b42b9baa8b25cb699c70a680fc17cfb5eb673e5bdc1be`.
