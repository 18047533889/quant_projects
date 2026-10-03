# Fused centered Pearson reduction: CUDA A/B

For each row, let $P$ be the indices where both inputs are finite. The RawKernel uses a first pass to accumulate counts and sums, reduces the means, then rereads the row to accumulate centered variances and covariance:

$$
n=|P|,\quad \bar{x}=\frac{1}{n}\sum_{i\in P}x_i,\quad
\bar{y}=\frac{1}{n}\sum_{i\in P}y_i,
$$
$$
v_x=\sum_{i\in P}(x_i-\bar{x})^2,\quad
v_y=\sum_{i\in P}(y_i-\bar{y})^2,\quad
\operatorname{cov}=\sum_{i\in P}(x_i-\bar{x})(y_i-\bar{y}),\quad
IC=\frac{\operatorname{cov}}{\sqrt{v_xv_y}}.
$$

Rows remain invalid when `n < min_obs` or either variance is nonpositive. The unchanged unsafe-row predicate still sends overflow, underflow, and high-offset rows to the existing scale-normalized repair. The fusion removes the full-size centered-value and product arrays from the ordinary Pearson reduction.

At $T=512,N=1000$, the synchronous `_pairwise_finite_sums` call measured 5.50x faster for 32 factors and 5.84x for 48 factors versus the implementation in pin `249113d554f2a88c03879e0a776042af26db002e`. These timings include pairwise-finite mask creation, Python dispatch, metadata allocations, and reduction work. They measure this batched reduction path only; they are not end-to-end evaluator, full application, or COS performance claims. Receipt revision 2 is authoritative; it replaces the earlier run after final count-width and grid-boundary changes.

Both f64 cases had equal valid counts and NaN masks, and maximum absolute IC error was `5.56e-17`. Each passed `rtol=1e-8, atol=1e-10`. The run used 2 warmups followed by 3 synchronous ABBA rounds. The conservative working-set estimate was 1.36 / 2.04 GiB; the CuPy pool high-water was 0.65 / 0.79 GiB against a 4 GiB per-process pool check. This is allocator accounting, not a physical device-memory guarantee. The run started with 37.39 GiB free on the L20. Full samples, seeds, source hashes, runtime versions, and memory observations are in [pearson_centered_fused_ab_20261003.json](pearson_centered_fused_ab_20261003.json); the reproducible runner is [benchmark_pearson_centered_fused_ab.py](../../scripts/benchmark_pearson_centered_fused_ab.py).

The public CPU/CUDA Pearson chain was also checked at `T=8, N=64, F=32/48` for `pearson_ic_series`, `pearson_ic`, `pearson_ic_std`, and `pearson_ic_ir`, with missing values, validity masks, and `min_assets=20`. The 19-observation row stayed invalid while the 20-observation row remained valid. Maximum absolute errors were `1.11e-15` (32 factors) and `5.55e-16` (48 factors); both passed the same house tolerance. This small correctness case is separate from the latency A/B.

Scope is limited to these two latency shapes, f64 random normal inputs, and the stated pairwise missing-value rates. f32/f64 parity, missing values, minimum-count boundaries, noncontiguous and negative-stride views, mixed f32/f64, and existing extreme/offset/repair cases were checked separately. The benchmark does not characterize repaired extreme rows, other GPUs, other factor/asset sizes, or full evaluator throughput.
