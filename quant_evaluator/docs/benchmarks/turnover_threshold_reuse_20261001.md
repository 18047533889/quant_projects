# GPU membership-turnover threshold reuse

QE computes each date/factor quantile threshold once and carries one threshold
row between bounded chunks. Previously it sorted both sides of each adjacent
date pair. The regression counter now checks exactly `T * F` sorted rows,
including forced one-pair chunks. It retains the 128 MiB workspace estimator;
one cross-section is the chunk floor, so this is not a hard allocator cap.

For adjacent dates, let V_t be the finite-asset set and S_t the selected set
using that date's finite-value linear quantile. Membership includes ties.
The default diagnostic is

$$r_t = |S_t \triangle S_{t+1}| / |V_t \cup V_{t+1}|.$$

Both dates must have at least ten finite assets. A selected asset whose next
value is unavailable makes the result unknown (NaN), rather than implying an
exit. This is a factor-membership diagnostic, not holdings turnover.

A synthetic float64 `(64, 8, 5000)` device-layout panel used 2% NaN and 0.05%
Inf values, one warmup and five timed calls. Independent CuPy pools each had
a 512 MiB limit. Paired results agreed. Median GPU event time was 3.462 ms
before reuse and 2.894 ms after; synchronized wall times were 3.471 and
2.918 ms. Pool reservation after calls was 115.1 MB before and 107.0 MB after.
Reservations are not live peaks. This kernel experiment does not measure
end-to-end real-COS evaluation latency.

The targeted 15-test run covered CPU parity, ties, missing/Inf values, top and
bottom quantiles, chunk boundaries and empty/small dimensions. A full QE
regression after the kernel change passed 4396 tests, with 26 skips.
