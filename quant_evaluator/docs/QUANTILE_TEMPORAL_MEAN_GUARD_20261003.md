# Quantile temporal mean guard (2026-10-03)

## Mean contract

For each quantile/factor column, the temporal mean is

\[
m = \frac{\sum_{t:\,x_t\ \mathrm{finite}} x_t}
         {\#\{t : x_t\ \mathrm{finite}\}}.
\]

NaN and both infinities are excluded from the numerator and denominator. The
returned count is the number of finite observations. If that count is below
`min_periods`, the mean remains NaN. The shared CPU helper requires
`min_periods` to be a positive integer; booleans and fractional values are
rejected.

Ordinary columns use vectorized Float64 summation and division. The helper
reuses `quantile_numeric.label_sum_error_bounds` to estimate a conservative
per-mean summation error. A sufficient column is repaired when that bound is
greater than `1e-12`, or when the vectorized result is nonfinite. Repair calls
`quantile_numeric.stable_finite_mean`, which sums the exact rational values of
the Float64 samples and converts the divided result to Float64 once. This
retains the ordinary vectorized path while avoiding overflow and risky
cancellation.

For example, the ordinary Float64 sum of 25 copies of `1e308` overflows to
infinity, although their mean is exactly `1e308`. Likewise, the finite sequence
`[1e308, 3, -1e308, 0]` has mean `0.75`; loss of the low-order `3` during an
ordinary reduction must not determine the result.

## Integration and verification

CPU full-return/spread adapters, the evaluator's nonwindowed and full-window
profiles, and public observation-count profiles use `finite_mean_axis0`.
Windowed inputs swap the window/sample axes before reduction and retain the
requirement that every sample in a full window be finite.

Shape stability also needs guarded leave-one-window-out profile means and
scale-safe Pearson correlation: a finite profile near `1e308` can overflow
inside ordinary correlation even after its time mean has been repaired.
`shape_evidence` reuses the existing IC Pearson implementation, preserving
the ordinary operation order and its scale-safe fallback. Fisher-z clipping
at correlation magnitudes `1 - 1e-9` is unchanged.

The GPU temporal helper remains device-local. It reuses the exact quantile-mean
kernel with a single synthetic bucket per temporal column, without copying
data values to CPU. Its conservative preflight is `40*T*C + 16*C` bytes.
Before repair, it releases transient references and subtracts the actual live
labels, bucket IDs, means, and counts from its admitted workspace. The repair
receives only the positive remainder and independently admits its normal and
conditional exact-kernel workspace, including measured RawKernel local memory.
The device session accounts for existing live allocations and output reserve.

Regressions include huge constants, cancellation, subnormal rounding, finite
mask/count thresholds, public CPU/GPU full/spread/monotonicity, public windowed
stability, identical three-window affine profiles, and budget forwarding.
The final integrated 12-file run reported **128 passed, 2 skipped**: both skips
require a second CUDA device not present on the test host. A separate shape
and ordinary-equivalence run reported **118 passed**, with two existing
empty-bootstrap-slice warnings. These overlap and must not be summed as unique
tests. Earlier focused suites reported 21 passed; 92 passed/1 skipped;
124 auto/calibration tests passed; and 199 quantile/reuse/equivalence tests passed.

## Measured GPU guard cost

The source/input/runtime-bound synthetic report is
[GPU quantile guard A/B](benchmarks/gpu_quantile_numeric_guard_20261003.json).
It uses T=128, N=5461, F=48, finance-scale labels with ties and nonfinite masks,
one ABBA warmup and three measured ABBA rounds. Each mode has six distinct
warm samples per quantile count. CUDA is synchronized around each callable;
host layout, HtoD, cold calls, and untimed CPU-reference computation are separate.

| Quantiles | Guard enabled median | Diagnostic bypass median |
| --- | ---: | ---: |
| 5 | 0.091535 s | 0.059095 s |
| 20 | 0.374446 s | 0.082255 s |

HtoD measured 0.086303 s. All 18 calls per Q (36 total) matched the current
NumPy CPU reference: counts exactly, returns within rtol=1e-9/atol=1e-12,
including NaNs. Separate Decimal oracles validate the extreme-risk paths.
Source, input, device-input, and actual runtime hashes were unchanged.

Bypass disables numeric repair **only for diagnosis**, never for production
routing. Q20 guard overhead remains open. Repeated full-width bucket scans
are a candidate hotspot, not a measured root cause until risk distribution
is inspected. This report is neither full-QE/COS timing nor a CPU/GPU backend
winner qualification; it cannot populate the default-auto winner registry.
It does not prove every metric globally fastest or bug-free. Broader
multi-year/full-market route qualification and the remaining metric-family
audit remain within the continuing scope.

## Risk distribution follow-up (2026-10-04)

[Risk-count report](benchmarks/gpu_quantile_guard_risk_profile_20261004.json)
uses the same input and metric-source hashes as the preceding ABBA cohort.
Both Q5 and Q20 executed seven guard calls, returned zero final-risk buckets,
and launched the exact fixed-mean kernel zero times. Thus exact-repair work
does not explain this cohort's guard overhead. Risk classification/refinement,
repeated scans and synchronization remain candidates; this probe does not
measure timing or identify their individual shares.

The probe reduces risk metadata on GPU and copies only Q-length count vectors
to host. Its RawKernel wrapper is restored in a finally block. Seven mock
tests cover wrapper behavior and accurate Linux MemAvailable admission before
allocation. An additional 34-test shape suite uses independent Decimal Pearson
and Fisher-z oracles for encoded huge profiles, plus binary-exact affine
profiles; decimal increments near 1e308 must not be presumed exactly affine
after binary64 quantization. The combined shape/probe run reported 41 passed.

## Bootstrap profile repair (2026-10-04)

The confidence and rank-agreement bootstrap metrics also used unsafe sample
profile means. Four identical, strictly increasing, ULP-exact huge profiles
returned NaN instead of rank agreement 1. Both sample reductions now reuse
the guarded finite mean; confidence retains vectorized resample reduction,
rank agreement retains its bounded per-sample memory behavior, and the shared
random draw schedule is unchanged. Exactly three common finite quantiles still
qualify, while two remain missing evidence. The two regressions were red before
the repair and green afterward. Final combined 16-file numerical, public GPU,
mask/cache, shape, bootstrap, benchmark-probe, and equivalence validation:
**287 passed, 2 skipped** (both require a second CUDA device), no warnings.
