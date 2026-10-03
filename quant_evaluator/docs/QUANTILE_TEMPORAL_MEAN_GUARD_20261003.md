# Quantile temporal mean guard (2026-10-03)

## Linear-shape arithmetic and bounded follow-up (2026-10-04)

Six CPU/GPU shape metrics now selectively repair overflow/cancellation and
subnormal arithmetic with exact binary-rational sums followed by division and
one float64 rounding. Constant huge curvature is zero, not negative infinity;
a finite 1e308 mean adjacent spread/extreme cliff stays finite. True result
overflow remains signed infinity. The formulas, finite stencil/pair sets and
risk threshold are in [metric conventions](METRIC_CONVENTIONS.md#41-六项线性形状指标的公式与数值边界2026-10-04).
The GPU finite-mean and shape kernels share a dyadic accumulator module;
workspace admission still uses actual compiled local-memory metadata. An
actual CUDA focused combined run reported 82 passed / 3 skipped; the skips
require a second device. This is not a proof of every metric/backend.
The final guard uses an inclusive error threshold, matching CPU at equality.
New tests check the previous/equal/next binary64 values at that threshold
and cancellation near the GPU underflow threshold against Fraction.
A separate CPU/mock regression run passed 62 tests with warnings as errors.

[CPU ABBA receipt](benchmarks/shape_linear_guard_ab_20261004.json) compares six
shape functions against frozen HEAD implementations at Q20/F48, including
normal and missing-value profiles. Every measured call retains ordinary-input
bitwise parity. New/old latency ratios for curvature/adjacent spread are
0.0925/0.1118 (normal) and 0.651/0.664 (masked); the four endpoint metrics also
improve. These are profile-kernel measurements, not full evaluation latency.

[T256 guarded ABBA](benchmarks/gpu_finance_guard_ab_T256_20261004.json) at
N5461/F48 gives Q20 0.760376 s generic versus 0.298384 s candidate, but Q5
0.179577 s versus 0.225107 s. [Q5 capacity-8 follow-up](benchmarks/gpu_finance_guard_specialized_Q5_T256_20261004.json)
still regresses: 0.192184 s versus 0.239586 s (24.7% slower), despite lower
compiled local memory (224 bytes/thread). All 18 calls per Q compare against
the CPU reference. Sources/input/runtime identities remain stable per receipt.
No blanket/default finance-guard replacement is justified. Cold-call timing
includes compile/setup variance. Whole-source COS auto qualification remains
a separate requirement, including memory, source drift and batch-size gates.

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

## Compiled local-memory metadata admission (2026-10-04)

Both the established numeric guard and the bounded-Q candidate must reject
missing, negative, boolean, fractional, string or nonfinite local-memory
metadata. Only a non-negative integral byte count is admitted. In particular,
a negative count must not reduce the workspace estimate. Actual scratch adds
the compiled per-thread byte count multiplied by the rounded launch size:
`ceil(rows / block_size) * block_size`. Global scratch and retained outputs
are additional reservations, not included in this local-memory term.

The CPU-only metadata regression reports **26 passed**, including valid alias
attributes, NumPy integral values, zero rows and block-rounding boundaries.
This validates admission semantics, not speed or GPU numerical accuracy.
The new finance guard remains a candidate pending guarded-versus-guarded A/B;
these checks alone do not qualify it for default dispatch.

## Run-local temporal profile reuse

`GPUExecutor.run()` shares the un-gated temporal profile and finite day counts
between `quantile_returns_full` and `quantile_monotonicity`. For each quantile
and factor, the profile is the finite daily return sum divided by the finite
day count, computed by the guarded finite mean. Each consumer independently
sets its result to NaN when the count is below that consumer's `min_periods`.
The key is `(n_quantiles, min_assets)`; cache lifetime is one run with fixed
staged inputs. It cannot leak between changed factor or label masks. Spread
still reduces its own top-minus-bottom series. Each retained pair reserves
`16 * Q * F` bytes, and one helper owns the cache-miss admission/reduction.
Six CPU executor mocks cover parameter/mask isolation, both metric orders,
independent gates and workspace reservations. The combined executor/parity
run reports **37 passed**. This avoids one duplicate reduction structurally;
no end-to-end speedup has yet been measured for the reuse change.

## Guarded-versus-guarded candidate timing

[The bounded A/B receipt](benchmarks/gpu_finance_guard_ab_20261004.json) compares
two implementations with exact repair enabled on T128/N5461/F48. Q5 existing
median is 0.093974 s versus candidate 0.115057 s (candidate 22.4% slower);
Q20 is 0.376684 s versus 0.147832 s (candidate 60.8% lower latency).
Each Q checks 18 calls total across both modes, including two cold calls,
one ABBA warmup and three measured ABBA rounds (six samples per mode).
Input/source/runtime identities remain stable. The candidate is **not**
enabled in production; Q5 regression forbids blanket replacement. Broader
shape qualification and full source/COS routing evidence remain required.
