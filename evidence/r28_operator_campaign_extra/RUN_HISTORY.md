# R28 extra run and triage history

## Attempt 01

The original 20-operator campaign is retained unchanged. All 20 main and edge
parity checks passed. Sixteen operators passed every execution check; four
needed protocol triage: two oracle mismatches and two all-NaN fixtures.

## Four-operator root cause analysis

- `ts_mass_concentration` is normalized HHI, `(HHI-1/n)/(1-1/n)`. The first
  independent oracle computed raw HHI only.
- `vv1_downside_vol_share` is the sample standard deviation of negative returns
  divided by the sum of separate negative- and positive-return sample standard
  deviations. The first oracle incorrectly used downside/upside RMS.
- `ts_vol_of_vol` uses the same `min_periods` inside its five-row inner window.
  Attempt 01 specified `min_periods=8`, making every inner estimate undefined.
  Attempt 02 uses the meaningful valid combination `inner_window=5,
  min_periods=4`.
- `ts_roll_effective_spread` is undefined when log-price changes have
  nonnegative lag-one covariance. Attempt 01's smooth path correctly produced
  NaN. Attempt 02 adds a positive alternating bounce-price path with negative
  lag-one covariance and independently verifies `2*sqrt(-cov)`, while also
  asserting that the smooth and constant paths remain all-NaN.

At the conclusion of attempts 01-02 these were classified as campaign oracle,
recipe, and fixture defects, and no operator source had yet been changed. A
subsequent API-contract review found the separate fail-open validation defect
documented under attempt 03 below.

## Attempt 02

- Script SHA-256: `0dc99f27b364f421b0a797242ed3edd1c256d7db6f7b1fe2adab0b11f1da7d61`.
- Recipe SHA-256: `1c4a07925e50cbb789387098b96bbc2c734faf4f0ea576576b753c85f92d63e4`.
- Runtime: 84.804096597014 seconds; sampled peak family RSS: 751,247,360 bytes;
  guard reason: none. The watchdog is sampled RSS, not a kernel hard cap.
- All eight backend executions had finite observations, causal-prefix
  invariance, a changed suffix, and independent numeric-oracle equality.
- All four pandas/Polars parity checks passed.
- Roll spread's two degenerate-path NaN assertions passed on both backends.

Both attempts' logs, watchdog JSON, and ledgers are retained.

## Root review: Attempt 04

Earlier triage records did not bind the parity row to exact backend fingerprints.
Root preserved those historical records and re-executed all four triaged operators
with attempt04.py, recording both fingerprints and the common explicit
future_prefix_invariant_all_inputs field. All eight backend records and four
paired comparisons passed, including the repaired ts_vol_of_vol implementation.
Runtime 79.289 seconds; sampled peak family RSS 749608960 bytes; no guard fired.
The watchdog remains sampled, not a kernel hard memory cap.

## Parameter-contract repair and attempt 03

Although `min_periods=8, inner_window=5` was a bad attempt-01 recipe, the public
API silently accepted this guaranteed-all-NaN domain. `TsVolOfVol` was repaired
locally in `factor_engine/cleaned_operators/alpha_language_volatility.py`:

- strict integer validation now rejects fractional and Boolean `min_periods`;
- runtime validation requires the shared `min_periods` to be no larger than
  either `inner_window` or `outer_window`;
- metadata declares both relations so pre-lowering/DSL validation and direct
  Python execution enforce the same domain.

The economic calculation and legitimate degenerate-window NaN behavior were
not changed. Targeted tests cover both backends, both relations, invalid scalar
types, a valid finite case, metadata, and pre-lowering validation.

Attempt 03 reran only the repaired `ts_vol_of_vol`:

- Source SHA-256: `36665a758a206dd447ac5e3de05af6b62508784e8176b439ace7e34e659dd95b`.
- Script SHA-256: `2770d169b9b0caa17ff7b05803f024fea40227c98aa0e680237e6f7b0509db09`.
- Recipe SHA-256: `fe78f4d704ca45c4f0ff8100a014a4641dbe1e9e68fc72fcd9fb178c958056b9`.
- Runtime: 77.94359269994311 seconds; sampled peak family RSS: 749,715,456
  bytes; guard reason: none.
- Both backends produced 464 finite values, preserved the causal prefix,
  changed on the future suffix, matched the independent oracle, and matched
  each other.

Attempt 03's log, watchdog JSON, and ledger are retained alongside the first
two attempts.
