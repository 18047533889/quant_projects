# `ts_skew` centered-moment repair evidence

## Defect and contract

The selected research `pandas_numpy` winner returned
`3.274180926381983e-11` for the symmetric finite window `[1, 2, 3]`.
An independent Decimal-50 centered-moment oracle returns exact zero. This was
not a `min_periods` mismatch: the rolling contract begins at three finite
samples, skips NaN/Inf, includes the current row, and uses unbiased
Fisher-Pearson sample skewness.

The repair uses finite filtering, anchor-centering, window-local scaling, and
centered normalized moments. If subtraction of opposite-sign finite extremes
overflows, it pre-scales only the current window before re-anchoring; no finite
sample is clipped or dropped. Constant windows return zero and fewer than
three finite samples return NaN.

## Bounded verification

- Project environment (pandas 2.3.3): 7 passed; sampled peak process-family
  RSS 471,814,144 bytes; 48.63 seconds; no watchdog stop reason.
- System environment (pandas 3.0.5): 7 passed; sampled peak process-family RSS
  410,689,536 bytes; 47.12 seconds; no watchdog stop reason.
- Cases cover exact symmetry, scales `1e-100`, `1`, and `1e100`, translation by
  `1e12`, constants, fewer than three finite observations, NaN/Inf masking, and
  the finite extreme window `[-1e308, 0, 1e308]`.

Evidence files are `evidence/r3/ts-skew-stability-final-py312.log`,
`evidence/r3/ts-skew-stability-final-py312-watchdog.json`,
`evidence/r3/ts-skew-stability-final-py305.log`, and
`evidence/r3/ts-skew-stability-final-py305-watchdog.json`.

## Performance boundary

The research pandas path now uses `rolling.apply` with one Python callback per
window. It is approximately O(N * window) and slower than pandas' native
rolling skew kernel; memory remains bounded to rolling-window work. This is a
correctness repair for a research candidate, not a claim that it is the
default fastest path or production-certified.
