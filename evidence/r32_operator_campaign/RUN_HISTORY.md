# R32 operator campaign run history

## Frozen result

The final current-source campaign is `ledger_final_current_source_attempt_2.jsonl`.
It contains 60 canonicals, 120/120 `EXECUTED_FINITE` backend results, 120/120
future-prefix invariance checks, and 60/60 `CANONICAL_PARITY` results.  Its four
15-operator logs are `run_final_current_source_attempt_2_batch_{0,15,30,45}.log`.
All four batches used the same final source, merged corrected recipe file, nonlinear
regression fixture, irregular explicit `date` coordinate, and bounded 96-row input.

The focused final-source suite is
`focused_tests_final_current_source_attempt_1.log`: 21 passed.  It includes independent
AR and Poly2 oracles, missing observations, irregular bar-coordinate timing, strict-prior
current-row perturbation, metadata preservation, and mismatched-axis rejection.

## Repairs and triage

- Ten AR implementations were true parity defects.  They now reuse the versioned
  NumPy AR kernel with exact current/prior alignment and warmup semantics.
- Three two-input implementations had true signature/dispatch defects:
  Huber in-sample residual, Ridge in-sample residual, and Quantile prior coefficient.
- Three Poly2 prior/forecast implementations had true parameter and bar-coordinate
  defects.  They now reuse the canonical NumPy kernels and match an independent
  `numpy.linalg.lstsq` oracle.
- The configured-history failures (including KAMA and multi-feature regressions) were
  recipe defects; corrected parameters are in `recipes_final_current_source.json`.
- Four regression Z outputs were all-NaN because the original response was exactly
  explained by a predictor.  This was a fixture defect; the final nonlinear fixture
  produces finite residual variance.
- `ts_best_lag_corr_excess` initially lacked its required time coordinate.  This was a
  fixture-protocol defect; the final campaign supplies an irregular explicit `date`.
- Original failed ledgers and retry ledgers remain present and are not rewritten.

## Physical execution declarations

The 16 repaired Polars classes declare `ExecutionKind.DELEGATE_PYTHON`, eager execution,
full-panel materialization, no lazy/streaming support, and non-native/non-production-
eligible status.  The Quantile prior coefficient declaration explicitly records
`Polars -> NumPy -> Pandas -> NumPy -> Polars`; the other repaired delegates record their
NumPy/Pandas CPU path.  These declarations do not advertise native Polars or GPU support;
automatic production selection must use the compliant pandas implementation.

## Time-identity coverage and known gap

The final 60-operator campaign supplies an irregular explicit `date` axis.  It is not
evidence that every operator was independently tested with every supported time-column
name or with a long-panel ordered instrument axis.  Focused tests additionally exercise
an explicit `timestamp` column for the three repaired Poly2 operators and the three
repaired two-input regression operators, including shifted-time and reordered-instrument
rejection for the two-input cases.

`ledger_final_current_source_attempt_1.jsonl` is deliberately retained.  Its all-60
`timestamp` run found a real unresolved metadata-isolation defect in the effective
`polars_robust_stats` registrations for `ts_poly2_coeff` and `ts_poly2_resid`: they treat
`timestamp` as a numeric value column.  This is not dismissed as a harmless fixture
failure and is outside the frozen source scope here; it needs a separate repair and
timestamp oracle in `polars_robust_stats.py`.

## Evidence-preservation exception

Two early focused-test outputs were accidentally overwritten before the unique-log rule
was restated.  Their failures were test-harness issues: direct access to a nonexistent
`physical_spec` property instead of the backend classifier, and a Quantile test fixture
with `window=8` below the required configured history of 10.  The original raw files are
not recoverable and were not reconstructed or represented as original evidence.  The
authentic final 8-pass Poly2/spec run is retained as `poly2_test_attempt_3_pass.log`;
all later logs and ledgers use unique names.

No commit, push, deployment, or production-factor publication was performed.
