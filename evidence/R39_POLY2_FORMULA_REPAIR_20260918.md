# R39 Poly2 formula, missingness and metadata correction

## Behavior-changing repair
Both NumPy and Polars ts_poly2_resid incorrectly treated numpy.polyfit's
[quadratic, linear, intercept] result as [intercept, linear, quadratic].
Both now evaluate the true fitted polynomial. Existing factor values using
this operator may change and should be recomputed; this is not metadata-only.

Polars also previously evaluated at the last finite predictor rather than
the current predictor when current x was missing. It now preserves the
current-row missing result. Robust-statistics value-column selection uses
the shared metadata vocabulary, excluding timestamp and other axis fields.

Both residual kernels reject windows with fewer than three distinct finite
predictor values before rank-deficient LAPACK fitting. Constant and two-value
predictors return missing results, not arbitrary coefficients or low-level
LAPACK diagnostics. No parameter or production-surface guard was relaxed.

## Verification and ledger
- Independent exact quadratic with distinct intercept (7.25) and quadratic
  coefficient (0.18) must yield zero residual.
- Independent least-squares oracle; date/timestamp, empty, NaN, y-only/x-only/
  paired missingness, shifted/reordered axes and future-prefix cases.
- Additional rank-deficient cases reproduced 2 failures before the guard.
- Root final regression: evidence/r39-poly2-rankguard-final.log:
  **26 passed**, including prior Poly2 and actual shared-DAG run_many cases.
- Latest paired evidence:
  evidence/r37_poly2_metadata/ledger_attempt_5.jsonl:
  **4 finite executions,4 prefix checks,2 parity pairs**.
  Root read and matched the real SHA256 hashes of both kernel files and
  paired_campaign_attempt5.py. Fixture/protocol/parameters are also recorded.

Earlier attempts are retained. R32's date-only results and R37 attempts1–4
are historical, not current-code certification after these changes. These
two canonicals were already in the historical 514 count, so no count increase.
