# R24 extra campaign run history

This note records facts observed during campaign construction that were not
retained as raw logs. It does not recreate or claim to be the deleted logs.

## Initial protocol failure

- Result: process exit code 1 before a ledger was created.
- Cause: the new script was named `campaign.py` and `import campaign as core`
  imported itself instead of the R23 shared campaign helper.
- Error: `AttributeError: module 'campaign' has no attribute 'execution_fingerprint'`.
- Recipe SHA-256: `ab6b05f3a5ca120b47324274eca8b415c38ad1f665f2fee066225c01349d7f68`.
- Script SHA-256: `39eb50cf9cbd8024813556148a76a34dfc29e61368859b4660b955ab468f5ec0`.
- Watchdog observations: 81.23 seconds, sampled peak family RSS
  747,765,760 bytes, no guard reason.

## First completed ledger and triage

The first completed 40-recipe run produced parity results for 38 operators and
six backend-level failure records:

- `open_close_return` and `overnight_return` failed closed on both backends
  because the recipes did not explicitly declare `price_basis="raw"`.
- `true_range_pct` passed backend parity on both backends but failed the
  independent oracle because the campaign oracle divided by current close;
  the operator contract divides by previous close.

These were campaign recipe/oracle defects, not operator implementation defects.
No operator was replaced. The R24 recipe and oracle were corrected, and the
generated ledger was rerun from a clean file.

## Final retained run

- Watchdog return code: 0.
- Runtime: 95.79116491600871 seconds.
- Sampled peak family RSS: 748,765,184 bytes.
- Guard reason: none. The watchdog is a sampled RSS guard, not a
  kernel-enforced hard cap.
- Recipe SHA-256: `ceb51b25e82c26e7212b01e3201095e9cec10e8dd409348826c93d98df28252b`.
- Script SHA-256: `21324c3b5cb50f9313f36b1d65ec7bf80dea151452995226086b720a3c9f4fcb`.
- Ledger SHA-256: `9d904dbc38a9056a153842b30f663319b0d50ba72e89436e9cd1cae0d04e422f`.
- Final ledger: 80 `EXECUTED_FINITE`, 40 `CANONICAL_PARITY`, all prefix,
  future-suffix-change, parity, and 20 backend-level oracle checks passed.

The raw watchdog and initial failure logs were temporary files and were removed
before this preservation requirement was clarified; they are not recoverable
from this evidence directory.
