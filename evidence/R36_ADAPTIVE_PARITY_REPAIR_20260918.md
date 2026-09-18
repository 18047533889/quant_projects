# R36 adaptive backend test repair

The old dual-backend test passed a pandas DataFrame to Polars implementations,
then skipped AttributeError containing 'with_columns'. Consequently five
backend comparisons never actually ran.

The test now supplies a real Polars panel with explicit date metadata, checks
date identity, and compares numeric results against pandas. Missing backend
slots or runtime errors fail rather than silently skip.

Root execution: evidence/r36-adaptive-parity-real.log: **45 passed**, no skips.
Five actual parity comparisons: ts_mcginley_dynamic, ts_vidya,
ts_one_euro_filter, ts_nlms_filter, ts_rls_filter.
Watchdog wall 54.138s; sampled process-family peak RSS 801284096 bytes.
No production implementation or eligibility change; no increment to the
strict paired campaign count from this test suite.
