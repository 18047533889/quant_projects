# Bidirectional gap-fill correction

The old active ts_gap_fill_ratio implementation counted continuation away from the previous close as gap completion, multiplied two rolling proportions instead of counting completed gaps, and returned all NaN for down-gap-only samples. An older parity test locked these incorrect outputs; matching backends alone did not establish financial correctness.

The corrected definition is: among days with an opening gap in the trailing full observation window, count an up gap as filled if close <= previous close and a down gap as filled if close >= previous close. Non-gap days do not enter the denominator. Require finite positive close/open/previous-close triplets across the full window; invalid data produces NaN until it leaves the window. A window with no gap remains undefined, not zero.

Semantic version is now 2. The existing public signature remains close, open, pre_close, window. This uses close-at-end-of-day completion, not an intraday high/low touch proxy.

Before: evidence/r56-gap-before-root.log, 12 failed / 2 passed.
Final: evidence/r56-gap-final-root.log, 97 passed / no skips. Includes 16 new hand-calculated financial-semantic cases plus existing backend, catalog recipe and semantic-identity regressions. Both active pandas_numpy and Polars paths pass. Coverage: both directions, zero gaps, no-gap denominator, NaN/Inf/nonpositive prices, timestamps, distinct stocks, prefix causality, price scales 1e-200 and 1e200.
Watchdog: sampled peak466452480 bytes, elapsed63.09786677593365 seconds including startup, no guard trigger. This is not a hard memory cap or throughput measurement.

R28 full-canonical fixture rerun and the 110k real-factor run are still incomplete. No CSV update or production factor publication was performed.
