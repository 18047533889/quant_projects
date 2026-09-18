# R49 membership transition repair

Strict whole-array parity exposed index_reconstitution_churn counting the first observed member as a new inclusion (r48-daily-parity-strict-root.log). Both Polars implementations also fabricated transitions around missing membership.

- Shared native Polars kernel counts changes only between two known adjacent states. First observation and missing boundaries contribute no event.
- Finite non-boolean values reject; null/NaN/Inf are unknown, including the pandas implementation.
- Canonical member parameter spelling reconciled in common Polars implementation; standard time/identity columns preserved.
- Semantic version bumped to 2.
- Independent explicit-event oracle, future perturbation, windows 1/3/6, invalid bool cases: 4 passed in evidence/r49-membership-churn-root-attempt2.log.
- First new test run exposed additional member/membership contract mismatch, retained in evidence/r49-membership-churn-root.log.

No production output publication; no CSV change.
