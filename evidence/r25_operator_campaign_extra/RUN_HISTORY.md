# R25 extra technical campaign run history

The campaign uses a deterministic legal daily OHLCV panel (64 rows by eight
assets), a suffix-only perturbation, a constant-price edge panel, and a panel
with one missing OHLCV bar. The watchdog is a sampled RSS guard, not a
kernel-enforced hard cap.

## Attempt 01 (retained)

- Recipe SHA-256: `9b8ea1d08c6e7fd13b5cdc8f0b823d9d780ddebf23ecce1015d207f702b00f0f`.
- Script SHA-256: `76c1dbbde0782a906325e39d4573ef59e62f11d6645bb3cc318f4977341fa363`.
- Runtime: 94.77944900793955 seconds; sampled peak family RSS: 749,092,864 bytes; guard reason: none.
- All 40 pandas/Polars main outputs and both edge panels had backend parity.
- Ten backend-level records failed only the suffix-change requirement:
  `donchian_breakout_up`, `keltner_breakout_strength`,
  `supertrend_days_since_flip`, `supertrend_direction`, and
  `supertrend_flip` on both backends. The initial legal perturbation did not
  cross their discrete breakout/regime thresholds.

No operator was replaced and no operator source was changed. Attempt 01's log,
watchdog JSON, and ledger remain in this directory.

## Attempt 02 (retained final protocol)

The suffix perturbation was strengthened to deterministic alternating large up
and down moves while preserving positive, nested OHLC bars. This exercises
breakouts and regime flips without changing the historical prefix.

- Recipe SHA-256: `9b8ea1d08c6e7fd13b5cdc8f0b823d9d780ddebf23ecce1015d207f702b00f0f`.
- Script SHA-256: `96bc53609e28bafb3e87b6b57c0efda696ab866b095707580111f22b6e5da047`.
- Runtime: 95.83314083213918 seconds; sampled peak family RSS: 749,494,272 bytes; guard reason: none.
- 80/80 backend executions produced finite observations, preserved the prefix,
  and changed on the perturbed suffix.
- 40/40 pandas/Polars main-output parity checks passed.
- 40/40 constant-panel and 40/40 single-missing-bar parity checks passed.
- Twelve independently specified numeric oracles passed on both backends
  (24/24 backend-level checks).
