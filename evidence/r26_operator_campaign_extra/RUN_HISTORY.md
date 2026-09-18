# R26 extra campaign run history

This locked 30-operator campaign uses deterministic legal daily financial and
OHLCV panels. It stops at 30 rather than guessing additional field semantics.
The watchdog is a sampled RSS guard, not a kernel-enforced hard cap.

## Attempt 01

- Recipe SHA-256: `21c71204407df1686b8a20e05c964655a4cac1e6976f530fb98163c037fd37f6`.
- Script SHA-256: `d816d2fdc9d56dfb09268073bf60ce08dcc6b1528e02207cc88a533306baf1fb`.
- Runtime: 63.97534114611335 seconds; sampled peak family RSS: 420,315,136 bytes; guard reason: none.
- 60/60 backend executions had finite observations, causal-prefix invariance,
  and a changed future suffix.
- 30/30 main pandas/Polars parity checks passed.
- 30/30 constant-panel and 30/30 single-missing-bar parity checks passed.
- Eleven independent numeric oracles passed on both backends (22/22 checks).
- The constant panel correctly produced all-NaN outputs for five degenerate
  correlation/variance ratios: `ep1_earnings_autocorr`,
  `es1_earnings_mean_reversion_speed`, `es1_earnings_smoothness`,
  `val1_pe_beta_to_market`, and `vr1_parkinson_close_scale`. Both backends
  agreed. The other 25 operators produced finite constant-panel observations;
  all 30 produced finite observations on the single-missing-bar panel.

No operator source was changed. The attempt log, watchdog JSON, and ledger are
retained in this directory.
