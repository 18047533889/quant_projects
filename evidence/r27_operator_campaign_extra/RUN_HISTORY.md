# R27 extra campaign run history

The locked batch contains 30 category, event, decay, and seasonal operators on
deterministic legal daily panels. The watchdog is a sampled RSS guard, not a
kernel-enforced hard cap.

## Attempt 01

- Recipe SHA-256: `c3d5be03ff796766accb3d38d4d73b992d29d3a988b699c89a528efc1cad966a`.
- Script SHA-256: `d01ae97f29a485b854ed0eeff1d3ad3950e4ee086f23a34a52f867127cf9f3a6`.
- Runtime: 65.45950585301034 seconds; sampled peak family RSS: 421,236,736 bytes; guard reason: none.
- 60/60 backend executions had finite observations, causal-prefix invariance,
  and nontrivial perturbed suffixes.
- 30/30 main, constant-panel, and single-missing-bar backend parity checks passed.
- Six independent numeric oracles passed on both backends (12/12 checks).
- Four operators were entirely undefined on their deliberately degenerate
  constant panels (`category_age`, `category_transition_surprise`,
  `cs1_week_cycle_phase`, and `event_cluster_duration`); both backends agreed.
  The other 26 operators had finite constant-panel observations, and every
  operator had finite observations on the single-missing-bar panel.

No operator source was changed. The attempt log, watchdog JSON, and ledger are retained.
