# R30 extra run history

## Attempt 01

Executed successfully at the process level.  This campaign covers the locked liquidity, order-flow,
A-share state, and event-clock operators with bounded deterministic panels,
causal future perturbations, constant/missing edges, backend parity, and four
independent numeric oracles.

All 30 pandas/Polars parity comparisons passed.  Twenty-six canonicals passed
every execution check.  Four cross-sectional liquidity operators failed only
the suffix-response check on both backends because the first future fixture
multiplied every asset in a row by the same scalar, leaving shares,
concentration, and cross-sectional skew invariant.  Attempt 01 is retained;
attempt 02 uses asset-asymmetric tail shocks to repair that protocol defect.

## Attempt 02

All 30 backend parity comparisons again passed and 29 canonicals passed every
execution check.  `lf1_volume_concentration` remained invariant because the
moderate asymmetry still required the same count of top assets to cross its
25% cumulative-volume threshold.  Attempt 02 is retained.  Attempt 03 applies
one dominant-asset tail shock so the economically discrete top-count output
must respond.

## Attempt 03

- Campaign SHA-256: `7389267f146b5902af5013aa912c3719b9f8cd141ef6ee434ea9d9228525e12c`.
- Recipe SHA-256: `7436af6a229e3a139faf678d592724b364b6a3418cb4cea5b91de230ae4b3f53`.
- Runtime: 90.97 seconds; process peak RSS: 416,999,424 bytes; exit status 0.
- All 60 backend executions produced finite observations, preserved the full
  causal prefix, and changed under the legal future suffix perturbation.
- All 30 pandas/Polars main and edge parity comparisons passed.
- Four independent numeric oracles passed on both backends:
  `lf1_herfindahl_volume`, `lf1_own_volume_share`, `ofi_volume_imbalance`, and
  `ashare_limit_asymmetry`.
- Constant and single-missing-bar edge executions are recorded for every
  backend in the immutable attempt-03 ledger.

Attempts 01 and 02 remain preserved as fixture-protocol failures; no production
operator source was changed in response to either failure.
