# R37 legacy fixture corrections

No production guards were weakened:
- Binned response tests now pass distinct response/explanatory panels.
  Six extra backend cases compare independent rank-correlation, quadratic
  least-squares curvature and standardized-slope formulas, preserve date
  metadata and verify future-prefix invariance.
- US daily OHLCV schema test targets the current authoritative external
  Massive table definition and SIP mapping, not the intentionally empty
  legacy tables_local map.
- Bivariate causality tests keep both fields on the same instrument axis;
  pointwise arithmetic receives no undeclared rolling-window keyword.
- The research-only micro_bvc_vpin test explicitly selects research mode and
  asserts it remains unavailable in production. Other pack members still
  use production registry lookups.

Evidence:
- r37-operators-b-c-root.log: 8 failed,25 passed; stop-on-first-8 at binned fixtures.
- r37-binned-response-repaired.log: 22 passed.
- r37-operators-b-c-after-binned.log: 8 failed,225 passed,146 skipped;
  exposed current-schema, axis/parameter and research-mode fixture errors.
- r37-legacy-fixtures-targeted.log: **72 passed,110 skipped**.
  Those 110 skips are pre-existing unimplemented fixture coverage in the
  broad Tier-1 sweep, NOT successful operator execution. This repair does
  not add their count to the strict paired campaign ledger.
