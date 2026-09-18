# Stateful filter input topology and real fixtures

Four two-panel state filters now declare required companion panels and scalar parameters: cost-aware deadband/slew, confidence-weighted EMA, uncertainty deadband. Missing companions fail with the formal parameter error instead of a raw Python TypeError. Numerical recurrences were not changed.

Root full suite evidence/r41-filter-full-after.log: 68 passed, no skips, sampled family peak 801075200 bytes. Includes paired backend independent default recurrences, NaN/Inf reset and recovery, missing-input refusal and empty panels. Restored legacy slew/EMA/uncertainty tests to provide their required companion inputs.

Envelope tests now use actual three-panel inputs and independent compression/pressure/dwell formulas, missing/all-missing and future-prefix checks. Retired example/package fixture dependencies are replaced with current load_config and AST call-target migration APIs, retaining field-name collision and idempotence checks.

Earlier root evidence/r41-ef-root.log contained 12 genuine legacy missing-input failures and 2 transient Universe module syntax failures during concurrent edits; it is NOT an accepted pass. The Universe scope tests are separately pending the snapshot-interface repair and are not counted in these 68.
