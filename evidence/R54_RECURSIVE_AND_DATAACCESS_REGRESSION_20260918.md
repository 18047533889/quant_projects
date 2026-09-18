# R54 recursive recovery and DataAccess regression

McGinley pandas and Polars now treat NaN and both infinities as a break in recursive state, wait for a complete finite trailing window, and reseed from a scale-safe mean. Leading infinities cannot poison initial warmup. No future rows enter a seed; no gap is bridged. Semantic version is now 2.

Normal-conftest root run, evidence/r54-recursive-public-identity-root.log: **117 passed, 0 skipped**, two unsupported-backend metadata warnings. Includes the complete existing adaptive-filter suite (not a -k subset), four recursive operators' prefix causality/backend/coordinate/recovery tests, 52 semantic-identity checks, and eight public-registry finite-share-domain checks.
Watchdog returncode 0; 71.859020447 s including setup; sampled family peak 803086336 bytes, no guard trigger (1536 MiB/180 s). This is a sampled guard, not a hard cap or performance benchmark.
Pre-fix McGinley reproduction was console-only and no saved before-log is claimed.

DataAccess/query-cache/manifest and FactorEngine source-cache regressions: **53 passed**, eight warnings (datetime conversion and multiprocessing fork warning), evidence/r54-dataaccess-cache-root.log. Watchdog returncode 0; 8.123099734 s; sampled peak 1213177856 bytes. No DataAccess production code was changed in this patch.

The eight public share-domain cases also passed independently: evidence/r54-public-share-domain-root.log (8 passed). These eight are already included in the 117 above and must not be double-counted.

The all-canonical campaign, the newly discovered cross-call run_many persistent-cache bug, all-backend certification, and 110k real-factor execution are separate ongoing work; these results do not claim their completion.
