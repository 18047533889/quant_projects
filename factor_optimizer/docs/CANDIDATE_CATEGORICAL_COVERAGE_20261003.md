# Automatic candidate coverage and bounded search

The execution audit exposed categorical branches that were runnable but not
proposed by the static automatic catalogue. `candidate_catalog.py` now adds:

- Both symmetric and asymmetric branches for each U/inverted-U center/power.
- All MAD/IQR/sample-standard-deviation × median/mean location combinations.
- Cross-sectional minimum-tie rank, alongside average-tie and temporal rank.

Existing per-family candidates keep their original relative order; additions
follow them deterministically. U-shape formulas reuse the existing prepared
rank cache, including asymmetric formulas. No new computational backend is
introduced. Minimum-tie rank must not be routed to FE average-tie rank; robust
scales must not be equated to unrelated winsor/zscore kernels by name.

Default static count rises from 46 to 64, or 50 to 68 after static decay sign
composition. The maximum candidate budget stays 128. A bounded authentic
diagnosis fixture combines 27 smoothing plans and their signs, two fitted
shape proposals, two tail-derived smoother scales and two layered-decay plans
with their signs: 132 raw proposals collapse to 128 distinct executions via
the existing scoped deduplication. At budget 127, the whole search is rejected
before even RAW eligibility scoring; RAW is retained and no partial candidate
search is reported as success. No silent truncation or budget increase occurs.

Independent joint tests: 13 passed, 53 existing warnings, 32.31 seconds.
The envelope fixture tests proposal admission and atomic rejection, not that
all 128 candidates outperform RAW. TRAIN fitting and single frozen-winner
VALIDATION confirmation remain unchanged; TEST labels never choose candidates.
A caller with a deliberately smaller budget may retain RAW explicitly.
