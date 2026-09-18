# R37 financial revision history

Five lag-derived revision canonicals now declare one prior row:
fin_revision_delta, fin_revision_pct, fin_revision_direction,
fin_expectation_revision, fin_expectation_revision_pct.

Six rolling revision canonicals now declare window_days prior rows (not
window_days-1 or zero), because their first rolling observation itself reads
a preceding source row:
fin_revision_count, fin_revision_magnitude, fin_restated_flag,
fin_expectation_revision_speed, fin_expectation_revision_count,
fin_expectation_revision_magnitude.

Forward impact matches those bounds. Invalid fractional windows stay
full-history/unbounded instead of being rounded.

Root regression: evidence/r37-financial-revision-root.log — **49 passed**.
Includes new per-canonical defaults, keyword W=10, invalid-window checks;
hand-calculated lag/gap/period-transition cases; W=3 and W=10 overlap and
independent count/signed percent sum/absolute percent sum/flag oracles.
Existing incremental, surprise and central history suites also ran.
Watchdog is a sampled RSS/time guard, not a hard memory cap.

Three age/staleness operators still need separate full-history/state review;
this repair does not certify them or the full catalog.
