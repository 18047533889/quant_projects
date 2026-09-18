# EVT threshold stability repair

The public contract allowed k_min=2, k_max=3 although the kernel requires three finite Hill estimates. Both pandas_numpy and Polars silently returned all NaN for this infeasible ladder. The relational contract now requires k_max >= k_min + 2; the statistical support floor is unchanged.

The Hill kernel also rejected legitimate strictly positive tiny values via an absolute 1e-12 threshold, making the statistic depend on the numeric unit. It now validates strict positivity and finiteness and uses log differences instead of forming a potentially overflowing ratio. Semantic version is 2 so older cached results cannot be reused.

Before: evidence/r56-evt-before-root.log, 5 failed / 4 passed.
Final: evidence/r56-evt-final-root.log, 92 passed / no skips (13 new EVT cases, existing EVT/moment support tests, 66 semantic-identity cases).
Coverage includes both public backends; positive scales 1e-200, 1, 1e200; a finite extreme-ratio Hill oracle; upper/lower tails; exact independent ladder formula; NaN/Inf recovery; preserved timestamp/date index; prefix causality; invalid two-threshold rejection.
Watchdog sampled peak 800735232 bytes, 53.331 seconds including startup. Sampling is not a hard memory cap or throughput benchmark. Two pre-existing ADX physical-spec warnings were not suppressed.

This is not full operator certification or execution of the 110k real-factor catalog. The full strict-canonical fixture campaign and production-bootstrap repair remain in progress.
