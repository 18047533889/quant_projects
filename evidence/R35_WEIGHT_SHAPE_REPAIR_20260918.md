# R35 weight-shape numeric repair

The root's independent review found genuine unit-dependence in two active canonical operators, not just outdated tests.

- ts_weighted_time_centroid: finite weights near 1e308 overflowed both the sum and weighted-position sum; near 1e-308 the fixed denominator epsilon incorrectly forced results toward -1. The measure is a relative-weight centroid, so changing units must not change it.
- ts_mass_concentration: an overflowing positive-weight sum yielded zero shares and an invalid negative normalized concentration.

Both now divide nonnegative weights by their positive maximum before computing shares. The centroid then uses normalized shares directly, without an absolute epsilon. Existing zero-total/negative-weight undefined-output behavior is preserved; missingness and minimum sample contracts are unchanged.

Reproduction: evidence/r35-weight-shape-before.log — 6 failed, 10 passed.
Repair verification: evidence/r35-weight-shape-after.log — 46 passed (new scale-free oracle tests plus full shape suite), 53.54s overall, sampled peak process-family RSS 465346560 bytes. The guard is sampled, not a kernel hard cap.

Tests cover both registered pandas_numpy and Polars paths at unit scale, 1e308 and 1e-308, with independently calculated centroid/HHI expectations. This does not add an all-parameter or production-factor certification.
