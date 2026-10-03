# TRAIN joint-scoring prepared label reuse (2026-10-04)

The existing immutable `_PairICPreparedSplit` now serves both paired IC and
joint scoring during `optimize_factor_batch` TRAIN search. `paired_series`
accepts optional `prepared_split`; its default path is unchanged. Reuse requires
the exact source batch and label objects and identical ordered indices, else
the request raises ValueError. Index validity and non-overlapping label checks
still run. Both TRAIN baseline and candidate call sites pass the prepared split;
VALIDATION remains on the ordinary independently materialized path.

This removes repeated label slicing, immutable-array copying, content hashing
and time-axis construction. It does not remove RAW cache content hashing or
candidate-specific missing-mask, portfolio or IC calculations. No training,
validation, budget or improvement acceptance threshold changed.

Bounded runtime probe: existing pair-IC fixture, 60 selected dates, 40 assets,
three candidate scales. Instrumented `_subset_labels` calls were 3 for ordinary
scoring and 0 for prepared scoring. All RAW and candidate output arrays were
exactly equal; exit zero. This is a work-count/equivalence result, not a timing
benchmark or claim of universal fastest behavior.

Root focused existing regression: fitness, pair-IC cache, fitness boundaries,
joint metric availability: 59 passed in 3.26 seconds (terminal exit zero).

Additional end-to-end bounded comparison: 240 dates, 40 assets, one reverse
signal; SIGN_ORIENTATION search with 99 bootstrap draws. The reference wrapper
removed only `prepared_split` from scoring calls. Both runs selected
SIGN_ORIENTATION; optimized values, candidate ledger, selected identity, TRAIN
gain, VALIDATION lower bound/coverage and joint diagnostics were exactly equal.
One TRAIN call used the prepared split; one VALIDATION call remained default.
Process exit zero. This synthetic comparison is not real-factor speed evidence.

New focused regression: 7 passed in 1.78 seconds (agent terminal zero), covering
context/type mismatch, ordinary fallback, extended precision, matching-target
reuse and cache correctness with candidate-dependent missing masks.

Root combined rerun of the seven new cases plus four existing focused suites:
66 passed in 4.17 seconds, terminal exit zero; production source hashes stable.
