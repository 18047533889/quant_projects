# Incremental batch recall: numeric correctness and bounded reuse

## Best-only selection follow-up

The public incremental assignment path consumes only the best member of each
exact-recall cluster. It now validates every cosine score, then selects the
first maximum directly and constructs one candidate per query/cluster instead
of constructing and sorting every member candidate. The legacy full-candidate
selector remains available and unchanged. Invalid nonwinning scores still fail
closed; ties retain original member order; floor/status semantics are unchanged.

For Q queries, C clusters and M total members, candidate construction falls
from O(Q M) to O(Q C). Cosine arithmetic still visits all present members;
this is not approximate search or a claim that mathematical work disappeared.
The retained cache budget is unchanged and is not a whole-process RSS limit.

The latest independent joint run passed 36 tests in 0.75 seconds (24 best-only
and 12 optimizer identity checks), including the adjacent-Float64-score
regression. This is a focused correctness run, not a full-library regression.
Full FactorAssets regression subsequently passed **1746 tests, 97 warnings,
78.50 seconds**. With 2048 members, 128 queries, 64 dimensions, 16 clusters and
three one-thread ABBA rounds, the final driver run measured baseline median
**0.8752228 seconds**, current **0.0770238 seconds**, ratio **11.363015x**.
Every public artifact matched exactly and the independent longdouble oracle
passed. Baseline `09f8cd51ed9d62040df5f3ece3fc077e5f4f0428` isolates best-only
selection from earlier cache/domain work. Evidence:
`benchmarks/incremental_best_only_verified_ab_20261003.json`; the preceding
run is retained in `benchmarks/incremental_best_only_ab_20261003.json`.
These remain synthetic embedding measurements on a shared host, not real COS
scoring or global fastest-backend certification. Timing windows were coordinated
with evaluator calibration; future runs must likewise avoid overlapping jobs.

## Reproduced defect

The former exact recall normalization used `np.linalg.norm` directly. Finite
vectors `[1e308, 1e308]` and `[1e-300, 1e-300]` were rejected because squaring
overflowed or underflowed, not because the vectors were invalid. Reusing the
existing `similarity.unit_vectors` scale-safe primitive fixes this defect
without adding another normalization backend.

For each finite nonzero row, the intended cosine calculation is

$$s=\max_i |x_i|,\qquad u_i=\frac{x_i/s}{\sqrt{\sum_j(x_j/s)^2}},\qquad
c(x,y)=\sum_i u_i(x)u_i(y).$$

Zero vectors, NaN and infinity must remain invalid. Missing member fingerprints
remain unmeasured, never invented as zero similarity. This is embedding recall,
not certified pairwise evidence or authorization to publish a production factor.

## Batch optimization

The request-local cache reuses immutable member rows and their normalized
vectors across new-factor queries. It must not outlive the request or silently
reuse data across snapshots, domains or changed membership. A private 32 MiB
retained-payload budget, lazy loading, LRU eviction and an uncached oversized
entry path protect memory. That budget does not claim to bound total process
RSS, source artifacts, current computation workspace or an approximate index.

Domain validation can compare each bound query and each present member once
rather than recomputing the same domain for every query-member pair. Mapping
keys, query content hashes and all domain fields must still be checked.

An approximate index query runtime failure uses exact recall for that query.
Malformed returned members and invalid governance evidence must fail closed;
they must not be swallowed by a broad fallback around result processing.

## Verification status

The numerical defect above was reproduced on server-c on 2026-10-03. Both
legacy exact recall and cached recall now share scale-safe normalization and
Float64 cosine endpoint handling: deviations within 64 machine epsilons are
clipped to [-1, 1], while nonfinite or larger deviations fail closed. This is
not the wider Float32 ANN backend tolerance and does not validate malformed
ANN members or upgrade recall hints to certified evidence.

Full FactorAssets regression: **1717 passed, 97 warnings, 83.31 seconds**.
Seven new recall regressions are included in that run. Five additional
benchmark-oracle guards were subsequently added; their joint run with the
seven recall regressions was **12 passed, 0.77 seconds**. These checks cover
retained-cache eviction/oversize behavior, member/domain reuse, extreme finite
scales, query-time ANN failure, malformed result rejection and report coverage.
They do not prove every possible input or backend is bug-free.

Controlled public A/B: **2048 members, 128 queries, 64 dimensions, 16 clusters,
three ABBA rounds**, one BLAS/OMP thread. Baseline median **2.0916016 seconds**,
current median **0.9738189 seconds**, ratio **2.147834x**. Public assignments,
candidates and content hashes match exactly on every measured call; an
independent extended-precision cosine oracle verifies the cluster winners.
The baseline is pinned to `0925681e5` with the current stable normalization,
so this comparison isolates batch reuse and domain-validation work rather
than mixing arithmetic corrections into the speed claim.

Evidence: `benchmarks/incremental_batch_recall_ab_20261003.json`.
`scripts/benchmark_incremental_batch_recall_oct03.py` reproduces the bounded
comparison, checks source stability and resources, and refuses report overwrite.
This is synthetic embedding evidence, not real COS factor evaluation, global
fastest-backend certification, production publication authority, or a full RSS
bound. The shared business host load is recorded, not completely controlled.
