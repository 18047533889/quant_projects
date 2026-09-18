# R32 checkpoint — verified repairs and open work

This is an intermediate checkpoint, not a declaration that all operators work.

## Committed production changes

- dad67a41: grouped demean uses per-group max-absolute-value normalization plus vectorized bincount instead of repeated group masks. Independent math.fsum-based tests include 1e308, 1e-308, empty panels and missing members. The synthetic aggregation-only microbenchmark is recorded in r32_group_demean_vectorization_results.json; its 15.85–149.85x result excludes loading, planning, grouping identity, and writes and is not an end-to-end speedup.
- e6729a30: shareholder snapshot slope and acceleration declare report-count/full-history requirements. Reproduction showed the previous stateless daily overlap lost needed reports. Analyzer and incremental planner now respect replay requirements. Generated pandas/Polars wrappers expose authored defaults, repairing omitted snapshot_date calls. This is conservative even for daily/no-snapshot mode; it is not a newly implemented checkpoint/restore path.
- 94ba579e: group_std, Polars-long scale and rank-weighted values avoid representable intermediate overflow. Shared scale contract excludes non-finite members, preserves nulls and zero semantics, and supports direct/fused long paths. Unrepresentable results are not clipped into fabricated finite numbers. Winsor-tail parameter gets an explicit threshold role.
- a5441d17: 24 exact-equivalence runtime default probes for 21 static inventory candidates. All were equivalent; no production changes were warranted.

## Real verification

Logs below are under evidence/. Watchdog RSS is sampled, not a hard kernel memory limit.

| Log | Result | Meaning |
| --- | --- | --- |
| r32-group-vectorized-regression.log | 256 passed, 1 skipped | group extreme and three-backend parity |
| r32-cross-sectional-integrated.log | 103 passed | extreme group/std/rank/scale plus vectorized independent oracle |
| r32-holder-history-before.log | 2 failed | reproduced unsafe daily overlap declaration |
| r32-holder-history-after.log | 12 passed | initial holder history repair |
| r32-holder-history-planner-final.log | 2 failed, 14 passed | exposed wrapper hiding snapshot_date default; retained |
| r32-holder-history-planner-repaired.log | 26 passed | repaired callable defaults, analyzer, incremental and holder contracts |
| r32-scale-finite-final.log | 73 passed | finite-members, extreme paths and 8 holder history/default tests |
| r32-scale-fused-final.log | 20 passed | scale shared contract and actual direct/fused Polars-long execution |
| r32-extreme-group-runmany.log | 3 passed | actual run_many shared-DAG execution on pandas, polars_long, auto; no production writes |
| r32-operator-suite-regression.log | 12 failed, 609 passed | stale generic fixtures diagnosed, not ignored |
| r32-operator-suite-regression-12-final.log | 12 passed | twelve corrected fixtures with legal panels/parameters |

The quantile cross-series tests now supply distinct aligned source/target panels and independently check known quantilogram/extremogram values. Causal tombstone tests preserve bfill/causal_bfill denylist and execute rejection checks instead of deleting safeguards. Their combined metadata/tombstone/quantile rerun passed 46 tests.

## Remaining work

Reviewed paired historical canonical coverage remains 454/1756 (1302 outside that coverage); a separate 60-operator campaign is in progress and not included yet. It has found real AR/regression/polynomial backend differences and three incompatible Polars callable signatures. Repairs and independent oracles are ongoing in common/polars_ts_advanced.py; unreviewed changes are not certified or committed in this checkpoint.

Broad generic and targeted operator suites are continuing in bounded batches. Generic finite-output tests alone do not prove financial correctness, all-parameter parity, all backend/GPU support, or full 110000-factor materialization. No CSV update or full production factor write is claimed.

Work stays in the server-c main checkout. Commits are pushed with user authorization; no branches, worktrees, whole-repository copies, deployment, or production factor publication were created.
