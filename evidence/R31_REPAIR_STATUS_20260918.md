# R31 verified checkpoint — 2026-09-18

## Scope and limitations

Work is in server-c /home/sunhaiwei/quant_projects on main, without extra branches or repository copies. User authorized commits and pushes. No deployment or production factor publication occurred.

454 of 1756 canonical operators now have reviewed historical, fingerprint-bound paired small-sample execution evidence; 1302 remain outside that coverage. This is not every parameter, every backend, GPU implementation, or production certification. Source changes may invalidate earlier fingerprints. R31_REVIEWED_EXECUTION_SUMMARY.json identifies the incremental evidence. The 110000-factor CSV has not been updated in this checkpoint and full real-catalog materialization has not been completed.

## Production repairs committed

- 2957334a / c878f6c5 / 50ee54fd: top-k/product/conditional default contracts, with actual run_many regression.
- dcfa4156: finite-member group fallback and label identity; ATR long-path OHLC gaps and argument binding; digital_count full time axis and exact d-prior-row history; buildup defaults and timestamp preservation.
- ceb9e7ab: Polars AR coefficient now follows single-lag intercept, pairwise support and warmup semantics. An independent centered least-squares oracle checks the result.
- 2d3e78b2: stable group mean/zscore/demean at 1e308 and 1e-308; shareholder slope/acceleration use only snapshots visible at each observation, preventing later revisions from rewriting the past; Polars snapshot implementations and strict axis/date checks. Test registry teardown avoids contaminating later field-catalog validation.
- 83b86b59: seventy additional paired execution recipes and retained evidence, not a production implementation change.

## Verification evidence

| Log in evidence/ | Result | Scope |
| --- | --- | --- |
| r31-group-holder-final-repaired.log | 333 passed, 1 skipped | group extremes, three backend parity, ATR, mixed batch, holder and field catalog |
| r30-group-digital-history-final.log | 342 passed, 18 skipped | group fallback/identity, digital and history |
| r30-ar-independent-lstsq-final.log | 4 passed | independent AR oracle |
| r30-runmany-auto-memory-regression.log | 64 passed | auto DAG, resource admission and memory accounting |
| r30-dataaccess-budget-scan-regression.log | 125 passed | query/read budgets and physical scan scope |
| r30-dataaccess-polars-budget-regression.log | 14 passed | Polars memory/deadline and managed-reader deadline |
| r30-real-gpu-report-smoke.log | 4 passed | actual L20 CUDA, strict no-fallback report adapter, 12 times × 40 assets × 3 factors |
| r30-100k-catalog-stream-regression.log | 43 passed | actual 100k-root SQLite DAG catalog; streaming backpressure uses FakeEngine, not real 100k-factor computation |

Watchdog memory figures are sampled process-family RSS, not kernel hard caps. Final group/holder run took 144.46s overall, sampled peak 806715392 bytes.

## Rejected optimization and open work

The native Polars rolling covariance AR prototype was about 17.28x faster on a small benchmark but failed all four large-offset cases, maximum error approximately 9.54e10. It was rejected and is not production code.

Corrected cross-sectional extreme diagnostics genuinely reproduced eight path/case failures: group_std, long-path scale and long-path rank-weighted-value. Their repair is underway, not counted as fixed here. Sixty further rolling/regression operators are being tested separately.

Disk was checked before continuing: approximately 490 GB free. No unrelated data or other users' temporary directories were deleted; this task's 34 MB catalog temporary directory was cleaned after its completed run.
