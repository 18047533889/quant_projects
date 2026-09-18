# R53 intraday / holder numerical repair evidence

## Scope and actual result

Direct edits in server-c /home/sunhaiwei/quant_projects, main; no branch, worktree, repository copy, production factor publication, or deployment.

Final combined normal-conftest regression: **255 passed, 0 skipped, 2 warnings**.
Log: evidence/r53-combined-root.log.
Watchdog: returncode 0; elapsed 61.466149386 s including import/setup; sampled process-family peak RSS 803545088 bytes; sampling interval 0.1 s; guard 1536 MiB/240 s, no guard trigger. This sampling guard is not a hard memory cap or a kernel throughput benchmark.
Warnings concern unsupported operators with missing explicit PhysicalImplementationSpec; this run does not promote production/backend eligibility.

## Numerical and binding corrections

- Next-stage jump/path expressions now compile and calculate finite-support statistics. Undefined zero running references stay NaN; missing days/assets retain their coordinates.
- Jump detection excludes Inf and uses the count of finite returns without reconnecting missing physical slots.
- Segment realized volatility follows the formal SessionPanel grid, coverage and timezone semantics, including afternoon boundary returns.
- Limit duration/first-hit/reopen use the exact daily reference. UTC-aware minute and limit panels bind on the same exchange-local trading date.
- Common holder placeholders now follow the full formal multi-panel contracts. Public-winner concentration retains the date coordinate.
- Three bounded holder ratios enforce finite nonnegative counts and 0 <= numerator <= denominator, preserve identity, and accept the formal positional and named arguments after full registry loading.
- Entropy rejects negative/infinite weights and scales before normalization; missing ranked holders retain the formal absent-rank rule.
- Pledge/freeze HHI rejects unknown/negative/infinite counts and scales before summation. Pledged-holder count and pledge churn no longer turn infinity into a large finite share count; unrepresentable churn results are NaN.
- Affected semantic versions change, with 51 identity regressions proving old and repaired contracts have different hashes.

## Conversion ledger

| Implementation | Actual execution | Boundary |
| --- | --- | --- |
| Common complex holder repairs (29 delegate classes) | POLARS_PANDAS_DELEGATE | CPU; full panel materialization; no lazy/streaming/GPU claim |
| Segment realized volatility | POLARS_PANDAS_DELEGATE | CPU SessionPanel reference; no native acceleration claim |
| Three limit kernels | POLARS_PANDAS_DELEGATE | CPU; labelled timezone-preserving conversion |
| Holder ratio / ranked HHI / pledged count | Native Polars expressions | identity checked; numeric value columns only |
| Public concentration | Existing NumPy-backed Polars path | date retained; not relabelled GPU |

The new 29-canonical holder parity suite checks exact argument names, at least one finite output per canonical, values/NaN masks, dates and instrument columns. Independent entropy and share-domain tests prevent two backends agreeing on the same invalid formula from being called correct.

## Preserved failures

- evidence/r52-holder-identity-root.log: 3 failed / 62 passed before formal ratio binding repair.
- evidence/r52-holder-29-parity.log: 1 failed / 28 passed before actual-winner date preservation.
- evidence/r53-share-domain-before.log: 6 failed before HHI/count/churn domain repairs.
- evidence/r53-share-domain-after.log: 6 passed.
- evidence/r52-holder-binding-root.log: 65 passed before adding the final expanded checks.
- evidence/r52-intraday-complete-root.log: 153 passed.

## Limits and work still in progress

The strict generic canonical campaign is separate and is not complete: actual pytest collection has 1826 items; an earlier standalone list had only 1789 and missed 37. Its fixture/source failures are being repaired without silent all-NaN/opaque acceptance. These 255 regressions do not mean all canonical/backend pairs are certified.
The prior source-bound paired coverage of 514/1756 is not increased by this ordinary unit-test run.
The 110k real-factor full execution, updated CSV, GPU throughput benchmark, and production acceptance are not delivered by this change.
The prior 257-factor/9-wave shared-DAG test remains a small synthetic reuse regression, not a 110k-real-factor claim.
