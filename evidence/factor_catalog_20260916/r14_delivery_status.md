# R14 remediation checkpoint (2026-09-16)

Status: work in progress. This is not certification of the complete catalog, all backends, or production publishing.

## Verified code changes

- Storage cache explicit zero budget now disables admission; None remains automatic. Negative and non-integer explicit budgets are rejected. Clearing one backing store releases only its own accounting, preserving independent caches on the same accounting layer.
- Red evidence: r14-cache-budget-red.log (7 failures before patch). Final regression: r14-cache-budget-green-r2.log (38 passed; peak 471146496 bytes).
- Storage source SHA256: a0944e23746f6812469eae3a4307857fe14c3aee73cf4c6505701fee55fef1df.
- Mixed-backend shared scalar literals now have actual scalar shape and measured Python payload size; container literals remain unestimated. This repairs automatic plan readiness without inventing panel row counts.
- Independent shape/literal regression: r14-root-literal-estimates-r2.log (8 passed, 5 deselected).

## Actual batch evidence

- DPO daily selection: 25 formulas, 8 symbols, 2025-01-01 through 2026-04-30.
- Before planner repair: all 25 BATCH_ABORTED due to region row estimate unavailable. These were batch failures, not 25 independently proven bad formulas.
- After repair: r14-dpo-all-daily-r3.jsonl.gz, 25 EXECUTED, 64000 values, 60592 finite. One run_many(auto), batch-only, no singleton retry, public performance defaults.
- Output SHA256: 2492eb95c64108504ea6f784710aa70f78d04624e55311473bbfcb3c69d1b362.
- Actual backend: pandas_numpy for this batch, not GPU evidence. Execution phase 25.578 seconds; watchdog total 76.996 seconds; peak 1430835200 bytes.

## Catalog chain

- r13f CSV SHA256: 4529988819993fc51219fd0561cd99c723565a8e063418c2116b47b5dbebba19.
- Frozen r14a CSV SHA256: 16261f5e3f7917c336e0e692578a57c7493eec2d0980559c7e0959b023f540cd.
- r14a retains all 114132 source rows and 113893 nonempty factor IDs. 67 WilliamsR revisions: 51 compile successfully, 16 still fail due to other expression issues.
- Distinguish valid HLC/window formula expansion from explanation-supported repair of malformed OHLCV bundles using standard HLC and default window 14. The latter is not numerical equivalence to an executable original five-argument formula.
- WilliamsR numerical verification includes tiny nonzero denominators, warmup, missing values, constant ranges and infinity cases. Safe division's default epsilon could alter tiny denominators; the repaired ratio uses raw divide and an explicit zero-denominator null branch.
- Frozen r14b SHA256: dccd5687209e1bbdf7b376fd2e8d3e212c74fe07a21398ff90ff607524006040. It merges only DPO25 successful evidence. All 67 Williams revisions remain NOT_RUN; batch failures are separate diagnostics.
- Williams classification sidecar SHA256: 41010a1c8e6bf56f105940e363b992f3588c82ae3a93ef99c42e1633bd932e30. All 67 formulas replay exactly; 35 valid-signature expansions and 32 explanation-supported malformed-call repairs are distinguished.
- Mac CSV synchronized to /Users/shw/Documents/新因子统一整合_更新版_R14b_20260916.csv without replacing original or R13d.

## Remaining boundaries

- Core streaming cross-wave cache now retains default CSE and uses normalized logical keys; 5 roots in 2+2+1 waves and two-worker oracle tests prove one shared ts_mean evaluation. Latest independent combined stream regression: r14-root-cross-wave-regressions-r2.log, 56 passed. Restricted to single proved snapshot and homogeneous execution scope; encoded SourceRefs remain excluded. This is not the durable multi-source path or full-catalog certification.
- Single-call adaptive DAG and bounded streaming do not yet prove that all 110k distinct real catalog factors form one globally reused DAG.
- Real cross-backend/GPU performance, multi-source snapshot-safe reuse, unresolved fields/operators, native crashes, and the large unexecuted portion remain separate outstanding work.
- No branch, worktree, repository/data copy, commit, push, deployment or production factor publication was performed. Changes remain in the server-c formal working tree.

## Additional verified repairs and active blockers

- Scalar cache ownership uses actual builtin immutable-object size and exact entry-generation lifetime, rather than pretending float NaN occupies zero bytes. Independent scalar/cache integration: 21 passed; ownership/concurrency regression log: r14-root-ownership-regressions.log.
- Region residency contraction introduced a cycle in an acyclic diamond A -> B -> A plus A -> A. Boundary-level partition fixes that case; 16 targeted/research tests and 79 region-contract regressions passed. Further review found multi-output-frontier grouping requiring repair, still in progress.
- Full Williams auto40 r2 remains BATCH_ABORTED. Read-only diagnostic identifies unevidenced encoded StockDailyBarAdj ret/vwap source costs; do not assign arbitrary anchor row counts or report the 40 formulas as executed.
