# Continuation verification (2026-09-16)

Formal worktree: server-c `/home/sunhaiwei/quant_projects`. No branch, commit, push, deployment, or production factor publication.

## Confirmed regression results

- `root-budget-final-r10.log`: 68 passed, exit 0. Broker RSS measurement, paired cgroup memory domains, auto budget, resource broker, and resource authority. Watchdog sampled peak family RSS 749588480 bytes; elapsed 13.662 seconds.
- `root-stream-parameter-isolation-r10.log`: 38 passed, 11 warnings, exit 0. Watchdog sampled peak family RSS 813494272 bytes; elapsed 56.149 seconds. Explicit unsupported-expression and operator-parameter failures are isolated at compile preflight; this does not certify arbitrary runtime-failure isolation.
- Previous independent CSE/budget regression `root-cse-budget-r9.log`: 79 passed, exit 0.
- Previous DataAccess resource regression `dataaccess-resource-post-budget-r9.log`: 23 passed, exit 0.

RSS unknown/permission failures now deny new execution budget; diagnostic ru_maxrss is not treated as a complete live process-family measurement. A single snapshot supplies paired cgroup remaining, preserving genuine zero headroom. The shared cgroup current is not subtracted from an unrelated process cap.

## Catalog evidence and pending replacements

- User explicitly authorized rewriting 99 legacy intraday-limit formulas according to current formal definitions, preserving original formulas and semantic redesign records. This does not authorize unrelated economic-definition changes.
- The latest ordinary daily sweep `resume-daily16546-external-r17` closed 20 chunks: 424 EXECUTED, 34 COMPILE_FAILED, 42 ALL_NONFINITE; next offset 17460. These are small real-data smokes, not full-history or all-backend certification.
- Donchian macro price-argument order was corrected with a numerical oracle. There are 280 affected catalog rows. Old matched-formula execution evidence must not be treated as valid semantic evidence. Replacement checkpoint work is ongoing.
- Donchian rerun offset 100 timed out during a known concurrent interface-transition window; current-version rerun is separate evidence. Offset 120 returned -11 (native crash). It is not proven to be a formula error or caused by a nearby admission warning; no blind retry or promotion of incomplete gzip.

## Newly established resource issue, remediation underway

The host coordinator adds cumulative scan bytes to estimated resident memory before requesting a memory lease. These are different quantities. DataAccess's host-backed admission path also skips its independent scan-cap check after a successful host lease, so changing the coordinator alone would bypass scan governance. The two paths require one atomic correction with regression coverage for scan rejection, concurrent admission rollback, and release accounting. No budget inflation or large scan is authorized as a workaround.

## Remaining limitations

Most catalog rows have not been executed. Cross-wave/global intermediate reuse is not implemented. The bounded sink path has a successful synthetic 10000-factor run; the synthetic 100000-factor run timed out after progress to 37000, and is not a success. Default return mode still retains outputs. Native crash investigation, missing source bindings, ambiguous legacy DSL definitions, and per-factor durable terminal outcomes after global abort remain open. No claim of all operators/backends passing, optimal performance, or OOM impossibility is made.

## Closed checkpoint and additional verification

Root independently streamed the completed `factor_catalog_review_checkpoint_r8_donchian_daily17460.csv.gz`: 114132 rows, 113893 nonempty unique IDs, SHA256 `9a46fb1e48cb56e2cc5b70ed9e4419705512f8731966b08e02fc0d55c6b3b133`.

All 280 Donchian rows now carry post-fix evidence or explicit failure states, with zero historical execution-validation scopes: 244 EXECUTED, 4 EXECUTED_ALL_NONFINITE, 3 COMPILE_FAILED, 9 EXECUTION_FAILED, 20 NATIVE_CRASH. NATIVE_CRASH denotes the affected batch, not a proven individual formula crash. The 99 authorized SEMANTIC_REDESIGN rows retain 71 PARSED_FIELDS_BOUND and 28 FIELDS_UNRESOLVED. Overall execution counts include 7212 EXECUTED and 106472 NOT_RUN; the file is a progress checkpoint, not certification.

Root independent numerical Donchian regression: `root-donchian-numeric-r10.log`, 2 passed, 2 warnings, exit 0, sampled peak 792686592 bytes.

The scan/memory dimension correction is now applied atomically in FE host coordinator and DA governor. Six new targeted tests passed after first reproducing four failures. Host-backed scans now obey the separate scan limit before host acquisition and again under the admission commit lock; failed second checks release acquired leases. Root independent cross-library regression `root-da-host-dimensions-r11.log`: 41 passed, exit 0, sampled peak 355753984 bytes. Two additional older R38 test assertions conflict with behavior present in both HEAD and the current worktree: stale decisions deliberately report PRESSURE_3, and no-active-job reads still acquire parent-broker leases when the bridge is installed. Test-only corrections and explicit bridge-absent fallback coverage are in progress; production behavior is not relaxed to satisfy stale assertions.

Native crash read-only findings are recorded in `native-crash-offset120-review.md`. The 120-second faulthandler snapshot is not a native crash backtrace; no specific extension has been established as the cause.

Root independently reran the corrected R38 tests: `root-r38-contract-r11.log`, 5 passed, exit 0, sampled peak 354775040 bytes. Assertions now cover conservative stale decisions, job child leases, no-active-job parent leases, and bridge-absent local admission. Production source was not changed for these test corrections.

## Next daily sweep completed

`resume-daily17460-external-r18`: 20 closed chunks, 500 factors, next offset 18760; 488 EXECUTED, 9 EXECUTED_ALL_NONFINITE, 3 COMPILE_FAILED. Watchdog exit 0, elapsed 211.226 seconds, sampled peak family RSS 862687232 bytes. No production factor values persisted. Compile failures are invalid legacy quantile parameters (two extreme-cluster q=0 calls and one tail-ratio q_low=2 call); bounds were not relaxed and alternative economic definitions were not invented. Inclusion into the next checkpoint preserves Donchian invalidation precedence and authorized intraday redesign records.

Disk check at this continuation: 754 GiB available. No whole-repository copies or large local artifacts created.
