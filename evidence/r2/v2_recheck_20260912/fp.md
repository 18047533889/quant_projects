# FP V2 delta recheck (2026-09-12)

Scope: formal server-c tree `/home/sunhaiwei/quant_projects`, HEAD observed as
`e94ac507d670fd1c16b1d6a63fc5d6286daa5970`.  No branch, commit, push,
deployment, production data, or repository copy was used.

## Code delta

- `factor_preprocess/registry/transforms.py`: retired `get_function()` now
  fails closed instead of returning an unvalidated FP-native production
  bypass.  Native parity kernels require the explicitly named
  `get_research_reference_function(..., allow_research=True)` path.
- `factor_preprocess/adapters/fe_operator.py`: mixed Python security-key types
  fail before sorting/pivoting or string coercion; integer-only and string-only
  identities remain lossless.
- `factor_preprocess/tests/test_v3_preprocess_contracts.py`: public bypass and
  mixed-key counterexamples added.

## Per-ID disposition and executable evidence

| ID | code / test evidence | actual result | remaining |
|---|---|---|---|
| FP-01 | `_long_to_wide`; integer and mixed-key counterexamples | integer keys preserved; mixed types deterministically rejected | DA canonical security catalog remains the upstream authority |
| FP-02 | `_long_to_wide`; duplicate-order test | all duplicate coordinates reject before pivot, independent of row order | explicit identical-row dedupe policy not enabled (safe default is reject) |
| FP-03 | `FeOperatorExecutor` signature binding and `max_lag -> max_periods`; public test exists | static contract present | FE bootstrap currently blocks bridge execution; see blocker below |
| FP-04 | `get_execution`, `get_recipe_execution`, retired `get_function`, explicit research reference API | new bypass tests pass | FE bridge rerun required after FE owner stabilizes registry |
| FP-05 | `FeRecipeExecutor` single panel boundary and runtime counters exist | source/test inspected | real GPU/batch memory and I/O benchmark remains W9 BLOCKED |
| FP-06 | explicit `exposure_cols`, exact panel axes; neutralization tests | metadata cannot be inferred as exposure; axis mismatch rejects | full DA ExposureBundle/knowledge-time integration is cross-owner BLOCKED |
| FP-07 | lineage dedupe only removes adjacent exact step including params | EWMA h3/h10 and nonlinear-separated repeat tests pass | closed for FP contract |
| FP-08 | `derive_output_properties` invalidates orthogonality after rank/nonlinear and marks unknown | postcondition tests pass | final QE exposure measurement is cross-owner |
| FP-09 | recipe tag must exactly match actual stage sequence | RAW-tag illegal order and preset suite pass | closed for FP grammar |
| FP-10 | new transforms default RESEARCH_ONLY; runtime bind/domain and implementation rehash guards | registry hardening tests pass | production evidence issuance remains governance integration |
| FP-11 | frozen semantic IDs, policy steps, nested params and canonical recipe hash | immutability/round-trip tests pass | closed for FP value objects |
| FP-12 | marshal-based canonical code plus defaults/closure/helper identity; opaque/mutable state rejects | dynamic constant difference and mutable closure tests pass | supported callable set remains deliberately bounded |
| QA-01 | real registry parameter-bridge tests and uniquely named test functions present | non-FE portions pass; FE-required collection fails, not skipped | BLOCKED by active FE registry bootstrap failure |
| V-06 | OLS/ridge diagnostics and root-output-property validation | 139 focused neutralization/lineage/representation/replay tests pass | paired QE/FO non-inferiority selection is cross-owner |
| V-07 | frozen EWMA full replay, future poison, daily/random chunk, restart, identity/integrity/concurrency guards | focused replay suite passes; FULL_REPLAY_ONLY is honestly reported | O(1) FE incremental-state backend not claimed |
| V-08 | four frozen representation profiles and train-only fitted-state apply | representation/fitted-state suites pass | modeling consumer integration is cross-owner |
| V-19 | frozen EWMA state identity, watermark, atomic checkpoint replace and restart replay | FP replay tests pass | complete daily FE/DA service, revision generations and production pointer are cross-owner BLOCKED |

## Test results

- `pytest` focused FP contract set: **139 passed** in 1.80s.
- New/adjacent delta selection: **7 passed, 10 deselected** in 0.74s.
- Broad FP run before this delta: **381 passed, 7 failed, 1 xfailed**; all
  seven failures share the same external FE bootstrap root cause.
- Required FE public bridge rerun: **BLOCKED at collection**, currently
  `TypeError: _calculate_series has no certifiable implementation identity`
  from `factor_engine/cleaned_operators/registry.py` while loading FE operators.
  This is not recorded as an FP pass or skip.

## Stable FE-bridge rerun after registry recovery

The failure above is retained as historical evidence. After the FE owner
reported registry recovery, the required complete FP suite, including
`test_fe_operator_parity.py` and the real public parameter bridge, was rerun
through `scripts/run_main_merge_checks.py`.

- First post-recovery run (`fp_stable`): all assertions passed
  (**408 passed, 1 xfailed, 4 warnings**), but the evidence gate correctly
  recorded `SOURCE_CHANGED_DURING_RUN` because
  `factor_engine/runtime/engine.py` changed concurrently. It is not the stable
  acceptance run. Log SHA-256:
  `7386a30bc5820026f8fedc90bec1aa64dd0e82922aa3127c3107c84e2e083b31`.
- Preserved retry (`fp_stable_retry`): **PASS**, **408 passed, 1 xfailed,
  4 warnings in 52.04s**; source digest was unchanged at
  `f2c0d96c40087bcd165d1c60c698a558c89af55cc8bbcaee86661990b5548f27`
  and `source_changes_during_run` was empty. Log SHA-256:
  `0acc0850abc49974a4db2a038f7d6229b90a0ed76b3d02a9a342d6760b0b72a0`.

Thus QA-01's FE bridge collection blocker is resolved for this stable tree
snapshot. The xfail and warnings remain explicitly non-passing evidence, and
the earlier cross-owner/real-data/performance limitations are unchanged.
