# R16 working evidence (2026-09-16)

Work remains in progress in /home/sunhaiwei/quant_projects. No branch,
worktree, repository copy, commit, push or production publication.

## Root-verified repairs

- CCI Polars now propagates null/nonfinite observations throughout the active
  window, matching the canonical pandas definition. Previous nanmean dropped
  Polars nulls. Independent window oracle covers null, NaN, +/-Inf, window
  1/3/20, tiny nonzero denominators, constant inputs and date preservation.
  r16-root-cci-red.log: 2 failed / 12 passed before repair.
  r16-root-cci-green.log: 14 passed. This is kernel evidence, not DSL admission.
  Central CCI semantic version advanced to 2.
- Cross-wave cache admission rejects flags and empty/mutable/unknown snapshot
  markers. Legacy explicit string/integer generations remain supported.
  An exposed authoritative snapshot_token takes precedence; missing/unknown
  tokens cannot fall back to a weaker data ID. Cache setup refreshes first
  and binds to the actual admitted scope.
- compute_data_scope includes the public manifest snapshot_token. A real
  run_many sink regression changes manifest while retaining data_snapshot_id:
  the second wave aborts without mixing new data with cached first-wave values.
  Initialization also works when the token is obtained only by refresh.
- TLS logical wrapper lookup is non-owning. Independent integration covers
  wrapper generation/lifetime, CSE leases, cross-wave cache and budget isolation.

## Test evidence

- r16-root-default-contracts.log: 88 passed.
- r16-root-source-lifetime-regressions.log: 54 passed.
- r16-root-snapshot-red.log: 11 failed / 16 passed before marker validation.
- r16-root-snapshot-green-r2.log: 51 passed.
- r16-root-manifest-lifetime-green.log: 78 passed.
- The first snapshot joint run encountered an intermediate CCI authoring-surface
  bootstrap failure. The incomplete surface change was removed, and the
  successful r2/full regressions above are the applicable evidence.
- Tests overlap; do not sum these counts as distinct coverage.

## Current validated catalog checkpoint

Frozen r15b CSV gzip SHA256:
819bb637a0d9bcab3331c596ab1371e0dd8dcff703a1e3a1e877583570f57c74

All 114132 source rows retained; 113893 nonempty unique factor IDs.
Among nonempty IDs, compile statuses are 109482 COMPILED,
4411 COMPILE_FAILED and 0 NOT_RUN.
Finite-output execution records: 8023. All-row NOT_RUN execution: 105744
(includes 239 blank-ID rows). Remaining statuses are preserved explicitly.
Both successful real batches below and all 3106 previous compile-NOT_RUN
records have been reconciled by exact identity/current formula.

Mac delivery (original untouched; compressed transfer hash verified):
/Users/shw/Documents/新因子统一整合_更新版_R15b_20260916.csv
Compile success is not execution or economic correctness certification.

Williams auto40 r5-r7 remain failed diagnostics. The global edge-validation
and per-root execution boundary repair closes auto40 r8: 40 EXECUTED,
102400 values / 97816 finite. Stochastic auto37 r2: 37 EXECUTED,
94720 values / 87627 finite. Each used one auto batch, eight symbols,
2025-01-01 through 2026-04-30, no singleton retries, actual pandas_numpy.
This does not establish GPU execution or all-backend parity.

Williams output SHA256:
186e150b50278d52d9fe69447c8badb63bf9612ad928c6af3193f4f8a04203bb
Stochastic output SHA256:
88595baba0f8385766b08db202253c5ea23d33afed22698d3adfbe46b2c5ca3d

Williams r8 predates the transfer-telemetry correction: its actual_bytes field
incorrectly repeated the prediction and must not be used as measured evidence.
Stochastic r2 carries corrected observations (identity zero-copy bytes,
materialized payload size, actual row/column count and measurement basis).
Frozen prior evidence is retained unchanged.

## Open work

- R16 catalog migration from frozen r15b: strict CCI recipe and eight remaining
  valid StochasticK signatures. Changed formulas must invalidate old execution
  evidence. CCI remains a recipe, not an admitted CCI primitive.
- R17 malformed limit-call migration: review directional explanation conflicts
  before checkpoint integration.
- Deliver later validated CSV checkpoints to Mac without replacing original.
- Qualified multi-source cross-wave reuse, full 110k-factor global scaling,
  all-backend/GPU verification and the large unexecuted set remain unproven.

## Source fingerprints

- streaming_batch_service.py: be81acb3ad63225304c876c512f40b7d692ae5ae0305535c17a295904c84d51c
- storage/data_scope.py: 8d5a0db2c6b3f4056c5148d40f9e2d2af55a3741d5d87347131e74ab578bdc97
- technical/polars_signal.py: 124a8a032a1ab385ea8eace59ee13c2a3b33629c3c512430a2a0f905ed9018b0
- operator_semantic_version.py: 06c077f8dbe9f1c55cc15646074bf96799c16bb51c6b4af766387bf2980a897d

## Additional root verification

- r16-root-stream-final-regressions.log: 78 passed. Eligibility is now reported
  separately from actual cross-wave hits: cross_wave_cache_enabled is a mode
  flag, while cross_wave_value_reuse is unknown when no hit counter exists.
  The independent counted-kernel tests still prove one shared ts_mean execution.
- r16-root-transfer-ledger-green.log: 14 passed. Identity transfers report zero
  copying and separate payload bytes; known materialized objects report their
  native memory footprint and shape; unknown values remain unmeasured and are
  never collected just for telemetry. Runtime event names are accepted by the
  compact ledger, with absent backend labels explicitly unreported.
- r16-root-physical-integration.log: 69 passed, peak 391655424 bytes.
- transfer_fallback.py: 6d2fad1308de585c4c09ff322d6c175dd1e4652642da57aac1b6d43c1d58bde2
- execution_ledger.py: 2a30e616332c3379cf46b536b042222f90220d1f80e5585ea202a74a26d6502a
- region_telemetry.py: c3a733ef77e117751a300ed1e4f4330591311ce6ce0ceec4589e750349d95eee

## Continuation verification

- r16-root-strict-cci-integration-r3.log: 37 passed; 53.33s watchdog,
  678993920 bytes sampled peak. Covers both strict primitive backends,
  research registration, exact history extension, public parameter binding,
  raw-kernel strictness and pandas run_many recipe/oracle boundaries.
  This does not certify production, GPU or real-catalog CCI execution.
- r16-root-evidence-validation.log: 15 passed. Combined reconciliation now
  requires caller-pinned execution shard hashes at CLI, finite count > 0
  bounded by integer value_count, valid result digest and matching summary.
  Existing Williams40 + Stochastic37 immutable evidence passes the stronger
  loader (77 records); frozen r15b is not rewritten.
- scaling/r15-expression-dag-100k.log: distinct add(shared_mean, constant)
  roots, 300003 nodes, 29.829s, 1335008 KiB peak. Forced pandas planner-only,
  small 2560-row panel cost hints; not public auto execution or full catalog.
- Backend-switch readiness now indexes node regions and planned boundaries
  once. Agent regression records 29 passing tests; root integration pending.

## Later R16/R17 continuation (read before using counts above)

R16a changes 76 formulas: 68 CCI (61 exact HLC/window expansions, seven
explicit OHLCV contract repairs) and eight exact StochasticK expansions.
Root independently compared all rows: same schema, source_row, ID, original
formula; 113817 untouched rows identical; changed rows invalidate old execution.
R16a counts: 109540 COMPILED / 4353 COMPILE_FAILED, 113893 nonempty IDs.
R16b adds only 40 pinned CCI execution records: 8063 EXECUTED.
R16b gzip SHA256:
146dfa35122c62aa77a33a5c118d2a1131d964d4cb1df19443ffcf9142b2c93b

Do not deliver R16b as final: review discovered 30 rows / 32 historical
quantile-transport rewrites whose window=60 -> recent20/old40 split was not
supported by source explanations. R17 must surgically withdraw only those
rewrites, preserve other edits, append correction history and invalidate
old execution evidence. The same unsupported rewrite is now prohibited
without explicit review-language support for both windows.

Root verification after physical compatibility-route optimization:
- r16-root-batch-route-final.log: 70 passed, 52.04s watchdog,
  483377152 bytes peak.
- r16-root-cci-auto40-r2.log: one run_many(auto), 40 EXECUTED, 102400 values /
  93858 finite, zero singleton retry. All result hashes match r1.
  98.78s watchdog, 1432014848 bytes peak; actual pandas_numpy.
  Output SHA256: 252e102540848caf293ce171a812cccc6ef4b96f3a2d7c7e75727a4f12275fdf
- Tested batch_service.py SHA256:
  9ca33f5664354c9b3326976cca7939554825af3371c4baf203b8c2298229e1dd
- Tested batch_global_optimizer.py SHA256:
  47305e47b2f38ca40ae04ef3799a635074c277a21266de37c0ce3a03db8374aa
- r17-root-recipes-integration.log: 41 passed (outer StochasticD/AROON,
  directional limit recipes and strict CCI), 56.55s watchdog, 799895552 bytes.
- r16-root-physical-integration-r2.log: 21 passed (narrower test selection).
- Evidence loader's applicable retained test set is 15 passing tests.
  r16-root-evidence-validation-r2.log also tested a proposed two-file rollback,
  subsequently withdrawn because ownership checks cannot exclude a concurrent
  replacement race. No rollback/deletion implementation remains; frozen
  output publication remains no-overwrite links. That log is not a certificate
  of the current publication implementation.

All real-data tests are research smoke tests, not durable production writes.
Global all-catalog auto DAG execution, qualified cross-wave reuse, every
backend and GPU remain open; no production profile was silently installed.

## R17b / R18 current verification (supersedes historical counts above)

R17b formula checkpoint SHA256:
90a9643fc075e423bbc589fb090bfada8b70eddb23ddc07e2ba250c4028c0ae9.
114132 source rows, 113893 nonempty unique IDs, 239 blank-ID rows.
Nonempty-ID compile counts: 109668 COMPILED / 4225 COMPILE_FAILED.
208 changed rows versus R16b were independently checked by root; unchanged
rows, original formulas and source identity were preserved. Ten historical
executions were invalidated by formula correction, not counted as current.

Closed root research batches under the same engine version:
- remaining CCI5 auto r2: 5 EXECUTED, 72.53s watchdog, 972214272 bytes peak.
- outer40 auto r1: 40 EXECUTED, 93.25s watchdog, 1470414848 bytes peak.
- mixed-source limit40 auto r1: 40 BATCH_ABORTED, PhysicalPlanRequiredError:
  region row estimate unavailable. Process RC0 does not mean factors passed.
- r17-root-engine-final-integration.log: 85 passed, 11 warnings,
  58.20s watchdog, 865632256 bytes peak.
Engine hashes for these batches:
- batch_service.py: ae47ef1fd3425df1f0af4fb09f0dfc69d11bb3bf53db800623b5fdbbd8a3c557
- batch_global_optimizer.py: b72c62fd79996096e9c128eaf7c3837b9a3c136c36755170af33e3c937818e5a
All batches use public auto defaults, eight symbols and 2025-01-01 through
2026-04-30, with no singleton retry. They are not whole-catalog certification.

Root R18 additional repair: source dependency traversal now visits unique
node identities iteratively, rejects cycles and preserves malformed-ref
errors. This avoids recursive depth failures and exponential shared-diamond
walks. V2 source market/provider/dataset/version and other semantic identity
fields now enter the dependency digest; legacy v1 manifest shape is retained.
r18-root-source-dependency.log: 110 passed, 7.02s watchdog,
355549184 bytes peak. Cases include 5000-deep expression, depth60 diamond
(one source decode), malformed cycle and nine independent V2 identity changes.
source_dependencies.py SHA256:
3eb72f32c449a04cecd46a7250a1324c9e4561424104b52b2fba5b20d7d4dcad.
This does not enable qualified cross-wave caching by itself.

Quant Evaluator: existing public CUDA smoke passed on NVIDIA L20/CuPy,
2.71s and 614453248 bytes peak (r17-root-quant-evaluator-gpu-smoke.log).
Independent Rank IC parity: 2 tests passed for 80 times x 24 assets x 2
factors, including ties/NaNs, measured positive H2D/D2H transfer bytes and
CPU/GPU numerical agreement (r17-root-gpu-evaluator-parity-r2.log).
Public EvaluationBundle returns owned NumPy host results deliberately.
Initial r1 test incorrectly required CuPy public outputs and was corrected;
no Quant Evaluator source change was needed. This is not all-metric GPU or
FactorEngine-to-GPU end-to-end coverage, nor asynchronous transfer proof.

R17c is being assembled from the three closed batches, preserving the 40
known failures. Further engine retry and R18 TRIX/ADXR work are separate and
must not silently revise frozen evidence. No commits, pushes or deployment.

### R17c delivered checkpoint

R17c CSV SHA256: 4eaf4ddc697de0f180df6f13cbe63590fe37ae56966bf3521b427808fb83568d.
Manifest SHA256: be22a1cb3f3daff7aca2753f3252aa619fa9d9d4940ed31ce42bf6213a0fffdb.
Root independently streamed all 114132 rows against R17b: exactly 85 rows
changed, only execution columns and appended migration history. All formula,
source, binding, static and compile columns unchanged. Compile remains
109668 COMPILED / 4225 COMPILE_FAILED for nonempty IDs.
Execution (all rows): EXECUTED 8098, EXECUTED_ALL_NONFINITE 186,
NOT_RUN 105629 (including 239 blank IDs), EXECUTION_FAILED 26,
COMPILE_FAILED 119, PREPARE_FAILED 14, NATIVE_CRASH 20, BATCH_ABORTED 40.
The 40 aborted rows have empty value/finite counts and result hashes.

Mac delivery: /Users/shw/Documents/新因子统一整合_更新版_R17c_20260916.csv
Both compressed transfer SHA and decompressed server/Mac SHA match.
Decompressed SHA256: db18bbadeff3b336b10f3f9cd04e5cb6ea7abe0286ac7bbf8cb5fd9a45335d41.
Original user CSV and previous checkpoints were not overwritten. Owned Mac
transfer gzip removed by successful decompression. R17c replaces R15b as the
latest delivered table; do not use the unsubstantiated historical 20/40
transport rewrites in R15b. Pending explicit user authorization remains open.
