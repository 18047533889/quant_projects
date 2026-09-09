# R5 continuation — V3 remediation

## Environment and boundaries

- User requests all V3 requirements, not merely passing package tests. Spec: `/Users/shw/Documents/量化平台_全量问题合并与AI逐项整改总任务书_V3_20260906.md` (177 issues, GOLD40, E2E A–H).
- Remote: `qs-server-c`, socket `/tmp/quant-v3-server-c.sock`; isolated worktree `/home/sunhaiwei/quant_projects_v3_remediation`, branch `codex/v3-remediation-20260906`, base `2db45f4e446309003381a51a5e03071d31859502`.
- Original Python: `/home/sunhaiwei/quant_projects/.venv/bin/python`. Test env `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONPATH=factor_optimizer:factor_preprocess:.`.
- Real GPU NVIDIA L20. Original production checkout/assets/models unchanged; no commits/push/deploy or production migrations authorized/performed.
- Root local mirror is incomplete; remote worktree is canonical. Never SCP stale agent-owned files. Apply patches and copy exact owned files only.
- Agents authorized by user: existing `v3_optimizer`, `v3_preprocess`, `v3_assets` use GPT-5.6 Sol. Reuse them only for bounded independent work.
- Temporary PostgreSQL `/tmp/quant-v3-pg-8VsqEo` is STOPPED. Start/DSN instructions remain in previous continuation/summary. Never point destructive live tests at production.

## Actual fixes since R3/R4

1. Legacy CPU/CUDA HAC no longer compresses internal missing dates. Leading/trailing finite contiguous segments allowed; irregular internal gaps are insufficient. Block bootstrap uses one shared original-time draw matrix for complete finite columns, invariant to factor order/count; invalid policies rejected. HAC and four multiple correction plus bootstrap metric versions bumped to 2.0.0. General irregular-time estimators still unsupported.
2. Public CPU/GPU pairwise RankIC checked against independent common-mask SciPy oracle with ties/constants/invalid poison/permutations.
3. Public actual CUDA exposure: nine metrics, device OLS/reductions, strict axes/masks, independent OLS/rank-deficiency/min-observation/drift/purity/factor-tile cases. Neutralized/residual RankIC CUDA remain unsupported.
4. CPU/public residual RankIC fixed: ExposurePanel.validity and LabelBundle.validity were ignored. Final residual-label Spearman now enforces declared min_obs. Both metrics version 2.0.0; independent masked OLS + Spearman poison tests.
5. Outbox.publish_pending/recover_expired_claims now accept exact idempotency_key, scoping BOTH expired recovery and candidate selection. OPS08 and H normal/recovery use exact `publish:{generation_id}`. SQLite/realPG tests prove unrelated pending/expired-claimed events untouched; shadow keeps production pointer unchanged.
6. OPS08 exact affected graph shadow executor persists audit-only envelopes and direct dependency payloads. Actual max_drawdown->FAhealth and recipe->existing FP bridge->changed treated values->public QE IC/coverage->FAhealth. Immutable old raw bytes, exact dry-run reconciliation and idempotent same-authority retries. Cluster/feature/model authorities and approved production rollback remain incomplete. Fixtures are not full historical production migration.
7. E2E-C real FE/Q10/adaptive Q20/risk exposure->six bounded TRAIN candidates->frozen OOS. Small/tied/missing-risk paths fail closed. No fabricated trial/FA/library refs; missing full trace remains partial.
8. H fitted `train_standardize@1.0.0` apply-only adapter validates FittedState identity, canonical factor-definition hashes, source implementation hash, strict cutoff, numeric scalar mean/scale, NaT/timezone. No refit.
9. H mature monitor reads actual published physical bytes for bounded retained generation history; verifies hash/size/staged bytes, axes, common recipe/state identity, overlap. Aggregate window identity binds actual values/axes/generations/policy. Slice mature labels BEFORE public QE rank_ic/rank_ic_series/coverage; gate actual finite IC dates. Non-QE drift dimensions explicitly external and status partial. Missing GCed history rejected; no invented sample history.
10. QA01 QAT07 runner now reconciles actual pytest collection vs AST definitions: 12 definitions /16 cases, no overwrite/missing. Includes negative duplicate/missing/unexpected tests; required collection errors/skips block.
11. Additional MODEL finite-fit defect: TRAIN nanmean/nanquantile included infinities and corrupted otherwise finite means. Five tests failed before, then fixed existing modeling.trainer.fit_preprocessing to normalize nonfinite features to NaN. Frozen steps record fit_semantics_version=finite-only-v2 in existing state hash; old serialized states unchanged. Sixth test proves old/new identity and legacy apply preservation.
12. V05 FO runner now respects typed PRUNED instead of unconditional passed=True; updates tier_name after promotion, preventing repeated expensive tier execution. PRUNED counts in multiplicity, cannot incumbent; low raw score alone does not override COMPLETE. Actual QE/FA nonlinear/event/complementary evidence production remains partial, not supplied by FO fixture refs.
13. Unified ledger guard now treats qualified *_VERIFIED labels like VERIFIED: without public paths/tests/history => EVIDENCE_INCOMPLETE; preserves reported_owner_status. Additional findings separate from authoritative177 inventory. Validator tests6 passed.

## Verification already saved

- `evidence/r2/v3_r5_qe_final.log`: 844 passed,2 optional absent-kernel skips,4 warnings (before new QA02 generator tests).
- `evidence/r2/v3_r5_platform_final.log`:469 passed,0skip,3warnings; then `v3_r5_postgres_final.log`:20 realPG passed including added scoped recovery. `v3_r5_outbox_scoped.log`:23passed.
- `evidence/r2/v3_r5_fo_fp_final.log`:1118passed,1 expected offline HP xfail,12warnings BEFORE later V05/V07 amendments.
- `evidence/r2/v3_r5_modeling_jobs_final.log`:447passed,59warnings BEFORE finite-fit amendment.
- `evidence/r2/v3_r5_modeling_finite_final.log`:427passed,3warnings after finite-fit code; subsequent one legacy identity test6 focused passed.
- Finite-fit old5failed log `v3_r5_finite_fit_before.log`, after13focusedpass `v3_r5_finite_fit_after.log`.
- Prior FA unchanged source:1375passed0skip49warnings with actual faiss/annoy/igraph isolated deps; evidence from R3 retained, not rerun current continuation.
- H10passed, monitoring6passed, fitted adapter19passed; latest agents own logs.
- Final post-freeze verification and QAT07 R5 still need recording below.

## Current finalization tasks

- v3_assets: root required actual CUDA generator positive test, reject CUDA->CPU relabel, identity-based array artifact dedup, revalidate mutable provenance, no fake L20 string on CPU data. QA02 generator existing authority now separates declarations vs evidence and downstream NOT_RUN; CSV regenerated. Await final actual GPU result and freeze.
- v3_preprocess: initial V07 attempt incorrectly set seven prefix-safe filters causal_safe=False/OFFLINE_ONLY and removed EWMA preset. Root rejected. Agent restoring ONLY its own last-turn downgrades, retaining prefix tests. Implementing existing FP EWMA numeric authority checkpoint full-history replay with finite max_history_rows budget; label FULL_REPLAY_ONLY, not O(1). FE ts_ema state exists but differs in lag/halflife/min_periods; do not alias silently. V07 must remain PARTIAL for efficient sufficient-state and other filters. Await tests and freeze.
- v3_optimizer V05 done, root requested owner ledger exact paths/tests/history/PARTIAL and freeze.
- Rerun final FO+FP, QE after generator tests, modeling/jobs after finite fit and runner changes; QAT07 eight wheels + actual parity/collection after all package freeze. Stop PG already done.
- Reconcile latest owner ledgers and root files, regenerate/validate177, fetch unified ledger locally, create truthful R5 report. Do not claim all177/GOLD40/E2E8 closed. Missing in-scope local adapters remain implementation work, not external blockers.

## Final frozen R5 results (supersede pending tasks above)

- All three agents finished and source frozen. V05 owner row PARTIAL in `evidence/r2/v3_optimizer_20260906.yaml`, actual runner/tests paths and limitations recorded.
- QA02 generator finalized with actual CPU/CUDA evidence/misuse tests, mutable provenance revalidation, identity-based ndarray artifact dedup. Neither empty auxiliary registry nor partial GPUExecutor list proves all public facade routes unsupported: absent runtime evidence => NOT_RUN; absent auxiliary metadata => NOT_DECLARED. Actual strict CUDA run can DONE even when auxiliary registry empty. Default CSV159rows has zero runtime evidence and no production completion claims. 11focusedpassed.
- V07 broad downgrade fully reverted; seven causal smoothers retain original causal_safe=True/PRODUCTION, original production_full preset restored. Additive `adapters/ewma_full_replay.py` reuses FP EWMA batch kernel. Strict numeric/identity/UTC checks, source implementation hash, append-only state, finite history budget, whole checkpoint transaction nonblocking file lock, unique owned temp+fsync/replace. Actual two-process competition ensures one success/one conflict-or-stale, no lost history. Adapter23passed; fullFP336passed1xfail. PARTIAL_FIXED_EWMA_FULL_REPLAY, NOT O(1) sufficient state.
- `evidence/r2/v3_r5_qe_capability_final.log`: **851passed2optional-skips4warnings115.93s**.
- `evidence/r2/v3_r5_frozen_fofp.log`: **1150passed1expected-offline-HP-xfail12warnings29.83s**.
- `evidence/r2/v3_r5_frozen_modeljobs.log`: **453passed59warnings118.82s** (includes final finite-fit/legacy identity and V05 changes).
- `evidence/r2/v3_qat07_r5_source_parity_20260907.json`: **PASS8wheels1750installedPythonfiles identical to checkout**, actual FP effective parameter/output equality, actual pytest collection12definitions16cases PASS.
- Package sources unchanged after these tests except reports/ledgers. PostgreSQL remains stopped. No production mutations/commits/push/deploy.
- Root final report `REMEDIATION_STATUS_20260907_R5.md`; unified ledger contains frozen_r5_verification and additional finite-fit finding separate from177IDs. Qualified *_VERIFIED without complete paths/tests/history degrades to EVIDENCE_INCOMPLETE, retaining original owner report separately. This may reduce apparent verified counts; it is honest evidence normalization, not code rollback.
- Remaining real work: complete A–H admission/library/feature/trial trace; OPS08 cluster/feature/model selective replay and approved same-lineage rollback; efficient other-filter state replay; general residual DAG and actualCUDA residual; complete per-capability runtime evidence/177/GOLD40 certification. Do not mark these externally blocked just because local implementation is missing.
