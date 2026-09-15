# V8 FP/QE remediation ledger

Scope: A29-A32 and A54-A63 only. Writable checkout `/home/sunhaiwei/quant_projects_v3_remediation`; no commit, push, deploy, or production migration.

## Before evidence

Source review reproduced the documented defects: merge-created RangeIndex was reindexed against caller labels; colliding exposure names could select the factor value; intercept fitting centered `y` without centering unscaled `X`; coordinate descent exposed neither objective normalization nor convergence; diagnostics inferred a design from raw exposures and used a normal approximation; HMM `ravel()` accepted `(T,2)`, cached the pre-clean array for xi, and treated likelihood decreases as convergence; regime labels were positionally consumed and cast through `int`; integer factor matrices allocated integer NaN/output outputs; global fallback was omitted from serialization; causal detection restarted its warm-up on every call.

## Implemented boundaries

- Regularized neutralization: cardinality-validated many-to-one alignment by private row ordinal, collision rejection, independent centering/scaling, explicit mean-loss objective, strict public hyperparameter guards, and per-date convergence/objective/stationarity status in `Series.attrs`.
- Diagnostics: actual-intercept and optional aligned-weight design support, full-column-rank and DOF separated from condition number, Student-t reference p-values explicitly marked descriptive/same-sample.
- HMM: univariate shape guard, one cleaned valid axis across forward/backward/xi, original-axis probability restoration, fit lifecycle reset, likelihood-decrease failure, transition validation, and explicit `filtered_asof` / `smoothed_posthoc` inference scope.
- Regime preprocessing: exact index/order and integer-label validation, effective finite observations, invalid/constant factor exclusion, float output allocation, serialized global fallback, copy-isolated read-only fitted arrays, unknown-state policy, valid-only rank masks.
- Causal detector: fit coordinates, rolling carry, chunk/Bar continuity, min-period pairwise-complete correlation, checkpoint restore, finite bounded strength, and missing/first-valid transition semantics. Historical overlap is an explicit stateless replay and does not mutate the live checkpoint.
- Representation selection: `decide_representation` delegates the unchanged canonical request to `DecisionProvider.decide`; FP contains no score table or default cost table.

## Verification

Environment: `BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 PYTHONPATH=factor_optimizer:factor_preprocess:factor_assets:quant_evaluator:.`

Final command: `python -m pytest -q factor_preprocess/tests quant_evaluator/tests/test_v8_hmm_scope.py --disable-warnings --maxfail=3`

Result: **349 passed, 1 xfailed**, 3 warnings, 23.87s. The xfail is the pre-existing HP-filter offline-only causal limitation, outside this scope.

Focused V8 public-entry tests cover non-RangeIndex/reordered exposure alignment, affine intercept behavior, duplicate-key rejection, invalid solver budgets, float weighting, label identity/type rejection, canonical DecisionProvider replay, HMM shape/missing-axis/refit lifecycle, and filtered prefix invariance.

## Source hashes

See `source_hashes.sha256`; hashes were captured from the remote writable checkout after the successful run.

## Historical impact and rollback

Recompute only artifacts variants that used these regularized FP entrypoints, regime weighting/switching, causal detector, or HMM posteriors. Historical factor assets outside those routes do not require recomputation. Existing results that lack actual solver convergence/effective-sample evidence or that treated smoothed posterior state as production as-of state are uncertified. Rollback is file-scoped using the pre-V8 baseline/source manifest; no database or deployed state changed.

## Uncertified boundaries

No GPU/backend parity claim is made. No claim is made that same-sample residual correlation proves independent alpha or out-of-sample quality. Smoothed HMM output remains research/post-hoc evidence and must be rejected by production consumers outside this owned module. End-to-end deployed FE routes and historical artifacts were not mutated in this task.
