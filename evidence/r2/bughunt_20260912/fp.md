# FP bug hunt evidence — 2026-09-12

Scope: formal server-c tree `/home/sunhaiwei/quant_projects`, base HEAD
`e94ac507d670fd1c16b1d6a63fc5d6286daa5970`. Changes are limited to
`factor_preprocess/**`; no branch, worktree, copy, commit, push, deployment or
production data action was used.

## Bugs reproduced and fixed

### 1. Production fitted-state identity omitted the fit window

`FittedState._derive_state_id()` did not include `fit_start_time` or
`fit_end_time`. Two production states with identical learned values and
caller-provided provenance strings but different actual training cutoffs had
the same `state_id`. The derived identity now binds the fit window, state kind,
fit-universe ref, config hash and producer identity/version.

Regression:
`test_production_state_identity_binds_actual_fit_window` constructs three
otherwise identical states and proves all three IDs differ.

Migration impact: persisted production fitted states and recipes referencing
their old IDs require explicit invalidation/rebuild. Old IDs must not be
silently treated as equivalent under the expanded identity.

### 2. Fitted apply trusted a claimed decision boundary instead of the panel axis

`apply_frozen_fitted_recipe()` compared `decision_start` with the fitted cutoff
but never checked the actual DataFrame index. A caller could claim a future
decision start while supplying rows before that boundary, including fit-period
rows. The adapter now requires a nonempty, NaT-free, unique, increasing
`DatetimeIndex`, enforces a common timezone policy, and rejects panel rows before
the declared decision segment or at/before the fitted cutoff.

Regressions cover a disguised pre-decision row, a row exactly at the fit cutoff,
integer index, decreasing timestamps and duplicate timestamps.

### 3. Frozen recipe compilation did not bind declared semantics to execution

`TreatmentRecipe.compile()` resolved `implementation_ref` and parameters but
did not compare `semantic_transform_id`, `stage` or `requires_fit` with the
registry metadata. Thus a recipe could advertise rank while executing forward
fill, or advertise a fitted operator as stateless. Compile now rejects all
three mismatches before returning an executor.

Regressions cover forged semantic ID, forged stage, stateless implementation
claimed fitted, and registered fitted implementation claimed stateless. The
existing positive fitted fixture now truthfully registers `requires_fit=True`.

## Executed evidence

- Direct targeted identity/time-axis run: **71 passed in 0.89s**.
- Expanded changed/public-boundary run: **45 passed, 2 warnings in 56.19s**.
  The warnings are the existing FE Polars physical-spec warning and are not
  counted as passes.
- Full FP assertions: **417 passed, 1 xfailed, 4 warnings in 57.20s**.
  Evidence status is deliberately `SOURCE_CHANGED_DURING_RUN`, not stable PASS,
  because independent owners modified DA/FE files during execution. Log SHA-256:
  `4868dd845df00bebd4364d278eecea4b312f35bcad8d612740f3a877b4d14fca`.
- A subsequent focused evidence-wrapper run had **92 passed, 2 warnings**, but
  was also marked `SOURCE_CHANGED_DURING_RUN` due independent FA/FE edits. It is
  retained as non-stable evidence, log SHA-256
  `73ef3e71dcd65f190623a00ac550da6abf33cb398e2ee9e691f644732f88a3b9`.

No FE/DA failure was bypassed or modified here. A final stable digest run is
still required after other owners stop changing the formal tree.
