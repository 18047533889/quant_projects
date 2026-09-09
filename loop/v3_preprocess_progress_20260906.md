# V3 factor_preprocess handoff

Implemented bounded fixes for integer asset identity, duplicate long keys, FP→FE parameter forwarding, strict FE error handling, explicit exposure columns, safe lineage dedupe, recipe-tag certification, deep immutability, and stable callable fingerprinting. Added public-path regressions.

Follow-up: extended the existing `FeatureBundle` authority with a fail-closed primary-plus-auxiliary constructor. Missing indicators are derived from the original validity mask, freshness is appended as an independent channel, and the primary values remain byte-for-byte separate. Added conservative root `OutputProperties`: rank/clip invalidate exact neutral orthogonality, while explicitly same-mask/same-weight affine scaling preserves it.

Shared dependencies remain: DA canonical security/AxisRef, QE measured output postconditions, and FO fold-local candidate policies. Historical values affected by identity, duplicate-key, parameter, dedupe, tag, or implementation-hash defects need controlled re-evaluation/re-signing; no production migration was executed.

Second follow-up: with coordinator authorization, `factor_engine/backend/cleaned_bridge.py` now compiles and runs all-FE stateless recipes through the existing `PlanNode` and `PandasBackend` authority. The public FP recipe executor performs one long/panel boundary conversion; its two-step regression records one input load. New registry entries default to research-only and runtime parameter domains are enforced. Canonical recipe roundtrip, unknown-operator rejection, and fitted-state binding are also covered. RCP-07/08 have no FP public implementation entrypoint and remain cross-owner FO/QE blockers; no duplicate selector was added.

DTA follow-up: the FP data-access exposure boundary now returns true named TN panels, preserves instrument dtype, requests real `UpdateTime`, rejects unknown provenance and duplicate-key conflicts, and freezes copied axes/value buffers. Existing DA PIT, cross-market, price-basis, and change-impact suites passed, so no competing r30/temporal-join implementation was added. DTA-04/05 remain downstream universe/execution integration items; DTA-11 is owned by QE/FO. Mergeable statuses are in `evidence/r2/v3_preprocess_ledger_20260907.json`.

Exact per-task status and commands are in `evidence/r2/v3_preprocess_20260906.yaml`.
