# R55 audit P0-5/P0-6 — progress log (task #95)

Working dir: /home/sunhaiwei/quant_projects/quant_platform

## Baseline
- pytest: 335 passed (baseline verified before edits).

## Classification (step 2)
PLATFORM-OWNED (platform may compute/validate):
- ArtifactRef (format-only ref) — already correct
- FeatureSetVersion.schema_hash / FeatureSetArtifact.content_hash / compute_feature_set_content_hash
  (platform governance artifact over carried member refs)
- ReportExportArtifact.content_hash (app/export — DO NOT TOUCH per scope)
- daily_snapshot manifest envelope hash (platform snapshot envelope)
- batch_fingerprint (ingest anti-replay token over CARRIED content hashes)
- _contenthash codec (platform transport codec)

DOMAIN-OWNED (platform must only CARRY):
- FactorDefinitionIdentity (spec 8.1) — factor_assets/identity/canonical.py owns it
- FactorValueIdentity (spec 8.2) — factor_assets + factor_engine own theirs
- EvaluationIdentity (spec 8.3) — quant_evaluator / factor_assets
- TreatmentIdentity (spec 8.4) — factor_assets treatment_selection
- cluster identity refs (LogicalCluster/ClusterVersion ids are carried refs)

## Violations found
1. app/contracts/identities.py — mints domain identities (canonicalize-then-hash over factor
   formula/params/calculation_semantics incl. hard-coded vwap->vwap basis).
2. app/candidate/candidate.py — re-negotiates FactorDefinitionIdentity in
   NormalizedCandidate.__post_init__ (semantic judgment + hard-coded vwap_to_vwap),
   derives semantic material in _semantic_id_fields, and calls NONEXISTENT
   _definition_identity_fields (latent NameError — module is dead code at runtime).
3. app/candidate/ingest.py — canonicalize(_representative_semantic_fields(...)) hashes
   discovery-side factor semantics into semantic_family_hint (platform minting).
4. app/orchestrator.py — pipeline_gate_decision() computes the whole promotion/admission
   decision in the platform (label maturity, return-basis judgment, |rank_ic| floor 0.02,
   duplicate-similarity threshold 0.7) + DEFAULT_MIN_RANK_IC + VWAP_TO_VWAP_BASIS.

## Plan
- identities.py -> carried-ref model (format-only validation), remove mint classes + canonicalize.
- NEW app/contracts/admission.py: AdmissionVerdict / AdmissionRequest / AdmissionAuthority Protocol.
- ingest.py: semantic_family_hint = CARRIED semantic id (format-only), fail closed when absent.
- candidate.py: drop negotiation + latent crash; carried semantic_hash only.
- orchestrator.py: delete gate logic/thresholds; inject admission_authority; record verdicts.
- tests: update candidate/orchestrator tests; new test_platform_identity_authority.py.

## Status — COMPLETE (2026-08-28)
- [x] map + classify
- [x] identities.py rewrite (carried refs, no hashing)
- [x] admission.py (AdmissionAuthority port + RefuseAdmission)
- [x] ingest/candidate (carried semantic id, no negotiation)
- [x] orchestrator (delegated verdicts, gate deleted)
- [x] tests updated + 27 new authority tests (tests/test_platform_identity_authority.py)
- [x] docs §4.9 rewritten + §4.9b admission delegation added
- [x] stale build/lib copy (old identities.py) removed
- [x] full suite green: 367 passed, 8 skipped (baseline 335)
- [x] data_access contract-importing tests still green (13 passed)