# Model Layer Major Redesign — Architecture Migration Report

- git_sha: `96939012325c5fd07171056d2e20f73088025a6f`
- generated: 2026-08-11T20:42:38+0800
- total canonicals classified: 5
- model-like canonicals: 5
- legacy local predictive (research-only, default_searchable=False): 5
- artifact-backed predictive learners (ModelRegistry): 5

## Execution classes (§1.1)

| class | meaning |
|---|---|
| LOCAL_ROLLING_ESTIMATOR | rolling/statistical local estimators (window is part of the definition) |
| RECURSIVE_STATE_ESTIMATOR | Kalman/CUSUM/stateful recursive filters |
| SAME_TIME_CROSS_SECTIONAL | same-date peer models (self-exclusion, PIT universe) |
| PREDICTIVE_SUPERVISED | artifact-backed supervised learners (this redesign) |
| RESEARCH_STRUCTURAL | DMD/SSA/RQA/TE/etc. research primitives |

## Migration phases (§65)

- Phase A (contracts): ModelExecutionClass / RichModelTiming / SampleAdequacy / SearchPolicy / WalkForward / Artifact — built additively; old results unchanged.
- Phase B (legacy classification): the five `panel_rolling_*` / `panel_regime_*` / `panel_mixture_*` operators classified LEGACY_LOCAL_ROLLING, default_searchable=False, research_only=True via `modeling/legacy.py` (operator registrations untouched).
- Phase C (predictive linear family): PCR / PLS / ElasticNet learners in `modeling/learners/` — pooled-panel, multi-year, artifact-backed, validation-only selection.
- Phase D (predictive Regime / MoE): `modeling/learners/regime.py`, `modeling/learners/mixture_of_experts.py` — §6 support contracts, fail-closed.
- Phase E (Factor DSL as-of artifact score): `modeling/dsl_bridge.py` provides `score_asof`/`ArtifactResolver` (§49/§50/§51); wiring into the DSL compiler remains pending (concurrent session owns `api/mining_integration.py`).
- Phase F (historical walk-forward regeneration): `modeling/evidence.py` + `run_walk_forward_evidence` produce per-fold as-of artifacts.

## Design principles (§85)

Predictive models are no longer defined by 'can compute a value in a short window'. They carry ModelSpec + TrainingSpec + WalkForwardSpec + LabelContract + DecisionClock + SampleAdequacyContract + ParameterSearchPolicy + PreprocessingSpec + ArtifactManifest + EvaluationEvidence, or they are NOT_PRODUCTION_READY (§81).
