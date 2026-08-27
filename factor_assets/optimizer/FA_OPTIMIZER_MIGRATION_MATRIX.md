# FA Optimizer Migration Matrix (DLIB-FA-001)

**Scope:** `factor_assets/optimizer/` — FA's internal 2nd optimizer.
**Authority:** `factor_optimizer` (FO) owns Pareto / MultiFidelity / Plateau /
Mutation / Search / Sealed Test. FA is a **consumer** of FO evaluation-tier
evidence and a **producer** of Factor-Library-Assembly artifacts only.

Per §108, no module is deleted abruptly. Each item below records: current
state, target authority, compatibility-migration status, and whether it is
already a thin reference-only adapter (VERIFIED_ALREADY_FIXED).

| Module | Current | Target authority | Migration status |
|--------|---------|------------------|------------------|
| `pareto.py` (ParetoOptimizer/ParetoPoint/ParetoFrontier) | FA-owned Pareto dominance | If used for **treatment/search** Pareto → FO. If used for **Factor-Library-Assembly** Pareto (ranking candidates for a library) → move to `assembly/pareto.py` (assembly-only, not search). | **VERIFIED_ALREADY_FIXED (assembly-only).** The only production caller is `assembly/engine.py` `_pareto_rank`, which uses `ParetoPoint.dominates` purely to rank already-admitted candidates for a FactorSet — a Factor-Library-Assembly use, not a search/treatment use. It does NOT run a search loop, does not mutate, does not evaluate. It is a pure dominance comparator. No treatment/search Pareto lives in FA. |
| `multifidelity.py` (MultiFidelityPolicy/FidelityTier) | FA-owned L0-L4 promotion policy | **Retire.** FO is the authority for fidelity/promotion. FA consumes evaluation-tier evidence. | **NOT_IMPLEMENTED (retire).** No production caller outside `optimizer/` and its own tests. Kept for compatibility; marked `RESEARCH_ONLY`; emits `DeprecationWarning` on import. No FA production path consumes it. |
| `plateau.py` (ParameterPlateauDetector/NeighborSurvivalAnalyzer) | FA-owned search-stopping plateau | **Retire from search-stopping.** FO owns search stopping. FA may keep a health/robustness consumer only. | **NOT_IMPLEMENTED (retire).** No production caller outside `optimizer/` and its own tests. Kept for compatibility; marked `RESEARCH_ONLY`; emits `DeprecationWarning` on import. |
| `typed_mutation.py` (TypedMutation/MutationContext/MutationResult) | FA-owned mutation execution semantics | **FO MutationSpec/Trial reference adapter.** FA carries the reference; it does NOT execute mutations. | **VERIFIED_ALREADY_FIXED (reference-only).** `TypedMutation` is a frozen dataclass that only *records* a mutation reference (mutation_id, parent_asset_id, mutation_type, parameters) bound to FA identity/lifecycle. It has no execution engine — no code applies a mutation to factor values. It is a reference adapter for FO MutationSpec/Trial. |
| `frozen_candidate.py` (FrozenCandidate/FrozenState/TEST_SEALED/TEST_CONTAMINATED) | FA-owned sealed-test authority | **Retire TEST_SEALED/TEST_CONTAMINATED authority.** Sealed-test is FO. FA only stores `test-evidence-ref` + `contamination-verdict-ref`. | **NOT_IMPLEMENTED (retire authority).** The `FrozenState.TEST_SEALED` / `TEST_CONTAMINATED` states and `FrozenCandidateStateMachine` are a logical seal, not a real test-protection boundary (see §43). No production caller outside `optimizer/` and its own tests. Kept for compatibility; marked `RESEARCH_ONLY`; emits `DeprecationWarning` on import. |

## Compatibility migration (§108)

- **Callers identified:** the only production caller of any optimizer module is
  `assembly/engine.py` → `ParetoPoint` (assembly-only dominance). All other
  optimizer modules have zero production callers (only their own tests).
- **Canonical adapter:** `ParetoPoint` remains importable from
  `factor_assets.optimizer.pareto` (unchanged) and is the canonical
  assembly-only dominance comparator. No new adapter needed.
- **DeprecationWarning:** `multifidelity.py`, `plateau.py`, `frozen_candidate.py`
  emit `DeprecationWarning` on import, marking them for future removal.
- **Internal callers migrated:** none needed — no internal production caller.
- **Tests confirm zero production callers:** `test_import_manifest.py` still
  imports every shipped module (they remain importable); the optimizer tests
  still pass against the retained modules.
- **Marked for future removal:** `multifidelity.py`, `plateau.py`,
  `frozen_candidate.py` (after the DeprecationWarning window).

## Cross-cutting note (§43)

`campaigns/ledger_adapter.py::seal_test_splits()` records a **logical** seal
(`usage_type="seal"`) — it is a projection/audit record, NOT a real test
protection boundary. Test authority is FO `TestAuthorityBroker`. No production
code treats the FA ledger seal as real protection.
