# Wave 0 Architecture Red-Team Audit

**Status:** Independent findings document
**Date:** 2026-08-13
**Scope:** Read-only analysis of CONTRACT_FREEZE_DRAFT.md, PACKAGE_SKELETON.md, PACKAGE_BOUNDARY_DRAFT.md, and all Wave 0 reports
**Binding snapshot:** LOCAL_REPO_SNAPSHOT bound to commit `ddb03749b7ff85e63b633770c43dfbbb7562af19`

---

## Executive Summary

Wave 0 has produced a clear, achievable architecture grounded in strong reuse of DataAccess and FactorEngine. The boundary matrix and integration contracts are well-conceived. However, **freeze is blocked by 23 unresolved decisions, 4 critical false dependencies, multiple unstable/stale snapshots, and foundational semantic gaps that block Phase 1 implementation.** 

Independent red-team findings rank the most severe blockers:

1. **P0: FA Migration Report Missing** — FA is explicitly pending independent review; all FA-downstream work (FO novelty, FP aggregation, Research Control ledger) cannot freeze without it.
2. **P0: Snapshot Drift** — Evidence bound to `ddb03749` (2026-08-13T13:48Z) but repo is 18 commits ahead with concurrent dirty changes in factor_engine. Contract is stale.
3. **P0: Unresolved Ownership Cycles** — Research Control placement, ModelInput termination, FA/FP aggregation boundary, and QE/FP role overlap create implicit cycles in the dependency DAG.
4. **P1: No Machine-Checkable Boundary** — 135-row matrix exists; no automated import/dependency/cycle checker. Auditors cannot verify compliance.
5. **P1: 23 Critical Unresolved Decisions** — Cannot implement without resolving terminal type name, canonical envelope, error taxonomy, metric tier, FA lifecycle representation, and others.
6. **P1: Legacy Import Leakage** — 769 sys.path hacks and 210 legacy import references; 58 production/operational path mutations remain in admissions/evaluators/agents.
7. **P1: Foundational Semantic Gaps** — RankIC ties/NaN coercion, fitted state serialization, label timing contracts, novelty vs similarity vs neighbor definitions, and A-share domain gaps unresolved.

**Honest assessment:** The architecture is **sound but immature**. Freeze is aspirational; implementation cannot start without resolving P0/P1 blockers and rebinding evidence to a later snapshot.

---

## Severity Ranking and Freeze Blockers

### P0 CRITICAL (Implementation impossible without resolution)

#### 1. FactorAssets Migration Report Missing / FA Independent Audit Not Complete

**Finding:** The entire FA module is explicitly pending its own migration report and independent audit. CONTRACT_FREEZE_DRAFT.md line 11 states:

> "FactorAssets migration and its identity/lineage/novelty storage boundary are not independently reviewed."

BLUEPRINT_COVERAGE.md item 354 notes "Concurrent reports were not part of the requested reading scope" — FA_LEGACY_MIGRATION.md exists but FA_INDEPENDENT_AUDIT does not.

**Concrete failure scenarios:**
- FA identity adapter shape unknown → QE evidence refs and FO lineage cannot freeze.
- FA raw-value retention boundary frozen as "forbidden" but FA report may require bounded fingerprints → contradiction.
- FA/FP aggregation boundary ("FA owns aggregation specifications, FP consumes selected factors") cannot be split without FA schema.
- Research Control trial/decision ledger synchronization with FA undefined.

**Evidence:**
- PACKAGE_BOUNDARY_DRAFT.md row 119: "Expected owner is FA... Final classification awaits FA report/audit."
- BLUEPRINT_COVERAGE.md contradiction #354: "Auditor overlap and closure" lists FA/FP and ledger authority as unresolved.

**Freeze blocker:** YES. Phase 1 cannot freeze schemas without FA architecture.

**Recommended gate:** FA migration report + independent audit + chief-integrator sign-off required before CONTRACT_FREEZE closes.

---

#### 2. Repository Snapshot Drift: Evidence Bound to Stale Commit

**Finding:** All Wave 0 reports are bound to commit `ddb03749b7ff85e63b633770c43dfbbb7562af19` (2026-08-13T13:48:17Z). Current HEAD is 18 commits ahead (2a5d43dc, most recent "fix: make crossing acceleration causal"). Repository has dirty changes in factor_engine/backend (pair_window_spec.py, polars_expr_emitter.py, sql_pushdown/emitter.py, stat_valid.py, layer_governance.py, operator_surface.py, test_neutralize_fastpaths.py).

**Concrete failure scenarios:**
- DA/FE inventory snapshots may be stale; new DA facade or FE identity changes not captured.
- Legacy import audit counted 769 sys.path hacks against old code; 18 commits may have added/removed path mutations.
- QE/FO/FP reference implementations may have changed; golden tests must be re-verified.
- Concurrent factor_engine backend edits may introduce new DA/FE API surface or break assumptions.

**Evidence:**
- CONTRACT_FREEZE_DRAFT.md line 5: "A snapshot delta binding this draft to the then-current tree is required before any freeze decision."
- LOCAL_REPO_SNAPSHOT.md bound to ddb03749; git log shows 18 newer commits.
- Git status shows 11 modified factor_engine files, not yet staged.

**Freeze blocker:** YES. Freeze cannot be final without a post-drift snapshot delta.

**Recommended gate:** Re-run all four legacy miners + boundary + leakage audits against current HEAD after stabilizing dirty changes.

---

#### 3. Dependency DAG Contains Unresolved Cycles / False Dependencies

**Finding:** Three design elements create implicit circular dependencies or missing owners:

**A. Research Control placement and ownership**
- PACKAGE_BOUNDARY_DRAFT.md row 131: "FO/FA ledger synchronization" classified as NEED_ADAPTER; owner listed as "Research Control with FO/FA."
- But Research Control is not one of the four runtime packages; its repository path, API, and dependency direction are "unresolved" (BLUEPRINT_COVERAGE.md #334).
- PACKAGE_SKELETON.md line 15 proposes `research_control/` at root level but qualifies it as "placement unresolved."
- CONTRACT_FREEZE_DRAFT.md #4: "Research Control repository path, API owner, and synchronization/idempotency with FO/FA" remains unresolved.
- **Implication:** FO and FA cannot finalize trial/decision records without knowing whether Research Control is a dependency or a consumer ledger. Both could legitimately claim authority over "mutation trial" records.

**B. ModelInput terminal type and owner**
- PACKAGE_BOUNDARY_DRAFT.md row 129: `FeatureBundle` versus `ModelInputBundle` unresolved; ModelInput owner undefined.
- BLUEPRINT_COVERAGE.md #1: "Final output name: architecture uses `ModelInputBundle`, while FP and integration contracts use `FeatureBundle`."
- PACKAGE_SKELETON.md line 233: FP exports `feature_bundle.py` but skeleton says "canonical terminal type (`FeatureBundle` versus `ModelInputBundle`) decision pending."
- CONTRACT_FREEZE_DRAFT.md line 259 and line 5: Explicit unresolved decision #1.
- **Implication:** FP cannot finalize public API until modeling/caller ownership is assigned. Is ModelInput a FP responsibility, caller responsibility, or a separate schema owner?

**C. FA/FP aggregation boundary**
- BLUEPRINT_COVERAGE.md #341: "FA/FP aggregation boundary: FA owns aggregation specifications/weights while FP consumes selected factors/representations. Define which adapter computes aggregate values and where they reside."
- PACKAGE_BOUNDARY_DRAFT.md row 122: "Factor selection/admission and aggregation specifications" assigned to FA; row 200: "FA aggregation versus FP representation responsibility."
- **Implication:** FA selects and weights factors; FP transforms them. But if FA specifies a weighted composite, does FP compute it? Is it stored in FP outputs? Who retains the aggregation state?

**D. QE core/temporal/runtime module ownership**
- BLUEPRINT_COVERAGE.md #349: "QE role overlap: runtime/core/temporal agents overlap around IC, exposures, neutralization-derived metrics, and robustness. Exact module ownership and handoff are needed."
- PACKAGE_SKELETON.md line 61: QE has both `runtime/` and root-level `metrics/`; temporal metrics and exposure metrics not clearly partitioned.
- PACKAGE_BOUNDARY_DRAFT.md row 104: "`EvaluationContext` and exposure context" is NEED_ADAPTER; row 110: "Exposure dependence and neutralization survival metrics" also belongs to QE but context ownership is "caller/modeling with DA."
- **Implication:** If QE owns both metric output and exposure input validation, is QE now a consumer of DA context? That creates a second DA→QE edge not reflected in the primary DAG.

**Evidence:**
- CONTRACT_FREEZE_DRAFT.md unresolved decisions #4, #5 (Research Control, ModelInput).
- PACKAGE_BOUNDARY_DRAFT.md row 131, 129, 122 explicit NEED_ADAPTER + unresolved owner.
- BLUEPRINT_COVERAGE.md contradictions #334, #1, #341, #349.
- PACKAGE_SKELETON.md sections 2, 5, 6 with forward-references to unresolved decisions.

**Freeze blocker:** YES. Cycles or unclear edges prevent teams from knowing which packages they cannot import.

**Recommended gates:**
1. Explicitly assign Research Control as either (a) a side ledger that QE/FO/FA append to (non-dependency), or (b) a core dependency that owns trial/decision truth (then add it to the DAG).
2. Assign ModelInput schema owner (FP or caller).
3. Freeze FA/FP aggregation boundary (one owns weights, one owns compute, or both are specified in a frozen protocol).
4. Partition QE temporal/exposure into explicit module ownership with clear responsibility/ownership matrix.

---

#### 4. No Automated Import/Boundary Checker: Manual Matrix Cannot Guarantee Compliance

**Finding:** PACKAGE_BOUNDARY_DRAFT.md provides a 135-row capability matrix with owner assignments and decision classifications (USE_EXISTING, NEED_ADAPTER, TRUE_GAP). However, there is **no machine-checkable implementation**, no automated linter, and no CI gate that validates:

- New packages do not import from forbidden trees (AutoFactorEvaluation-RECONSTRUCT internals, raw_data_layer runtime, etc.).
- Adapter modules do not cause core imports to fail.
- Cross-package dependencies remain acyclic.
- `sys.path` mutations do not occur in production code.
- Legacy symbols are not accidentally re-imported.

**Concrete failure scenario:**
During implementation, an agent builds `quant_evaluator.metrics.ic` and unknowingly imports `from AutoFactorEvaluation-RECONSTRUCT.factor_engine.cleaned_operators import ...` thinking it's a "USE_EXISTING" FE capability. The boundary matrix forbids it, but only code review would catch it. If 100+ files are written in parallel, such violations accumulate silently until integration.

**Evidence:**
- BLUEPRINT_COVERAGE.md #317: "Exact cross-package import allowlist and CI checker" is an unresolved skeleton decision.
- LEAKAGE_BOUNDARY_AUDIT.md: Found 769 sys.path hacks and 210 legacy import references but reports them as read-only evidence. No automated gate prevents new violations.
- PACKAGE_BOUNDARY_DRAFT.md section 4.1 prohibits legacy imports but provides no validator.
- No `.pre-commit`, `.pylintrc`, `pyproject.toml [tool.import-linter]`, or CI workflow found in blueprint directory.

**Freeze blocker:** MEDIUM→HIGH. Manual compliance during parallel implementation will fail.

**Recommended gate:** Before Phase 2 begins, implement:
1. Import-linter CI check (e.g., `import-linter` Python package or hand-rolled AST walker).
2. Forbidden-path allowlist in `.pylintrc` or `pyproject.toml`.
3. Test fixture that verifies `import quant_evaluator` works without monorepo PYTHONPATH.

---

#### 5. 23 Unresolved Decisions Block Schemas and Public APIs

**Finding:** CONTRACT_FREEZE_DRAFT.md § 9 lists 21 unresolved decisions; BLUEPRINT_COVERAGE.md § 327 adds 28 contradictions including overlaps. Key unresolved items:

| Decision | Impact | Blocker? |
|---|---|---|
| 1. `FeatureBundle` vs `ModelInputBundle` | FP public API, terminal type | YES |
| 2. `FactorSet` vs `FactorSetArtifact` | FA/FP contract, selection output | YES |
| 3. Versioned envelope without `quant_contracts` | Cross-package metadata, field list | YES |
| 4. Research Control placement | Ledger ownership, dependency DAG | YES |
| 5. ModelInput owner | Terminal handoff, caller integration | YES |
| 6. `PRODUCTION_CERTIFIED` as state or flag | FA lifecycle representation | YES |
| 7. Versioned thresholds and tolerances | FA selection policy, metric tiers | YES |
| 8. Metric tier vocabulary | QE registry semantics, FO search | YES |
| 9. Permitted FA fingerprints and retention | FA scale design, storage policy | YES |
| 10. FA/FP aggregation boundary | Signal computation responsibility | YES |
| 12. FO L0-L4 vs fast/full/robust mapping | FO search nomenclature | YES (low) |
| 13. Concrete error code strings | QE/FO/FA exception taxonomy | YES (medium) |
| 19. QE role overlap | Module ownership, temporal/core split | YES |
| 20. FP optimization handoff | Performance agent authority | YES (medium) |

**Evidence:**
- CONTRACT_FREEZE_DRAFT.md § 9, lines 257–281.
- BLUEPRINT_COVERAGE.md § 327, items 1–28.

**Freeze blocker:** YES, all 23 are critical. Cannot generate schemas, public APIs, or finalize error handling without these.

**Recommended gate:** Lead Architect + Chief Integrator closed-door decision session. Publish decisions as a separate "DESIGN_DECISIONS.md" with rationale and fallback options. Then update all downstream contracts.

---

### P1 HIGH (Blocks orderly implementation, creates rework)

#### 6. Legacy Import Leakage: 58 Production Path Mutations in Evaluators/Agents

**Finding:** LEAKAGE_BOUNDARY_AUDIT.md categorized 769 sys.path occurrences. Of those, 58 are marked **UNACCEPTABLE** production/operational code path mutations (not test/example bootstraps):

- `/home/shw/quant_projects/factor_layer/factor_admission/run_from_config.py` (lines 8–9): mutates sys.path.
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` (lines 152–153, 316–317, 468–469, 622–623): four separate sys.path mutations in evaluation pipeline.
- `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/platform_bootstrap.py` (lines 140–141, 144): platform bootstrap mutates sys.path.
- `/home/shw/quant_projects/factor_layer/factor_agent/main.py` (lines 13–14): agent main entry mutates sys.path.
- `/home/shw/quant_projects/ashare_lqtp_kit/tools/probe.py` (lines 20–21, 46): LQTP tools mutate sys.path.
- 50+ others in gateway, raw_data_layer, dataaccess scripts.

**Concrete failure scenario:**
When QE/FO/FA/FP packages are installed in isolation (per extraction acceptance requirement), any legacy code that depended on these path mutations will break. If QE accidentally imports legacy evaluator code that mutates sys.path, QE will fail in isolated venv. Extraction audit will expose this.

**Evidence:**
- LEAKAGE_BOUNDARY_AUDIT.md table, lines 9–160, 58 UNACCEPTABLE verdicts.
- All violations in production/operational code, not test fixtures.

**Freeze blocker:** MEDIUM→HIGH. Extraction phase will fail if legacy path mutations are not removed or quarantined before packages are built.

**Recommended gate:** Before extraction testing, audit and remove all 58 sys.path mutations from production code. Quarantine any still needed as corpus/test-only.

---

#### 7. RankIC Semantics and NaN Coercion Unspecified

**Finding:** The core QE metric—RankIC (daily cross-sectional average Spearman)—has multiple unresolved semantics:

**A. Tie handling**
- BLUEPRINT_COVERAGE.md § 169: "RankIC is daily cross-sectional average-rank Spearman with pairwise finite filtering; constants and insufficient assets produce NaN."
- QE_LEGACY_MIGRATION.md row 70: `daily_ic()` seed does "best compact legacy Pearson/average-tie RankIC" and requires "ties, constants, min-assets, sign flip, monotonic transform, order" tests.
- But "average-rank" vs "min-rank" vs "max-rank" is not specified. SciPy's tie-breaking is deterministic but not documented in contracts.

**B. NaN aggregation across days**
- BLUEPRINT_COVERAGE.md § 169: "aggregate NaNs remain NaN."
- But what if 1/252 days is NaN (insufficient assets that day)? Does ICIR drop that day or carry NaN? ICIR formula (mean_IC / std_IC) with partial NaNs is undefined.

**C. Pairwise-finite filtering for Spearman**
- "pairwise finite filtering" means remove (signal, return) pairs where either is NaN, compute correlation on remaining pairs.
- But if 90% of assets are missing on a given day, is correlation meaningful? No minimum-observation threshold is specified.

**Evidence:**
- BLUEPRINT_COVERAGE.md § 169 and § 107.
- QE_LEGACY_MIGRATION.md row 70 lists required tests but implementations diverge.
- Contract does not name concrete SciPy/scipy.stats.spearmanr parameters (nan_policy, alternative).

**Freeze blocker:** MEDIUM. QE metric contracts are incomplete without explicit tie-handling and NaN-aggregation policy. Golden tests cannot be written.

**Recommended gate:** Freeze RankIC contract with:
1. Exact tie-breaking rule (average-rank with SciPy default).
2. Minimum observations per day.
3. NaN aggregation rule (drop days with NaN or carry NaN).
4. Golden-test oracle (historical reference implementation or statsmodels).

---

#### 8. FittedState Serialization and Checkpoint Strategy Unspecified

**Finding:** FP is required to implement stateless and fitted transforms, with FittedState "immutable once published" (CONTRACT_FREEZE_DRAFT.md § 5). However:

**A. No serialization format defined**
- PACKAGE_SKELETON.md line 232: FP.contracts.state.py will define `FittedState`, but serialization is not specified.
- PACKAGE_BOUNDARY_DRAFT.md row 127: "Fitted state is bound to fit window, feature IDs/order, transform version/config hash, training-universe reference and learned-parameter hash/ref."
- But how is FittedState stored? JSON? Pickle? Parquet + protobuf? Arrow IPC?

**B. Learned-parameter storage undefined**
- "learned-parameter hash/ref" suggests either (a) embedding parameters in FittedState, or (b) storing them externally and only storing a hash.
- If external, who stores them? FA? A new artifact store? For how long?

**C. Cross-fold checkpointing undefined**
- If FP uses k-fold cross-validation, does it save k FittedState objects? One per fold? Does FP runtime load them?
- BLUEPRINT_COVERAGE.md § 273 prohibits "full-sample fitting before a split" but says nothing about fold-local checkpointing.

**Evidence:**
- CONTRACT_FREEZE_DRAFT.md § 5, line 170.
- PACKAGE_SKELETON.md line 232 forward-references unspecified state.py.
- PACKAGE_BOUNDARY_DRAFT.md row 127 lists required fields but no serialization contract.

**Freeze blocker:** MEDIUM. FP implementation cannot proceed without a frozen serialization contract. Golden tests need round-trip (fit, save, load, transform) parity.

**Recommended gate:** Define and commit to a `FittedState` Pydantic model with an explicit serialization strategy (e.g., Arrow IPC for arrays, JSON for metadata, external S3 refs for large parameters).

---

#### 9. Label Timing Contracts Incomplete: Ambiguous Execution/Decision/Horizon Semantics

**Finding:** QE consumes an explicit `LabelBundle` (CONTRACT_FREEZE_DRAFT.md § 5) with fields: `target_id, values, horizon, execution_delay, decision_time, execution_time, label_start_time, label_end_time, validity, source_ref, calendar_ref`.

However, the semantics of these fields in real backtesting are underspecified:

**A. Execution delay vs. decision delay**
- Signal generates at end-of-day t (e.g., 15:00 SZ). Decision time is t. Execution time could be t (same-day late order) or t+1 (next-day open). Label start/end unclear.
- No contract distinguishes "I generate a signal at t, decide at t, but it executes at t+1" from "I decide at t, it executes at t+1, but I can only observe label from t+1 onwards."

**B. Horizon definition**
- Is horizon measured in trading days (calendar-adjusted) or wall-clock days?
- Does horizon=5 mean "5 trading days from execution_time" or "5 calendar days"?
- BLUEPRINT_COVERAGE.md § 177: "Horizon and execution delay are distinct dimensions; delays must cover at least T+1/T+2/T+3 where applicable" but does not define horizon numerics.

**C. A-share specific gaps**
- BLUEPRINT_COVERAGE.md § 179: "Raw IC and tradable IC are separate. QE context carries suspension, ST/status, actual limits, can_buy/can_sell/can_hold, board, and IPO age."
- But who populates these fields? Who validates that tradability is known at `decision_time`?

**Evidence:**
- CONTRACT_FREEZE_DRAFT.md § 5 (LabelBundle) and § 4 (Acceptance before freeze).
- BLUEPRINT_COVERAGE.md § 176–180 (Time/PIT/A-Share rules).
- PACKAGE_BOUNDARY_DRAFT.md row 70 (Label timing and execution convention: NEED_ADAPTER).

**Freeze blocker:** MEDIUM→HIGH. QE golden tests cannot be written without exact label timing. Leakage auditor cannot verify causality.

**Recommended gate:** Freeze `LabelBundle` schema with exact definitions:
1. Horizon: calendar days or trading days (specify).
2. Execution delay: discrete options (T+0, T+1, T+2, T+3 for A-share).
3. Tradability validation: caller responsibility or QE adapter?

---

#### 10. Novelty vs. Similarity vs. Neighbor Definitions Conflated

**Finding:** PACKAGE_BOUNDARY_DRAFT.md and PACKAGE_SKELETON.md use three terms with overlapping meanings but no clear distinction:

- **Exact identity:** FA has seen this factor before (same canonical hash).
- **Structural similarity:** Factor has same operators/domains but different parameters (e.g., `sma(close, 5)` vs `sma(close, 10)`).
- **Behavioral neighbor:** Factor behaves similarly to existing ones (high correlation in historical returns).
- **Novelty:** Factor is new relative to existing set (inverse of neighbor/duplicate).

**Concrete failure scenarios:**
1. FO generates a mutation with different parameters → is it "not novel"? If so, should it skip evaluation? But it may have different risk/return profiles.
2. QE evaluates novelty by requesting "exact residual IC" (row 121) through a provider → but residual IC is slow (O(K) evaluations). Is ANN shortlist the same as novelty?
3. FA owns "conditional novelty and exact similarity orchestration" (row 121) → but what is "conditional"? Novel relative to which universe?

**Evidence:**
- PACKAGE_BOUNDARY_DRAFT.md row 120, 121, 122 all use "novelty/neighbor/similarity" without clear definition.
- BLUEPRINT_COVERAGE.md § 188: "exact/canonical identity, structural similarity, behavior, and conditional novelty remain distinct" — states they are distinct but does not define.
- PACKAGE_SKELETON.md line 180: FA has `novelty/provider.py` but interface is not specified.

**Freeze blocker:** MEDIUM. FO cannot implement mutation generation without knowing which neighbors trigger a "duplicate" rejection.

**Recommended gate:** Publish a "NOVELTY_DEFINITIONS.md" with:
1. Exact identity (canonical hash match).
2. Structural similarity (same operators, within parameter bounds).
3. Behavioral neighbor (e.g., Spearman > 0.8 vs existing factors).
4. Conditional novelty (novel relative to selected universe).
5. Decision rule: when does FO skip evaluation vs. evaluate anyway?

---

### P2 MEDIUM (Creates testing/QA burden, delays integration)

#### 11. Unstable/Stale Snapshot: Wave 0 Reports Not Mutually Coherent

**Finding:** Wave 0 produced nine reports (QE_LEGACY_MIGRATION, FO_LEGACY_MIGRATION, FP_LEGACY_MIGRATION, FA_LEGACY_MIGRATION, LEAKAGE_BOUNDARY_AUDIT, LICENSE_PROVENANCE_LEDGER, DA_INVENTORY, FE_INVENTORY, CORPUS_INVENTORY, LOCAL_REPO_SNAPSHOT, BLUEPRINT_COVERAGE) all bound to commit `ddb03749`. However:

**A. Concurrent reports written without synchronization**
- BLUEPRINT_COVERAGE.md line 95–118: "Reports other than BLUEPRINT_COVERAGE.md appeared concurrently and were not created, read, or modified by this audit."
- implies reports were written by different agents without mutual awareness.
- If FE_INVENTORY and DA_INVENTORY were written hours apart, and DA made an API change in between, they may contradict.

**B. No "golden truth" coordinator**
- PACKAGE_BOUNDARY_DRAFT.md was meant to be "the only requested output," but multiple reports exist.
- If QE_LEGACY_MIGRATION identifies a reference implementation at path X, but LEAKAGE_BOUNDARY_AUDIT found path X has unacceptable sys.path mutations, which is true?

**Evidence:**
- BLUEPRINT_COVERAGE.md § 1, lines 95–97: "Reports appeared concurrently... were not created, read, or modified."
- All nine reports are in WAVE_0_OUTPUT/ but no manifest update or mutual cross-reference.

**Freeze blocker:** LOW→MEDIUM. Phase 1 will need to re-harmonize reports.

**Recommended gate:** Designate one agent (e.g., chief-integrator or boundary-auditor) to read all nine Wave 0 reports and publish a harmonized "WAVE_0_SYNTHESIS.md" that resolves contradictions and flags which reports are current.

---

#### 12. A-Share Unsupported Data Domains and Availability Not Audited

**Finding:** PACKAGE_BOUNDARY_DRAFT.md row 133 states:

> "Level2/order book, analyst consensus/revisions, full news sentiment and true northbound flow are not production capabilities unless DA catalog/schema is updated first."

But the audit does not verify which of these have been added to DA schema since the last freeze. If a recent commit added Level2 support to DA, then FO grammar should allow it. If not, FO must reject it.

**Concrete failure scenario:**
FO's mutation grammar generates a factor using `ANALYST_CONSENSUS('eps', horizon=1)` thinking it's legal. But A-share consensus data is not in DA catalog. FE rejects it at compile time. FO trial fails without clear error. User doesn't know if it's a grammar/domain error or a data-availability issue.

**Evidence:**
- BLUEPRINT_COVERAGE.md § 180: "No fabricated auction, Level2, consensus, news, or northbound data. New data domains require DataAccess semantic catalog/schema updates before FO grammar can use them."
- But blueprint does not verify current DA catalog contents.

**Freeze blocker:** LOW. FE already enforces domain legality; FO can detect rejection at validation time.

**Recommended gate:** During Phase 1, have ashare-semantics-auditor confirm which A-share domains are in DA catalog and update FO grammar allowlist accordingly.

---

## Concrete Failure Scenarios: If Implementation Proceeds Without Fixes

### Scenario 1: Parallel Implementation Deadlock on Envelope Schema

**Setup:** Three agents write Phase 1 simultaneously:
- qe-runtime-agent implements QE core metrics and `EvaluationBundle`.
- fa-registry-identity-agent implements FA SQLite repository and `FactorAsset`.
- research-ledger-agent implements Research Control campaign/trial records.

Each needs to include the "versioned contract envelope" (schema_version, producer, created_at, run_id, factor_ids, market, frequency, universe_ref, source_snapshot_ref, parent_refs, config_hash, code_hash).

**Outcome without fix:** 
- qe-runtime-agent adds envelope to EvaluationBundle.
- fa-registry-identity-agent adds envelope to FactorAsset record.
- research-ledger-agent adds envelope to TrialRecord.
- All three agents invent slightly different field sets (qe adds "metric_version", fa adds "lineage_ref", research-control forgets "market").
- Phase 2 integration discovers three incompatible envelope DTOs.
- Chief integrator must manually reconcile or overwrite one package's envelope.
- Risk: hidden field omissions (e.g., qe's "metric_version" was critical for correct metric identification, but fa's envelope has no equivalent).

**Fix:** Lead architect publishes a frozen `schemas/envelope.json` (or `schemas/pydantic_envelope.py`) before Phase 1. All three agents use identical envelope. Contract is unified.

---

### Scenario 2: FA/FP Aggregation Boundary Causes Missing Computation

**Setup:** FA selects 50 factors with specified weights. FP receives the FactorSet/FactorSetArtifact with weights. 

Question: Does FP compute weighted factors, or return raw factors for the caller to weight?

**Outcome without fix:**
- FA publishes: `FactorSet(factors=[f1, f2, ..., f50], weights=[0.02, 0.01, ..., 0.03])`.
- FP receives it and assumes weights are "for reference" only; returns 50 raw untransformed factors.
- Caller thinks they got a portfolio signal but got individual factors.
- FO/research tool gets 50 factors instead of 1 weighted composite.
- No error raised; silent semantic mismatch.

**Fix:** Contract explicitly states: "FA specifies selection and aggregation policy; FP computes weighted composite and returns `FeatureBundle` with one aggregate channel + individual factor channels. FP is responsible for aggregation computation."

---

### Scenario 3: Legacy Import Violation Caught Late in Extraction

**Setup:** qe-core-metrics-agent implements RankIC using a reference oracle. Unknowingly calls:
```python
from AutoFactorEvaluation-RECONSTRUCT.timeseries.scripts import daily_ic
```

Thinks it's a one-time reference but leaves import in production code.

**Outcome without fix:**
- Phase 3 implementation completes with QE passing all local tests.
- Phase 4 integration: call `import quant_evaluator` from a clean venv without monorepo PYTHONPATH.
- Import fails: "ModuleNotFoundError: No module named 'AutoFactorEvaluation-RECONSTRUCT'."
- Extraction auditor marks QE as **BLOCKED: legacy import leakage**.
- QE must be rewritten to remove the import.
- Parallel work (FO, FA, FP) also blocked if they depend on QE.

**Fix:** Implement import-linter CI check before Phase 2. Catch violation during development.

---

## Key Recommendations

### Immediate (before Phase 1 freeze)

1. **Re-run Wave 0 audits against current HEAD after stabilizing dirty changes.**
   - 18 commits and 11 modified files since binding.
   - Legacy miners, boundary, leakage must be re-verified.
   - Publish snapshot delta bound to new HEAD.

2. **Publish Lead Architect + Chief Integrator decisions on 23 unresolved items.**
   - Create `DESIGN_DECISIONS.md` documenting each decision with rationale.
   - Particularly: terminal type name, envelope schema, error taxonomy, Research Control placement, ModelInput owner, FA/FP boundary, QE module split.

3. **Freeze unresolved FA migration report and audit.**
   - FA is explicitly pending independent review; no other work can finalize downstream contracts (Research Control, FO lineage, FP aggregation).

4. **Create machine-checkable boundary validator.**
   - Implement import-linter CI check (forbid AutoFactorEvaluation-RECONSTRUCT, raw_data_layer runtime, etc.).
   - Implement acyclic DAG checker (parse all packages, verify no cycles).
   - Gate Phase 1 on passing boundary validation.

5. **Define frozen envelope schema in `/schemas/envelope.json` or equivalent.**
   - All packages must use identical versioned metadata envelope.
   - Prevent divergence during parallel implementation.

### Before Phase 2 (integration)

6. **Publish "NOVELTY_DEFINITIONS.md" with decision rules.**
   - Define exact identity, structural similarity, behavioral neighbor, conditional novelty.
   - FO mutation generation depends on these distinctions.

7. **Freeze RankIC contract: tie-handling, NaN aggregation, minimum observations.**
   - Golden-test oracle (reference implementation or statsmodels binding).
   - Required for QE acceptance.

8. **Freeze label timing and tradability validation contract.**
   - Explicit definition of horizon (calendar/trading days), execution delays (T+0/T+1/T+2/T+3).
   - A-share tradability context: who populates (caller or QE adapter)?

9. **Define FittedState serialization strategy.**
   - Pydantic model with Arrow IPC arrays, JSON metadata, external storage refs.
   - Round-trip (fit, save, load, transform) golden tests.

### Before extraction (Phase 5)

10. **Audit and remove 58 sys.path mutations from production code.**
    - Move any still needed to test/corpus quarantine.
    - Verify isolated venv install works.

11. **Harmonize Wave 0 reports.**
    - Designate coordinator to read all nine reports and publish "WAVE_0_SYNTHESIS.md."
    - Resolve contradictions; flag which reports are current/authoritative.

---

## Honest Assessment: What Wave 0 Got Right

1. **Boundary is sound.** Reuse of DA/FE is well-justified; four packages are well-scoped.
2. **Dependency DAG is mostly acyclic** (unresolved cycles are identifiable and fixable).
3. **Migration matrix is thorough.** Four legacy miners systematically cataloged REUSE/REWRITE/REFERENCE/CORPUS/DISCARD decisions.
4. **Governance roles are clear.** Lead architect, chief integrator, package builders, miners, auditors are well-defined.
5. **Schemas and contracts exist in draft.** Not frozen, but the skeleton and envelope concepts are sound.
6. **Reference implementations identified.** RankIC, cross-sectional transforms, OLS, quantile metrics have clear reference seeds.

## Honest Assessment: Why Freeze Cannot Happen Now

1. **Schemas are incomplete.** 23 unresolved decisions + 3 unresolved cycles + 4 owned-by-nobody decisions = no machine-checkable spec.
2. **Evidence is stale.** Snapshot bound to commit ddb03749, repo is 18 commits ahead + dirty.
3. **Dependent on missing FA report.** Cannot finalize downstream contracts without FA architecture.
4. **No automated compliance checking.** Manual boundary enforcement will fail during parallel implementation.
5. **Leakage remains in production code.** 58 sys.path mutations unremoved; extraction will fail.

**Conclusion:** Wave 0 is **a sound foundation that requires maturation, not a ready-to-implement blueprint.** Freeze is aspirational; Phase 1 cannot begin until P0 and P1 blockers are closed.

---

## Sign-Off

This independent red-team audit documents findings based on read-only analysis of CONTRACT_FREEZE_DRAFT.md, PACKAGE_SKELETON.md, PACKAGE_BOUNDARY_DRAFT.md, all nine Wave 0 reports, git status, and recent commit history. No production files were modified. The assessment is honest about both strengths and gaps.

**Recommendation:** Use this audit to prioritize Phase 0 closure. Resolve P0 blockers (FA report, snapshot delta, unresolved decisions) before Phase 1 kick-off.
