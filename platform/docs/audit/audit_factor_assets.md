# Factor Assets (factor_assets/) Phase-0 Audit

**Audit date:** 2026-08-26
**Git HEAD:** d58eae0821da2c71182e200751af97a577e1a616 (`Add 量化研究与生产报告中心`)
**Mode:** READ-ONLY (no source modified, no install)
**Baseline confirmed:** `../.venv/bin/python -m pytest tests -q` = **827 passed, 43 skipped, 38 warnings** (re-run this session)
**Code size:** ~31,378 Python lines across factor_assets/ (incl. tests, excl. build/)

Scope: identity, registry, lifecycle, novelty, selection, similarity, graph, seen_index,
clustering, assembly, aggregation, contracts. Mapped to master spec §8/9/10/13/14/15/16/17/18/36.

---

## 1. Identity (`identity/`)

| Item | Exists? | Location |
|---|---|---|
| `FactorIdentity` | YES | `identity/canonical.py:61` |
| `FactorIdentityProvider` (FE protocol boundary) | YES | `identity/canonical.py:12` |
| `FactorDefinitionIdentity` | YES | `identity/canonical.py:103` |
| `FactorCompilerIdentity` (separate axis) | YES | `identity/canonical.py:130` |
| `FactorValueIdentity` | YES | `identity/canonical.py:151` |
| `EvaluationIdentity` | **NO** | spec §8.3; not present anywhere in repo |
| `ResultIdentity` (output/result-based) | YES (in novelty) | `novelty/conditional.py:42` |
| Sign-invariant identity | YES | `identity/adapters.py:19` |
| Structural identity | YES | `identity/adapters.py:80` |

- Canonicalization delegates to FE via `FactorIdentityProvider` protocol (does not re-implement FE parser/hash — good, honors §7/§37 boundary). `identity/canonical.py:1-6`.
- `FactorDefinitionIdentity` enforces `factor_version` mandatory and refuses to fall back to compiler generation (`canonical.py:119-126`) — matches §8.1/8.6 intent.
- `FactorValueIdentity` requires at least one of snapshot/universe/run ref (`canonical.py:163-170`) — partial match to §8.2.
- **GAP:** `EvaluationIdentity` (FactorValueIdentity + EvaluationPolicyIdentity + LabelDefinitionIdentity + EvaluationProfileIdentity) missing. The `ResultIdentity` in novelty is a value-hash identity used for novelty dedup, not the spec's EvaluationIdentity.

## 2. Registry / Lifecycle

Registry (`registry/`): `AssetRepository` (`repository.py:89`), `LifecycleRepository` protocol (`repository.py:46`), `LifecycleOrchestrator` (`lifecycle.py:56`), `SQLiteLifecycleRepository` (`sqlite_repository.py`), `SnapshotManager` (`snapshots.py:67`), `LineageGraph` (`lineage.py:63`), `create_repository` factory (`factory.py`).

**Lifecycle state machine** (`contracts/lifecycle.py:27-44`, `lifecycle/state_machine.py`):

FA states: `REGISTERED, EVALUATED, APPROVED, PRODUCTION_READY, DEPRECATED, RETIRED` (6 states).

Spec §9 mandates a richer machine with states:
`DISCOVERED, VALIDATING, VALIDATED, COMPILED, MATERIALIZING, MATERIALIZED, RAW_EVALUATING, RAW_EVALUATED, TREATMENT_SEARCHING, TREATMENT_SELECTED, TREATED_EVALUATING, TREATED_EVALUATED, NOVELTY_CHECKING, ADMISSION_REVIEW, CLUSTER_PENDING, CLUSTERED, LIBRARY_CANDIDATE, SHADOW, PRODUCTION_ELIGIBLE, PRODUCTION, DEGRADED, RETIRED, QUARANTINED`.

- **MISSING ~15 spec states**: DISCOVERED, VALIDATING, VALIDATED, COMPILED, MATERIALIZING, MATERIALIZED, RAW_EVALUATING, RAW_EVALUATED, TREATMENT_SEARCHING, TREATMENT_SELECTED, TREATED_EVALUATING, TREATED_EVALUATED, NOVELTY_CHECKING, ADMISSION_REVIEW, CLUSTER_PENDING, CLUSTERED, LIBRARY_CANDIDATE, SHADOW, PRODUCTION_ELIGIBLE, PRODUCTION, DEGRADED, QUARANTINED. None of these tokens appear in FA source (verified by grep).
- **HealthState mismatch**: FA has `ACTIVE/DEPRECATED/RETIRED` (`contracts/lifecycle.py:60-70`); spec §9 wants `UNKNOWN/HEALTHY/WATCH/DEGRADED/CRITICAL/STALE`. FA additionally has a `ValidationStatus` (`UNVALIDATED/PARTIAL/VALIDATED/STALE`, lines 46-57) — orthogonal dimension not in spec.
- **JobStatus dimension missing entirely** (spec wants `PENDING/RUNNING/SUCCEEDED/FAILED_RETRYABLE/...` separate from lifecycle).

## 3. Novelty / Admission

- `FactorAdmissionArtifact` + `AdmissionDecision` (APPROVED/REJECTED/SHADOWED): `contracts/admission.py` (self-describing `content_hash`; fail-closed on mismatch).
- Exact result-identity novelty assessor: `novelty/conditional.py` — `ResultIdentity`, `ResultIdentityCache` (reverse cache, line 111), `ResultIdentityNoveltyAssessor` (line 202).
- Residual/conditional novelty (QE-wired, SHADOW/SHADOWED_COEXIST chain): `adapters/residual_novelty.py:55` `ResidualICNoveltyProducer` (incremental IC residualization). Optional adapter (QE optional dep, fail-closed `OptionalDependencyMissing`).
- `SelectionPolicy.make_decision` (`selection/policy.py`) supports similarity-threshold + novelty floor: high similarity is NOT auto-reject — SHADOW vs SHADOWED_COEXIST vs REJECTED_SIMILARITY. Matches spec §13 Stage 9 (no `corr>threshold => reject`).

## 4. Similarity

- `SimilarityArtifact` + `SimilarityView`: `contracts/similarity.py`. Canonical view keys (`SIMILARITY_VIEW_KEYS`, line ~25): `rank_corr, pnl_corr, top_overlap, bottom_overlap, residual_similarity, horizon_similarity`. `similarity_spec_hash` derived/fail-closed; deep-frozen views.
- Exact correlation: `similarity/exact.py` — `SimilarityMethod` PEARSON/SPEARMAN/KENDALL, `QEPairwiseSimilarity`, `CorrelationSimilarity`.
- ANN: `similarity/ann.py` — Faiss (`FaissANNIndex`, line 167) and Annoy (`AnnoyANNIndex`, line 318) backends; `create_ann_index` factory; `ANN_AVAILABLE` guard in `similarity/__init__.py`.
- **GAP vs spec §10 multi-view**: artifact views cover rank/PnL/top+bottom overlap/residual/horizon but are **missing `pearson` view, `quantile_overlap`, and `regime_conditional` similarity**. No regime-conditional similarity anywhere (grep found only `regime_sensitivity` in `optimizer/diagnosis_mapper.py:21`).
- **GAP orchestration**: spec §10 incremental pipeline `fingerprint → ANN shortlist → exact multi-view → sparse graph update` is NOT assembled into one flow. ANN index and exact similarity are separate modules with no combined incremental pipeline; `SimilarityFingerprintArtifact` missing.

## 5. Clustering (`clustering/`)

- `ConnectedComponents` (`families.py:182`), `ModularityClustering` (`:252`), `LeidenClustering` (`:418`, PRODUCTION_CAPABLE, backend igraph/leidenalg, **fails closed** if neither installed), `HierarchicalClustering` (`:824`, `Dendrogram` :693), `LineageDetector` (`lineage.py:100`).
- `ClusterArtifact` (`families.py:75`): hash-addressed, records `graph_identity, algorithm, backend, backend_version, seed, resolution, assignments, representatives`, `content_hash` derived/fail-closed. Good provenance for a raw clustering result.
- `SparseCorrelationGraph` (`graph/sparse.py:42`) with `graph_identity` content hash; rejects duplicate edges with conflicting corr; keeps isolated nodes — matches §13 Stage 12 prohibitions (no missing-edge-as-zero, no isolated-node drop).

**CRITICAL QRP-P6 GAPS** (spec §14 + §36 + Stage 11):
- **No algorithm-cluster-label vs logical-cluster-id vs cluster-version-id separation.** `ClusterArtifact` stores a raw `algorithm` label and integer `assignments`, but there is no `logical_cluster_id`, no `CL_*` scheme, no `cluster_version_id`. Grep for `logical_cluster|ClusterVersion|algorithm_cluster` in clustering = empty.
- **No `ClusterVersionArtifact`, no `ClusterLineageArtifact`** (spec §36 contract list).
- **No incremental cluster assignment** states `ASSIGNED/AMBIGUOUS/BRIDGE/SINGLETON/OUTLIER/PENDING_GLOBAL_REFRESH` (grep empty).
- **No cross-version cluster matching** transitions `UNCHANGED/MIGRATED/SPLIT/MERGED/NEW/DISSOLVED`.
- `ClusterArtifact`'s `representatives` field exists but there is no version-diff capability.

## 6. Assembly & Aggregation

`FactorSetAssembler` (`assembly/engine.py:15`). Selection policies: `manual`, `pareto_front`, `family_robust`, `diverse` (MMR) (`_KNOWN_POLICIES`, line 34).

- **`manual`**: legacy deterministic factor_id order.
- **`pareto_front`**: Pareto dominance over decision `metadata["objectives"]` (`_pareto_rank`, :734).
- **`family_robust`**: reads composite `metadata["family_robust"]` (a float combining quality, stability, worst_slice, recent_decay, tradability, residual_novelty, family_budget — `_family_robust_rank`, :786). Falls safe to recency when absent (never invents a score).
- **`diverse`** (MMR): `lambda*quality - (1-lambda)*max_similarity`, `quality` from `metadata["quality"]` or recency rank (`_diverse_mmr_rank`, :819); fails closed without similarity_provider and on UNKNOWN similarity.
- `_apply_family_constraints` (`:594`): `max_per_family=N` grammar only; fails closed on other strings.
- Production mode requires `FactorAdmissionArtifact` (health_state_ref/cluster_id/orientation/factor_version) + `TreatmentSelectionArtifact` (`treatment_selection_ref`), consume-only, fails closed on missing (`_build_memberships`, :404).

**Spec §36 verdict:** FA no longer does "family round-robin + recency" only. It *can* rank by quality/stability/robustness/novelty/tradability/cluster-diversity/health — but the composite scores are read from caller-supplied decision `metadata`, **NOT computed by FA** (`selection/policy.py` `make_decision` never writes `family_robust`/`quality`/`objectives`; those keys are injected in tests via `object.__setattr__`). So the scoring surface exists but the producer of those scores is outside FA. **Capacity is not represented anywhere** in assembly.

`aggregation/` (`representatives.py`, `specs.py`): family representative selection (MAX_IC/MIN_CORRELATION/EQUAL_WEIGHT/FIRST/RANDOM) and weighting specs (EQUAL/IC_WEIGHTED/INVERSE_VARIANCE/RANK_IC/...). These are separate utility specs, not wired into the assembly path.

## 7. Contracts (`contracts/`)

| Spec §36 contract | Present? | Location |
|---|---|---|
| FactorSetArtifact | YES (canonical) | `contracts/factor_set.py:104` |
| FactorSet (legacy compat) | YES | `contracts/factor_set.py:197` |
| TreatmentSelectionArtifact | YES | `contracts/treatment_selection.py:119` |
| FactorAdmissionArtifact | YES | `contracts/admission.py` |
| SimilarityArtifact | YES | `contracts/similarity.py` |
| SimilarityFingerprintArtifact | **NO** | — |
| SimilarityGraphArtifact | **NO** | — |
| IncrementalClusterAssignment | **NO** | — |
| ClusterVersionArtifact | **NO** | — |
| ClusterLineageArtifact | **NO** | — |
| FactorLibraryDefinition | **NO** | — |
| FactorLibraryVersionArtifact | **NO** | — |
| FactorLibraryMembership | **NO** | — |
| FeatureSetCandidate | **NO** | — |
| PromotionDecision | **NO** | — |

Other contracts: `asset.py`, `evidence_ref.py`, `envelope.py`, `lineage.py`, `treatment_policy.py`, `lifecycle.py`. All frozen dataclasses; artifacts enforce derived-only content_hash (fail-closed) — consistent with §7.3/7.4.

## 8. Mapping to spec §36 / §14 / §15 / §16 / §17 / §18

| Spec | Status | Notes |
|---|---|---|
| §8 Identity/Version (all 9-10 identities) | PARTIAL | 4 present, `EvaluationIdentity` missing; no Treatment/SimilarityGraph/ClusterVersion/LibraryVersion/FeatureSet identity types |
| §9 Lifecycle state machine | PARTIAL/MISSING | 6 of ~21 states; HealthState values wrong; no JobStatus |
| §13 Stage 9 Novelty/Admission | EXISTS | SHADOW/SHADOWED_COEXIST chain, residual-IC novelty |
| §13 Stage 10 Incremental Similarity | PARTIAL | ANN + exact + sparse graph modules exist; **no orchestrated incremental pipeline, no fingerprint artifact, no Pearson/quantile/regime views** |
| §13 Stage 11 Incremental Cluster Assignment | MISSING | no ASSIGNED/AMBIGUOUS/BRIDGE/SINGLETON/OUTLIER/PENDING_GLOBAL_REFRESH |
| §13 Stage 12 Global Cluster Refresh | PARTIAL | Leiden + sparse graph + isolated-node handling exist; **no global-vs-incremental refresh orchestration, no cluster versioning** |
| §13 Stage 13 Library vs Cluster | MISSING | cluster exists; library governance layer absent |
| §13 Stage 14 FeatureSet Promotion | MISSING | no FeatureSet/FeatureSetArtifact, no promotion event |
| §14 Cluster Version / logical ID | MISSING | no logical_cluster_id, no ClusterVersionArtifact, no cross-version matching |
| §15 Factor Library | MISSING | no LibraryDefinition/Version/Membership |
| §16 Library incremental + Promotion | MISSING | no candidate/shadow/approved/production promotion flow |
| §17 FeatureSet | MISSING | no FeatureSetArtifact / ordered_feature_manifest |
| §18 Model retrain trigger | MISSING | no FeatureSet-change trigger |
| §36 FA contract list | PARTIAL | 4 of 12 new contracts exist; 8 missing |
| §36 Assembly quality dims | PARTIAL | scoring surface exists but composite scores not computed in FA; capacity absent |

## 9. GAP rows (Requirement → Current → Gap → Action → Reuse/Modify/Add → Priority)

| # | Requirement | Current (files) | Gap | Action | Reuse/Modify/Add | Pri |
|---|---|---|---|---|---|---|
| G1 | EvaluationIdentity | only FactorValueIdentity (canonical.py:151) + ResultIdentity (novelty/conditional.py:42) | no EvalPolicy/Label/Profile axes | add EvaluationIdentity contract | Add | HIGH |
| G2 | Lifecycle states (§9) | 6 states (contracts/lifecycle.py:27) | ~15 states missing | extend enum + transition table | Modify | HIGH |
| G3 | HealthState values (§9) | ACTIVE/DEPRECATED/RETIRED (lifecycle.py:60) | values differ | align to UNKNOWN/HEALTHY/WATCH/DEGRADED/CRITICAL/STALE | Modify | MED |
| G4 | JobStatus dimension | none | missing | add JobStatus enum (separate from lifecycle) | Add | MED |
| G5 | Logical cluster id / ClusterVersion | ClusterArtifact (families.py:75) stores raw label only | no logical id, no version | add algorithm_cluster_label vs logical_cluster_id vs cluster_version_id; ClusterVersionArtifact | Modify/Add | **P6 CRITICAL** |
| G6 | Cluster cross-version matching | none | no UNCHANGED/SPLIT/MERGED/... | add version-diff matching | Add | **P6 CRITICAL** |
| G7 | Incremental cluster assignment | none | no ASSIGNED/AMBIGUOUS/... states | add IncrementalClusterAssignment | Add | **P6 CRITICAL** |
| G8 | Multi-view similarity (Pearson/quantile/regime) | SimilarityArtifact views (similarity.py) lack pearson/quantile_overlap/regime_conditional | 3 views missing | extend SIMILARITY_VIEW_KEYS | Modify | HIGH |
| G9 | Incremental similarity pipeline | ANN (ann.py) + exact (exact.py) + graph (sparse.py) separate | no fingerprint→ANN→exact→graph flow | orchestrate; add SimilarityFingerprintArtifact | Add | MED |
| G10 | Factor Library governance (§15/16) | none | no Library/Definition/Version/Membership | add contracts + promotion flow | Add | HIGH |
| G11 | FeatureSet / FeatureSetArtifact (§17) | none | missing ordered feature manifest | add FeatureSetArtifact + FeatureSetCandidate | Add | HIGH |
| G12 | PromotionDecision | none | missing | add PromotionDecision contract | Add | HIGH |
| G13 | Capacity in assembly | assembly reads quality/stability/... but no capacity (engine.py:786) | capacity absent | thread capacity into assembly/metadata | Add | MED |
| G14 | Composite assembly scores computed in FA | family_robust/quality/objectives read from external metadata (engine.py:693-807) | scores not produced in FA | decide ownership (FA compute vs consume) — currently consume-only | Modify | MED |

## 10. Second-authority duplicates / FactorSet canonical vs compat

- **FactorSetArtifact vs FactorSet**: cleanly resolved. `FactorSetArtifact` is documented as "the CANONICAL authority" (`contracts/factor_set.py:104-106`); legacy `FactorSet` (`:197`) is explicitly "DEPRECATED — read-only compatibility view ... produced via `FactorSetArtifact.to_legacy_view()`. It is NOT a second authority and should not be constructed directly" (`:200-203`, `:260-263`). Assembler produces the artifact only; legacy view is a projection. **No dual-truth.**
- **LifecycleConflictError**: exported twice with aliases `LifecycleConflictError` and `LifecycleConflictErrorNew` in `__init__.py` — cosmetic alias, single source in `errors.py`. Low risk.
- **No library-layer second authority** because the library/featureset layer does not exist yet (G10/G11).
- **No other FactorSet duplicate** found: `FactorSetSpec` is the input spec, distinct from the output artifact.

---

## Summary

factor_assets is a well-structured, immutability-first package (derived content-hashes, fail-closed contracts, ~827 green tests). Identity (4/4 core), admission/novelty (Stage 9), Leiden clustering over a sparse graph, and a canonical FactorSet artifact with a compat-only legacy view are solid. The assembly engine already moved past "family round-robin + recency" and supports pareto/family_robust/MMR with health/cluster/orientation provenance — though the composite quality scores are consumed from external metadata, not computed in FA.

**Main gaps** are the pipeline-forward layer spec §14-18/§36: EvaluationIdentity, the full §9 lifecycle/Health/JobStatus dimensions, and — most critical for QRP-P6 — the entire cluster-version/logical-id/incremental-assignment layer, plus the Factor Library / FeatureSet / Promotion governance that §36 lists but FA has not yet implemented (8 of 12 new contracts missing). Similarity views also lack Pearson/quantile/regime-conditional.

[audit completed; report written to platform/docs/audit/audit_factor_assets.md; HEAD d58eae08]
