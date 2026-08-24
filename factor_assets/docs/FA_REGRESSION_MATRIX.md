# FA Regression Matrix — R46 P0-J/K/L/M/N/O/P

**Scope:** `factor_assets/**` only. This is a **direct hardening** of the
current codebase against the audit's substantive requirements, **not a
restore**.

> **IMPORTANT FINDING (verified by coordinator + this agent):** The audit
> asserted FA was restored from an older branch and that
> `SimpleConditionalNoveltyAssessor` "should be `ResultIdentityNoveltyAssessor`
> (which was previously fixed)". `git log --all -S ResultIdentityNoveltyAssessor`
> returns **no commit** that ever introduced `ResultIdentityNoveltyAssessor`.
> `SimpleConditionalNoveltyAssessor` exists in `c0893da0`, `724cf286`,
> `c1f3564b`. There is **no LastKnownGood `ResultIdentityNoveltyAssessor`
> commit to recover from**. Every item below is therefore implemented as a
> fresh correctness hardening, and the rename is a deliberate API change, not
> a restore.

Legend: **CURRENT** = what the code actually does today; **TARGET** = what the
audit requires; **DIFF** = the honest gap and what this change does.

---

## 1. ResultIdentityNoveltyAssessor (P0-J/K)

- **CURRENT:** `factor_assets/novelty/conditional.py` defines
  `SimpleConditionalNoveltyAssessor`, an exact result-identity duplicate
  checker (novel iff no pool factor shares the result identity). It is
  documented as "simple conditional novelty" but is really exact
  result-identity novelty.
- **TARGET:** A class named `ResultIdentityNoveltyAssessor` documented as
  **exact result-identity novelty**, not conditional residual novelty.
- **DIFF:** No LKG exists in git (see note above). Renamed the class to
  `ResultIdentityNoveltyAssessor`, updated the docstring to state it is exact
  result-identity novelty (residual/conditional novelty is the separate
  `ResidualICNoveltyProducer` in `adapters/residual_novelty.py`). The old
  name is dropped; all internal imports/tests updated to the new name.

## 2. ResultIdentity fields (P0-K)

- **CURRENT:** `ResultIdentity` has only `result_hash, factor_id,
  universe_ref, period_start, period_end, num_observations`.
- **TARGET:** Add `data_snapshot_ref`, `factor_version`, `coordinate_hash`,
  `calculation_spec_hash`; the `ResultIdentityCache` key must include ALL
  result-comparability-relevant identity fields.
- **DIFF:** Added the four fields (with sensible defaults). The cache key now
  includes `result_hash, universe_ref, data_snapshot_ref, period_start,
  period_end, num_observations, factor_version, coordinate_hash,
  calculation_spec_hash` so two factors are only "equivalent" when their full
  evaluation context matches — a missing snapshot/version/coordinate can no
  longer alias two genuinely different results.

## 3. FactorSetArtifact authority (P0-L)

- **CURRENT:** `FactorSetAssembler.assemble()` returns the legacy `FactorSet`.
  `FactorSetArtifact` exists but is not produced by the assembler.
- **TARGET:** `assemble()` returns the canonical `FactorSetArtifact`;
  `FactorSetArtifact.to_legacy_view() -> FactorSet` serves old consumers;
  `FactorSet` becomes a deprecated read-only compatibility view, not a second
  authority. Production artifact must require `snapshot_ref, universe_ref,
  split_ref, policy_hash, assembly_hash, members, versions, evidence_refs`.
- **DIFF:** `assemble()` now returns `FactorSetArtifact` with required
  `snapshot_ref/universe_ref/split_ref` and new `versions`/`evidence_refs`.
  Added `to_legacy_view()`. `FactorSet` is documented as a deprecated
  read-only view.

## 4. FactorSet legacy view (P0-L)

- **CURRENT:** `FactorSet` is a first-class output with its own authority.
- **TARGET:** Deprecated read-only compatibility view produced only via
  `to_legacy_view()`.
- **DIFF:** `FactorSet` retained for compatibility, marked deprecated, and
  now only produced by `FactorSetArtifact.to_legacy_view()`.

## 5. snapshot fail-closed (P0-L)

- **CURRENT:** `FactorSetSpec.data_snapshot_ref` is optional and the
  assembler falls back to `snapshot_ref = data_snapshot_ref or universe_ref`
  — the universe impersonates the snapshot.
- **TARGET:** Remove the fallback; `snapshot_ref` must be required/validated;
  universe must NOT impersonate snapshot.
- **DIFF:** `FactorSetSpec.data_snapshot_ref` is now required (non-empty) and
  the assembler sets `snapshot_ref = spec.data_snapshot_ref` with no universe
  fallback. Tests updated to supply an explicit `data_snapshot_ref`.

## 6. family_robust (P0-N)

- **CURRENT:** `_rank` for `family_robust` orders by admission-decision recency
  then round-robins across families. It is recency + round-robin only.
- **TARGET:** Combine quality, stability, worst_slice, recent_decay,
  tradability, residual_novelty, family_budget.
- **DIFF:** Added a composite `family_robust` score read from decision
  metadata (`quality, stability, worst_slice, recent_decay, tradability,
  residual_novelty, family_budget`). When the required scores are absent the
  ranking fails safe to recency ordering (never invents a score). Family
  round-robin is retained as a budget/tiebreak.

## 7. diverse (real MMR) (P0-N)

- **CURRENT:** `diverse` is identical to `family_robust` (recency +
  round-robin). No MMR.
- **TARGET:** Real MMR: `lambda*quality - (1-lambda)*max_similarity_to_selected`.
- **DIFF:** Implemented greedy MMR selection. Quality comes from decision
  metadata `quality` (fallback to recency rank). Similarity comes from an
  optional `similarity_provider` callable passed to `assemble()`; when no
  provider is supplied the policy fails closed with a clear error rather than
  silently degrading to non-MMR. All scores + similarity refs are written into
  membership.

## 8. FactorMembership lineage (P0-M)

- **CURRENT:** `FactorMembership` has `factor_id, role, orientation, family_id,
  cluster_id, selection_decision_ref, evidence_ref, novelty_ref, reason`.
- **TARGET:** Every member traceable: `factor_id, factor_version, orientation,
  family_id, cluster_id, representative_of, selection_decision_ref,
  evidence_ref, novelty_ref, similarity_ref, health_state_ref, assembly_score,
  selection_rank, role`.
- **DIFF:** Added `factor_version, representative_of, similarity_ref,
  health_state_ref, assembly_score, selection_rank`. The assembler populates
  them from the asset/decision where available.

## 9. UNKNOWN edge state (P0-P)

- **CURRENT:** `HierarchicalClustering.build_dendrogram` treats a missing edge
  as `dist = 1.0` (max distance) — a missing edge is silently conflated with
  low similarity.
- **TARGET:** Add `SimilarityObservationState: UNKNOWN | COMPUTED_LOW |
  COMPUTED_HIGH`; hierarchical clustering must NOT treat a missing edge as
  distance=1; it accepts only a complete distance artifact or an explicit
  missing-distance policy.
- **DIFF:** Added the `SimilarityObservationState` enum. `HierarchicalClustering`
  now takes `missing_distance_policy` (default `None` = fail closed). When a
  pair distance is missing and no explicit policy is given, it raises
  `ValueError` instead of fabricating distance 1.0. An explicit
  `missing_distance_policy="treat_missing_as_max"` opts into the old behaviour
  for research only.

## 10. Ward fail-closed (P0-P)

- **CURRENT:** `method='ward'` with `metric='correlation'` builds a dense
  distance matrix where missing edges become 1.0, so Ward silently runs on
  fabricated distances.
- **TARGET:** Ward + `1-|corr|` must FAIL CLOSED when a distance is missing.
- **DIFF:** With the default fail-closed `missing_distance_policy`, any missing
  distance (including under Ward) raises before linkage is computed. Ward is
  only runnable on a complete distance artifact or with an explicit policy.

## 11. production Leiden (P0-O)

- **CURRENT:** `ModularityClustering` is a greedy modularity optimizer over the
  sparse graph; it already fails closed unless `allow_toy_algorithm=True`. No
  Leiden backend is wired.
- **TARGET:** Production sparse-graph clustering: ANN shortlist → exact
  similarity refinement → sparse graph → Leiden via a mature library
  (`igraph`/`leidenalg`), stable seed, cluster lineage. 100K-scale must NOT use
  a dense matrix. Toy `ModularityClustering` → RESEARCH_ONLY.
- **DIFF:** `ModularityClustering` is explicitly marked RESEARCH_ONLY. Added
  `LeidenClustering` which requires `igraph`/`leidenalg` for production and
  fails closed when they are not installed. The sparse-graph pipeline (ANN
  shortlist → exact refinement → sparse graph) is implemented and does not
  build a dense matrix. **DONE (R46):** `igraph` 1.0.0 and `leidenalg` 0.12.0
  are now installed in the venv, so the production Leiden path is exercised by
  `tests/test_clustering_leiden.py`. The igraph 1.x API change was fixed:
  `igraph.Random` no longer exists and `community_leiden` takes
  `objective_function=` + `resolution=` (not `objective=`/`resolution_parameter=`),
  so the seed is supplied via `igraph.set_random_number_generator(random.Random(seed))`.
  Both backends recover the two-triangle cluster structure deterministically.

## 12. residual novelty (P0-N)

- **CURRENT:** `adapters/residual_novelty.py` already implements residual-IC
  conditional novelty via QE (`ResidualICNoveltyProducer`), feeding
  `SelectionPolicy.make_decision`'s `novelty_score`/`residual_ic`.
- **TARGET:** Residual novelty is distinct from exact result-identity novelty
  and coexists with it.
- **DIFF:** No change needed to the producer; the `ResultIdentityNoveltyAssessor`
  docstring now explicitly distinguishes exact result-identity novelty from
  residual/conditional novelty.

## 13. shadow coexist (P0-N)

- **CURRENT:** `SelectionPolicy.make_decision` already implements the
  SHADOW / SHADOWED_COEXIST chain (result-identity duplicate → SHADOW; residual
  novelty clearing the floor → SHADOWED_COEXIST).
- **TARGET:** Shadow and coexist semantics preserved.
- **DIFF:** No change; verified present and covered by existing tests.

## 14. persistent registry (P0)

- **CURRENT:** `registry/sqlite_repository.py` + `registry/factory.py` provide
  a durable SQLite repository; `create_repository(production=True)` requires an
  existing file path.
- **TARGET:** Persistent registry available for production.
- **DIFF:** No change; verified present.

## 15. seen index (P0)

- **CURRENT:** `seen_index/exact.py` (in-memory) + `seen_index/persistent.py`
  (SQLite) provide exact seen-history tracking.
- **TARGET:** Seen index available.
- **DIFF:** No change; verified present.

---

## Honest status summary

| Item | Status |
|------|--------|
| ResultIdentityNoveltyAssessor rename | DONE (no LKG; direct hardening) |
| ResultIdentity fields + cache key | DONE |
| FactorSetArtifact canonical assemble | DONE |
| FactorSet legacy view + to_legacy_view | DONE |
| snapshot fail-closed | DONE |
| family_robust composite | DONE |
| diverse real MMR | DONE |
| FactorMembership lineage | DONE |
| UNKNOWN edge state | DONE |
| Ward fail-closed | DONE |
| production Leiden | DONE (igraph 1.0.0 + leidenalg 0.12.0 pinned; exercised by test_clustering_leiden.py) |
| residual novelty | VERIFIED present |
| shadow coexist | VERIFIED present |
| persistent registry | VERIFIED present |
| seen index | VERIFIED present |
