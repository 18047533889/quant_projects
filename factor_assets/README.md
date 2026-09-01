# factor_assets

Factor asset registry: identity, admission, de-duplication, similarity, clustering,
and library governance for quantitative factors. The single source of truth for
"what factors exist, how they relate, and what state they are in".

**Version:** 0.1.0 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/factor_assets (private)

## What it owns

- **Identity** — canonical factor identity from the factor expression
  (`FactorIdentityProvider`), deterministic `create_factor_id(canonical_hash)`,
  `FactorValueIdentity` / `FactorDefinitionIdentity`.
- **Registry** — `AssetRepository` (append-only, immutable identity), optional
  `SQLiteLifecycleRepository`; snapshots & migrations.
- **Seen index (de-dup)** — `SeenIndex` exact + `PersistentSeenIndex` (sqlite)
  for "have we already admitted this factor".
- **Similarity** — `QEPairwiseSimilarity` (via quant_evaluator), ANN backends
  (faiss / annoy), `SimilarityArtifact`.
- **Clustering** — `LeidenClustering` (igraph/leidenalg, production-certified
  graph only) + hierarchical/modularity fallbacks; `incremental_assign` for
  incremental cluster versions; `certification` (graph content hash).
- **Lifecycle** — `LifecycleState` REGISTERED → EVALUATED → APPROVED →
  PRODUCTION_READY → DEPRECATED / RETIRED, with state machine + transition guards.
- **Library governance** — `PromotionGate` / `RollbackGate` (evidence-gated
  promotion decisions).
- **Assembly & aggregation** — `FactorSetAssembler`, composite aggregation
  (horizon/orientation/regime), family representative selection.
- **Campaigns** — research campaign governance / budgets / coordinator.

## Install & first steps

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_assets.git
cd factor_assets
pip install -e .                     # core only
pip install -e ".[adapters]"         # + quant_evaluator / factor_engine / data_access
```

```python
from factor_assets.registry.factory import create_repository
from factor_assets.identity.canonical import FactorIdentityProvider, create_factor_id

repo = create_repository()
identity = FactorIdentityProvider().get_full_identity(expr_node)  # from factor_engine
factor_id = create_factor_id(identity.canonical_hash)
repo.register(factor_id, metadata={...})
```

Adapters (optional, lazy import, protocol-based): `QEEvidenceProvider`
(bundle → EvidenceRef), `FEIdentityProvider` (Expr → identity),
`DAFactorValueReader` / `DACatalogReader` (protocols). See `adapters/README.md`.

## Layout

```
contracts/     FactorAsset, FactorAdmissionArtifact, FactorSetArtifact,
               SimilarityArtifact, TreatmentSelectionArtifact, lifecycle, lineage
registry/      repository, lifecycle, sqlite_repository, factory, snapshots
identity/      canonical (hash), adapters
seen_index/    exact + persistent (sqlite) de-dup
similarity/    exact (QE), ANN (faiss/annoy)
clustering/    families (Leiden/modularity/hierarchical), incremental, certification, lineage
library/       promotion / rollback gates
lifecycle/     state machine
assembly/      FactorSetAssembler
aggregation/   composite evaluation, representatives
novelty/       conditional novelty / residual-IC novelty (adapters/residual_novelty.py)
selection/     selection policy + gates
campaigns/     campaign coordinator / budget / governance
adapters/      factor_engine / quant_evaluator / data_access (optional)
graph/         sparse correlation graph, edge filters
optimizer/     pareto / plateau / multifidelity / typed_mutation helpers
docs/          ARCHITECTURE, API_REFERENCE, QUICKSTART, TESTING, CHANGELOG
tests/         63 test files
```

## Hard rules

- Append-only: no silent mutation of admitted factors; lineage preserved.
- Core never imports adapters; adapters never import each other.
- Evidence is referenced (not duplicated) — `EvidenceRef` points to QE bundles.
- Production clustering requires a certified graph (graph content hash enforced).

## Related repos

- **factor_engine** — canonical identity source (expression system)
- **quant_evaluator** — evidence source (`QEEvidenceProvider`)
- **data_access** — factor value / catalog reads (protocols)
- **factor_optimizer** — upstream search that produces treatment candidates
- **quant_platform** — platform-side cluster/library DTOs map onto FA artifacts
