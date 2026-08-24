# research_platform

Pure-stdlib research governance / artifact layer for the factor-mining
pipeline. **No dependency on any other package in `quant_projects`** — it only
imports `dataclasses`, `hashlib`, `enum`, `datetime`, `typing`, and the
standard library. It is self-contained and can be imported in isolation.

## Modules

| Module | Responsibility |
|--------|----------------|
| `artifacts.py` | Unified frozen `Artifact` model + 14 domain subclasses; `to_dict`/`from_dict` round-trip; SHA-256 `content_hash` over semantic fields. |
| `graph.py` | `ArtifactGraph` (lineage / dependents) + `DataInvalidationGraph` (reverse-lookup: "snapshot replaced → recompute what?"). |
| `campaign.py` | `ResearchCampaignArtifact` + `TrialLedger` (multiple-testing count, admission rate, per-generator breakdown). |
| `health.py` | `FactorHealthState` enum + `FactorHealthMonitor` (IC-decay / drift / stale / revive / failure thresholds). |
| `firewall.py` | `FactorCandidateArtifact` (provenance-only, **no chain-of-thought**) + `AlphaGenerationFirewall` (10-stage validation chain) + `FirewallResult`. |
| `scorecard.py` | `GeneratorScorecard`: aggregate pass rates, duplicate rates, novelty, incremental IC, cost per admitted, family diversity. |

## Artifacts

All artifacts are **frozen dataclasses**. The base `Artifact` carries identity
and provenance:

```
artifact_id, artifact_type, schema_version, content_hash,
producer, producer_version, source_refs, data_snapshot_ref,
universe_ref, policy_ref, environment_ref, created_at
```

`content_hash` is the SHA-256 over all **semantic** fields (everything except
`artifact_id`, `content_hash` and `created_at`). Two artifacts with identical
semantics hash identically regardless of id or creation time — that is what
makes cheap result-identity duplicate detection possible.

Subclasses: `DataSnapshotArtifact`, `FactorDefinitionArtifact`,
`FactorValueArtifact`, `EvaluationBundle`, `SimilarityArtifact`,
`AdmissionDecisionArtifact`, `FactorSetArtifact`, `FeatureBundle`,
`ModelDatasetArtifact`, `ModelTrainingArtifact`, `BacktestArtifact`,
`PortfolioArtifact`, `RiskSnapshotArtifact`, `ExpectedReturnArtifact`.

All datetimes are timezone-aware (UTC); naive datetimes are rejected.

## Lineage & invalidation (`graph.py`)

```
ArtifactGraph.add(dependent, *sources)     # edge source → dependent
lineage(id)      -> transitive ancestors, oldest first
dependents(id)   -> node + everything downstream
```

`DataInvalidationGraph` adds a snapshot→artifact index. When a data snapshot is
replaced, `mark_invalid(snapshot_ref)` returns every artifact (and its full
downstream) that must be recomputed, in dependency order.

## Campaigns & multiple testing (`campaign.py`)

`TrialLedger.append(trial_id, generator_type, admitted, stage)` is the raw
running record. `export_campaign()` snapshots the ledger into a frozen
`ResearchCampaignArtifact` with `candidate_count`, `failed_trials`,
`admitted_trial_ids`, `admission_rate`, and `content_hash`. This directly
exposes the **multiple-testing multiplicity** that must be corrected for when
evaluating admission significance.

## Factor health (`health.py`)

`FactorHealthMonitor.update(factor_id, snapshot, at)` keeps a bounded rolling
(ts, IC) history per factor and returns one of: `HEALTHY`, `DECAYING`,
`DRIFTED`, `STALE`, `REVIVED`, `FAILED`. Thresholds are configurable
(`stale_after_days`, `decay_ratio`, `failure_threshold`, `drift_floor`,
`history_keep`, `decay_window`).

## Firewall (`firewall.py`)

`FactorCandidateArtifact` stores **only** the research *hypothesis*, the
*operations* (formula + mutation operator), and full *provenance* (generator,
`prompt_hash`, `system_prompt_hash`, model, temperature, seed, retrieval
sources, tool versions). It deliberately does **not** store chain-of-thought or
any private reasoning trace.

`AlphaGenerationFirewall.check(candidate, context)` runs 10 stages and returns
a full (non-short-circuited) `FirewallResult` list:

1. `grammar_legality`
2. `type_dimension`
3. `field_availability`
4. `pit_future_leakage_static`
5. `complexity_budget`
6. `formula_exact_duplicate`
7. `result_identity_duplicate`
8. `cheap_fingerprint_novelty`
9. `compute_cost_estimate`
10. `fe_compilation` (default passes `True`; inject `compile_checker` for a
    real compiler)

`check_and_report()` returns a summary dict with `passed`, `passed_stages`,
`failed_stages`, and per-stage results.

## Scorecard (`scorecard.py`)

`GeneratorScorecard.record(...)` is fed one event per candidate; `summary()`
aggregates: `proposal_count`, `legality_pass_rate`, `compile_pass_rate`,
`exact_duplicate_rate`, `behavior_duplicate_rate`, `l0..l3_pass_rate`,
`final_admission_rate`, `mean_novelty`, `mean_incremental_ic`,
`compute_cost_per_admitted`, `wall_clock_cost`, `api_token_cost`,
`family_diversity`.

## Scope notes (deliberate skeletons)

- **FE compilation** (`fe_compilation`) defaults to `True` and must be injected
  with a real predicate for production. Marked clearly in the source.
- `type_dimension`, `pit_future_leakage_static`, and complexity/fingerprint
  checks are documented **heuristics**, not full static analyses; a production
  deployment should replace them with the FactorEngine operator type system and
  a real point-in-time field registry.

## Tests

```
cd /home/sunhaiwei/quant_projects
/tmp/fe2/bin/python -m pytest research_platform/tests -q -p no:cacheprovider
```

Coverage per module: artifact hash stability + round-trip + naive-datetime
rejection; graph lineage/dependents/cycle/invalidation; campaign counts &
duplicate rejection; health state transitions (healthy/stale/decaying/failed/
revived); firewall full-pass + each rejection path + injected compiler; scorecard
rates + divide-by-zero safety.
