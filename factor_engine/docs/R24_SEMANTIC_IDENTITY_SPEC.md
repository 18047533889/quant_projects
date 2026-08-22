# R24 SemanticIdentityDigest Spec

R24-132..134: a stable, order-independent digest of the semantic dimensions that CHANGE the economic
definition of a factor / cache / materialization identity.

## Included (change the economic definition)
market, concept_id, field_id, provider_id, dataset, unit_semantic, price_basis, flow_semantics,
period_duration, timeframe, universe_id, availability_policy, availability_precision,
availability_expr, knowledge_time_policy, revision_policy, source_vintage, snapshot_id,
missing_policy, group_fallback, temporal_model.

## Excluded (do NOT change the economic definition)
debug notes, description, cost estimate, display tags.

## Digest enters (R24-135)
IR canonical identity, Plan cache key, CSE key (where semantically relevant), FactorId,
MaterializationId, Checkpoint key, Incremental state key, Cold-start dedup key,
Mining candidate dedup.

## Collision tests (R24-205..209)
- fin_growth(revenue) under A YTD vs US quarterly vs US TTM -> DIFFERENT digests.
- rank(roe) under CSI300 vs All-A -> DIFFERENT digests.
- PubDate next-session vs exact same-day -> DIFFERENT digests.
- first_available vs latest_available_asof revision -> DIFFERENT digests.
- missing group -> nan vs global demean -> DIFFERENT digests.

## CSE split (R24-210)
StructuralMathHash (pure-math CSE sharing) vs SemanticExecutionHash (must NOT share materialized
results across different source-vintage / universe / availability contexts).
