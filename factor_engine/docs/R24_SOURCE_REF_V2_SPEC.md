# R24 SourceRef v2 Spec

R24-136..140: SourceRef v2 carries the market-scoped semantic identity; v1 refs decode
back-compat but NEW production encodes emit v2.

## v2 fields
market, concept_id, field_id, provider_id, dataset, timeframe, temporal_policy_digest,
catalog_hash, source_version.

## v1 back-compat
A v1 ref (no market) decodes to a spec with is_v2=False and None identity fields.

## Production decode strictness (R24-139/140)
decode_source_ref_production(name) rejects:
- a non-SourceRef string;
- an unknown transform;
- unknown/invalid transform params;
- an un-approved dialect_version.

## Cross-market collision (R24-241)
A StockIncome.revenue vs US StockIncome.revenue encode to DIFFERENT identities.
