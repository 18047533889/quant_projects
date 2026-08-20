# R24 Mining Mode Authority Audit

R24-158..166: the mining validators run the PRODUCTION analyzer with a resolved market.

- validate_production_dsl now runs Analyzer(production=True, market=...) (R24-158/159).
- bare-field checks resolve through MULTI_MARKET_FIELD_REGISTRY.registry_for(market) (R24-160).
- the default typed mining search space takes an explicit market (R24-162/163).
- operator role resolution failures hard-fail search-space generation (R24-166).
- legacy fundamental presets require a TemporalSourceCertificate (R24-171/172).
- default mining presets are built via build_certified_mining_preset(market, concepts, context) (R24-173).
