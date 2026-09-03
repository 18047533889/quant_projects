# FactorEngine Operator Admission Summary

- total registered canonical = 1740

## Role (factor_role) distribution

- alpha = 1110
- alpha_high_cost = 99
- condition = 15
- denied = 15
- diagnostic = 22
- event = 38
- fundamental_pit = 144
- global_state = 11
- group_state = 22
- internal = 18
- intraday_eod = 149
- legacy = 1
- recipe_internal = 63
- research = 7
- source_transform = 11
- state = 15

## Target mining lane distribution

- denied = 33
- diagnostic_only = 22
- direct_daily = 1254
- direct_high_cost = 99
- direct_intraday_eod = 149
- direct_specialized = 101
- legacy_only = 1
- recipe_only = 63
- research_pending = 7
- source_transform = 11

## Hard blockers (quality defects)

- B01 PERMANENT_PIT_UNSAFE: 15
- B02 NON_FACTOR_OPERATOR: 122
- B08 IMPLEMENTATION_EVIDENCE_MISSING: 1654
- B09 SEMANTIC_GOLDEN_MISSING: 1654
- B10 TEMPORAL_PREFIX_MISSING: 1654
- B11 SOURCE_PIT_MISSING: 1654
- B13 EDGE_EVIDENCE_MISSING: 1654
- B14 BACKEND_EVIDENCE_MISSING: 1654
- B15 SOURCE_FIELD_MISSING: 1740
- B33 COST_CONTRACT_MISSING: 651

## Pools

- direct_daily: 1254
- direct_specialized: 101
- direct_intraday_eod: 149
- direct_high_cost: 99
- recipe_only: 63
- diagnostic_only: 33
- denied: 41

## Every canonical in a non-direct pool carries exact blocker(s) + a
recommended_action in operator_admission_matrix.json/csv — no vague
'not production' / 'research only' answers.
