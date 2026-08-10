# R18 Direct-Use Matrix

Fingerprint: `e0d56b06ccc2f5ceaeda0870e460949533e3edaa` (dirty=True)

Registered canonicals: **1426**  |  retained DIRECT_*: **1368**

## DirectUseStatus histogram

| status | count |
|---|---|
| delete_no_data | 3 |
| delete_obsolete | 1 |
| direct_alpha | 1099 |
| direct_alpha_high_cost | 95 |
| direct_condition | 15 |
| direct_control_flow | 1 |
| direct_event | 38 |
| direct_global_state | 11 |
| direct_group_state | 22 |
| direct_intermediate | 58 |
| direct_recipe | 5 |
| direct_source_transform | 9 |
| direct_state | 15 |
| move_internal | 16 |
| research_tool | 38 |

## Delete / remediation plan

| canonical | status | replacement | reason |
|---|---|---|---|
| arg | move_internal | - | DSL/grammar primitive — internal supporting layer |
| circulating_cap_unlock_proxy | move_internal | - | internal supporting layer — not public mining |
| constant | move_internal | - | DSL/grammar primitive — internal supporting layer |
| cube | delete_obsolete | - | legacy surface — obsolete or aliased |
| fin_total_operating_accruals | delete_no_data | - | required concepts (depreciation, amortization) have no provider in either market |
| fin_ttm | move_internal | - | internal supporting layer — not public mining |
| holder_concentration_change | delete_no_data | - | implementation raises (requires distinct relation snapshots); no snapshot provider — delete until real data exists |
| holder_count_change_rate | delete_no_data | - | TopTen rows are not total shareholder count; no total-holder provider — delete rather than fake |
| holder_pledge_churn | move_internal | - | internal supporting layer — not public mining |
| identity | move_internal | - | internal supporting layer — not public mining |
| intra_negative_jump_variation | move_internal | - | internal supporting layer — not public mining |
| intra_positive_jump_variation | move_internal | - | internal supporting layer — not public mining |
| intra_return_profile_cosine | move_internal | - | internal supporting layer — not public mining |
| intra_signed_jump_ratio | move_internal | - | internal supporting layer — not public mining |
| protected_div | move_internal | - | internal supporting layer — not public mining |
| ts_dmd_dominant_frequency | move_internal | - | untyped DMD — compat alias; use the ts_dmd_level/return typed variants |
| ts_dmd_dominant_growth_rate | move_internal | - | untyped DMD — compat alias; use the ts_dmd_level/return typed variants |
| ts_dmd_mode_concentration | move_internal | - | untyped DMD mode concentration — use the ts_dmd_level/return typed variants |
| valuation_pcf_definition_gap | move_internal | - | internal supporting layer — not public mining |
| valuation_pe_ttm_lyr_gap | move_internal | - | internal supporting layer — not public mining |
