# R18 Delete / remediation plan

Fingerprint: `0d22dc674d0e3e6d1437c2c67e801ace7bc4cc37` (dirty=True)

**30** canonicals to delete / move.

| canonical | status | replacement | delete reason |
|---|---|---|---|
| acos | move_internal | - | generic raw arccos — no factor semantics; use acos_bounded on a bounded input |
| arg | move_internal | - | DSL/grammar primitive — internal supporting layer |
| asin | move_internal | - | generic raw arcsin — no factor semantics; use asin_bounded on a bounded input |
| circulating_cap_unlock_proxy | move_internal | - | internal supporting layer — not public mining |
| constant | move_internal | - | DSL/grammar primitive — internal supporting layer |
| cos | move_internal | - | generic raw cosine — no factor semantics; use cos_phase inside a phase/seasonality recipe |
| cosh | move_internal | - | hyperbolic cosine — raw math, not a factor canonical |
| cot | move_internal | - | cotangent — raw math, not a factor canonical |
| csc | move_internal | - | cosecant — raw math, not a factor canonical |
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
| sec | move_internal | - | secant — raw math, not a factor canonical |
| sin | move_internal | - | generic raw sine — no factor semantics; use sin_phase inside a phase/seasonality recipe |
| sinh | move_internal | - | hyperbolic sine — raw math, not a factor canonical |
| tan | move_internal | - | tangent — raw math, not a factor canonical |
| ts_dmd_dominant_frequency | move_internal | - | untyped DMD — compat alias; use the ts_dmd_level/return typed variants |
| ts_dmd_dominant_growth_rate | move_internal | - | untyped DMD — compat alias; use the ts_dmd_level/return typed variants |
| ts_dmd_mode_concentration | move_internal | - | untyped DMD mode concentration — use the ts_dmd_level/return typed variants |
| valuation_pcf_definition_gap | move_internal | - | internal supporting layer — not public mining |
| valuation_pe_ttm_lyr_gap | move_internal | - | internal supporting layer — not public mining |
