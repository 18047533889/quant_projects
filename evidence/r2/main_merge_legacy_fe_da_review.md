# M20 bounded FE/DA legacy semantic review

Review date: 2026-09-09. Current source baseline named by the disposition:
`6b610d0c8be7a3e6a7516ee2f4f8de6b58765f4f`. The live working tree contains
uncommitted merge work, so the test result below certifies the loaded files,
not that immutable commit by itself.

Sources read in full for routing were
`LEGACY-CURRENT-DISPOSITION-20260909.json` and its original
`LEGACY-BRANCH-CONTENT-20260909.json` inventory. Byte inequality was used only
to select work; it is not evidence that behavior was lost. Historical refs
were not changed and no historical implementation was copied into main.

## Reviewed functional families

| Historical branch family | Current public authority | Counterexamples represented by current tests | Actual focused result | Disposition |
|---|---|---|---|---|
| `agent/harden-data-access-factor-engine`: deterministic key policy, strict parameter validation, guarded scans, FE data-scope identity | `data_access.read.KeyPolicy`, `arrow_table_to_multiindex_columns`, current request/scan contracts, and `factor_engine.storage.data_scope` | highest revision wins; equal/null revisions reject; invalid policy values reject; strict parameters and guarded scan handles; stable FE scope identity | Included in the combined main-only run below | **COVERED/REPLACED.** Current packages expose the behavior through reorganized public modules (`data_access.read.*`) rather than the historical top-level file layout. No concrete historical counterexample was found missing in this bounded family. |
| `agent/factor-engine-composite-fastpath`: fused composites and native backend parity | `factor_engine.cleaned_operators.load_all`, `OperatorRegistry`, retained `ATR_WILDER`/`RSI_WILDER` composite implementations | only intended fused canonicals remain active; recipe-replaced names do not re-register; Polars implementation contains no pandas bridge; retained composites match pandas numerically | Included in the combined main-only run below | **COVERED/REPLACED.** Current registry source/provenance and cross-backend parity tests are stronger than merely retaining the old files. No lost behavior was demonstrated. |
| `agent/gtja191-production-complete`: GTJA compatibility arithmetic and snapshot-cache behavior | `OperatorRegistry` public lookup plus current `factor_engine.storage.sources.data_access_source` | arg-extreme recent-tie and all-missing behavior; regression common-sample set; signed/zero rolling product; scalar comparisons; snapshot cache identity | Included in the combined main-only run below | **COVERED for tested counterexamples.** Current canonical implementations and source adapter retain the selected behaviors. This does not certify every GTJA191 formula or production dataset. |
| COS runtime/schema branches | Current `data_access.cos_*` contract, PIT, registry and storage APIs | typed-date fingerprint, panel/effective-date fail-closed, read-asof guard, registry axes, sparse/month/complete-marker layout | 11 passed; `test_cos_pit_runtime.py` separately failed collection because it imports missing bare module `store` | **PARTIAL.** Covered contract/storage cases remain; five PIT-runtime cases are not executable from the main test entry. This is a concrete test-integration gap, not proof of a numerical failure. |
| FE recipe/stateful branches | `FactorRecipeRegistry`, `RecipeCompiler`, `compile_recipe_plans`, stateful checkpoint APIs | unsafe syntax/cycles, shared DAG, deterministic IDs, planner bridge, fingerprint/schema/order rejection and segmented EMA/Wilder/MACD | Stateful and recipe negative cases passed; six recipe expansion/batch cases failed on the current three-backend evidence gate | **PARTIAL / TEST MIGRATION.** Current production policy is stricter than historical direct expansion. Tests need scoped certified evidence or non-production fixtures; the gate must not be bypassed. |
| LQTP canonical-surface branches | Current LQTP compatibility APIs, `OperatorRegistry`, `build_operator_spec` | alias identity, macros, source refs, ambiguity rejection, Chinese SMA arithmetic, scalar broadcast and canonicalized fastpath | 18 passed; two production-classification assertions failed | **PARTIAL / DECISION REQUIRED.** `ts_quantile` and `ts_sma_cn` currently resolve experimental and denied while retained tests expect production. Arithmetic compatibility is covered, but policy owners must decide whether certification or tests are stale. |

## Executed evidence

Command used the formal tree interpreter and only current-main tests:

```
.venv/bin/python -m pytest -q \
  data_access/tests/unit/test_key_policy_deterministic.py \
  data_access/tests/unit/test_params_validation_strict.py \
  data_access/tests/unit/test_scan_handle_guard.py \
  factor_engine/tests/test_data_scope_stability.py \
  factor_engine/tests/operators/test_composite_fastpath.py \
  factor_engine/tests/test_gtja_compat_semantics.py \
  factor_engine/tests/test_scalar_comparison_semantics.py \
  factor_engine/tests/test_data_access_source_snapshot_cache.py \
  -p no:cacheprovider
```

Result: **30 passed, 2 warnings in 28.64 s**. Both warnings were the same
production-mode notice for a test probe named `unknown` lacking a Polars
`PhysicalImplementationSpec`; no test skipped or failed.

Second combined main-only run: **38 passed, 8 failed, 4 warnings in 31.86 s**.
Failures are retained and classified in the table. Separate COS PIT-runtime
collection failed with `ModuleNotFoundError: No module named store`.

Consolidated SHA-256 for sorted `sha256sum` records of the ten reviewed current
implementation files (`data_access/cos_{contract,runtime,panel_runtime,
registry_runtime,storage_runtime}.py`, FE recipe compiler/registry/planner
bridge, `stateful_contract.py`, and `api/lqtp_compat.py`):
`f18f889629f1249fb3b8de53f2c216fd07663a13c882c2c0997ce01df3fbc4519`.
This identifies the selected loaded sources, not the dirty tree as a whole.

## Honest boundary / remaining manual review

This remains a bounded family review, not all 23 retained refs.
Operator overhaul, semantic hardening, final-audit manifests/evidence, and
operator-surface cleanup remain **MANUAL_REVIEW / NOT_RUN** here. In particular,
passing synthetic/current unit tests is not proof of COS credentials, real-data
PIT behavior, installed-wheel identity, CUDA behavior, or all historical
formulas. The disposition file must therefore retain overall
`NEEDS_SEMANTIC_REVIEW`; this note only closes the three scoped rows above.

## Migration rerun addendum

The routine fixture gaps were migrated without changing production admission.
COS PIT now imports `_period_selection_sql` from `data_access.store`. Recipe
positives run under explicit `FACTOR_ENGINE_EXPAND_RECIPE_USAGE=1` research
scope; a negative case removes that scope and proves production without
three-backend evidence is rejected. LQTP retains mathematical positives while
`ts_quantile` and `ts_sma_cn` remain production-denied until certified.

Actual rerun: **41 passed, 4 warnings, 0 failed/skipped in 32.19 seconds** with
the formal-tree `.venv/bin/python`. Earlier failures are superseded only for
these exact migrated tests.
