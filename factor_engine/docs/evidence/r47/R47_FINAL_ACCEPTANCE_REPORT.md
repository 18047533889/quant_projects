# FactorEngine R47 Final Acceptance Report

- git_sha: `5f44df633ee4d1763a5d2d1cdeb55eb579f1e4e5`
- implemented operators: 47
- registered: 47 / 47
- pit_safe: 47 / 47

## Hard gates

| gate | value |
|---|---|
| R47_ALL_IMPLEMENTED_REGISTERED | TRUE |
| R47_ALL_IMPLEMENTED_PIT_SAFE | TRUE |
| R47_ZERO_UNCLASSIFIED_SURFACE | TRUE |
| R47_EXTENDED_SURFACE | TRUE |
| R47_RESEARCH_MINING_VISIBLE | TRUE |
| R47_ZERO_PIT_CAUSALITY_AUDIT_ERRORS | TRUE |

## Backend matrix (honest)

- pandas reference: YES for all 47 operators
- polars: gap-coverage delegate (`polars_udf_pandas_delegate`), NOT claimed as native
- duckdb SQL: NO (complex intraday kernels; §6.3 SQL emitter != production certified)
- production-safe: NO (no six-gate evidence; fail-closed)

## Regression

- `pytest tests/operators/test_r47_state_ops.py test_r47_event_response.py test_r47_slice_profile.py test_r47_indicators_chip_panel.py` -> 224 passed
- pre-existing baseline failures (not caused by R47): `test_production_all_runtime_excludes_research_tools`, `test_registry_rejects_implicit_duplicate_after_bootstrap`
