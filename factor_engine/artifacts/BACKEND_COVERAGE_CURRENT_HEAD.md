# Backend coverage — CURRENT HEAD (regenerated 2026-09-03)

> MACHINE-GENERATED from a live registry load at current HEAD.  Do not edit by
> hand.  This report is the P0#1 rebuild of `docs/BACKEND_COVERAGE.md`, which is
> STALE (it documents HEAD `f798b527…` / 2026-08-20 / **1624** canonicals from a
> different fact plane).  This file documents current HEAD `7628674b…`.

## Load plane (authoritative)

- **HEAD:** `7628674b95348cc3227881546a7d21ed8cd50881` (`jobs: rebuild_flagged_full_window_pages …`)
- **Working tree:** DIRTY (5 modified files + untracked prompt/artifacts/test —
  `data_access/runtime/resource_governor.py`, `data_access/store.py`,
  `cleaned_operators/polars_gap_coverage.py`, `cleaned_operators/rolling_pack.py`,
  `runtime/perf_config.py`; the two cleaned_operators edits are the R40/R13
  freeze/`PhysicalImplementationSpec` fixes required for a *second*
  `load_all()` in one process to succeed).
- **Load mode:** `load_all(include_research=False)` → the production surface,
  **1673 canonical**.  The default `include_research=True` load registers a
  further 67 research-extension module canonicals → **1740**; those 67 are the
  only difference between the two planes.
- **registry_fingerprint:** `14e9f4fd85d3b0a0` (computed over the frozen read
  state; sha256[:16]) — see `artifacts/preflight/inventory_manifest.json`.

## Counts (current HEAD, 1673 plane)

| metric | count |
|---|---:|
| total canonicals | **1673** |
| lifecycle: production | 137 |
| lifecycle: experimental | 1532 |
| lifecycle: deprecated | 1 |
| lifecycle: research | 3 |
| surface: daily | 1242 |
| surface: extended | 415 |
| surface: research | 5 |
| surface: internal | 3 |
| surface: unsafe | 7 |
| surface: legacy | 1 |
| scope: time_series | 820 |
| scope: elementwise | 493 |
| scope: fundamental_period | 136 |
| scope: cross_sectional | 95 |
| scope: session_intraday | 70 |
| scope: group | 59 |
| alias entries (registry `_aliases`) | 270 |
| `production_certified is True` (catalog six-gate) | 86 |
| `backend_passed is True` | 86 |
| `backend_passed False/None` | 1517 / 70 |
| `backend_signature_consistent True` | 1666 (7 False) |
| runtime impl: pandas_numpy only | 1 |
| runtime impl: polars only | 2 |
| runtime impl: pandas_numpy+polars | 1248 |
| runtime impl: pandas_numpy+polars+sql | 422 |

## Backend matrix (from `backend_matrix.json`; per canonical `backend_meta`)

86 catalog `production_certified` pandas slots carry
`certification_source = 'primitive_verified.json (R23 P1: …)'` (suffixed).
`docs/BACKEND_COVERAGE.md`'s old per-backend table (pandas 1622 implemented /
1603 selectable, polars 1623/1605) is NOT reproducible at current HEAD because
its generator read an older inventory plane and an older evidence state.

Current per-backend implementation counts (runtime `_operators` registry slots):

| backends present on canonical | canonicals |
|---|---:|
| pandas_numpy, polars | 1248 |
| pandas_numpy, polars, sql | 422 |
| polars | 2 |
| pandas_numpy | 1 |

## Direct-use ladder (R18 authority — `artifacts/preflight/direct_use_matrix.json`)

| metric | count |
|---|---:|
| mining_visible | 1610 |
| composition_usable | 1609 |
| terminal_allowed (direct_alpha + direct_alpha_high_cost) | 1446 |
| terminal_usable | 1445 |
| **production_admitted** | **0** |
| **directly_usable** | **0** |
| context_admitted | 0 |
| direct_alpha rows | 1397 |
| direct_alpha_high_cost rows | 49 |
| r64 causality: causal | 1566 |
| r64 causality: unknown | 91 |
| r64 causality: leak_detected | 16 |
| production_mining_allowlist (live `build_production_mining_allowlist()`) | **0** |

**production_admitted = 0 is the honest fail-closed state of the registry at
current HEAD, NOT an artifact bug.**  The DirectUse `production_admitted`
conjunct requires BOTH the physical-evidence gate
(`_has_physical_production_evidence` → `enumerate_physical_inventory()` rows
with `admission.admitted=True`, of which there are **0 of 1001** rows) AND
`context_admitted` (unit/availability/calendar/universe/source-context
declared — all absent).  Two candidate root causes were identified
(reported for the §110/GO risk register; NOT fixed here — P0 is read-only):

1. `backend/operator_capability._pandas_status()` compares the pandas
   `certification_source` string by EXACT equality against
   `{"primitive_verified.json", "factor_operator_verified.json"}`; live
   backend_meta stores a SUFFIXED source (`'primitive_verified.json (R23 P1:
   math/elementwise_math certification)'`) for all 86 certified pandas slots →
   status degrades to `implemented` → `production_eligible_backends()=()`
   → `allow_in_production=False`.
2. `evidence_artifact_valid()` = **False** (389 mismatches across **79**
   operators: `implementation_hash_pandas` 47, `implementation_hash_polars` 22,
   `implementation_hash_duckdb` 79, `parameter_signature_hash` 79,
   `semantic_contract_hash` 79, `test_source_hash` 79, 1 emitter-level) →
   `PRIMITIVE_BACKEND_EXECUTION_CERTIFIED` is empty at runtime → the
   evidence fallback path also yields no production-safe backend.

## Evidence state (current HEAD)

`evidence/primitive_verified.json` binds the 2026-08-28 code (HEAD `fa9b6eb`).
Current code has drifted in `cleaned_operators/common/elementwise.py`,
`backend/sql_pushdown/emitter.py` (duckdb emitter hash `cd74…`→`1809…`), test
sources and parameter/semantic contracts, so the physical-evidence validation
fails closed at HEAD.  `factor_engine/factor_engine/docs/operator_manifest.json`
(1737 operators, HEAD `fa9b6eb`, 2026-08-28) also records
`allow_in_production: 0` for every operator — the zero-admission state is
pre-existing and consistent, not an artifact of this rebuild.

## Old vs new `BACKEND_COVERAGE.md` — quantified differences

| dimension | OLD `docs/BACKEND_COVERAGE.md` | NEW (this file / current HEAD) |
|---|---|---|
| doc HEAD | `f798b527…` (2026-08-20) | `7628674b…` (2026-09-03) |
| load plane | 1624 canonical | 1673 canonical (`research=False`) |
| surface distribution | not stated | daily 1242 / extended 415 / … |
| production_admitted | 0 (old matrix) | 0 (fail-closed, root causes above) |
| per-backend rows | 4033 physical rows | runtime slots (see above) |
| evidence generation | stale provenance caveat | 389 mismatches / 79 operators, quantified |

## README / static catalog drift (listed, not fixed — later deliverable)

README static numbers (1737 canonical / 1421 DSL allowlist / daily 1238 /
extended 480) and `cleaned_operators/docs/operators_catalog.json`
(1737 / daily 1238 / extended 480, HEAD `fa9b6eb` 2026-08-28) predate current
HEAD.  Complete reconciliation (verified arithmetically):

- static **1737** = live1673 − 3 new canonicals (`avg2`, `fillna`,
  `ts_positive_streak`) + 67 research-extension canonicals (64 extended-surface
  experimental + 3 research-surface research; load only under
  `include_research=True`).
- live daily **1242** = static daily 1238 + 3 new + 1 migration
  (`ts_abs_concentration`: extended → daily).
- live extended **415** = static extended 480 − 64 research-extension − 1
  migration.
- DSL allowlist: `docs/dsl_allowlist.json` lists 1421 (schema
  `factor_engine.dsl_allowlist.v1`, `us/daily`); the LIVE
  `build_dsl_allowlist(surface='daily')` now yields **1483** — 68 names in live
  not in the static file, 6 in the static file not live (`WMA`, `ewm_corr`,
  `ewm_cov`, `fin_mad`, `ts_ewm_corr`, `ts_ewm_cov`).
- Live `build_production_mining_allowlist()` = 0; live
  `build_research_mining_allowlist()` = 1997 spellings.

## Honest caveats

- 70 canonicals carry `backend_passed = None` (undeclared); 7 carry
  `backend_signature_consistent = False` (`benchmark_relative_price`,
  `fin_cash_earnings_gap`, `float_share_ratio`, `free_float_share_ratio`,
  `intra_segment_amount_share`, `intra_segment_volume_share`, `ts_sharpe`).
- 91 canonicals have r64 causality `unknown`; 16 are `leak_detected`
  (list in `operator_matrix.csv`, column `prefix_causality_pass`).  All 86
  catalog `production_certified` canonicals are `causal`.
- `input_fields` is `[]` for ALL 1673 canonicals in the live catalog — field
  metadata lives in the DirectUse input-slot layer
  (`artifacts/preflight/direct_use_matrix.json` rows, `input_slots`), not in
  the catalog `input_fields` key.  `field_registry.json` records this
  truthfully.
- This report is a **view**; the authoritative artifacts are
  `artifacts/preflight/*.json` and
  `artifacts/operator_audit/current_head/operator_matrix.csv` (all carry the
  HEAD/dirty/registry fingerprint).
