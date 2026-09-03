# P1 Operator-Family Correctness Audit

Generated: 2026-09-03 · HEAD `dbbc3fe75f38d75c88f62ad60e3e0eedc666f026` (dirty tree)
Scope: 62 terminal agent-direct admitted operators (agent_direct_allowlist.json) + intraday + rolling families.
Dimensions: alias coherence · DSL coverage · hot-path · evidence drift.

## Findings

| id | family | dimension | severity | status |
|----|--------|-----------|----------|--------|
| P1-01 | all (270 aliases) | alias | info | open (verified-clean) |
| P1-02 | ts_beta/ts_topk*/ts_bottomk*/MACD_line | alias | info | open (verified-clean) |
| P1-03 | all 62 terminal | DSL | info | open (verified-clean) |
| P1-04 | intraday daily_agg | hot-path | info | open (verified-clean) |
| P1-05 | primitive evidence | evidence | **high** | **fixed** |

### P1-01 — alias coherence (270 live aliases)
- 0 dangling (alias→missing canonical), 0 self-referential, 0 alias==canonical collision.
- All 32 alias-bearing terminal operators: every alias resolves to the same canonical via `resolve_canonical` (0 mismatches).

### P1-02 — phys-vs-DSL alias inventory mismatch (6 canonicals)
Preflight `alias_inventory.json` flagged 6 canonicals where physical `aliases` differ from `dsl_aliases_pointing_here`:
`MACD_line`, `ts_beta`, `ts_bottomk_mean`, `ts_bottomk_sum`, `ts_topk_mean`, `ts_topk_std`.
Verified live: all 6 are **additive** — the extra DSL aliases (`Beta`, `ts_macd`, `m_top_n_avg`, `m_bottom_n_avg`, `m_top_n_std`, `m_bottom_n_sum`, `ts_rolling_beta`) resolve correctly to the canonical. No impl divergence, no drift. The preflight inventory is simply a superset view (DSL aliases registered in `_aliases.py`/`_dedupe.py` beyond the physical `aliases` field). **No fix needed.**

### P1-03 — DSL coverage (62 terminal)
- All 62 `resolve_canonical(canonical)` OK; all terminal aliases resolve to correct canonical (0 mismatches).
- Sample DSL round-trips (`add`, `abs`, `log`, `sqrt`, `ts_mean`, `ts_std`, `rank`, `group_neutralize`) parse to `CleanedCall` via `api/dsl_parser.parse_expr`.
- All 62 terminal are `surface=daily` (in `DAILY_CANONICALS`).

### P1-04 — hot-path (intraday daily_agg vector kernels)
- `perf_vec_equiv` harness: **6/6 PASS** (session_mean_reversion, price_delay, volume_imbalance, dynamic_stock_graph_features, common_trading_intensity, time_above_vwap).
- `count_bound()=6` — all 6 whitelisted kernels bound at import (`intraday/__init__.py` calls `bind_whitelist()`).
- `_core.daily_agg{,_two,_three}` dispatch to `_vec_*` fast path when `__vec__` present; scalar per-(inst,day) loop is the honest fallback.
- `perf_vec_bench.py` passes `fn` directly (not a lambda) so the vec leg genuinely times the vector path (100k GO §6.2 fix already in tree).

### P1-05 — evidence drift (FIXED)
- Spot-check 10/72 operators claiming `primitive_verified` certification: **0 drift** (all have complete hash evidence in `evidence/primitive_verified.json`).
- 48/62 terminal have `primitive_verified.json` evidence; 14/62 intraday have `evidence/intraday_minute_parity.json` evidence (all 14 present, `status=certified`).
- **Real drift found:** committed `evidence/primitive_case_registry.json` was stale — `fillna` has a live polars/duckdb parity case (`test_production_safe_bulk_parity.py`) but was missing from the 3 case-registry lists that already carried `fillna_const`. `test_primitive_case_registry_matches_cases` failed on the missing `fillna`.
- **Fix:** added `fillna` to `polars_reference_parity` / `duckdb_reference_parity` / `duckdb_real_sql_verified` only. `avg2`/`ts_positive_streak` were NOT added (pytest daily filter drops them — not in merged case list under staging; adding would over-declare and re-break the six-way intersection).

## Fixes applied
- `evidence/primitive_case_registry.json` — added `fillna` to 3 lists (evidence drift).

## Evidence YAMLs
- `evidence/r2/P1_FAMILY_AUDIT_EVIDENCE_DRIFT.yaml`

## Test results
- `tests/backend/test_primitive_evidence_contract.py`: **4 passed, 1 skipped** (primitive_verified.json not regenerated), 0 failed.
- `tests/backend/test_parameter_aliases.py` + `tests/backend_parity/test_alias_semantic_equivalence.py` + `tests/backend/test_primitive_evidence_contract.py`: **15 passed, 1 skipped**, 0 failed.
- `perf_vec_equiv` harness: **6/6 PASS**.

## Open items
- `primitive_verified.json` not regenerated (would require running the full certify pipeline; out of scope for this audit — the case-registry fix is the contract-level drift).
- 14 intraday terminal ops rely on `intraday_minute_parity.json` (not `primitive_verified.json`) — evidence binding present, no drift.
