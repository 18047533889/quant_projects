# R2 Master Queue (compact)

**Updated:** 2026-08-20  
**Loop:** [`orchestration.md`](orchestration.md)  
**Workflow:** `DISCOVERED → FIXING → TESTING → REVIEW → CLOSED`  
**Ticket line:** `ID | state | owned_paths | pytest_cmd | evidence`  
**History:** evidence YAMLs only; do not load archives.

## Active / open (pointers)

| ID | Status | Pointer |
|---|---|---|
| Q2 residency / PlanNode / verifier | LOCAL focused | `Q2-P0-009`, `Q2-P0-019`, `Q2-P0-020*`, `Q2-P0-021-022*`, `Q2-P0-002-verifier*` |
| Q `ts_beta` | quarantined | `Q2-P0-008.yaml` |
| POL2 moment/kurt / rolling | LOCAL focused | `POL2-P0-002`…`004` YAMLs |
| DA catalog identity / PIT | LOCAL focused | `DA2_P0_002_MANIFEST.yaml`, `R2-P0-039_MANIFEST.yaml` |
| FE multi-region physical | LOCAL focused | `R2-P0-059-062-063_manifest.yaml` |
| FA2 adapter / assembly | LOCAL focused | `FA2-P0-003.yaml`, `FA2-P0-004.yaml` |
| QE cache / streaming / mean_ic | LOCAL focused | `quant_evaluator/tests/test_*` |
| CI2 / live COS / real Redis / broad QE | `NOT_RUN` | — |

## QE post-P0 cleanup (2026-08-20 coverage-gap audit — open)

| ID | sev | item |
|---|---|---|
| ~~QE-P1-27~~ | closed | `_ic_wrapper` follows `spec.ic_method` (spearman on rank-family specs) — landed 2026-08-20, `tests/test_p1_cleanup.py` 15 passed |
| ~~QE-P1-28~~ | closed | NaN-safe first-occurrence-min loop in `max_drawdown_idx` — landed 2026-08-20 |
| ~~QE-P1-29~~ | closed | observation_count bases documented + pinned (mean_ic=480 / ic.pearson.mean=480 / pearson_ic=40) — landed 2026-08-20 |
| ~~QE-P1-30~~ | closed | stale `build/lib/quant_evaluator` deleted (2026-08-20); `dist/*.whl` still v-pre-batch — **rebuild wheel before any distribution** |
| ~~QE-P2-a~~ | closed | dead branch + unused import removed 2026-08-20 |
| ~~QE-P2-b~~ | closed | `compute_coverage` DeprecationWarning landed; `quick_reference` does not exist as module; `examples/` scripts standalone (no import fix needed); verified 2026-08-20 |
| ~~QE-P2-c~~ | closed | `compute_per_time_coverage` one-liner over `_valid_pair_mask` |
| ~~QE-P2-d~~ | closed | global RNG eliminated: `verify_parallel_implementation.py` → local `default_rng` (script re-run green); `tests/test_diagnosis.py` 11×randn → module `default_rng(20260820)` (32 passed); `tests/test_batch_plan.py` 1× (13 passed); residual global RNG only in standalone benchmark/verify_gpu scripts (not imported by tests) |
| ~~QE-P2-e~~ | closed | full-suite order-dependence fixed at root; OOM root cause = polars perf tests 252×3000×1000 shrunk to 252×500×100; full suite **1540 passed** 2026-08-20 |

Verified consistent (no action): coverage registry→adapter→per-factor path; min_assets day-count vs ic.py min_obs; all compute_coverage callers unpack 3-tuple.

## Closed-enough locally (not production)

Older R2-P0-032/033/036/037 and OPT2-P0-001 remain regression-tested historically; broad DA / live COS still `NOT_RUN`. See archive if needed.

## Truth

No surface is `CLOSED_VERIFIED` for production. Q remains fail-closed and not production-certified.
