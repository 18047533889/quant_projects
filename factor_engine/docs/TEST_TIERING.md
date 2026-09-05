# Test Tiering (R61-P1 #55)

## Problem

The last full factor_engine suite run (2026-09-04) was `16478 passed / 4961 failed /
1321 skipped`. All 4961 failures were triaged as **pre-existing working-tree drift**
(test-vs-code signature drift, polars 1.42 API removals, auto-harness tests passing
insufficient parameters — see `artifacts/GO_NO_GO.md` R1). Because they sit in the
default `pytest factor_engine/tests` run, a genuine production regression can hide
inside the 4961.

## Solution: three additive tiers

| Tier | File | Meaning | Gate behavior |
|------|------|---------|---------------|
| `production_critical` | `factor_engine/tests/production_critical.txt` | Agent-direct operator admission, DA PIT, backend parity, run_many/batch, CSE, streaming sink, minute/session & compute_many, A-share contract gates, multiworker governance, evidence-integrity gate, multibackend execution. | **MUST be 100% green.** Any failure or missing list file ⇒ non-zero exit (fail-closed). |
| `research_extended` | `factor_engine/tests/research_extended.txt` | Known-fail / evidence-boundary / deep-API-drift node ids (each with a one-line reason after `#`). | Exits `0`; prints the exact failing list for audit. |
| `legacy_quarantine` | `factor_engine/tests/legacy_quarantine.txt` | Legacy / deprecated / auto-harness / registry-collection-error files (auto_polars_all 2469F, polars_structure 34F, cs_polars_native_batch1 + weighted_cs_operators collection errors, r28 all_canonicals_execute 89F, `_old` superseded twin, r28 evidence-current). | Exits `0`; still collected for reference. |

The three lists are **machine-generated** (one pytest path or node id per line,
`#` starts a reason comment) so they are complete and verifiable — see "How to
rebuild" below.

## Gate command (GO process)

```bash
factor_engine/scripts/check_production_critical.sh            # exit != 0 => BLOCK GO
factor_engine/scripts/check_production_critical.sh --workers 4 --junitxml-out /tmp/fe_gate
```

Equivalently, the Python dispatcher:

```bash
python factor_engine/scripts/run_test_tiers.py --gate production_critical --workers 4
python factor_engine/scripts/run_test_tiers.py --tier all          # full tiered pass
python factor_engine/scripts/run_test_tiers.py --tier research_extended --junitxml-out /tmp/fe_rx
```

`production_critical` failing (or a file in the list missing from disk) ⇒ gate exit
`2`/`1`. The default `pytest factor_engine/tests` run is **unchanged** — the tiers are
additive entrypoints.

## Demotion policy (never silently drop)

1. If a production_critical test fails, first decide: is it a **real product bug**?
   Fix it small (example this run: `ResourceLeakDetector` used a non-reentrant
   `Lock`, deadlocking `get_stats()`→`detect_leaks()`; `ts_mean`/`ts_std` marked the
   optional `min_periods`/`ddof` scalars `required=True`, making every windowed
   production plan demand a phantom second positional input).
2. Test drift vs current semantics → fix the test small (pandas `.group_by` →
   `.groupby`, `min_samples`→`min_periods`, duckdb `arrow()` returning
   `RecordBatchReader`, evidence at repo-root).
3. Deep / pre-existing / evidence-boundary (e.g. R37 parameter-domain store has zero
   `duckdb_sql` certification points while R40 `is_sql_production_safe` requires
   them) → **MOVE the specific node id to `research_extended.txt`** with a one-line
   reason, and record it in the changelog. Never silently drop a test.

## How to rebuild the lists (data-driven)

```bash
# 1. Enumerate:
.venv/bin/python -m pytest factor_engine/tests --co -q | sort > /tmp/fe_all_nodes.txt

# 2. production_critical seed terms (whole matching FILES when the file is clearly
#    production scope): grep tests for run_many, cse, streaming, session, pit,
#    contract_gate, multiworker, governance, parity, evidence, direct_use, backend,
#    capability, scheduler, telemetry, warmup, incremental, compute_many, minute,
#    fast_path, production, intraday, multibackend.

# 3. Run the seeded set; every red node is a finding. Fix (small) or demote with reason.

# 4. legacy_quarantine: tests/operators/ + deprecated/legacy/unsafe/placeholder
#    name matches plus the triaged pre-existing mass-failure files
#    (auto_polars_all, polars_structure, collection-error files).

# 5. research_extended: every demoted node id with its one-line reason.
```

## Current status (2026-09-05)

- `production_critical.txt`: **45 entries (files + green node ids)**.
- `research_extended.txt`: 46 demoted node ids, each with a one-line reason.
- `legacy_quarantine.txt`: 7 files.

### production_critical gate result (2026-09-05, real run)

```
OMP_NUM_THREADS=4 .venv/bin/python factor_engine/scripts/run_test_tiers.py \
  --gate production_critical --timeout 300 --workers 1
[gate:production_critical] running 45 entries ...
330 passed, 31 warnings in 318.83s (0:05:18)
[gate:production_critical] PASSED — all 45 entries green   (exit 0)
```

45 entries green end-to-end (exit 0, fail-closed gate). The 45 entries expand to
330 collected tests in the files/node-ids listed, all passing; 0 failed / 0 errors /
0 skipped-missing.

Real product bugs fixed while tiering (all in production_critical scope):
`multibackend_governance.ResourceLeakDetector` Lock→RLock deadlock;
`operator_signatures_phase1` ts_mean/ts_std optional-scalar arity;
`runtime.engine` init no longer builds `Analyzer(production=True, market=None)`
(raised immediately under R40 #174); `runtime.production_policy.assert_production_factors`
now threads the factor's market into `validate_production_dsl` + hash Analyzer.
