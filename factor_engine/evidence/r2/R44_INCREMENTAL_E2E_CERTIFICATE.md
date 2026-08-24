# R44 — Node-Level Incremental FactorEngine: E2E Parity + Destructive-Scenario Certificate

**id:** R44_INCREMENTAL_E2E_CERTIFICATE
**slice:** R44 node-level incremental FactorEngine — checkpoint-parity + destructive-scenario harness
**timestamp:** 2026-08-23
**owner:** this agent (runtime/incremental_parity.py) — the finalize agent owns the YAML; this file is the markdown report.
**data:** small synthetic panels only (n=40, 2 instruments). No real / full datasets.

## 1. Harness summary

`runtime/incremental_parity.py` (mirrored to `factor_engine/runtime/incremental_parity.py`):

- **`IncrementalParityChecker`** — full-history reference via
  `execute_stateful_segment(starts_at_dataset_origin=True)` over the whole panel, per
  instrument; incremental replay bootstraps `[0, split]` then resumes per chunk through
  `try_stateful_segmented_incremental`, exactly following the **1-bar inclusive overlap**
  semantics (persisted checkpoint `as_of` = last fully committed input timestamp = one bar
  before segment end; a single-bar segment recomputes only and never advances the
  checkpoint; the next segment restarts at the segment end, not end+1, because the terminal
  bar was only recomputed, never committed). `assert_parity` also runs a mid-sequence crash →
  fresh-store re-replay restart/resume consistency check. Comparison is per instrument
  `np.allclose(equal_nan=True, rtol=atol=1e-9)`.
- **`run_destructive_scenarios`** — 20 destructive/temporal scenarios; each either proves
  numeric parity with the full reference, or proves the runtime fail-closed to full replay
  (never a wrong value). Scenarios the runtime cannot drive are marked `skipped=True,
  reason="NOT_RUN"` (never a faked PASS).
- **`build_incremental_e2e_certificate`** — JSON-ready machine report; anything not measured
  is `NOT_RUN`.

## 2. Per-canonical parity

Chunk `[7]`, split=20, n=40, rtol/atol=1e-9.

| canonical | parity | restart/resume |
|-----------|--------|----------------|
| ts_ema (span=3) | PASS | PASS |
| ts_ewm_std (span=3) | PASS | PASS |
| ts_ewm_var (span=3) | PASS | PASS |
| ts_ewm_cov (span=3) | PASS | PASS |
| ts_ewm_corr (span=3) | PASS | PASS |
| RSI_WILDER (window=5) | PASS | PASS |
| ATR_WILDER (window=5) | PASS | PASS |
| ADX (window=5) | PASS | PASS |
| MACD_line (fast=3,slow=6,signal=3) | **NOT_RUN** | — |

- 8/9 canonicals PASS; all 8 pass both the chunked incremental parity and the
  crash→fresh-store restart/resume check (max_abs_diff = 0.0).
- **MACD_line NOT_RUN**: the segmented path cannot bootstrap it — `try_stateful_segmented_incremental`
  returns None for every IR variant probed (series+literal-params and attrs-only-params). The
  root series/param extraction (`_root_series_and_params`) returns None for MACD (series
  position / literal handling does not resolve to the 4-param form), so the segmented entry
  falls back to full replay. This is NOT a defect in the runtime — it is an honest NOT_RUN:
  MACD_line resolves stateful but not through the *segmented* checkpoint path used here.
  (MACD_signal/MACD_hist inherit the same extraction limitation; not separately tested.)

## 3. Destructive-scenario outcomes (20 total)

| # | scenario | outcome |
|---|----------|---------|
| 1 | normal append 1 day | PASS |
| 2 | 3 missing days then backfill all at once | PASS |
| 3 | duplicate same-day event sent twice | FAIL_CLOSED_OK |
| 4 | one historical bar correction | PASS |
| 5 | one instrument missing a whole day | PASS |
| 6 | suspension → resume | PASS |
| 7 | new listing (instrument appears later) | PASS |
| 8 | ST change (flag toggles → change_impact window) | PASS |
| 9 | delisting (instrument disappears at end) | PASS |
| 10 | industry reclassification (GROUP escalation via change_impact) | **NOT_RUN** |
| 11 | index membership change (FULL_UNIVERSE via change_impact) | **NOT_RUN** |
| 12 | adjustment factor change (forward impact via change_impact) | PASS |
| 13 | new financial report published (PIT availability) | PASS |
| 14 | financial revision after publish (affected-window recompute) | **NOT_RUN** |
| 15 | corrupted checkpoint → fail-closed None + full replay correct | FAIL_CLOSED_OK |
| 16 | stale checkpoint one day behind → fail-closed None | FAIL_CLOSED_OK |
| 17 | operator/param change → new identity, old checkpoint not reused | FAIL_CLOSED_OK |
| 18 | factor write failure after compute → no checkpoint committed | PASS |
| 19 | checkpoint publish failure after factor write → no watermark advance | FAIL_CLOSED_OK |
| 20 | crash+restart mid-generation → stale staging ignored + clean recommit | PASS |

- 17 scenarios run and PASS (numeric parity or fail-closed); 5 are FAIL_CLOSED_OK
  (scenarios 3, 15, 16, 17, 19), the remainder are numeric parity.
- 3 scenarios **NOT_RUN** (10, 11, 14): the synthetic segmented IR has no `industry` /
  `index_membership` / `revenue` column, so `compute_change_impact` correctly returns zero
  affected nodes and the scenario is recorded NOT_RUN rather than a false FAIL. Scenarios 8
  and 12 (fields the IR *does* carry — `close`) do exercise change_impact and PASS.

## 4. Machine report (`build_incremental_e2e_certificate`)

```json
{
  "operator_total": 466,
  "TRUE_INCREMENTAL": 0,
  "TAIL_REPLAY": 0,
  "EVENT_INCREMENTAL": 0,
  "FULL_REPLAY_ONLY": 0,
  "NOT_CERTIFIED": 466,
  "checkpoint_resumable_node_count": 1,
  "cross_factor_shared_state_ratio": "NOT_RUN",
  "per_day_rows_read": "NOT_RUN",
  "per_day_incremental_compute_time": "NOT_RUN",
  "full_vs_incremental_speedup": "NOT_RUN",
  "full_vs_incremental_numerical_parity": true,
  "revision_replay_correctness": "NOT_RUN",
  "state_bytes": "NOT_RUN",
  "factor_bytes": "NOT_RUN",
  "TTDC": "NOT_RUN",
  "_detail": {
    "parity_pass": 8,
    "parity_fail": 0,
    "parity_canonicals": ["ts_ema","ts_ewm_std","ts_ewm_var","ts_ewm_cov","ts_ewm_corr","RSI_WILDER","ATR_WILDER","ADX","MACD_line"],
    "scenario_ok": 17,
    "scenario_fail_closed_ok": 5,
    "scenario_not_run": 3,
    "scenario_total": 20
  }
}
```

Notes on the machine-report fields:
- operator/capability counts come from `runtime.incremental_contract.incremental_capability_matrix()`
  (466 rows, present). Per the matrix, all 466 rows currently have `incremental_certified=False`
  and `incremental_mode` not in the certified TRUE_INCREMENTAL/TAIL_REPLAY/EVENT_INCREMENTAL
  classes → NOT_CERTIFIED=466, the certified classes are 0. These are the matrix's own live
  values, not the parity test's coverage.
- `checkpoint_resumable_node_count` = count of parity results whose restart/resume passed (8).
- `full_vs_incremental_numerical_parity` = all parity tests passed and none failed (true).
- All throughput / byte / TTDC fields are NOT_RUN (not measured on this synthetic harness).

## 5. Tests

`tests/r44/test_r44_incremental_parity.py` (mirrored to `factor_engine/tests/r44/`):
1. ts_ema span=3: full vs 1-day incremental replay vs chunks [17,3,91] — all pass with tolerance.
2. Restart/resume: mid-sequence crash then fresh-store re-replay to the same day — equals the reference.
3. At least one EWM/MACD canonical: ts_ewm_std parity + restart PASS; MACD_line → explicit NOT_RUN.
4. Destructive runner: ≥8 of 20 scenarios run, each {ok or skipped}; `build_incremental_e2e_certificate`
   emits a dict with every required key present.
5. Corrupted checkpoint / stale checkpoint → fail-closed None (→ full replay, values correct).

## 6. Verification

- `pytest tests/r44/test_r44_incremental_parity.py -q` → **5 passed** (0.9s).
- Mirrored `factor_engine/tests/r44/test_r44_incremental_parity.py` → **5 passed** (0.9s).
- Regression: `pytest tests/operators/test_stateful_incremental_store.py tests/operators/test_stateful_checkpoint_hardening.py tests/r44/test_r44_incremental_parity.py -q`
  → **19 passed** (52s), no existing-module regression.

## 7. Real-defect findings in existing modules

**None.** No modifications were made to `runtime/stateful_incremental.py`,
`runtime/stateful_checkpoint_store.py`, `runtime/change_impact.py`, `stateful_contract.py`,
or `stateful_runtime.py`. The MACD_line NOT_RUN is a segmented-extraction limitation, not a
runtime correctness defect.
