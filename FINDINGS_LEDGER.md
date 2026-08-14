# R2 Findings Ledger

**Local HEAD reviewed:** `d78ed761b7d098d27e3acf916f3961d75096b3b3`
**Remote/GitHub evidence:** not used

## Confirmed P0

### R2-REC-001 — Q executor deleted while consumers remain

- Status: `FIXING`
- Evidence: `q_executor.py` is 51 lines at HEAD vs 372 at `HEAD^`.
- Reproduction: `PYTHONPATH=factor_engine python3 -c 'import backend.q_backend.q_backend'` raises missing `QExecutor`.
- Consumers still import `QExecutor`, `QExecutionFallbackPolicy`, and `get_q_executor`.
- Hard-coded PASS strings at lines 48-51 are not executable evidence.

### Q2-P0-001..005 — Q capability authority and gates are fake/disconnected

- Status: `REPRODUCED`
- `compute_q_capability_evidence()` sets compile/runtime/parity false/TODO.
- `QPhysicalImplementationRegistry` starts empty and is not populated.
- Empty registry makes missing-evidence gate vacuous.
- `gate_q_capability_single_authority` contains a literal `pass` and returns success.
- Compiler admission still uses `_operator_map`, not evidence authority.

### MODEL2-P0-006 — Modeling split-brain not closed

- Status: `PARTIALLY_FIXED`
- Approved narrowly: `CrossSectionalScaler.transform(..., apply_start_time=None)` rejects; 21 focused tests passed.
- Rejected closure: both packages retain executable contracts and preprocessing logic.
- Gap enforcement misses train→test when no validation split, accepts negative `gap_days`, and does not enforce validation→test gap.

### ABI2-P0-001 — return_decomp hard-gate test has wrong oracle

- Status: `REJECTED`
- Focused suite: 13 passed, 1 failed.
- For preclose=100, open=105, close=110, overnight=5%, intraday=4.7619%, total=10% multiplicatively.
- Test incorrectly expects 15.5%; production formula must not be changed to satisfy it.

### QE2-P0-003 — singleton shape collapse

- Status: `FIXING`
- `assign_quantiles_batch(np.ones((1,1,1))).shape == ()` due to unrestricted `.squeeze()`.
- Expected documented shape retains T/N axes.

## Narrowly Approved

### QE2-P0-001 — top-bottom orientation

- Status: `LOCAL_TESTED`
- Focused orientation suite: 7 passed.
- Highest quantile minus lowest quantile is now default.
- Broader QE certification remains open because singleton shape ABI is broken.

## Additional Confirmed Source Patterns

- Q backend assumes legacy/nonexistent PlanNode attributes.
- Registered Polars rolling statistics include formulas based on time-varying historical means.
- FactorAssets adapter imports `dataaccess` while local public package/API is different; it uses unsupported `DataRequest(source/start_date/end_date)` and `date.today()` defaults.

## Truth Rule

No item is `CLOSED_VERIFIED` until targeted regression, import smoke, independent review, and current-local-HEAD integration proof pass.
